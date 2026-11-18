import csv
import hashlib
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from kda_hybrid_lm.models import HybridLM
from .optim import build_optimizer


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batch_at(data, batch_size, length, generator, device):
    starts = torch.randint(len(data) - length, (batch_size,), generator=generator).tolist()
    chunks = []
    for start in starts:
        chunks.append(data[start:start + length + 1])
    chunks = np.stack(chunks).astype(np.int64)
    batch = torch.from_numpy(chunks).to(device)
    return batch[:, :-1], batch[:, 1:]


@torch.inference_mode()
def evaluate(model, data, length, batch_size, max_batches):
    """固定、不重叠的目标 token，按有效 token 数加权。"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    device = next(model.parameters()).device
    blocks = min((len(data) - 1) // length, batch_size * max_batches)
    if blocks == 0:
        raise ValueError("Validation data is shorter than one sequence")
    for start in range(0, blocks, batch_size):
        rows = []
        end = min(start + batch_size, blocks)
        for index in range(start, end):
            token_start = index * length
            row = np.asarray(data[token_start:token_start + length + 1])
            rows.append(row)
        batch = torch.from_numpy(np.stack(rows).astype(np.int64)).to(device)
        output = model(batch[:, :-1], labels=batch[:, 1:], shift_labels=False)
        count = batch.size(0) * length
        total_loss += output.loss.item() * count
        total_tokens += count
    model.train()
    nll = total_loss / total_tokens
    if not math.isfinite(nll):
        raise FloatingPointError("Nonfinite validation loss")
    return {"nll": nll, "ppl": math.exp(min(nll, 700)), "tokens": total_tokens}


class WeightTrace:
    """每个可训练参数张量一行；step=t 对应第 t 次更新后的权重和矩。"""

    def __init__(self, model, optimizer, path):
        self.weights = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.weights[name] = parameter
        self.optimizer = optimizer
        self.groups = {}
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                self.groups[id(parameter)] = group
        self.initial = torch.stack([p.detach().float().norm() for p in self.weights.values()])
        self.file = Path(path).open("w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.writer.writerow([
            "step", "parameter", "numel", "parameter_step", "has_grad",
            "norm", "norm_ratio", "relative_update", "rms", "pre_norm", "update_norm",
            "grad_norm", "grad_rms", "momentum_norm", "momentum_rms",
            "momentum_hat_norm", "momentum_hat_rms", "second_moment_norm",
            "second_moment_rms", "second_moment_hat_norm", "sqrt_second_moment_hat_rms",
            "adam_direction_norm", "adam_direction_rms", "effective_lr",
        ])
        self.write(0)

    def snapshot(self):
        return [p.detach().float().clone() for p in self.weights.values()]

    def _weight_statistics(self, before):
        norms = torch.stack([p.detach().float().norm() for p in self.weights.values()])
        ratio = torch.where(self.initial > 0, norms / self.initial, float("nan"))
        previous = torch.full_like(norms, float("nan"))
        delta = previous
        relative = previous
        if before is not None:
            previous = torch.stack([p.norm() for p in before])
            changes = []
            for parameter, old in zip(self.weights.values(), before):
                changes.append((parameter.detach().float() - old).norm())
            delta = torch.stack(changes)
            relative = torch.where(previous > 0, delta / previous, float("nan"))
        return norms, ratio, previous, delta, relative

    def _optimizer_statistics(self, parameter, norm):
        """梯度、Adam 一二阶矩、偏差校正后的方向，列顺序与旧 CSV 一致。"""
        group = self.groups[id(parameter)]
        state = self.optimizer.state.get(parameter, {})
        parameter_step = 0
        if state:
            parameter_step = int(state["step"].item())
        root_n = math.sqrt(parameter.numel())
        zero = norm.new_zeros(())
        nan = norm.new_full((), float("nan"))
        grad_norm = nan
        if parameter.grad is not None:
            grad_norm = parameter.grad.detach().float().norm()
        momentum_norm = zero
        second_moment_norm = zero
        momentum_hat_norm = nan
        second_moment_hat_norm = nan
        denominator_rms = nan
        direction_norm = nan
        if parameter_step:
            beta1, beta2 = group["betas"]
            momentum = state["exp_avg"].float()
            second_moment = state["exp_avg_sq"].float()
            momentum_hat = momentum / (1 - beta1 ** parameter_step)
            second_moment_hat = second_moment / (1 - beta2 ** parameter_step)
            momentum_norm = momentum.norm()
            second_moment_norm = second_moment.norm()
            momentum_hat_norm = momentum_hat.norm()
            second_moment_hat_norm = second_moment_hat.norm()
            denominator_rms = second_moment_hat.mean().sqrt()
            if parameter.grad is not None:
                denominator = second_moment_hat.sqrt() + group["eps"]
                direction_norm = (momentum_hat / denominator).norm()
        values = [
            grad_norm, grad_norm / root_n,
            momentum_norm, momentum_norm / root_n,
            momentum_hat_norm, momentum_hat_norm / root_n,
            second_moment_norm, second_moment_norm / root_n,
            second_moment_hat_norm, denominator_rms,
            direction_norm, direction_norm / root_n,
        ]
        return parameter_step, values

    @torch.no_grad()
    def write(self, step, before=None):
        norms, ratio, previous, delta, relative = self._weight_statistics(before)
        rows = []
        parameter_steps = []
        for index, parameter in enumerate(self.weights.values()):
            parameter_step, optimizer_values = self._optimizer_statistics(parameter, norms[index])
            parameter_steps.append(parameter_step)
            root_n = math.sqrt(parameter.numel())
            weight_values = [
                norms[index], ratio[index], relative[index],
                norms[index] / root_n, previous[index], delta[index],
            ]
            rows.append(torch.stack(weight_values + optimizer_values))

        # 统一搬到 CPU，保留原来的 GPU 同步边界和 CSV 列布局。
        values = torch.stack(rows).cpu().tolist()
        for (name, parameter), parameter_step, row in zip(self.weights.items(), parameter_steps, values):
            has_gradient = int(parameter.grad is not None)
            effective_lr = self.groups[id(parameter)]["lr"]
            self.writer.writerow([
                step, name, parameter.numel(), parameter_step, has_gradient,
                *row, effective_lr,
            ])
        self.file.flush()

    def close(self):
        self.file.close()


def _parameter_digest(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        values = parameter.detach().cpu().numpy().tobytes()
        digest.update(values)
    return digest


def _accumulate_gradients(model, train, batch_size, length, rng, device, accumulation, step):
    loss_sum = 0.0
    for _ in range(accumulation):
        inputs, targets = batch_at(train, batch_size, length, rng, device)
        output = model(inputs, labels=targets, shift_labels=False)
        loss = output.loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite loss at step {step}")
        scaled_loss = loss / accumulation
        scaled_loss.backward()
        loss_sum += loss.detach().item()
    return loss_sum


def _save_checkpoint(model, optimizer, config, metadata, steps, rng, path):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": asdict(config),
        "metadata": metadata,
        "step": steps,
        "torch_rng": torch.get_rng_state(),
        "data_rng": rng.get_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    torch.save(checkpoint, path)


def run_training(config, train, valid, output_dir, *, kind, lr, seed, shuffle_seed,
                 steps, batch_size, accumulation, length, eval_every, eval_batches,
                 device, trace_every=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model = HybridLM(config).to(device)
    digest = _parameter_digest(model)
    optimizer, mapping = build_optimizer(model, kind, lr, shuffle_seed)
    rng = torch.Generator().manual_seed(seed + 100000)
    metadata = dict(model=asdict(config), optimizer=kind, lr=lr, seed=seed,
                    shuffle_seed=shuffle_seed, steps=steps, batch_size=batch_size,
                    accumulation=accumulation, length=length, device=str(device),
                    initial_sha256=digest.hexdigest(), scales=mapping,
                    torch_version=str(torch.__version__), precision="float32",
                    trace_every=trace_every, weight_decay=0,
                    data_order_seed=seed + 100000)
    (output_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    trace = WeightTrace(model, optimizer, output_dir / "weights.csv")
    start_time = time.perf_counter()
    tokens = 0
    try:
        with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
            baseline = evaluate(model, valid, length, batch_size, eval_batches)
            log.write(json.dumps({"step": 0, "tokens": 0, "validation": baseline}) + "\n")
            for step in range(1, steps + 1):
                optimizer.zero_grad(set_to_none=True)
                loss_sum = _accumulate_gradients(
                    model, train, batch_size, length, rng, device, accumulation, step,
                )
                record = step <= 20 or step % trace_every == 0 or step == steps
                before = None
                if record:
                    before = trace.snapshot()
                optimizer.step()
                if record:
                    trace.write(step, before)
                tokens += batch_size * length * accumulation
                row = {"step": step, "tokens": tokens, "train_loss": loss_sum / accumulation}
                if step % eval_every == 0 or step == steps:
                    metrics = evaluate(model, valid, length, batch_size, eval_batches)
                    row["validation"] = metrics
                    print(f"{kind} seed={seed} lr={lr:g} step={step}/{steps} "
                          f"loss={row['train_loss']:.4f} val={metrics['nll']:.4f}", flush=True)
                log.write(json.dumps(row) + "\n")
                log.flush()
            elapsed = time.perf_counter() - start_time
            result = dict(optimizer=kind, lr=lr, seed=seed, shuffle_seed=shuffle_seed,
                          tokens=tokens, seconds=elapsed, tokens_per_second=tokens / elapsed,
                          validation=metrics, initial_sha256=digest.hexdigest(), path=str(output_dir))
            _save_checkpoint(
                model, optimizer, config, metadata, steps, rng, output_dir / "last.pt",
            )
            (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    finally:
        trace.close()
