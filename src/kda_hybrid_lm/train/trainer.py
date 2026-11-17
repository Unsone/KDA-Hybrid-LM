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
    chunks = np.stack([data[s:s + length + 1] for s in starts]).astype(np.int64)
    batch = torch.from_numpy(chunks).to(device)
    return batch[:, :-1], batch[:, 1:]


@torch.inference_mode()
def evaluate(model, data, length, batch_size, max_batches):
    """固定、不重叠的目标 token，按有效 token 数加权。"""
    model.eval()
    total_loss = total_tokens = 0
    device = next(model.parameters()).device
    blocks = min((len(data) - 1) // length, batch_size * max_batches)
    if blocks == 0:
        raise ValueError("Validation data is shorter than one sequence")
    for start in range(0, blocks, batch_size):
        rows = [np.asarray(data[i * length:i * length + length + 1])
                for i in range(start, min(start + batch_size, blocks))]
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
        self.weights = {n: p for n, p in model.named_parameters() if p.requires_grad}
        self.optimizer = optimizer
        self.groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
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

    @torch.no_grad()
    def write(self, step, before=None):
        norms = torch.stack([p.detach().float().norm() for p in self.weights.values()])
        ratio = torch.where(self.initial > 0, norms / self.initial, float("nan"))
        previous = delta = relative = torch.full_like(norms, float("nan"))
        if before is not None:
            previous = torch.stack([p.norm() for p in before])
            delta = torch.stack([(p.detach().float() - old).norm()
                                 for p, old in zip(self.weights.values(), before)])
            relative = torch.where(previous > 0, delta / previous, float("nan"))
        rows = []
        parameter_steps = []
        for i, p in enumerate(self.weights.values()):
            group = self.groups[id(p)]
            state = self.optimizer.state.get(p, {})
            t = int(state["step"].item()) if state else 0
            parameter_steps.append(t)
            root_n = math.sqrt(p.numel())
            zero = norms.new_zeros(())
            nan = norms.new_full((), float("nan"))
            grad = p.grad.detach().float().norm() if p.grad is not None else nan
            m_norm = v_norm = zero
            m_hat_norm = v_hat_norm = denominator_rms = direction_norm = nan
            if t:
                beta1, beta2 = group["betas"]
                m, v = state["exp_avg"].float(), state["exp_avg_sq"].float()
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                m_norm, v_norm = m.norm(), v.norm()
                m_hat_norm, v_hat_norm = m_hat.norm(), v_hat.norm()
                denominator_rms = v_hat.mean().sqrt()
                if p.grad is not None:
                    direction_norm = (m_hat / (v_hat.sqrt() + group["eps"])).norm()
            rows.append(torch.stack([
                norms[i], ratio[i], relative[i], norms[i] / root_n, previous[i], delta[i],
                grad, grad / root_n, m_norm, m_norm / root_n,
                m_hat_norm, m_hat_norm / root_n, v_norm, v_norm / root_n,
                v_hat_norm, denominator_rms, direction_norm, direction_norm / root_n,
            ]))
        # 批量搬到 CPU，避免对每个统计量单独触发 GPU 同步。
        values = torch.stack(rows).cpu().tolist()
        for (name, p), t, row in zip(self.weights.items(), parameter_steps, values):
            self.writer.writerow([step, name, p.numel(), t, int(p.grad is not None),
                                  *row, self.groups[id(p)]["lr"]])
        self.file.flush()

    def close(self):
        self.file.close()


def run_training(config, train, valid, output_dir, *, kind, lr, seed, shuffle_seed,
                 steps, batch_size, accumulation, length, eval_every, eval_batches,
                 device, trace_every=1):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model = HybridLM(config).to(device)
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
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
                loss_sum = 0.0
                for _ in range(accumulation):
                    x, y = batch_at(train, batch_size, length, rng, device)
                    loss = model(x, labels=y, shift_labels=False).loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at step {step}")
                    (loss / accumulation).backward()
                    loss_sum += loss.detach().item()
                record = step <= 20 or step % trace_every == 0 or step == steps
                before = trace.snapshot() if record else None
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
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": asdict(config), "metadata": metadata, "step": steps,
                        "torch_rng": torch.get_rng_state(), "data_rng": rng.get_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                       output_dir / "last.pt")
            (output_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            return result
    finally:
        trace.close()
