"""固定步长的矩阵 × RMSNorm 四组机制实验；默认补跑 B/C，复用配对 A/D。"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from kda_hybrid_lm.models import HybridLM, ModelConfig
from kda_hybrid_lm.modules.norm import RMSNorm
from kda_hybrid_lm.train.trainer import evaluate, run_training, seed_all
from kda_hybrid_lm.train.ablation import ARMS, make_contrasts, plot_contrasts


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def parameter_hash(model):
    digest = hashlib.sha256()
    for name, p in model.named_parameters():
        digest.update(name.encode())
        digest.update(p.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def verify_reference(reference, data_dir):
    """先核对数据、模型、预算、逐张量 LR 与实际初始化，再开始长训练。"""
    launch = read_json(reference / "launch.json")
    if launch["mode"] != "compare":
        raise ValueError("Reference must be a completed real-data compare experiment")
    data = read_json(data_dir / "metadata.json")
    old_data = read_json(reference / "data-metadata.json")
    for key in ("vocab_size", "token_counts", "sha256", "tokenizer_sha256"):
        if data[key] != old_data[key]:
            raise ValueError(f"Prepared data differs from reference: {key}")
    arrays = {}
    for split, expected in data["sha256"].items():
        tokens = np.load(data_dir / f"{split}.npy", mmap_mode="r")
        if hashlib.sha256(tokens).hexdigest() != expected:
            raise ValueError(f"Token file hash differs from reference: {split}")
        arrays[split] = tokens
    baseline = {(r["optimizer"], r["seed"]): r for r in read_json(reference / "summary.json")}
    config = None
    expected_tokens = launch["steps"] * launch["batch_size"] * launch["accumulation"] * launch["length"]
    checks = []
    for seed in launch["seeds"]:
        hashes = []
        for arm in ("A", "D"):
            kind, lr = ARMS[arm]
            row = baseline[kind, seed]
            folder = reference / "repeat" / f"{kind}-seed{seed}"
            old = read_json(folder / "config.json")
            candidate = ModelConfig(**old["model"])
            if config is None:
                config = candidate
            if candidate != config or config.bias:
                raise ValueError("Reference must use the same bias-free model in every run")
            if old["torch_version"] != str(torch.__version__):
                raise ValueError("Use the original PyTorch version: " + old["torch_version"])
            for key in ("seed", "steps", "batch_size", "accumulation", "length", "trace_every"):
                expected = seed if key == "seed" else launch[key]
                if old[key] != expected:
                    raise ValueError(f"Reference config mismatch: {key}")
            if (row["lr"] != lr or old["lr"] != lr or row["tokens"] != expected_tokens
                    or old["data_order_seed"] != seed + 100000 or old["precision"] != "float32"
                    or old["weight_decay"] != 0):
                raise ValueError("Reference A/D must use the planned fixed LR, data order and budget")
            seed_all(seed)
            model = HybridLM(config)
            digest = parameter_hash(model)
            if digest != old["initial_sha256"] or digest != row["initial_sha256"]:
                raise ValueError(f"Initial parameters differ from reference: {kind} seed={seed}")
            for name, scale in old["scales"].items():
                expected_scale = 1.0 if arm == "A" else scale["initial_rms"]
                if scale["scale"] != expected_scale:
                    raise ValueError(f"Unexpected reference scale: {kind}/{name}")
            # 所有 RMSNorm 初值为 1，D 才对应固定 Norm LR=0.03。
            for module in model.modules():
                if isinstance(module, RMSNorm) and not torch.equal(module.weight, torch.ones_like(module.weight)):
                    raise ValueError("Reference RMSNorm must initialize to all ones")
            hashes.append(digest)
            del model
        if len(set(hashes)) != 1:
            raise ValueError(f"Unpaired reference initialization: seed={seed}")
        checks.append(dict(seed=seed, initial_sha256=hashes[0], reference_tokens=expected_tokens))
    return launch, config, arrays, baseline, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tinystories")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--arms", choices=list(ARMS), nargs="+", default=["B", "C"])
    parser.add_argument("--smoke", action="store_true", help="Synthetic data, all four arms, CPU by default")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if len(set(args.arms)) != len(args.arms):
        parser.error("--arms must not contain duplicates")
    reference = args.reference.resolve() if args.reference else None
    if not args.smoke and reference is None:
        parser.error("Real-data runs need --reference to verify pairing")
    if args.smoke:
        launch = dict(steps=8, batch_size=2, accumulation=1, length=32,
                      eval_every=4, eval_batches=2, trace_every=1, threads=4,
                      seeds=[42,43], shuffle_seed=1000)
        config = ModelConfig(vocab_size=256, hidden_dim=32, intermediate_dim=80,
                             num_heads=4, layer_types=("softmax",) * 2)
        text = b"Once upon a time, a bird shared an apple with a friend.\n"
        arrays = {split: np.frombuffer(text * 100, dtype=np.uint8)
                  for split in ("train", "validation", "test")}
        baseline, checks = {}, []
        arms = list(ARMS)
        device = "cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device
    else:
        launch, config, arrays, baseline, checks = verify_reference(reference, args.data_dir)
        arms, device = args.arms, args.device
    torch.set_num_threads(launch["threads"])
    torch.use_deterministic_algorithms(True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
    prefix = "norm-ablation-smoke" if args.smoke else "norm-ablation"
    output = (args.output or ROOT / "outputs/metrics" / f"{prefix}-{stamp}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(
        experiment="matrix_x_rmsnorm_fixed_lr", reference=str(reference),
        smoke=args.smoke, device=device, arms_trained=arms,
        arms={a: dict(optimizer=k, base_lr=lr, matrix_lr="0.03 * initial_rms" if a in ("B","D") else 0.001,
                       norm_lr=0.03 if a in ("C","D") else 0.001) for a,(k,lr) in ARMS.items()},
        settings=launch, model=asdict(config), search_budget=0,
        hypothesis="Larger Norm LR contributes to early learning and may affect late optimization",
        checkpoints=[s for s in (100,1000,3000,launch["steps"]) if s <= launch["steps"]],
        contrasts=["B-A", "C-A", "D-C", "D-B", "D-B-C+A"],
        contrast_sign="first arm minus second arm; negative NLL/PPL favors first arm",
        interaction_sign="D-B-C+A; negative NLL is a favorable interaction on the additive NLL scale",
        scope="Mechanism test with fixed effective LRs; not a new best-LR performance comparison",
    )
    (output / "protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    shutil.copytree(ROOT / "src/kda_hybrid_lm", output / "source/kda_hybrid_lm",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, output / "source/run_norm_ablation.py")
    if not args.smoke:
        shutil.copy2(args.data_dir / "metadata.json", output / "data-metadata.json")
    print(f"Device: {device}; output: {output}\nFixed-LR experiment; no LR search; training arms: {arms}",flush=True)
    results, combined = [], []
    actual_checks = []
    common = {key: launch[key] for key in ("steps","batch_size","accumulation","length",
                                           "eval_every","eval_batches","trace_every")}
    expected_tokens = common["steps"] * common["batch_size"] * common["accumulation"] * common["length"]
    for seed in launch["seeds"]:
        for arm,(kind,lr) in ARMS.items():
            if arm not in arms:
                row = dict(baseline[kind,seed])
                folder = reference / "repeat" / f"{kind}-seed{seed}"
                row.update(path=str(folder), reused=True, arm=arm)
                # 保留小体积的旧 loss 日志，下载新结果后也能追溯四组差值。
                metric_copy = output / "reference-metrics" / f"{kind}-seed{seed}.jsonl"
                metric_copy.parent.mkdir(exist_ok=True)
                shutil.copy2(folder / "metrics.jsonl", metric_copy)
                row["metrics_path"] = str(metric_copy)
            else:
                folder = output / "repeat" / f"{kind}-seed{seed}"
                row = run_training(config,arrays["train"],arrays["validation"],folder,
                                   kind=kind,lr=lr,seed=seed,shuffle_seed=launch["shuffle_seed"]+seed,
                                   device=device,**common)
                saved = torch.load(folder / "last.pt",map_location=device,weights_only=True)
                model = HybridLM(config).to(device)
                model.load_state_dict(saved["model"])
                row["test"] = evaluate(model,arrays["test"],launch["length"],launch["batch_size"],launch["eval_batches"])
                row.update(reused=False,arm=arm)
                results.append(row)
                (output / "summary.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
                del model,saved
            if row["tokens"] != expected_tokens:
                raise ValueError("Training token budget mismatch")
            combined.append(row)
        hashes = {r["initial_sha256"] for r in combined if r["seed"] == seed}
        if len(hashes) != 1:
            raise ValueError(f"Actual four-arm initialization mismatch: seed={seed}")
        actual_checks.append(dict(seed=seed,same_initialization=True,same_tokens=True))
    (output / "comparison_summary.json").write_text(json.dumps(combined,indent=2),encoding="utf-8")
    (output / "reference_check.json").write_text(json.dumps(dict(
        reference=str(reference),data_hashes_verified=not args.smoke,
        preflight=checks,pairs=actual_checks),indent=2),encoding="utf-8")
    make_contrasts(combined,output)
    print(f"Completed. Results: {output}; four-arm pairing verified.",flush=True)
    if not args.no_plots:
        from kda_hybrid_lm.train.plot import plot_results
        plot_contrasts(output)
        plot_results(combined,output / "figures")


if __name__ == "__main__":
    main()
