"""可直接在 Windows/Linux 运行：python scripts/train_lm.py --mode smoke。"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from kda_hybrid_lm.models import ModelConfig
from kda_hybrid_lm.train.trainer import evaluate, run_training


def parser():
    p = argparse.ArgumentParser(description="Adam / Adam-S / shuffled Transformer experiment")
    p.add_argument("--mode", choices=["smoke", "compare", "train"], default="smoke")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data/processed/tinystories")
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="auto")
    p.add_argument("--steps", type=int)
    p.add_argument("--search-steps", type=int)
    p.add_argument("--seeds", type=int, nargs="+")
    p.add_argument("--tune-seed", type=int, default=7)
    p.add_argument("--shuffle-seed", type=int, default=1000)
    p.add_argument("--adam-lrs", type=float, nargs="+", default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    p.add_argument("--scaled-lrs", type=float, nargs="+", default=[3e-3, 1e-2, 3e-2, 1e-1, 3e-1])
    p.add_argument("--batch-size", type=int)
    p.add_argument("--accumulation", type=int)
    p.add_argument("--length", type=int)
    p.add_argument("--layers", type=int)
    p.add_argument("--hidden-dim", type=int)
    p.add_argument("--intermediate-dim", type=int)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--eval-every", type=int)
    p.add_argument("--eval-batches", type=int)
    p.add_argument("--trace-every", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--optimizer", choices=["adam", "adam_s", "shuffled"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-plots", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    smoke = args.mode == "smoke"
    defaults = dict(steps=8 if smoke else 6104, search_steps=3 if smoke else 611,
                    batch_size=2 if smoke else 8, accumulation=1 if smoke else 4,
                    length=32 if smoke else 256, layers=2 if smoke else 6,
                    hidden_dim=32 if smoke else 256, intermediate_dim=80 if smoke else 688,
                    eval_every=4 if smoke else 100, eval_batches=2 if smoke else 32)
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.seeds = args.seeds or ([42, 43] if smoke else [42, 43, 44])
    for key in (*defaults, "trace_every", "threads"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if len(args.seeds) != len(set(args.seeds)) or args.tune_seed in args.seeds:
        p.error("Use distinct repeat seeds excluding --tune-seed")
    if not smoke and len(args.adam_lrs) != len(args.scaled_lrs):
        p.error("Adam and scaled optimizers need equal numbers of LR candidates")
    if min(args.adam_lrs + args.scaled_lrs + [args.lr]) <= 0:
        p.error("Learning rates must be positive")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if smoke:
        text = b"Once upon a time, a little bird found a red apple. It shared the apple with a friend.\n"
        train = np.frombuffer(text * 200, dtype=np.uint8)
        valid = np.frombuffer(b"A little bird and a friend shared a red apple.\n" * 30, dtype=np.uint8)
        test = np.frombuffer(b"A friend found an apple and shared it with a bird.\n" * 30, dtype=np.uint8)
        vocab_size = 256
    else:
        if not (args.data_dir / "metadata.json").exists():
            p.error("Prepared data missing. Run python scripts/prepare_data.py --help first")
        metadata = json.loads((args.data_dir / "metadata.json").read_text(encoding="utf-8"))
        train, valid, test = [np.load(args.data_dir / f"{split}.npy", mmap_mode="r")
                              for split in ("train", "validation", "test")]
        vocab_size = metadata["vocab_size"]
    if min(len(train), len(valid), len(test)) <= args.length:
        p.error("Each data split must be longer than --length")
    config = ModelConfig(vocab_size=vocab_size, hidden_dim=args.hidden_dim,
                         intermediate_dim=args.intermediate_dim, num_heads=args.heads,
                         layer_types=("softmax",) * args.layers)
    output = args.output or ROOT / "outputs/metrics" / datetime.now(timezone.utc).strftime(f"{args.mode}-%Y%m%d-%H%M%S-%f")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "launch.json").write_text(json.dumps({**vars(args), "device": device}, default=str, indent=2), encoding="utf-8")
    try:
        version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        diff = subprocess.check_output(["git", "-c", "core.safecrlf=false", "diff", "--binary"], cwd=ROOT, stderr=subprocess.DEVNULL)
        (output / "code-version.txt").write_text(version, encoding="utf-8")
        (output / "working-tree.patch").write_bytes(diff)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    # 保存实际运行的源码（包括尚未提交的新文件），不依赖 Git patch 复现。
    import shutil
    shutil.copytree(ROOT / "src/kda_hybrid_lm", output / "source/kda_hybrid_lm",
                    ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(__file__, output / "source/train_lm.py")
    if not smoke:
        shutil.copy2(args.data_dir / "metadata.json", output / "data-metadata.json")
    print(f"Device: {device}; output: {output}\n"
          f"Budget/run: {args.steps * args.batch_size * args.accumulation * args.length:,} target tokens\n"
          + ("SMOKE: synthetic byte-level data; pipeline check only." if smoke else "Real-data run; FP32, constant LR, no weight decay."), flush=True)
    common = dict(config=config, train=train, valid=valid, batch_size=args.batch_size,
                  accumulation=args.accumulation, length=args.length, eval_every=args.eval_every,
                  eval_batches=args.eval_batches, device=device, trace_every=args.trace_every)

    def run(folder, kind, lr, seed, steps):
        return run_training(output_dir=output / folder, kind=kind, lr=lr, seed=seed,
                            shuffle_seed=args.shuffle_seed + seed, steps=steps, **common)

    results = []
    if args.mode == "train":
        results.append(run("train", args.optimizer, args.lr, args.seeds[0], args.steps))
    else:
        grids = {"adam": [0.003] if smoke else args.adam_lrs,
                 "adam_s": [0.03] if smoke else args.scaled_lrs,
                 "shuffled": [0.03] if smoke else args.scaled_lrs}
        selected = {}
        searches = []
        for kind, lrs in grids.items():
            candidates = []
            for i, lr in enumerate(lrs):
                folder = f"search/{kind}-{i}"
                try:
                    candidate = run(folder, kind, lr, args.tune_seed, args.search_steps)
                    candidates.append(candidate)
                    searches.append(candidate)
                except FloatingPointError as error:
                    failure = dict(optimizer=kind, lr=lr, error=str(error))
                    searches.append(failure)
                    (output / folder / "failure.json").write_text(json.dumps(failure), encoding="utf-8")
                (output / "search.json").write_text(json.dumps(searches, indent=2), encoding="utf-8")
            if not candidates:
                raise RuntimeError(f"All {kind} candidates diverged; inspect {output}")
            selected[kind] = min(candidates, key=lambda r: r["validation"]["nll"])["lr"]
        (output / "selected_lrs.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        for seed in args.seeds:
            for kind, lr in selected.items():
                results.append(run(f"repeat/{kind}-seed{seed}", kind, lr, seed, args.steps))
    # 最终评估集只在选择 LR 且完成训练后使用。
    from kda_hybrid_lm.models import HybridLM
    for row in results:
        saved = torch.load(Path(row["path"]) / "last.pt", map_location=device, weights_only=True)
        model = HybridLM(config).to(device)
        model.load_state_dict(saved["model"])
        row["test"] = evaluate(model, test, args.length, args.batch_size, args.eval_batches)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Completed. Results: {output / 'summary.json'}", flush=True)
    if not args.no_plots:
        from kda_hybrid_lm.train.plot import plot_results
        plot_results(results, output / "figures")


if __name__ == "__main__":
    main()
