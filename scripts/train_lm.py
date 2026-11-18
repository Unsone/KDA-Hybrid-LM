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
from kda_hybrid_lm.train.optim import OPTIMIZERS
from kda_hybrid_lm.train.trainer import evaluate, run_training


def parser():
    p = argparse.ArgumentParser(description="Adam / Adam-S / shuffled / theoretical-RMS experiment")
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
    p.add_argument("--optimizer", choices=OPTIMIZERS, default="adam")
    p.add_argument("--optimizers", choices=OPTIMIZERS, nargs="+",
                   default=["adam", "adam_s", "shuffled"],
                   help="Arms to tune and repeat in compare/smoke; each gets the same search budget")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--no-plots", action="store_true")
    return p


def _apply_defaults(args, p):
    smoke = args.mode == "smoke"
    if smoke:
        defaults = {
            "steps": 8,
            "search_steps": 3,
            "batch_size": 2,
            "accumulation": 1,
            "length": 32,
            "layers": 2,
            "hidden_dim": 32,
            "intermediate_dim": 80,
            "eval_every": 4,
            "eval_batches": 2,
        }
    else:
        defaults = {
            "steps": 6104,
            "search_steps": 611,
            "batch_size": 8,
            "accumulation": 4,
            "length": 256,
            "layers": 6,
            "hidden_dim": 256,
            "intermediate_dim": 688,
            "eval_every": 100,
            "eval_batches": 32,
        }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if not args.seeds:
        if smoke:
            args.seeds = [42, 43]
        else:
            args.seeds = [42, 43, 44]
    positive_options = list(defaults) + ["trace_every", "threads"]
    for key in positive_options:
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if len(args.seeds) != len(set(args.seeds)) or args.tune_seed in args.seeds:
        p.error("Use distinct repeat seeds excluding --tune-seed")
    if len(args.optimizers) != len(set(args.optimizers)):
        p.error("--optimizers must not contain duplicates")
    if not smoke and len(args.adam_lrs) != len(args.scaled_lrs):
        p.error("Adam and scaled optimizers need equal numbers of LR candidates")
    if min(args.adam_lrs + args.scaled_lrs + [args.lr]) <= 0:
        p.error("Learning rates must be positive")
    return smoke


def _load_data(args, p, smoke):
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
    return train, valid, test, vocab_size


def _prepare_output(args, device, smoke):
    output = args.output
    if not output:
        timestamp = datetime.now(timezone.utc).strftime(f"{args.mode}-%Y%m%d-%H%M%S-%f")
        output = ROOT / "outputs/metrics" / timestamp
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    launch = vars(args).copy()
    launch["device"] = device
    launch_json = json.dumps(launch, default=str, indent=2)
    (output / "launch.json").write_text(launch_json, encoding="utf-8")
    try:
        version = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          text=True, stderr=subprocess.DEVNULL).strip()
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
    return output


def _run_experiment(args, config, train, valid, device, output, smoke):
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
        grids = {}
        for kind in args.optimizers:
            if smoke:
                learning_rate = 0.03
                if kind == "adam":
                    learning_rate = 0.003
                grids[kind] = [learning_rate]
            elif kind == "adam":
                grids[kind] = args.adam_lrs
            else:
                grids[kind] = args.scaled_lrs
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
    return results


def _evaluate_final_runs(results, config, test, args, device):
    # 最终评估集只在选择 LR 且完成训练后使用。
    from kda_hybrid_lm.models import HybridLM
    for row in results:
        saved = torch.load(Path(row["path"]) / "last.pt", map_location=device, weights_only=True)
        model = HybridLM(config).to(device)
        model.load_state_dict(saved["model"])
        row["test"] = evaluate(model, test, args.length, args.batch_size, args.eval_batches)


def main():
    p = parser()
    args = p.parse_args()
    smoke = _apply_defaults(args, p)

    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == "auto":
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"

    train, valid, test, vocab_size = _load_data(args, p, smoke)
    config = ModelConfig(
        vocab_size=vocab_size,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        num_heads=args.heads,
        layer_types=("softmax",) * args.layers,
    )
    output = _prepare_output(args, device, smoke)
    results = _run_experiment(args, config, train, valid, device, output, smoke)
    _evaluate_final_runs(results, config, test, args, device)
    (output / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Completed. Results: {output / 'summary.json'}", flush=True)
    if not args.no_plots:
        from kda_hybrid_lm.train.plot import plot_results
        plot_results(results, output / "figures")


if __name__ == "__main__":
    main()
