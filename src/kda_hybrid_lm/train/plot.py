import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WEIGHT_KEYS = ("norm", "norm_ratio", "relative_update")
OPTIMIZER_KEYS = ("grad_rms", "momentum_rms", "momentum_hat_rms",
                  "sqrt_second_moment_hat_rms", "adam_direction_rms")


def load_results(roots, optimizers):
    """按下载后的目录定位日志，不依赖 summary 中服务器的绝对路径。"""
    results = []
    seen = set()
    for root in map(Path, roots):
        for row in json.loads((root / "summary.json").read_text(encoding="utf-8")):
            if row["optimizer"] not in optimizers:
                continue
            identity = (row["optimizer"], row["seed"])
            if identity in seen:
                raise ValueError(f"Duplicate run {identity}; choose one result directory per run")
            seen.add(identity)
            # 同时支持 compare 的 repeat 子目录和单次 train 模式。
            folder = root / "repeat" / f"{row['optimizer']}-seed{row['seed']}"
            if not folder.is_dir():
                folder = root / "train"
            if not (folder / "metrics.jsonl").is_file() or not (folder / "weights.csv").is_file():
                raise FileNotFoundError(f"Missing downloaded logs for {identity} under {root}")
            results.append({**row, "path": str(folder)})
    missing = set(optimizers) - {r["optimizer"] for r in results}
    if missing:
        raise ValueError(f"No results for optimizers: {sorted(missing)}")
    return results


def read_traces(path):
    # 只保留画图需要的数值列，避免 6 份大 CSV 常驻数百万个字符串字典。
    by_name = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as source:
        reader = csv.reader(source)
        header = next(reader)
        keys = WEIGHT_KEYS + (OPTIMIZER_KEYS if all(k in header for k in OPTIMIZER_KEYS) else ())
        indices = [header.index(k) for k in ("step", *keys)]
        name_index = header.index("parameter")
        for row in reader:
            by_name[row[name_index]].append(tuple(float(row[i]) for i in indices))
    return keys, {name: np.asarray(rows) for name, rows in by_name.items()}


def plot_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = {"adam": "tab:blue", "adam_s": "tab:orange", "shuffled": "tab:green",
              "adam_s_theory": "tab:purple"}
    traces = defaultdict(list)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for run in results:
        label = f"{run['optimizer']} seed={run['seed']}"
        color = colors[run["optimizer"]]
        path = Path(run["path"])
        rows = [json.loads(line) for line in (path / "metrics.jsonl").read_text().splitlines()]
        train = [r for r in rows if "train_loss" in r]
        valid = [r for r in rows if "validation" in r]
        axes[0].plot([r["tokens"] for r in train], [r["train_loss"] for r in train],
                     label=label, color=color, alpha=0.7)
        axes[1].plot([r["tokens"] for r in valid], [r["validation"]["nll"] for r in valid],
                     label=label, color=color, alpha=0.7)
        keys, by_name = read_traces(path / "weights.csv")
        for name, entries in by_name.items():
            traces[name].append((label, color, keys, entries))
    for axis, title in zip(axes, ["Training loss", "Validation NLL"]):
        axis.set(xlabel="Target tokens", ylabel=title)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=130)
    plt.close(fig)
    for name, runs in traces.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for label, color, keys, entries in runs:
            steps = entries[:, 0]
            for axis, key in zip(axes, WEIGHT_KEYS):
                axis.plot(steps, entries[:, 1 + keys.index(key)],
                          label=label, color=color, alpha=0.7)
                axis.set_ylabel(key)
                axis.grid(alpha=0.2)
        axes[0].set_title(name)
        axes[0].legend(fontsize=7)
        axes[-1].set_xlabel("Optimizer step")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=120)
        plt.close(fig)
        # 老版本 CSV 仍可绘制原有的三项指标。
        optimizer_runs = [run for run in runs if "grad_rms" in run[2]]
        if optimizer_runs:
            fig, axes = plt.subplots(len(OPTIMIZER_KEYS), 1, figsize=(9, 11), sharex=True)
            for label, color, keys, entries in optimizer_runs:
                steps = entries[:, 0]
                for axis, key in zip(axes, OPTIMIZER_KEYS):
                    axis.plot(steps, entries[:, 1 + keys.index(key)],
                              label=label, color=color, alpha=0.7)
                    axis.set_ylabel(key, fontsize=8)
                    axis.grid(alpha=0.2)
            axes[0].set_title(name)
            axes[0].legend(fontsize=7)
            axes[-1].set_xlabel("Optimizer step")
            fig.tight_layout()
            fig.savefig(output / f"{name}_optimizer.png", dpi=120)
            plt.close(fig)
    print(f"Figures: {output}", flush=True)
