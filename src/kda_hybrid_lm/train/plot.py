import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_results(results, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    colors = {"adam": "tab:blue", "adam_s": "tab:orange", "shuffled": "tab:green"}
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
        by_name = defaultdict(list)
        with (path / "weights.csv").open(encoding="utf-8") as source:
            for row in csv.DictReader(source):
                by_name[row["parameter"]].append(row)
        for name, entries in by_name.items():
            traces[name].append((label, color, entries))
    for axis, title in zip(axes, ["Training loss", "Validation NLL"]):
        axis.set(xlabel="Target tokens", ylabel=title)
        axis.grid(alpha=0.2)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "loss.png", dpi=130)
    plt.close(fig)
    for name, runs in traces.items():
        fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for label, color, entries in runs:
            steps = [int(row["step"]) for row in entries]
            for axis, key in zip(axes, ["norm", "norm_ratio", "relative_update"]):
                axis.plot(steps, [float(row[key]) for row in entries],
                          label=label, color=color, alpha=0.7)
                axis.set_ylabel(key)
                axis.grid(alpha=0.2)
        axes[0].set_title(name)
        axes[0].legend(fontsize=7)
        axes[-1].set_xlabel("Optimizer step")
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=120)
        plt.close(fig)
    print(f"Figures: {output}", flush=True)
