"""四组固定步长实验及预先约定的配对差值。"""
import csv
import json
from pathlib import Path

import numpy as np


# 矩阵包括 Embedding、所有注意力/MLP 投影和输出头；Norm 包括最终 RMSNorm。
ARMS = {
    "A": ("adam", 0.001),
    "B": ("adam_s_matrix", 0.03),
    "C": ("adam_norm", 0.001),
    "D": ("adam_s", 0.03),
}
CONTRASTS = {"B-A": {"B":1,"A":-1}, "C-A": {"C":1,"A":-1},
             "D-C": {"D":1,"C":-1}, "D-B": {"D":1,"B":-1},
             "D-B-C+A": {"D":1,"B":-1,"C":-1,"A":1}}


def write_csv(path, rows):
    with Path(path).open("w",newline="",encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_contrasts(results, output):
    output = Path(output)
    curves, final, tidy, training, grids = {}, {}, [], {}, {}
    for row in results:
        arm,seed = row["arm"],row["seed"]
        metric_path = Path(row.get("metrics_path", str(Path(row["path"]) / "metrics.jsonl")))
        logs = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines()]
        training[arm,seed] = {r["step"]:r["train_loss"] for r in logs if "train_loss" in r}
        valid = [r for r in logs if "validation" in r]
        grids[arm,seed] = [(r["step"],r["tokens"],r["validation"]["tokens"]) for r in valid]
        curves[arm,seed] = {r["step"]:r["validation"]["nll"] for r in valid}
        for r in valid:
            tidy.append(dict(arm=arm,optimizer=row["optimizer"],seed=seed,step=r["step"],
                             tokens=r["tokens"],validation_nll=r["validation"]["nll"],
                             validation_ppl=r["validation"]["ppl"]))
        final[arm,seed] = row
    seeds = sorted({r["seed"] for r in results})
    if len(results) != 4 * len(seeds):
        raise ValueError("Need exactly four arms for every seed")
    stages = sorted(curves["A",seeds[0]])
    for seed in seeds:
        for arm in ARMS:
            if sorted(curves[arm,seed]) != stages or grids[arm,seed] != grids["A",seeds[0]]:
                raise ValueError("Validation grids differ across arms")
        losses = [training[arm,seed][1] for arm in ARMS]
        if max(losses) - min(losses) > 1e-6:
            raise ValueError(f"First training losses differ: seed={seed}; inspect data/RNG pairing")
    rows, final_rows = [], []
    for name,weights in CONTRASTS.items():
        for seed in seeds:
            for step in stages:
                delta = sum(w*curves[a,seed][step] for a,w in weights.items())
                rows.append(dict(contrast=name,seed=seed,step=step,validation_nll_delta=delta))
            final_rows.append(dict(contrast=name,seed=seed,
                                   test_nll_delta=sum(w*final[a,seed]["test"]["nll"] for a,w in weights.items()),
                                   test_ppl_delta=sum(w*final[a,seed]["test"]["ppl"] for a,w in weights.items())))
    write_csv(output / "validation_curves.csv",tidy)
    write_csv(output / "validation_contrasts.csv",rows)
    stage_rows=[r for r in rows if r["step"] in (100,1000,3000,stages[-1])]
    write_csv(output / "stage_contrasts.csv",stage_rows)
    write_csv(output / "test_contrasts.csv",final_rows)
    aggregates=[]
    for name in CONTRASTS:
        for step in stages:
            values=np.array([r["validation_nll_delta"] for r in rows if r["contrast"]==name and r["step"]==step])
            aggregates.append(dict(contrast=name,step=step,mean=float(values.mean()),
                                   sample_sd=float(values.std(ddof=1)) if len(values)>1 else None))
    (output / "contrast_summary.json").write_text(json.dumps(aggregates,indent=2),encoding="utf-8")


def plot_contrasts(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output=Path(output)
    with (output / "validation_curves.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    seeds=sorted({int(r["seed"]) for r in rows})
    steps=sorted({int(r["step"]) for r in rows})
    colors={"A":"tab:blue","B":"tab:cyan","C":"tab:red","D":"tab:orange"}
    labels={"A":"A: Adam","B":"B: scaled matrices","C":"C: larger Norm LR","D":"D: Adam-S"}
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for arm in ARMS:
        a=np.array([[float(r["validation_nll"]) for r in rows if r["arm"]==arm and int(r["seed"])==seed]
                    for seed in seeds])
        for ax in axes:
            ax.plot(steps,a.mean(axis=0),color=colors[arm],label=labels[arm])
            ax.fill_between(steps,a.min(axis=0),a.max(axis=0),color=colors[arm],alpha=.12)
    axes[0].set_title("Full training; mean and seed range")
    axes[1].set(title="Early training",xlim=(0,min(1000,steps[-1])))
    late=max(0,steps[-1]//2)
    latevalues=[float(r["validation_nll"]) for r in rows if int(r["step"])>=late]
    lo,hi=min(latevalues),max(latevalues)
    padding=max((hi-lo)*.08,.001)
    axes[2].set(title="Late training",xlim=(late,steps[-1]),ylim=(lo-padding,hi+padding))
    for ax in axes:
        ax.set(xlabel="Optimizer step",ylabel="Validation NLL")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.tight_layout();fig.savefig(output / "validation_overview.png",dpi=150);plt.close(fig)

    with (output / "validation_contrasts.csv").open(encoding="utf-8-sig") as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(3,1,figsize=(9,10),sharex=True)
    groups=[("Matrix intervention",["B-A","D-C"]),
            ("Norm intervention",["C-A","D-B"]),
            ("Interaction on additive NLL scale",["D-B-C+A"])]
    for ax,(title,contrasts) in zip(axes,groups):
        for name in contrasts:
            a=np.array([[float(r["validation_nll_delta"]) for r in rows
                         if r["contrast"]==name and int(r["seed"])==seed] for seed in seeds])
            line,=ax.plot(steps,a.mean(axis=0),label=name)
            ax.fill_between(steps,a.min(axis=0),a.max(axis=0),color=line.get_color(),alpha=.15)
        ax.axhline(0,color="black",lw=.8)
        ax.set(title=title,ylabel="Paired NLL difference")
        ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel("Optimizer step; bands show seed min-max, not confidence intervals")
    fig.tight_layout();fig.savefig(output / "factorial_effects.png",dpi=150);plt.close(fig)
