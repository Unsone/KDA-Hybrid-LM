import random

import torch


def build_optimizer(model, kind, lr, shuffle_seed=0):
    """用逐张量 LR 实现固定尺度 Adam；bias 不缩放，weight decay=0。"""
    weights = {n: p for n, p in model.named_parameters() if n == "weight" or n.endswith(".weight")}
    scales = {n: p.detach().float().square().mean().sqrt().item() for n, p in weights.items()}
    sources = {n: n for n in weights}
    if kind == "shuffled":
        nonzero = [n for n in weights if scales[n] > 0]
        shuffled = nonzero.copy()
        random.Random(shuffle_seed).shuffle(shuffled)
        sources.update(zip(nonzero, shuffled))
    elif kind not in ("adam", "adam_s"):
        raise ValueError(f"Unknown optimizer: {kind}")
    mapping = {n: {"initial_rms": scales[n], "source": sources[n],
                   "scale": 1.0 if kind == "adam" else scales[sources[n]]} for n in weights}
    groups = [{"params": [p], "lr": lr * (mapping[n]["scale"] if n in mapping else 1.0),
               "name": n} for n, p in model.named_parameters()]
    return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0), mapping
