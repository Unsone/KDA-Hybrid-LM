import math
import random

import torch
from torch import nn

from kda_hybrid_lm.modules.norm import RMSNorm


OPTIMIZERS = ("adam", "adam_s", "shuffled", "adam_s_theory", "adam_s_matrix", "adam_norm")


def theoretical_rms(model, name):
    """HybridLM 初始化分布的 sqrt(E[w²])；不读取样本值，不修改参数或 RNG。"""
    module_name, _, _ = name.rpartition(".")
    module = model.get_submodule(module_name)
    if isinstance(module, nn.Linear):
        # HybridLM 保留 Linear 的 U(-1/sqrt(fan_in), 1/sqrt(fan_in))。
        return 1 / math.sqrt(3 * module.in_features)
    if isinstance(module, nn.Embedding):
        return model.config.embedding_std
    if isinstance(module, RMSNorm):
        return 1.0
    raise ValueError(f"No initialization rule for {name}: {type(module).__name__}")


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
    elif kind not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {kind}")
    mapping = {n: {"initial_rms": scales[n], "source": sources[n],
                   "scale": 1.0 if kind == "adam" else scales[sources[n]]} for n in weights}
    if kind == "adam_s_theory":
        for name, row in mapping.items():
            row.update(scale=theoretical_rms(model, name), scale_source="initialization_rule")
    if kind in ("adam_s_matrix", "adam_norm"):
        for name, row in mapping.items():
            module_name, _, _ = name.rpartition(".")
            is_norm = isinstance(model.get_submodule(module_name), RMSNorm)
            if is_norm:
                # 绝对 LR 固定，用 scale 表示以保持逐张量映射与日志一致。
                norm_lr = 0.001 if kind == "adam_s_matrix" else 0.03
                row["scale"] = norm_lr / lr
            elif kind == "adam_norm":
                row["scale"] = 1.0
            row["parameter_role"] = "rmsnorm" if is_norm else "matrix"
            row["effective_lr"] = lr * row["scale"]
    groups = [{"params": [p], "lr": lr * (mapping[n]["scale"] if n in mapping else 1.0),
               "name": n} for n, p in model.named_parameters()]
    return torch.optim.Adam(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0), mapping
