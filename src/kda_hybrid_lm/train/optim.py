import math
import random

import torch
from torch import nn

from kda_hybrid_lm.modules.norm import RMSNorm


OPTIMIZERS = ("adam", "adam_s", "shuffled", "adam_s_theory")


def theoretical_rms(model, name):
    """HybridLM 初始化分布的 sqrt(E[w²])；不读取样本值，不修改参数或 RNG。"""
    module_name, _, _ = name.rpartition(".")
    module = model.get_submodule(module_name)
    if isinstance(module, nn.Linear):
        # Linear 的初始化为 U(-1/sqrt(fan_in), 1/sqrt(fan_in))。
        return 1 / math.sqrt(3 * module.in_features)
    if isinstance(module, nn.Embedding):
        return model.config.embedding_std
    if isinstance(module, RMSNorm):
        return 1.0
    raise ValueError(f"No initialization rule for {name}: {type(module).__name__}")


def _initial_weight_scales(model):
    """按模型参数顺序，计算每个 weight 张量的初始 RMS。"""
    scales = {}
    for name, parameter in model.named_parameters():
        if name == "weight" or name.endswith(".weight"):
            values = parameter.detach().float()
            rms = values.square().mean().sqrt().item()
            scales[name] = rms
    return scales


def _scale_mapping(model, kind, scales, shuffle_seed):
    sources = {}
    for name in scales:
        sources[name] = name
    if kind == "shuffled":
        nonzero = [name for name in scales if scales[name] > 0]
        shuffled = nonzero.copy()
        random.Random(shuffle_seed).shuffle(shuffled)
        sources.update(zip(nonzero, shuffled))
    elif kind not in OPTIMIZERS:
        raise ValueError(f"Unknown optimizer: {kind}")

    mapping = {}
    for name in scales:
        source = sources[name]
        scale = scales[source]
        if kind == "adam":
            scale = 1.0
        mapping[name] = {
            "initial_rms": scales[name],
            "source": source,
            "scale": scale,
        }
    if kind == "adam_s_theory":
        for name, row in mapping.items():
            row.update(scale=theoretical_rms(model, name), scale_source="initialization_rule")
    return mapping


def build_optimizer(model, kind, lr, shuffle_seed=0):
    """用逐张量 LR 实现固定尺度 Adam；bias 不缩放，weight decay=0。"""
    scales = _initial_weight_scales(model)
    mapping = _scale_mapping(model, kind, scales, shuffle_seed)
    groups = []
    for name, parameter in model.named_parameters():
        scale = 1.0
        if name in mapping:
            scale = mapping[name]["scale"]
        groups.append({
            "params": [parameter],
            "lr": lr * scale,
            "name": name,
        })
    optimizer = torch.optim.Adam(
        groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,
    )
    return optimizer, mapping
