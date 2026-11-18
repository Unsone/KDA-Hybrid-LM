import copy
import math

import pytest
import torch
from torch import nn

from kda_hybrid_lm.train.optim import build_optimizer
from kda_hybrid_lm.models import HybridLM, ModelConfig


def test_scaled_adam_preserves_moments_and_scales_actual_update():
    torch.manual_seed(8)
    base = nn.Linear(3, 2)
    scaled = copy.deepcopy(base)
    adam, _ = build_optimizer(base, "adam", 0.01)
    adam_s, mapping = build_optimizer(scaled, "adam_s", 0.01)
    before = base.weight.detach().clone()
    # 人工固定相同梯度，验证多步矩估计与 Adam 完全相同。
    for i in range(3):
        for p, q in zip(base.parameters(), scaled.parameters()):
            p.grad = torch.full_like(p, 0.2 * (i + 1))
            q.grad = p.grad.clone()
        adam.step()
        adam_s.step()
        for p, q in zip(base.parameters(), scaled.parameters()):
            for key in ("exp_avg", "exp_avg_sq", "step"):
                torch.testing.assert_close(adam.state[p][key], adam_s.state[q][key])
    s = mapping["weight"]["scale"]
    torch.testing.assert_close(scaled.weight - before, (base.weight - before) * s)
    torch.testing.assert_close(base.bias, scaled.bias)


def test_scaled_weight_update_shuffle_and_zero_weights():
    model = nn.Sequential(nn.Linear(3, 2, bias=True), nn.Linear(2, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(0.2)
        model[1].weight.zero_()
    other = copy.deepcopy(model)
    base, _ = build_optimizer(model, "adam", 0.01)
    scaled, mapping = build_optimizer(other, "adam_s", 0.01)
    before = model[0].weight.detach().clone()
    for p, q in zip(model.parameters(), other.parameters()):
        p.grad = torch.ones_like(p)
        q.grad = p.grad.clone()
    base.step()
    scaled.step()
    torch.testing.assert_close(other[0].weight - before, (model[0].weight - before) * 0.2)
    torch.testing.assert_close(other[0].bias, model[0].bias)
    assert torch.count_nonzero(other[1].weight) == 0
    assert mapping["1.weight"]["scale"] == 0
    initial_rng = torch.get_rng_state().clone()
    _, shuffled = build_optimizer(other, "shuffled", 0.01, shuffle_seed=37)
    torch.testing.assert_close(torch.get_rng_state(), initial_rng)
    for name, row in shuffled.items():
        assert row["scale"] == shuffled[row["source"]]["initial_rms"]
    assert shuffled["1.weight"]["scale"] == 0


def test_theoretical_scales_preserve_initial_weights_rng_and_adam_moments():
    torch.manual_seed(12)
    model = HybridLM(ModelConfig(vocab_size=16, hidden_dim=8, intermediate_dim=24,
                                 num_heads=2, layer_types=("softmax",), embedding_std=0.07))
    original = copy.deepcopy(model)
    rng = torch.get_rng_state().clone()
    optimizer, mapping = build_optimizer(model, "adam_s_theory", 0.01)
    baseline, _ = build_optimizer(original, "adam", 0.01)
    torch.testing.assert_close(torch.get_rng_state(), rng)
    assert mapping["embedding.weight"]["scale"] == 0.07
    assert mapping["norm.weight"]["scale"] == 1
    assert mapping["lm_head.weight"]["scale"] == pytest.approx(1 / math.sqrt(24))
    assert mapping["layers.0.mlp.down_proj.weight"]["scale"] == pytest.approx(1 / math.sqrt(72))
    assert mapping["embedding.weight"]["initial_rms"] != mapping["embedding.weight"]["scale"]
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for (n, p), (_, q) in zip(model.named_parameters(), original.named_parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    for step in range(3):
        for p, q in zip(model.parameters(), original.parameters()):
            p.grad = torch.full_like(p, 0.3 * (step + 1))
            q.grad = p.grad.clone()
        optimizer.step()
        baseline.step()
    for (n, p), (_, q) in zip(model.named_parameters(), original.named_parameters()):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            torch.testing.assert_close(optimizer.state[p][key], baseline.state[q][key])
        torch.testing.assert_close(p - before[n], (q - before[n]) * mapping[n]["scale"])
