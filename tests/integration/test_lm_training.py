import copy

import pytest
import torch

from kda_hybrid_lm.models import HybridLM, ModelConfig


@pytest.mark.parametrize("layer_types", [("softmax", "softmax"), ("gdn", "softmax")])
def test_paired_model_can_train_with_three_fixed_scale_assignments(layer_types):
    """仅验收模型可训练；这里用 Adam 参数组模拟固定 s，不是正式实验。"""
    torch.manual_seed(23)
    model = HybridLM(ModelConfig(vocab_size=8, hidden_dim=16, intermediate_dim=40,
                                num_heads=4, layer_types=layer_types))
    initial = copy.deepcopy(model.state_dict())
    tokens = torch.arange(8).repeat(2, 2)
    names = dict(model.named_parameters())
    scales = {name: p.detach().square().mean().sqrt().item() if name.endswith(".weight") else 1.0
              for name, p in names.items()}
    nonzero = [name for name, s in scales.items() if name.endswith(".weight") and s > 0]
    order = torch.randperm(len(nonzero), generator=torch.Generator().manual_seed(101)).tolist()
    shuffled = dict(scales)
    shuffled.update({name: scales[nonzero[j]] for name, j in zip(nonzero, order)})
    for mode in ("adam", "adam_s", "shuffled"):
        model.load_state_dict(initial)
        torch.manual_seed(29)
        assigned = dict.fromkeys(names, 1.0) if mode == "adam" else (scales if mode == "adam_s" else shuffled)
        optimizer = torch.optim.Adam([
            {"params": [p], "lr": 0.01 * assigned[name]} for name, p in names.items()
        ], weight_decay=0)
        before = model(tokens, labels=tokens).loss.item()
        for _ in range(12):
            optimizer.zero_grad(set_to_none=True)
            loss = model(tokens, labels=tokens).loss
            loss.backward()
            optimizer.step()
        after = model(tokens, labels=tokens).loss.item()
        assert after < before, (mode, before, after)
        assert all(torch.isfinite(p).all() for p in model.parameters())
