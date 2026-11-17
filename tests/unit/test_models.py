import copy
from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from kda_hybrid_lm.models import HybridLM, ModelConfig, generate


def small_config(**kwargs):
    return ModelConfig(vocab_size=17, hidden_dim=16, intermediate_dim=40,
                       num_heads=4, **kwargs)


@pytest.mark.parametrize("kind", ["softmax", "linear", "delta", "gdn", "kda"])
def test_model_loss_and_all_parameter_gradients(kind):
    torch.manual_seed(3)
    model = HybridLM(small_config(layer_types=(kind, "softmax"), bias=True))
    tokens = torch.randint(17, (2, 6))
    labels = tokens.clone()
    labels[:, 3] = -100
    output = model(tokens, labels=labels)
    expected = F.cross_entropy(output.logits[:, :-1].reshape(-1, 17), labels[:, 1:].reshape(-1))
    torch.testing.assert_close(output.loss, expected)
    assert output.state is None
    output.loss.backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    aligned = model(tokens, labels=labels, shift_labels=False)
    torch.testing.assert_close(aligned.loss, F.cross_entropy(aligned.logits.reshape(-1, 17), labels.reshape(-1)))
    ignored = model(tokens, labels=torch.full_like(tokens, -100))
    assert ignored.loss.item() == 0
    ignored.loss.backward()


def test_mixed_model_streaming_causality_and_generation():
    torch.manual_seed(7)
    model = HybridLM(small_config(layer_types=("linear", "delta", "gdn", "kda", "softmax"))).eval()
    tokens = torch.randint(17, (2, 8))
    with torch.no_grad():
        full = model(tokens, use_cache=True)
        state = None
        pieces = []
        for start, end in ((0, 3), (3, 5), (5, 8)):
            part = model(tokens[:, start:end], state=state, use_cache=True)
            pieces.append(part.logits)
            state = part.state
        torch.testing.assert_close(full.logits, torch.cat(pieces, dim=1), atol=2e-6, rtol=2e-5)
        assert state.tokens_seen == 8
        for actual, expected in zip(state.layers, full.state.layers):
            torch.testing.assert_close(actual, expected)
        changed = tokens.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 17
        torch.testing.assert_close(full.logits[:, :4], model(changed).logits[:, :4])
        expected = tokens
        for _ in range(3):
            next_token = model(expected).logits[:, -1].argmax(-1, keepdim=True)
            expected = torch.cat((expected, next_token), dim=1)
    model.train()
    torch.testing.assert_close(generate(model, tokens, 3), expected)
    assert model.training


@pytest.mark.parametrize("tied", [False, True])
def test_reproducible_initialization_and_checkpoint(tmp_path, tied):
    config = small_config(tie_embeddings=tied)
    torch.manual_seed(19)
    first = HybridLM(config)
    torch.manual_seed(19)
    second = HybridLM(config)
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert (first.embedding.weight is first.lm_head.weight) == tied
    path = tmp_path / "model.pt"
    torch.save({"config": asdict(config), "model": copy.deepcopy(first.state_dict())}, path)
    saved = torch.load(path, weights_only=True)
    restored = HybridLM(ModelConfig(**saved["config"]))
    restored.load_state_dict(saved["model"])
    tokens = torch.randint(17, (2, 4))
    torch.testing.assert_close(first(tokens).logits, restored(tokens).logits)


def test_invalid_model_config():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(hidden_dim=15)
    with pytest.raises(ValueError, match="even"):
        ModelConfig(hidden_dim=12, num_heads=4)
    with pytest.raises(ValueError, match="layer_types"):
        ModelConfig(layer_types=("unknown",))
