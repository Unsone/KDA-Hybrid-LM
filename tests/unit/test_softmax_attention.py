import torch
from torch.nn import functional as F

from kda_hybrid_lm.mixers import SoftmaxAttention, TokenMixer


def test_matches_pytorch_attention():
    torch.manual_seed(31)
    model = SoftmaxAttention(16, 4, bias=True).eval()
    assert isinstance(model, TokenMixer)
    x = torch.randn(2, 7, 16)
    q = model._split_heads(model.q_proj(x), 2, 7)
    k = model._split_heads(model.k_proj(x), 2, 7)
    v = model._split_heads(model.v_proj(x), 2, 7)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    expected = model.out_proj(expected.transpose(1, 2).reshape(2, 7, 16))
    actual, state = model(x, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state[0], k)
    torch.testing.assert_close(state[1], v)


def test_streaming_cache():
    torch.manual_seed(37)
    for batch, length, dim, heads in ((2, 7, 16, 4), (1, 1, 4, 1)):
        model = SoftmaxAttention(dim, heads).eval()
        x = torch.randn(batch, length, dim)
        full, final = model(x, use_cache=True)
        for chunk_size in (1, 3):
            state = None
            outputs = []
            for start in range(0, length, chunk_size):
                output, state = model(x[:, start:start+chunk_size], state, use_cache=True)
                outputs.append(output)
            torch.testing.assert_close(full, torch.cat(outputs, dim=1))
            torch.testing.assert_close(final[0], state[0])
            torch.testing.assert_close(final[1], state[1])
        saved = (state[0].clone(), state[1].clone())
        _, no_cache = model(x[:, :1], state)
        assert no_cache is None
        torch.testing.assert_close(state[0], saved[0])
        torch.testing.assert_close(state[1], saved[1])


def test_causality_and_gradients():
    torch.manual_seed(41)
    model = SoftmaxAttention(16, 4, bias=True)
    x = torch.randn(2, 7, 16, requires_grad=True)
    output, _ = model(x)
    changed = x.detach().clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])
    changed_output, _ = model(changed)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])
    output[:, :3].square().mean().backward()
    torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))
    assert torch.isfinite(x.grad).all()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_attention_dropout():
    torch.manual_seed(43)
    model = SoftmaxAttention(16, 4, dropout=0.5)
    x = torch.randn(2, 7, 16)
    first, first_state = model(x, use_cache=True)
    second, second_state = model(x, use_cache=True)
    assert not torch.equal(first, second)
    torch.testing.assert_close(first_state[0], second_state[0])
    torch.testing.assert_close(first_state[1], second_state[1])
    model.eval()
    first, _ = model(x)
    second, _ = model(x)
    torch.testing.assert_close(first, second)
