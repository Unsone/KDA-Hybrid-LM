import math

import torch
from torch.nn import functional as F

from kda_hybrid_lm.modules import RotaryEmbedding, ShortConvolution, SwiGLU


def test_swiglu_token_independence_and_gradients():
    torch.manual_seed(67)
    model = SwiGLU(8, 20, bias=True)
    x = torch.randn(2, 5, 8, requires_grad=True)
    output = model(x)
    assert output.shape == x.shape
    changed = x.detach().clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])
    torch.testing.assert_close(output[:, :3], model(changed)[:, :3])
    output[:, :3].square().mean().backward()
    torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_swiglu_dropout():
    torch.manual_seed(71)
    model = SwiGLU(8, 20, dropout=0.5)
    x = torch.randn(2, 5, 8)
    first = model(x)
    second = model(x)
    assert torch.count_nonzero(first == 0) > 0
    assert not torch.equal(first, second)
    model.eval()
    torch.testing.assert_close(model(x), model(x))


def test_rotary_two_dimensional_rotation():
    rope = RotaryEmbedding(2)
    q = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], requires_grad=True)
    k = torch.tensor([[[[0.0, 1.0], [0.0, 1.0]]]], requires_grad=True)
    rotated_q, rotated_k = rope(q, k)
    expected_q = torch.tensor([[[[1.0, 0.0], [math.cos(1), math.sin(1)]]]])
    expected_k = torch.tensor([[[[0.0, 1.0], [-math.sin(1), math.cos(1)]]]])
    torch.testing.assert_close(rotated_q, expected_q)
    torch.testing.assert_close(rotated_k, expected_k)
    (rotated_q.square().sum() + rotated_k.square().sum()).backward()
    torch.testing.assert_close(q.grad, 2 * q)
    torch.testing.assert_close(k.grad, 2 * k)


def test_rotary_offsets_and_relative_positions():
    torch.manual_seed(73)
    rope = RotaryEmbedding(8)
    q = torch.randn(2, 3, 7, 8)
    k = torch.randn(2, 1, 7, 8)
    full_q, full_k = rope(q, k)
    first_q, first_k = rope(q[:, :, :3], k[:, :, :3])
    next_q, next_k = rope(q[:, :, 3:], k[:, :, 3:], offset=3)
    torch.testing.assert_close(full_q, torch.cat((first_q, next_q), dim=2))
    torch.testing.assert_close(full_k, torch.cat((first_k, next_k), dim=2))
    torch.testing.assert_close(full_q.square().sum(dim=3), q.square().sum(dim=3))
    shifted_q, shifted_k = rope(q, k, offset=11)
    scores = torch.matmul(full_q, full_k.transpose(2, 3))
    shifted_scores = torch.matmul(shifted_q, shifted_k.transpose(2, 3))
    torch.testing.assert_close(scores, shifted_scores, rtol=1e-5, atol=1e-5)


def test_rotary_low_precision():
    torch.manual_seed(79)
    for dtype in (torch.float16, torch.bfloat16):
        rope = RotaryEmbedding(8).to(dtype=dtype)
        q = torch.randn(1, 2, 5, 8, dtype=dtype, requires_grad=True)
        k = torch.randn_like(q, requires_grad=True)
        output_q, output_k = rope(q, k, offset=1000)
        expected_q, expected_k = rope(q.float(), k.float(), offset=1000)
        assert output_q.dtype == dtype and output_k.dtype == dtype
        torch.testing.assert_close(output_q, expected_q.to(dtype))
        torch.testing.assert_close(output_k, expected_k.to(dtype))
        (output_q.float().sum() + output_k.float().sum()).backward()
        assert torch.isfinite(q.grad).all() and torch.isfinite(k.grad).all()


def test_short_convolution_hand_calculation():
    model = ShortConvolution(1, kernel_size=3, bias=True)
    with torch.no_grad():
        model.conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]]]))
        model.conv.bias.fill_(0.5)
    x = torch.tensor([[[1.0], [2.0], [4.0]]])
    output, state = model(x, use_cache=True)
    expected = F.silu(torch.tensor([[[3.5], [8.5], [17.5]]]))
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(state, torch.tensor([[[2.0, 4.0]]]))


def test_short_convolution_streaming():
    torch.manual_seed(83)
    for kernel_size in (1, 3, 4):
        model = ShortConvolution(4, kernel_size)
        x = torch.randn(2, 7, 4)
        full, final = model(x, use_cache=True)
        assert final.shape == (2, 4, kernel_size - 1)
        for chunk_size in (1, 2, 5):
            outputs = []
            state = None
            for start in range(0, x.size(1), chunk_size):
                output, state = model(x[:, start:start+chunk_size], state, use_cache=True)
                outputs.append(output)
            torch.testing.assert_close(full, torch.cat(outputs, dim=1))
            torch.testing.assert_close(final, state)
        saved = state.clone()
        _, no_cache = model(x[:, :1], state)
        assert no_cache is None
        torch.testing.assert_close(state, saved)


def test_short_convolution_causality_and_gradients():
    torch.manual_seed(89)
    model = ShortConvolution(4, kernel_size=4, bias=True)
    x = torch.randn(2, 7, 4, requires_grad=True)
    output, _ = model(x)
    changed = x.detach().clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:])
    changed_output, _ = model(changed)
    torch.testing.assert_close(output[:, :3], changed_output[:, :3])
    output[:, :3].square().mean().backward()
    torch.testing.assert_close(x.grad[:, 3:], torch.zeros_like(x.grad[:, 3:]))
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
