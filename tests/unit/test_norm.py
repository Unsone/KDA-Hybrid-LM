import torch
from torch.nn import functional as F

from kda_hybrid_lm.modules import RMSNorm


def test_matches_pytorch_outputs_and_gradients():
    torch.manual_seed(59)
    for shape in ((2, 3, 8), (3, 8), (8,), (1, 1, 1)):
        norm = RMSNorm(shape[-1])
        with torch.no_grad():
            norm.weight.copy_(torch.randn(shape[-1]))
        x = torch.randn(shape, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        reference_weight = norm.weight.detach().clone().requires_grad_()
        actual = norm(x)
        expected = F.rms_norm(reference_x, (shape[-1],), reference_weight, norm.eps)
        torch.testing.assert_close(actual, expected)
        gradient = torch.randn_like(actual)
        actual.backward(gradient)
        expected.backward(gradient)
        torch.testing.assert_close(x.grad, reference_x.grad)
        torch.testing.assert_close(norm.weight.grad, reference_weight.grad)


def test_zero_input():
    norm = RMSNorm(8)
    torch.testing.assert_close(norm.weight, torch.ones(8))
    x = torch.zeros(2, 3, 8, requires_grad=True)
    output = norm(x)
    torch.testing.assert_close(output, torch.zeros_like(x))
    output.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(norm.weight.grad).all()


def test_low_precision():
    torch.manual_seed(61)
    for dtype in (torch.float16, torch.bfloat16):
        norm = RMSNorm(8)
        x = torch.randn(2, 3, 8, dtype=dtype, requires_grad=True)
        output = norm(x)
        expected = F.rms_norm(x.float(), (8,), norm.weight, norm.eps).to(dtype)
        assert output.dtype == dtype
        torch.testing.assert_close(output, expected)
        output.float().square().mean().backward()
        assert torch.isfinite(x.grad).all()
        assert torch.isfinite(norm.weight.grad).all()
