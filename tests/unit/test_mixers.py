import torch
from torch.nn import functional as F

from kda_hybrid_lm.mixers.delta_net import DeltaNet
from kda_hybrid_lm.mixers.gated_delta_net import GatedDeltaNet
from kda_hybrid_lm.mixers.kda import KDA
from kda_hybrid_lm.mixers.linear_attention import LinearAttention


def test_streaming_and_causality():
    torch.manual_seed(42)
    x = torch.randn(2, 7, 16)
    for mixer_class in (LinearAttention, DeltaNet, GatedDeltaNet, KDA):
        model = mixer_class(16, 4).eval()
        initial = torch.randn(2, 4, 4, 4)
        saved = initial.clone()
        full, final = model(x, initial, use_cache=True)
        first, state = model(x[:, :3], initial, use_cache=True)
        second, state = model(x[:, 3:], state, use_cache=True)
        torch.testing.assert_close(full, torch.cat((first, second), dim=1))
        torch.testing.assert_close(final, state)
        torch.testing.assert_close(initial, saved)
        pieces = []
        state = initial
        for t in range(x.size(1)):
            piece, state = model(x[:, t:t+1], state, use_cache=True)
            pieces.append(piece)
        torch.testing.assert_close(full, torch.cat(pieces, dim=1))
        torch.testing.assert_close(final, state)
        changed = x.clone()
        changed[:, 3:] = torch.randn_like(changed[:, 3:])
        changed_output, no_cache = model(changed, initial)
        torch.testing.assert_close(full[:, :3], changed_output[:, :3])
        assert no_cache is None


def test_gradients_and_output_dropout():
    torch.manual_seed(7)
    for mixer_class in (LinearAttention, DeltaNet, GatedDeltaNet, KDA):
        model = mixer_class(16, 4, dropout=0.5, bias=True)
        x = torch.randn(2, 7, 16, requires_grad=True)
        output, state = model(x, use_cache=True)
        assert output.shape == x.shape
        assert torch.count_nonzero(output == 0) > 0
        other_output, other_state = model(x, use_cache=True)
        torch.testing.assert_close(state, other_state)
        assert not torch.equal(output, other_output)
        output.square().mean().backward()
        assert torch.isfinite(x.grad).all()
        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        model.eval()
        first, _ = model(x)
        second, _ = model(x)
        torch.testing.assert_close(first, second)


def test_linear_matches_causal_attention():
    torch.manual_seed(11)
    model = LinearAttention(8, 2)
    x = torch.randn(2, 5, 8)
    q = F.normalize(model._split_heads(model.q_proj(x), 2, 5), dim=3)
    k = F.normalize(model._split_heads(model.k_proj(x), 2, 5), dim=3)
    v = model._split_heads(model.v_proj(x), 2, 5)
    weights = torch.matmul(q, k.transpose(2, 3))
    weights = torch.tril(weights)
    expected = torch.matmul(weights, v).transpose(1, 2).reshape(2, 5, 8)
    expected = model.out_proj(expected)
    actual, state = model(x, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, torch.matmul(v.transpose(2, 3), k))


def test_gated_delta_matches_transition_matrix():
    torch.manual_seed(17)
    model = GatedDeltaNet(4, 2)
    x = torch.randn(2, 1, 4)
    initial = torch.randn(2, 2, 2, 2)
    q = F.normalize(model._split_heads(model.q_proj(x), 2, 1), dim=3)
    k = F.normalize(model._split_heads(model.k_proj(x), 2, 1), dim=3)
    v = model._split_heads(model.v_proj(x), 2, 1)
    alpha = torch.sigmoid(model.alpha_proj(x)).transpose(1, 2).reshape(2, 2, 1, 1)
    beta = torch.sigmoid(model.beta_proj(x)).transpose(1, 2).reshape(2, 2, 1, 1)
    transition = torch.eye(2) - beta * torch.matmul(k.transpose(2, 3), k)
    expected_state = alpha * torch.matmul(initial, transition)
    expected_state = expected_state + beta * torch.matmul(v.transpose(2, 3), k)
    expected = torch.matmul(expected_state, q.transpose(2, 3))
    expected = model.out_proj(expected.transpose(2, 3).transpose(1, 2).reshape(2, 1, 4))
    actual, state = model(x, initial, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)


def test_kda_matches_diagonal_transition():
    torch.manual_seed(23)
    model = KDA(4, 2)
    x = torch.randn(2, 1, 4)
    initial = torch.randn(2, 2, 2, 2)
    q = F.normalize(model._split_heads(model.q_proj(x), 2, 1), dim=3)
    k = F.normalize(model._split_heads(model.k_proj(x), 2, 1), dim=3)
    v = model._split_heads(model.v_proj(x), 2, 1)
    alpha = model._split_heads(torch.sigmoid(model.alpha_proj(x)), 2, 1)
    decay = torch.diag_embed(alpha[:, :, 0, :])
    beta = torch.sigmoid(model.beta_proj(x)).transpose(1, 2).reshape(2, 2, 1, 1)
    transition = torch.eye(2) - beta * torch.matmul(k.transpose(2, 3), k)
    expected_state = torch.matmul(torch.matmul(initial, decay), transition)
    expected_state = expected_state + beta * torch.matmul(v.transpose(2, 3), k)
    expected = torch.matmul(q, expected_state.transpose(2, 3))
    expected = model.out_proj(expected.transpose(1, 2).reshape(2, 1, 4))
    actual, state = model(x, initial, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)


def test_kda_uniform_channels_match_gated_delta():
    torch.manual_seed(29)
    gdn = GatedDeltaNet(8, 2)
    kda = KDA(8, 2)
    weights = gdn.state_dict()
    weights['alpha_proj.weight'] = weights['alpha_proj.weight'].repeat_interleave(4, dim=0)
    weights['alpha_proj.bias'] = weights['alpha_proj.bias'].repeat_interleave(4, dim=0)
    kda.load_state_dict(weights)
    x = torch.randn(2, 7, 8)
    initial = torch.randn(2, 2, 4, 4)
    expected, expected_state = gdn(x, initial, use_cache=True)
    actual, state = kda(x, initial, use_cache=True)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(state, expected_state)
