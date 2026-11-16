import math

import torch

from .base import TokenMixer


class SoftmaxAttention(TokenMixer):
    """多头因果自注意力；输入输出为 [B, T, C]，缓存为 [B, H, S, D]。"""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        self.scale = 1 / math.sqrt(self.head_dim)

    def forward(self, x, state=None, use_cache=False):
        # Compute Q, K, V
        batch, length, dim = x.shape
        # self.q_proj(x) equals to x@self.q_proj.weight.T+self.q_proj.bias.
        q = self._split_heads(self.q_proj(x), batch, length)
        k = self._split_heads(self.k_proj(x), batch, length)
        v = self._split_heads(self.v_proj(x), batch, length)

        # Get new KVCache, using torch.cat in dim=2, which means tokens are appended.
        # state[0] is Kcache, whose shape is [B, H, T, D].
        past_len = 0
        if state is not None:
            past_len = state[0].size(2)
            k = torch.cat((state[0], k), dim=2)
            v = torch.cat((state[1], v), dim=2)

        # Compute ATTENTION scores.
        scores = torch.matmul(q.float(), k.float().transpose(2, 3)) * self.scale

        # Build the CASUAL MASK to prevent attending to future tokens.
        query_pos = past_len + torch.arange(length, device=x.device).reshape(length, 1)
        key_pos = torch.arange(k.size(2), device=x.device).reshape(1, k.size(2))
        future_mask = key_pos > query_pos
        scores = scores.masked_fill(future_mask, float("-inf"))

        # Compute the softmax and apply dropout.
        # dim=3 is the dimension of the key sequence length.
        weights = torch.softmax(scores, dim=3)
        # turn type if weights to v's type.
        # dropout means set random elements to 0 with a probability of 'dropout'.
        weights = self.dropout(weights.to(v.dtype))
        # Compute and Reshape the output.
        output = torch.matmul(weights, v)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        # Make a Linear Projection using W_o.
        output = self.out_proj(output)

        new_state = None
        if use_cache:
            new_state = (k, v)
        return output, new_state
