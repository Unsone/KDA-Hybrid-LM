import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    """相邻特征成对旋转的 RoPE；head_dim 为偶数，Q/K 为 [B, H, T, D]。

    对当前片段的 Q/K 应用旋转后，再将新 K 加入 KV cache。
    offset 是当前片段首个 token 的绝对位置，Q/K 的时间长度相同。
    """

    def __init__(self, head_dim: int, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        # 整数索引随模块移动设备，且不会被 half()/bfloat16() 降低精度。
        self.register_buffer('frequency_indices', torch.arange(0, head_dim, 2), persistent=False)

    def _rotate(self, x, cos, sin):
        even = x[:, :, :, 0::2].float()
        odd = x[:, :, :, 1::2].float()
        output = torch.empty_like(x)
        output[:, :, :, 0::2] = even * cos - odd * sin
        output[:, :, :, 1::2] = even * sin + odd * cos
        return output

    def forward(self, q, k, offset: int = 0):
        length = q.size(2)
        positions = torch.arange(offset, offset + length, device=q.device, dtype=torch.float32)
        exponent = -self.frequency_indices.float() / self.head_dim
        inverse_frequency = self.base ** exponent
        positions = positions.reshape(1, 1, length, 1)
        inverse_frequency = inverse_frequency.reshape(1, 1, 1, self.head_dim // 2)
        angles = positions * inverse_frequency
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        rotated_q = self._rotate(q, cos, sin)
        rotated_k = self._rotate(k, cos, sin)
        return rotated_q, rotated_k
