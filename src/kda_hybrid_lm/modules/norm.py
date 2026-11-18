import torch
from torch import nn


class RMSNorm(nn.Module):
    """沿最后一维做 RMS 归一化，再乘以可学习的通道权重。"""

    def __init__(self, hidden_dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, x):
        # 用 FP32 计算均方值和归一化，最终恢复输入精度。
        values = x.float()
        mean_square = values.square().mean(dim=-1, keepdim=True)
        inverse_rms = torch.rsqrt(mean_square + self.eps)
        output = values * inverse_rms
        output = output * self.weight
        return output.to(x.dtype)
