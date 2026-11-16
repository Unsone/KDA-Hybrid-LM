import torch
from torch import nn
from torch.nn import functional as F


class ShortConvolution(nn.Module):
    """因果逐通道短卷积加 SiLU；输入输出为 [B, T, C]。

    state 保存卷积前的历史输入，形状为 [B, C, kernel_size - 1]。
    每个通道使用独立卷积核，不在通道之间混合信息。
    """

    def __init__(self, hidden_dim: int, kernel_size: int = 4, bias: bool = False):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, groups=hidden_dim, bias=bias)

    def forward(self, x, state=None, use_cache=False):
        batch, length, dim = x.shape
        values = x.transpose(1, 2)
        if state is None:
            state = values.new_zeros(batch, dim, self.kernel_size - 1)
        values = torch.cat((state, values), dim=2)
        output = F.silu(self.conv(values)).transpose(1, 2)

        new_state = None
        if use_cache:
            # 只复制末尾历史，避免小缓存的视图持有整个输入片段。
            new_state = values[:, :, length:].clone()
        return output, new_state
