import torch
from torch import nn
from torch.nn import functional as F

from .base import TokenMixer


class GatedDeltaNet(TokenMixer):
    """
    标量 sigmoid 遗忘门的递推 GDN；状态为 [B, H, D, D]。
    事实上，相较于 DeltaNet，GDN 仅仅在状态更新时增加了一个遗忘门 alpha。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)

    def forward(self, x, state=None, use_cache=False):
        batch, length, dim = x.shape
        q = self._split_heads(self.q_proj(x), batch, length)
        k = self._split_heads(self.k_proj(x), batch, length)
        v = self._split_heads(self.v_proj(x), batch, length)
        q = F.normalize(q.float(), dim=3)
        k = F.normalize(k.float(), dim=3)
        v = v.float()

        # beta 控制写入修正，alpha 控制旧状态保留比例。
        beta = torch.sigmoid(self.beta_proj(x).float())
        beta = beta.view(batch, length, self.num_heads, 1).transpose(1, 2)
        alpha = torch.sigmoid(self.alpha_proj(x).float())
        alpha = alpha.view(batch, length, self.num_heads, 1).transpose(1, 2)

        if state is None:
            state = torch.zeros(
                batch, self.num_heads, self.head_dim, self.head_dim,
                device=x.device, dtype=torch.float32,
            )

        outputs = []
        for t in range(length):
            q_t = q[:, :, t:t+1, :]
            k_t = k[:, :, t:t+1, :]
            v_t = v[:, :, t:t+1, :]
            beta_t = beta[:, :, t:t+1, :]
            alpha_t = alpha[:, :, t:t+1, :]

            # [B, H, 1, 1]：每个头用一个标量衰减整个状态矩阵。
            state = alpha_t * state

            # 根据衰减后的状态计算误差，再进行 delta 修正。
            prediction = torch.matmul(k_t, state.transpose(2, 3))
            error = v_t - prediction
            correction = beta_t * error
            state = state + torch.matmul(correction.transpose(2, 3), k_t)

            # 计算本轮输出，输出为 [B, H, 1, D]。
            output_t = torch.matmul(q_t, state.transpose(2, 3))
            outputs.append(output_t)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2).reshape(batch, length, dim)
        output = self.out_proj(output.to(x.dtype))
        output = self.dropout(output)

        new_state = None
        if use_cache:
            new_state = state
        return output, new_state
