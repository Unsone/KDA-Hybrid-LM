import torch
from torch import nn
from torch.nn import functional as F

from .base import TokenMixer


class GatedDeltaNet(TokenMixer):
    """
    标量 sigmoid 遗忘门的递推 GDN；状态为 [B, H, D, D]。
    事实上，相较于 DeltaNet，GDN 仅仅在状态更新时增加了一个遗忘门 alpha。
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.alpha_proj = nn.Linear(hidden_dim, num_heads, bias=True)

    def forward(self, x, state=None, use_cache=False):
        batch, length, dim = x.shape
        q_projected = self.q_proj(x)
        q = self._split_heads(q_projected, batch, length)
        k_projected = self.k_proj(x)
        k = self._split_heads(k_projected, batch, length)
        v_projected = self.v_proj(x)
        v = self._split_heads(v_projected, batch, length)
        q = F.normalize(q.float(), dim=3)
        k = F.normalize(k.float(), dim=3)
        v = v.float()

        # beta 控制写入修正，alpha 控制旧状态保留比例。
        beta_logits = self.beta_proj(x).float()
        beta = torch.sigmoid(beta_logits)
        beta = beta.view(batch, length, self.num_heads, 1)
        beta = beta.transpose(1, 2)
        alpha_logits = self.alpha_proj(x).float()
        alpha = torch.sigmoid(alpha_logits)
        alpha = alpha.view(batch, length, self.num_heads, 1)
        alpha = alpha.transpose(1, 2)

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
            memory_for_reading = state.transpose(2, 3)
            prediction = torch.matmul(k_t, memory_for_reading)
            error = v_t - prediction
            correction = beta_t * error
            correction_column = correction.transpose(2, 3)
            update = torch.matmul(correction_column, k_t)
            state = state + update

            # 计算本轮输出，输出为 [B, H, 1, D]。
            memory_for_reading = state.transpose(2, 3)
            output_t = torch.matmul(q_t, memory_for_reading)
            outputs.append(output_t)

        output = torch.cat(outputs, dim=2)
        output = output.transpose(1, 2)
        output = output.reshape(batch, length, dim)
        output = output.to(x.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        new_state = None
        if use_cache:
            new_state = state
        return output, new_state
