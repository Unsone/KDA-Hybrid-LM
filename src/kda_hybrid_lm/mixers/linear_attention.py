import torch
from torch.nn import functional as F

from .base import TokenMixer


class LinearAttention(TokenMixer):
    """
    L2 归一化 Q/K 的线性注意力基线，不带分母归一化；状态为 [B, H, D, D]。
    该基线直接累加 key-value 外积，DeltaNet 则根据预测误差修正记忆。
    因此，以下代码暂不加注释，阅读时可参考 DeltaNet。
    """

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

            # 累加当前 key-value 外积，不计算预测误差或遗忘。
            value_column = v_t.transpose(2, 3)
            update = torch.matmul(value_column, k_t)
            state = state + update
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
