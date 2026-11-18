import torch
from torch import nn
from torch.nn import functional as F

from .base import TokenMixer


class DeltaNet(TokenMixer):
    """
    多头递推 DeltaNet；输入输出为 [B, T, C]，状态为 [B, H, D, D]。
    beta 控制预测误差的修正幅度，不包含 GDN 的全局遗忘门。
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

    def forward(self, x, state=None, use_cache=False):
        # 计算 Q, K, V。
        batch, length, dim = x.shape
        q_projected = self.q_proj(x)
        q = self._split_heads(q_projected, batch, length)
        k_projected = self.k_proj(x)
        k = self._split_heads(k_projected, batch, length)
        v_projected = self.v_proj(x)
        v = self._split_heads(v_projected, batch, length)

        # 归一化 Q、K，使用 FP32 进行状态计算。
        q = F.normalize(q.float(), dim=3)
        k = F.normalize(k.float(), dim=3)
        v = v.float()

        # 每个 token、每个头对应一个写入强度：[B, H, T, 1]。
        beta_logits = self.beta_proj(x).float()
        beta = torch.sigmoid(beta_logits)
        beta = beta.view(batch, length, self.num_heads, 1)
        beta = beta.transpose(1, 2)

        # 初始化记忆矩阵，形状为 [B, H, D, D]。
        # 把 S 理解为一个从 key 空间映射到 value 空间的线性映射。
        if state is None:
            state = torch.zeros(batch, self.num_heads, self.head_dim, self.head_dim, 
                                device=x.device, dtype=torch.float32)

        outputs = []

        # 逐个处理每个 token。
        for t in range(length):
            # 保留时间维：Q/K/V 为 [B, H, 1, D]，beta 为 [B, H, 1, 1]。
            q_t = q[:, :, t:t+1, :]
            k_t = k[:, :, t:t+1, :]
            v_t = v[:, :, t:t+1, :]
            beta_t = beta[:, :, t:t+1, :]

            # 读取旧记忆，根据误差修正状态。
            # 使用行向量形式读取记忆：[B, H, 1, D]。
            memory_for_reading = state.transpose(2, 3)
            prediction = torch.matmul(k_t, memory_for_reading)
            # 计算误差，并使用 beta 修正状态。
            error = v_t - prediction
            correction = beta_t * error
            # 更新记忆，[B,H,D,1] × [B,H,1,D] -> [B,H,D,D]。
            correction_column = correction.transpose(2, 3)
            update = torch.matmul(correction_column, k_t)
            state = state + update

            # 使用当前 query 读取更新后的记忆。
            memory_for_reading = state.transpose(2, 3)
            output_t = torch.matmul(q_t, memory_for_reading)
            # 每个 token 的输出形状为 [B, H, 1, D]。
            outputs.append(output_t)

        # 沿已有的时间维拼接，得到 [B, H, T, D]。
        output = torch.cat(outputs, dim=2)
        # 交换维度、线性变换、dropout，得到最终输出。
        output = output.transpose(1, 2)
        output = output.reshape(batch, length, dim)
        output = output.to(x.dtype)
        output = self.out_proj(output)
        output = self.dropout(output)

        new_state = None
        if use_cache:
            new_state = state
        return output, new_state
