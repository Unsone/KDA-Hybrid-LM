import torch
from torch import nn
from torch.nn import functional as F

from .base import TokenMixer


class DeltaNet(TokenMixer):
    """
    多头递推 DeltaNet；输入输出为 [B, T, C]，状态为 [B, H, D, D]。
    beta 控制预测误差的修正幅度，不包含 GDN 的全局遗忘门。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)

    def forward(self, x, state=None, use_cache=False):
        # 计算 Q, K, V。
        batch, length, dim = x.shape
        q = self._split_heads(self.q_proj(x), batch, length)
        k = self._split_heads(self.k_proj(x), batch, length)
        v = self._split_heads(self.v_proj(x), batch, length)

        # 归一化 Q、K，使用 FP32 进行状态计算。
        q = F.normalize(q.float(), dim=3)
        k = F.normalize(k.float(), dim=3)
        v = v.float()

        # 每个 token、每个头对应一个写入强度：[B, H, T, 1]。
        beta = torch.sigmoid(self.beta_proj(x).float())
        beta = beta.view(batch, length, self.num_heads, 1).transpose(1, 2)

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
            prediction = torch.matmul(k_t, state.transpose(2, 3))
            # 计算误差，并使用 beta 修正状态。
            error = v_t - prediction
            correction = beta_t * error
            # 更新记忆，[B,H,D,1] × [B,H,1,D] -> [B,H,D,D]。
            state = state + torch.matmul(correction.transpose(2, 3), k_t)

            # 使用当前 query 读取更新后的记忆。
            output_t = torch.matmul(q_t, state.transpose(2, 3))
            # 每个 token 的输出形状为 [B, H, 1, D]。
            outputs.append(output_t)

        # 沿已有的时间维拼接，得到 [B, H, T, D]。
        output = torch.cat(outputs, dim=2)
        # 交换维度、线性变换、dropout，得到最终输出。
        output = output.transpose(1, 2).reshape(batch, length, dim)
        output = self.out_proj(output.to(x.dtype))
        output = self.dropout(output)

        new_state = None
        if use_cache:
            new_state = state
        return output, new_state
