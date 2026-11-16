import torch
from torch import nn
from torch.nn import functional as F

from .base import TokenMixer


class KDA(TokenMixer):
    """
    通道级 sigmoid 遗忘门的递推 KDA；状态为 [B, H, D, D]。
    实现 KDA 的核心更新规则，门控参数化与现有 GDN 保持一致；
    尚未包含原论文完整网络结构与门控参数化。

    与 GDN 的唯一实质区别（本简化实现）是 alpha 的粒度：
    - GDN: alpha 形状 [B, H, T, 1]，每个 (batch, head, token) 一个标量，整体衰减状态；
    - KDA: alpha 形状 [B, H, T, D]，每个 (batch, head, token) 有 D 个系数，
      分别对应 key 的 D 个坐标（即状态矩阵的每一列），因此每个 key 坐标
      可以拥有不同的遗忘速度。

    记号（先分清"维度"与"个数"）：
    - D = head_dim = hidden_dim // num_heads，是每个 key / value 向量的维度（坐标数），
      不是 key 的个数；T = 序列长度，也是每个 (batch, head) 下 key 的个数；
    - 状态 S [D, D]：行 = value 维、列 = key 维，S @ k ≈ v；
      state * alpha_t 的形状是 [B,H,1,D]，沿 value 维（行）广播、逐列相乘，
      等价于 S @ diag(alpha_t)，即每个 key 坐标独立衰减。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__(hidden_dim, num_heads, dropout, bias)
        self.beta_proj = nn.Linear(hidden_dim, num_heads, bias=True)
        self.alpha_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x, state=None, use_cache=False):
        batch, length, dim = x.shape
        q = self._split_heads(self.q_proj(x), batch, length)
        k = self._split_heads(self.k_proj(x), batch, length)
        v = self._split_heads(self.v_proj(x), batch, length)
        q = F.normalize(q.float(), dim=3)
        k = F.normalize(k.float(), dim=3)
        v = v.float()

        # beta 仍是头级标量（每个 token、每个 head 一个），alpha 则为每个 key 坐标生成一个系数。
        # beta: [B,T,H] -> [B,T,H,1] -> [B,H,T,1]
        beta = torch.sigmoid(self.beta_proj(x).float())
        beta = beta.view(batch, length, self.num_heads, 1).transpose(1, 2)
        # alpha: [B,T,C] -> [B,H,T,D]；D 是每个 key 的维度，不是 key 的个数（key 个数是 T）。
        alpha = torch.sigmoid(self.alpha_proj(x).float())
        alpha = self._split_heads(alpha, batch, length)

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

            # alpha_t: [B,H,1,D]，沿 value 维（行）广播、逐列乘各自的系数，
            # 等价于 S <- S @ diag(alpha_t)：每个 key 坐标独立衰减。
            state = state * alpha_t

            # 根据衰减后的状态计算误差，再进行 delta 修正。
            prediction = torch.matmul(k_t, state.transpose(2, 3))
            error = v_t - prediction
            correction = beta_t * error
            state = state + torch.matmul(correction.transpose(2, 3), k_t)
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
