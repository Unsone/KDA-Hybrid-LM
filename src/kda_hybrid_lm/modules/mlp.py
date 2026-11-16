from torch import nn
from torch.nn import functional as F


class SwiGLU(nn.Module):
    """
    逐 token 的门控前馈网络（FFN）：[B, T, C] -> [B, T, C]。

    计算式：y = down_proj( silu(gate_proj(x)) * up_proj(x) )，
    可以理解成"经典 MLP + 一次逐通道缩放"：
    - up_proj 生成每个中间通道的"内容"；
    - gate_proj 生成每个通道的"旋钮"，经 silu 后作为带符号的缩放系数；
    - 两者逐元素相乘，再由 down_proj 混合回 C 维。

    注意：这里的门控不是 0~1 的开关。silu(z) = z * sigmoid(z) 的输出
    可正可负、也可大于 1，所以它同时能放大、抑制和翻转某个通道。

    命名：*_proj 是 projection（线性投影）的缩写，就是 nn.Linear；
    up/down 指维度相对 hidden_dim 的升高/降低（C -> I、I -> C）。

    参数量 = 3 * C * I（bias=False 时，C = hidden_dim，I = intermediate_dim）；
    通常取 I ≈ 8/3 * C，使总参数量与经典 4C 的 FFN 持平。
    本模块只做逐 token 变换、不混合序列；残差和归一化在 block 层组合时添加。
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        # gate_proj：C -> I，产出门控分数（silu 之前）
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        # up_proj：C -> I，产生每个中间通道的内容（不加激活）
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        # down_proj：I -> C，把门控后的中间表示混合回 hidden_dim
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        # 训练时以概率 p 随机置零，eval() 时自动变成恒等；p=0 表示不启用
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]；nn.Linear 只作用最后一维，等价于 B*T 个 C 维向量各过一次 MLP。
        # 门控分支：先做线性投影，再套 silu(z) = z * sigmoid(z)：
        # z 越大越接近"按原样放行"，z 很负时输出接近 0 但仍保留小的负值。
        gate = F.silu(self.gate_proj(x))  # [B, T, I]
        # 值分支：只做线性，不加激活
        values = self.up_proj(x)  # [B, T, I]
        # 逐通道相乘：gate 是缩放系数、values 是内容；再投影回 C 维
        output = self.down_proj(gate * values)  # [B, T, C]
        return self.dropout(output)
