from torch import nn


class TokenMixer(nn.Module):
    """Mixer 的公共投影和多头形状处理；具体计算由子类实现。"""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, t, batch, length):
        # [B, T, C] -> [B, H, T, D]
        t = t.view(batch, length, self.num_heads, self.head_dim)
        return t.transpose(1, 2)
