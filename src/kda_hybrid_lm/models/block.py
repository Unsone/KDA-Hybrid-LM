from torch import nn
from kda_hybrid_lm.mixers import TokenMixer
from kda_hybrid_lm.modules import RMSNorm, SwiGLU

class HybridBlock(nn.Module):
    """
    Pre-Norm Block.
    其输入、输出均为 [B, L, D]。
    """

    def __init__(self, hidden_dim:int, intermediate_dim:int, mixer:TokenMixer, 
                dropout:float=0.0, bias:bool=False):
        super().__init__()
        self.mixer = mixer
        self.mixer_norm = RMSNorm(hidden_dim)
        self.mlp_norm = RMSNorm(hidden_dim)
        self.mlp = SwiGLU(hidden_dim, intermediate_dim, dropout, bias)

    def forward(self, x, state=None, use_cache=False):
        """
        x: [B, L, D]
        state: [B, L, D] or None
        use_cache: bool
        """
        mixed, new_state = self.mixer(self.mixer_norm(x), state, use_cache)
        x = x + mixed
        x = x + self.mlp(self.mlp_norm(x))
        return x, new_state
