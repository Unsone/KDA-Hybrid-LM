from dataclasses import dataclass


MIXER_TYPES = ("softmax", "linear", "delta", "gdn", "kda")


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 256
    hidden_dim: int = 128
    intermediate_dim: int = 344
    num_heads: int = 4
    # 每一项对应一层；默认纯 Transformer，混合架构需显式指定。
    layer_types: tuple[str, ...] = ("softmax", "softmax")
    dropout: float = 0.0
    bias: bool = False
    rope: bool = True
    rope_base: float = 10000.0
    tie_embeddings: bool = False
    embedding_std: float = 0.02

    def __post_init__(self):
        object.__setattr__(self, "layer_types", tuple(self.layer_types))
        if min(self.vocab_size, self.hidden_dim, self.intermediate_dim, self.num_heads) <= 0:
            raise ValueError("Model dimensions must be positive")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not self.layer_types or any(t not in MIXER_TYPES for t in self.layer_types):
            raise ValueError(f"layer_types must contain names from {MIXER_TYPES}")
        if self.rope and "softmax" in self.layer_types and (self.hidden_dim // self.num_heads) % 2:
            raise ValueError("RoPE requires an even head_dim")
        if not 0 <= self.dropout < 1 or self.embedding_std <= 0 or self.rope_base <= 0:
            raise ValueError("Invalid dropout, embedding_std or rope_base")
