from dataclasses import dataclass

from torch import Tensor


MixerState = Tensor | tuple[Tensor, Tensor]


@dataclass
class ModelState:
    """按模型层顺序保存 KV/矩阵缓存；不自动 detach。"""

    layers: tuple[MixerState, ...]
    tokens_seen: int
