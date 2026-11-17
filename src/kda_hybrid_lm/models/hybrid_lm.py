from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from kda_hybrid_lm.mixers import SoftmaxAttention, LinearAttention, DeltaNet, GatedDeltaNet, KDA
from kda_hybrid_lm.modules import RMSNorm
from .block import HybridBlock
from .config import ModelConfig
from .state import ModelState


@dataclass
class LMOutput:
    logits: Tensor
    loss: Tensor | None = None
    state: ModelState | None = None


class HybridLM(nn.Module):
    """Embedding → Pre-Norm 混合层 → RMSNorm → 词表 logits。"""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        mixer_classes = dict(zip(
            ("softmax", "linear", "delta", "gdn", "kda"),
            (SoftmaxAttention, LinearAttention, DeltaNet, GatedDeltaNet, KDA),
        ))
        self.layers = nn.ModuleList()
        for kind in config.layer_types:
            kwargs = {"rope": config.rope, "rope_base": config.rope_base} if kind == "softmax" else {}
            mixer = mixer_classes[kind](
                config.hidden_dim, config.num_heads, config.dropout, config.bias, **kwargs,
            )
            self.layers.append(HybridBlock(
                config.hidden_dim, config.intermediate_dim, mixer, config.dropout, config.bias,
            ))
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def _init_weights(self, module):
        # Linear 保留 PyTorch 的 fan-in 初始化，使不同形状有不同初始尺度。
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.embedding_std)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, labels=None, state=None, use_cache=False,
                shift_labels=True, ignore_index=-100):
        """默认 labels 与输入同位置，内部移位一次预测下一 token。

        MQAR 或已移位的目标用 shift_labels=False；无 padding mask，文本需预先打包。
        训练批次通常不传 state；缓存不会自动截断梯度。
        """
        if state is not None and len(state.layers) != len(self.layers):
            raise ValueError("Cache layer count does not match model")
        x = self.embedding(input_ids)
        new_states = []
        for i, layer in enumerate(self.layers):
            x, layer_state = layer(x, None if state is None else state.layers[i], use_cache)
            if use_cache:
                new_states.append(layer_state)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            predictions, targets = (logits[:, :-1], labels[:, 1:]) if shift_labels else (logits, labels)
            # 全部忽略（或 T=1 移位后为空）时返回可反传的零，避免 NaN。
            if (targets != ignore_index).any():
                loss = F.cross_entropy(
                    predictions.float().reshape(-1, self.config.vocab_size),
                    targets.reshape(-1), ignore_index=ignore_index,
                )
            else:
                loss = logits.sum() * 0
        new_state = None
        if use_cache:
            tokens_seen = (0 if state is None else state.tokens_seen) + input_ids.size(1)
            new_state = ModelState(tuple(new_states), tokens_seen)
        return LMOutput(logits, loss, new_state)
