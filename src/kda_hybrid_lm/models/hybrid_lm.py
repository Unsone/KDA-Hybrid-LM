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
        self.layers = nn.ModuleList()
        for kind in config.layer_types:
            layer = self._build_layer(kind)
            self.layers.append(layer)
        self.norm = RMSNorm(config.hidden_dim)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def _build_layer(self, kind):
        """为这一层选择 Mixer，再与 Norm、MLP 和残差组成 Block。"""
        config = self.config
        mixer_classes = {
            "softmax": SoftmaxAttention,
            "linear": LinearAttention,
            "delta": DeltaNet,
            "gdn": GatedDeltaNet,
            "kda": KDA,
        }
        mixer_options = {}
        if kind == "softmax":
            mixer_options["rope"] = config.rope
            mixer_options["rope_base"] = config.rope_base
        mixer_class = mixer_classes[kind]
        mixer = mixer_class(
            config.hidden_dim,
            config.num_heads,
            config.dropout,
            config.bias,
            **mixer_options,
        )
        return HybridBlock(
            config.hidden_dim,
            config.intermediate_dim,
            mixer,
            config.dropout,
            config.bias,
        )

    def _init_weights(self, module):
        # 保留初始化顺序和分布，避免改变已经运行的配对实验。
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.embedding_std)
        elif isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def _forward_layers(self, x, state, use_cache):
        """每层读取自己的旧状态，按相同顺序收集新状态。"""
        new_states = []
        for index, layer in enumerate(self.layers):
            previous_state = None
            if state is not None:
                previous_state = state.layers[index]
            x, layer_state = layer(x, previous_state, use_cache)
            if use_cache:
                new_states.append(layer_state)
        return x, new_states

    def _compute_loss(self, logits, labels, shift_labels, ignore_index):
        if labels is None:
            return None
        if shift_labels:
            predictions = logits[:, :-1]
            targets = labels[:, 1:]
        else:
            predictions = logits
            targets = labels

        # 保持原来的空目标处理：返回仍与计算图相连的零。
        if (targets != ignore_index).any():
            predictions = predictions.float()
            predictions = predictions.reshape(-1, self.config.vocab_size)
            targets = targets.reshape(-1)
            return F.cross_entropy(predictions, targets, ignore_index=ignore_index)
        return logits.sum() * 0

    def forward(self, input_ids, labels=None, state=None, use_cache=False,
                shift_labels=True, ignore_index=-100):
        """默认将 labels 移位一次预测下一 token；MQAR 用 shift_labels=False。

        无 padding mask，文本需预先打包；缓存不会自动截断梯度。
        """
        if state is not None and len(state.layers) != len(self.layers):
            raise ValueError("Cache layer count does not match model")

        x = self.embedding(input_ids)
        x, new_states = self._forward_layers(x, state, use_cache)
        normalized = self.norm(x)
        logits = self.lm_head(normalized)
        loss = self._compute_loss(logits, labels, shift_labels, ignore_index)

        new_state = None
        if use_cache:
            previous_tokens = 0
            if state is not None:
                previous_tokens = state.tokens_seen
            tokens_seen = previous_tokens + input_ids.size(1)
            new_state = ModelState(tuple(new_states), tokens_seen)
        return LMOutput(logits, loss, new_state)
