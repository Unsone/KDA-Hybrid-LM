# 工程架构与调用链

## 当前边界

已实现内容集中在 `mixers/` 和 `modules/`；训练器、任务、完整模型、配置加载以及 kernels 仍为占位。目录存在不等于功能可运行。

## Mixer 接口

五种组件都采用：

```python
mixer = MixerClass(hidden_dim, num_heads, dropout=0.0, bias=False)
output, new_state = mixer(x, state=None, use_cache=False)
```

`x` 和输出均为 `[B,T,C]`，要求隐藏维度能被头数整除。`T` 是本次输入长度，不是包含历史的总长度。传入 state 后，只输入尚未处理的新 token。

- `state` 决定是否读取历史；`use_cache` 决定是否返回新的历史状态。
- 返回缓存不会自动 detach；跨片段反向传播仍可相连。推理时由调用者使用 `eval()` 和 `torch.inference_mode()`。
- 独立训练样本不共享历史状态，通常以 `state=None` 开始。
- 状态格式由具体 Mixer 决定，基类不统一转换。

| 组件 | state | 增长规律 |
|---|---|---|
| SoftmaxAttention | `(K,V)`，各为 `[B,H,L,D]` | L 随历史增长 |
| 其余四种 Mixer | 单个 `[B,H,D,D]` 张量，行是 value、列是 key | 形状固定 |

`TokenMixer` 定义头数、独立 Q/K/V 与输出投影、Dropout 模块及 `_split_heads()`。各子类保留独立 forward，直接展示自己的公式。没有 QKV 合并、统一数学开关或多层嵌套继承。

Dropout 位置不同：SoftmaxAttention 在注意力权重上应用；四种递推组件在输出投影后应用。相同 p 不代表相同正则化行为。

## 基础组件

| 类 | 输入与输出 | 主要参数 / 约束 |
|---|---|---|
| RMSNorm | 最后一维为 C，形状不变 | `hidden_dim, eps=1e-6`；可学习 weight 为 `[C]` |
| SwiGLU | `[B,T,C] -> [B,T,C]` | `hidden_dim, intermediate_dim, dropout, bias` |
| RotaryEmbedding | Q/K 为 `[B,H,T,D]`，返回旋转后的二者 | 偶数 `head_dim`、base、调用时的 offset；Q/K 的 T 相同，可有不同头数 |
| ShortConvolution | `[B,T,C]`，返回输出与可选 state | `hidden_dim, kernel_size=4, bias`；因果 depthwise Conv1d + SiLU |

短卷积 state 是卷积前历史输入 `[B,C,K-1]`，不是卷积输出。K=1 时缓存长度为零。RoPE 的 offset 是片段首位置；应只旋转新 Q/K，然后再追加新 K 到缓存，不能每步重转已有 K。

## 计划中的模型块

```text
x → RMSNorm → Mixer → 与 x 残差相加
  → RMSNorm → SwiGLU → 与该分支输入残差相加
```

目前尚未组装该 Block。位置编码、短卷积和残差分支 Dropout 的启用方式，将在集成阶段明确配置，不能因组件已经实现就默认网络已使用它们。

全模型计划为 Embedding → 多个 Block → 最终 Norm → 输出头。3:1 混合表示每三层线性 Mixer 接一层全注意力；各层持有自己的状态。后续 `models/state.py` 管理复合状态，包括 KV/矩阵记忆、卷积历史和位置偏移。

## 训练全链路与文件职责（待实现）

```text
scripts/train_*.py
  → 配置与随机种子
  → 数据批次
  → Embedding / Block / 输出头
  → tasks 中的标签对齐与损失
  → loss.backward()
  → optimizer.step() / 学习率调度
  → 日志、定期验证、检查点
```

- `data/`：真实文本读取、分词、缓存与 packing。
- `tasks/`：MQAR/needle 样例、查询标签、LM 目标与损失。下一 token 标签移位只做一次。
- `models/`：模型配置、网络组装、生成与状态容器。
- `train/`：训练循环、优化器、调度、检查点。
- `evaluation/`：验证指标；`experiments/`：批量实验和汇总；`bench/`：性能测量。
- `utils/`：配置合并、种子、日志、环境记录。

## kernels 的后续作用

当前数学计算直接位于各 Mixer 的 forward。后续提取 kernels，是为同一投影网络替换递推、分块和 Triton 实现，独立对拍、测速；不是因为 PyTorch 运算必须放在 kernels 中。

学习阶段允许适量重复，让状态更新清楚可见。当前编程约定是简洁、少防御性分支，不为减少行数引入晦涩表达；递推核心统一使用四维行向量，保留单 token 的时间维，不使用 squeeze/unsqueeze。
