# Adam / Adam-S 实验的模型准备

模型、最小训练循环、三种优化器、TXT/BPE 预处理、学习率搜索、统计与绘图入口已实现，运行方式见 [启动指南](launch_training.md)。已验证合成数据短试跑；真实 TinyStories 下载与正式 GPU 长训练尚未执行，测试中的小批次拟合不是实验结论。

## 已实现的模型接口

```python
import copy
import torch
from kda_hybrid_lm.models import HybridLM, ModelConfig

torch.manual_seed(42)
config = ModelConfig(
    vocab_size=256, hidden_dim=128, intermediate_dim=344, num_heads=4,
    layer_types=("softmax",) * 4,
    dropout=0.0, tie_embeddings=False,
)
model = HybridLM(config)
initial = copy.deepcopy(model.state_dict())
tokens = torch.randint(config.vocab_size, (2, 32))
output = model(tokens, labels=tokens)
output.loss.backward()
# 每个优化器及学习率候选均新建 optimizer，再 load_state_dict(initial)。
```

`layer_types` 按层显式指定 `softmax/linear/delta/gdn/kda`，长度即深度。模型使用 Embedding、Pre-Norm 残差 Block、最终 RMSNorm 和独立词表输出头。可选共享输出头，但正式尺度实验默认关闭，以避免同一参数被重复定义尺度。

优化器对照以纯 Transformer 为主架构：所有层使用因果多头 Softmax Attention，其余为 RMSNorm、RoPE 和 SwiGLU。这是常见的现代 decoder-only Transformer 组合，不是完整复刻某个现成模型。`ModelConfig()` 默认两层、hidden_dim=128、4 个头、FFN 中间维度 344，是便于本机验收的小配置；上例显式使用四层。正式训练规模需在数据和预算确定后固定，三臂保持一致。GDN/KDA 仍可选，但不作为本次主实验的必要变量。

Linear 保留 PyTorch 默认 fan-in 均匀初始化；Embedding 为标准差 `embedding_std=0.02` 的正态初始化；Norm 权重为 1，bias 为 0。不同形状的投影保留初始尺度差异，而不是统一改为相同标准差。全注意力默认接入 RoPE，仅旋转新增 Q/K，offset 来自该层 KV 历史长度；递推层沿用现有简化更新，不接入 RoPE 或短卷积。

输出 `LMOutput(logits, loss, state)`：

- `logits` 为 `[B,T,V]`，不是概率。
- 默认 `labels` 与输入对齐，模型内移位一次。已移位标签或 MQAR 查询位置标签使用 `shift_labels=False`。
- 默认忽略标签 `-100`；若任务使用 `-1`，显式设置 `ignore_index=-1`。全忽略批次返回可反传零损失，训练器仍应跳过此类无有效目标的批次。
- `ModelState.layers` 按层保存矩阵或 KV 元组，`tokens_seen` 保存已处理长度。只输入新增 token；训练独立样本时不共享缓存。缓存不自动 detach。
- 当前输入应是等长、无 padding 的打包序列；不支持 attention mask 或文档边界重置。
- `generate(model, input_ids, max_new_tokens)` 支持带缓存的定长贪心生成。

模型没有内部优化器、随机数据生成或自定义参数更新。`named_parameters()` 提供稳定的逐张量名称，例如 `layers.0.mixer.q_proj.weight`、`layers.0.mlp.down_proj.weight`、`embedding.weight` 和 `norm.weight`；共享参数时 PyTorch 默认只枚举一次。

## 正式实验的固定口径

三臂为 Adam、Adam-S、Adam-S shuffled。Adam-S 仅将 Adam 的最终自适应更新乘以初始化时测得的固定 `s=RMS(W0)`，一阶/二阶矩、bias correction、epsilon 均不变，不随训练重算 s。可用每张量 Adam 参数组 `lr=global_lr*s` 实现；统一调度须保持这个比例，检查点需保存 s 与打乱映射。

“权重”定义为所有 `.weight` 参数，包括 embedding、输出头和 Norm；bias 使用相同的全局学习率，不参与打乱。零权重张量按原公式 s=0 保持冻结，并单独标记。若后续改为所有可训练参数都缩放，必须作为明确的不同配置记录，因为零初始化 bias 会被冻结。

shuffled 在非零权重张量间以独立 RNG 做均匀随机排列，保留尺度多重集，不要求无固定点；记录目标张量、原尺度、来源张量、实际尺度及打乱 seed。三臂使用同一个模型初始快照、逐批相同的数据和相同 Dropout RNG；打乱 RNG 不消耗训练 RNG。每个候选都重置模型、优化器及所有 RNG，使用相同训练 token/step 预算和验证时点。

每臂各用相同数量的全局 LR 候选和相同单次预算，允许各自不同的搜索区间（s 会改变有效 LR）。按预先固定的验证 NLL 选 LR，保留失败和发散候选。用未参与选 LR 的至少三个配对训练 seed 复验，shuffled 还需覆盖多个打乱 seed；报告全部结果与均值/波动，禁止拿最终测试指标选 LR。

逐权重张量记录 step=0 与每次 optimizer step：

- `norm = ||vec(W_t)||₂`：张量展平后的欧氏范数（矩阵即 Frobenius 范数，不是谱范数）。
- `norm_ratio = ||W_t||₂ / ||W_0||₂`。
- `relative_update = ||W_{t+1}-W_t||₂ / ||W_t||₂`：分母取更新前，差分取实际更新，包含所有实际参数变化。
- `rms = norm / sqrt(numel)`，用于跨尺寸比较尺度排序；不可拿不同尺寸的原始范数代替 RMS 对应关系。

零分母记缺失值并显式标记，不用 epsilon 伪造比值。范数采样至少覆盖所有早期步骤，避免漏掉差异最早出现的阶段。按参数名逐层绘制三项曲线，与训练损失、固定验证 NLL/PPL（或 MQAR 查询准确率）对齐；验证 NLL 按有效 token 加权。以早期、中期、末期报告初始/当前 RMS 的排序相关性与比值漂移，逐种子比较最早持续分离的层和步骤，避免把单个噪声尖峰称为差异起点。

判断尺度是否持续代表权重大小，需同时看各张量相对初值的漂移与跨张量尺度排序；判断收益是否依赖对应关系，重点比较经过等预算调优的 Adam-S 与 shuffled。日志和逐张量绘图已实现，尺度排序相关性、阶段分析与差异起始检测仍是分析计划，当前没有正式损失、验证或层级差异结论。
