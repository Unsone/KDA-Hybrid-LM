# KDA-Hybrid-LM

从零实现线性注意力与 delta 记忆更新，逐步构建可复现的小型混合语言模型。

本项目围绕一个问题展开：**把历史压缩到固定大小的状态矩阵后，能保留多少信息？少量全注意力层能否补足检索能力，同时降低推理缓存成本？**

实现路线从可读的 PyTorch 递推公式开始，再计划推进 MQAR 记忆实验、混合语言模型、分块并行和 Triton 优化。DeltaNet、Gated DeltaNet 与 KDA 均来自已有研究；项目的目标是独立实现、数值验证与对照分析，不将这些算法本身作为原创贡献。

## 当前进度

`AdamS_Lab` 分支新增了[理论 RMS 对照与 Adam / Adam-S 筛选绘图](docs/adam_s_theory_control.md)：可读取既有实验配置，只补跑 `adam_s_theory`，不需要重跑三组基线。以下历史进度记录保留。

截至 **2026-11-17**，本机 CPU 环境执行 `pytest tests -q`：**39 项测试通过**。

| 部分 | 当前状态 |
|---|---|
| 五种 Mixer | 已实现手写 SoftmaxAttention，以及 LinearAttention、DeltaNet、GatedDeltaNet、KDA 的 PyTorch 递推版 |
| 公共基类 | 已提取独立 Q/K/V 和输出投影、Dropout 定义、多头拆分；没有合并 QKV |
| 基础组件 | RMSNorm、SwiGLU、RoPE、因果逐通道短卷积已实现并测试 |
| 模型与集成 | 已实现 HybridBlock、可配置 HybridLM、损失、混合缓存与贪心生成；全注意力已接入可选 RoPE，短卷积尚未接入 |
| 数据与训练 | 已实现最小 LM 训练入口、三种优化器、等预算搜索、多种子复验、TinyStories TXT/BPE 预处理和逐张量记录；MQAR 数据尚未补齐 |
| 高效内核 | `kernels/` 仍为占位，尚无分块或 Triton 实现，也未与 FLA 对拍 |
| 实验结果 | 尚无正式训练曲线、困惑度、记忆任务准确率或 GPU 性能结果 |

测试通过代表当前覆盖范围内的组件与模型行为正确，不代表正式对照实验、GPU/BF16 内核或论文级效果已经验证。`main.py` 目前只输出初始化时的 Hello 信息。

## 已实现的组件

| Mixer | 核心机制 | 推理状态 |
|---|---|---|
| SoftmaxAttention | 因果缩放点积注意力，显式计算分数矩阵 | `(K, V)`，各为 `[B,H,L,D]` |
| LinearAttention | L2 归一化 Q/K，累加 key-value 外积 | `[B,H,D,D]` |
| DeltaNet | 根据旧记忆的预测误差修正关联 | `[B,H,D,D]` |
| GatedDeltaNet | 头级标量遗忘，再进行 delta 修正 | `[B,H,D,D]` |
| KDA | key 通道级遗忘，再进行 delta 修正 | `[B,H,D,D]` |

当前 LinearAttention 是不带分母归一化的关联记忆基线。GDN/KDA 使用简化的 sigmoid 遗忘门，没有复现原论文完整门控参数化、卷积、输出门控等结构。SoftmaxAttention 仍是参考实现，独立调用默认关闭 RoPE，HybridLM 中默认启用；尚无高效注意力后端。

## 快速开始

**Linux GPU 服务器请使用 [Linux 安装与训练指南](docs/linux_server.md)**，clone 后执行：

```bash
bash scripts/setup_linux.sh
bash start_training.sh smoke
bash start_training.sh prepare --download
bash start_training.sh compare --device cuda
```

安装器使用独立虚拟环境和 CUDA PyTorch；无需复制本机环境。完整依赖、CUDA 通道选项和后台运行方式见指南。

本机开发采用 Windows、Python 3.11 和 CPU 版 PyTorch，依赖安装在项目内 `.venv`。以下为 PowerShell 命令：

```powershell
Set-Location D:\UV\KDA-Hybrid-LM
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps -e .
.\.venv\Scripts\python.exe -m pytest tests -q
```

已有环境可直接执行最后一条命令。完整说明、远程训练分工及精度限制见 [环境文档](docs/environment.md)。不要将本机 CPU 安装命令原样用于 GPU 训练。

### 一键试跑

```powershell
.\start_training.ps1
```

默认使用内置合成句子进行三种优化器的 CPU/GPU 自动设备短试跑，生成日志、检查点与逐张量图表；这不是正式实验。真实 TinyStories 数据准备、GPU 启动和参数说明见 [启动指南](docs/launch_training.md)。跨平台入口为 `python scripts/train_lm.py --mode smoke`。

### 组件使用示例

```python
import torch
from kda_hybrid_lm.mixers import KDA
from kda_hybrid_lm.modules import RMSNorm, SwiGLU

mixer = KDA(hidden_dim=32, num_heads=4, dropout=0.0).eval()
norm = RMSNorm(32)
mlp = SwiGLU(hidden_dim=32, intermediate_dim=80)
x = torch.randn(2, 8, 32)

with torch.inference_mode():
    full, final_state = mixer(x, use_cache=True)
    first, state = mixer(x[:, :3], use_cache=True)
    second, state = mixer(x[:, 3:], state=state, use_cache=True)
    torch.testing.assert_close(full, torch.cat((first, second), dim=1))
    torch.testing.assert_close(final_state, state)
    features = mlp(norm(full))

print(full.shape, state.shape, features.shape)
# [2, 8, 32]、[2, 4, 8, 8]、[2, 8, 32]
```

示例演示组件调用，不是完整语言模型或训练任务。历史 `state` 只能传回对应组件；全注意力的 KV 元组与线性层的矩阵状态不能互换。

完整模型使用 `HybridLM(ModelConfig(...))`，默认是两层纯 Transformer（因果全注意力 + RoPE、RMSNorm、SwiGLU），也支持按层指定五种 Mixer。调用 `model(tokens, labels=tokens)` 返回 logits 与下一 token 损失。模型示例与初始化口径见 [模型与优化器实验准备](docs/model_optimizer_experiment.md)。正式启动入口默认使用六层纯 Transformer；目前已验证合成数据短试跑及绘图，尚未执行真实语料的正式搜索或多种子长训练。

## 计划研究的问题

1. **记忆容量**：在 MQAR 上，准确率如何随序列长度、键值对数量和状态维度变化？
2. **门控粒度**：在结构和预算受控的设置下，标量遗忘与通道遗忘是否产生稳定差异？
3. **混合比例**：纯 Transformer、纯 GDN、7:1 与 3:1 混合模型在质量、检索与成本之间如何取舍？KDA 替换为可选扩展。
4. **实现效率**：递推、分块 PyTorch、自写 Triton 和参考库的数值结果与效率分别怎样？

这里的混合指不同层选择不同 Mixer，不是在单层内求平均。目标规模约 100M 参数、每个配置约 1B token，均为受算力约束的计划；先用约 20M 模型试跑。

## 工程结构

```text
src/kda_hybrid_lm/
  mixers/       已实现：五种 Mixer 与公共基类
  modules/      已实现：Norm、MLP、RoPE、短卷积
  kernels/      待实现：分块与 GPU 内核
  models/       已实现：配置、Block、混合 LM、生成与状态管理
  data/         待实现：tokenizer、预处理、序列拼接与文本数据集
  tasks/        待实现：任务数据、标签与损失
  train/        已实现：最小训练、三种 Adam 步长分配、最终检查点与绘图；调度/续跑待实现
  evaluation/   待实现：任务指标
  utils/        待实现：配置、随机种子、日志与环境记录
configs/        配置占位，尚不可用于运行
scripts/        已有 LM 训练与 TinyStories 预处理；其它入口待实现
experiments/    批量实验与结果分析占位
bench/          性能基准占位
tests/unit/     当前可执行的组件测试
tests/integration/  集成测试占位
docs/           项目原理、架构、环境、实验与学习笔记
```

数据、缓存、检查点和生成结果分别放在 `data/`、`checkpoints/`、`outputs/`，默认不进入 Git；正式结果经检查后再选择性发布。

## 下一阶段

- [x] 五种 Mixer、公共基类与统一导出。
- [x] 四种基础模块及组件测试。
- [x] 组装 HybridBlock、完整 LM，验证三种固定步长分配可反向训练。
- [ ] 完成 MQAR 数据与正式训练循环。
- [ ] 固定验证集与多种子记忆实验。
- [x] 全注意力接入 RoPE，验证混合 LM 缓存与全序列输出一致。
- [ ] 接入短卷积及其组合状态。
- [x] 实现 Adam / Adam-S / shuffled 启动入口、等预算搜索、逐层统计与多种子评估流程。
- [ ] 在真实数据/GPU 上执行正式实验与差异阶段分析。
- [ ] 20M 试跑后开展预算允许的混合比例实验。
- [ ] 分块算法、梯度对拍、Triton 前向与效率基准。
- [ ] 发布带真实结果和复现记录的技术报告。

## 文档与参考

阅读入口为 [docs 索引](docs/README.md)：[架构与调用链](docs/architecture.md)、[数学与张量约定](docs/math.md)、[环境](docs/environment.md)、[实验设计](docs/experiments.md)、[基准规范](docs/benchmarks.md)、[报告框架](docs/report.md)、[语法学习笔记](docs/grammar/README.md)。

算法参考：[DeltaNet](https://arxiv.org/abs/2102.11174)、[Gated Delta Networks](https://arxiv.org/abs/2412.06464)、[Kimi Linear / KDA](https://arxiv.org/abs/2510.26692)、[Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)、[RoPE](https://arxiv.org/abs/2104.09864)、[GLU Variants](https://arxiv.org/abs/2002.05202)。论文结果不等同于本仓库结果。
