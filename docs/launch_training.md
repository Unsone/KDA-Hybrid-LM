# 快速启动训练

Linux 服务器从 clone、安装到后台训练请直接阅读 [Linux 服务器指南](linux_server.md)。Bash 入口为 `bash start_training.sh smoke/prepare/compare`；以下保留本机 PowerShell 用法及通用实验口径。

## 本机直接试跑

在项目根目录的 PowerShell 中运行：

```powershell
.\start_training.ps1
```

它优先使用项目 `.venv/Scripts/python.exe`，否则使用 PATH 中的 Python。不自动安装依赖、下载数据或开始长训练。默认 smoke 模式在 CPU/GPU 自动选择的设备上运行微型纯 Transformer，对 Adam、Adam-S 和 shuffled 各做一个候选的短搜索，再以 seed 42、43 配对复验；输入是内置的合成英文句子，使用 byte token。这只用于检查流程，不能用于研究结论。

跨平台等价入口（使用已安装项目依赖的 Python）：

```bash
python scripts/train_lm.py --mode smoke
```

输出目录会打印在终端，默认 `outputs/metrics/smoke-时间戳/`，可指定 `--output`；已有输出目录不会覆盖。按 Ctrl+C 停止；当前没有断点恢复入口。

## 准备真实数据

真实训练需要额外的 BPE 分词依赖，先在训练所用环境安装：

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv pip install --python .venv/Scripts/python.exe -r requirements-training.txt
.\.venv\Scripts\python.exe scripts/prepare_data.py --download
```

已有激活环境时，也可使用 `python -m pip install -r requirements-training.txt` 和 `python scripts/prepare_data.py --download`。GPU 机器需先安装适配设备的 PyTorch；此脚本不会把项目 CPU 环境自动换成 CUDA 环境。

数据来源为 [TinyStories 原版 TXT](https://huggingface.co/datasets/roneneldan/TinyStories)，默认锁定 revision `f54c09f`，首次下载约 2GB。若网络无法访问 Hugging Face，可先取得 `TinyStories-train.txt` 和 `TinyStories-valid.txt`，然后指定本地文件：

```powershell
.\.venv\Scripts\python.exe scripts/prepare_data.py --train-text data/raw/TinyStories-train.txt --valid-text data/raw/TinyStories-valid.txt
```

文件使用 `<|endoftext|>` 分隔故事。预处理仅用训练故事拟合 8K ByteLevel BPE；移除训练与官方验证之间的完全重复故事，并对训练内部去重。官方验证故事按内容哈希分为调参 validation 与最终 test，相同内容不会跨这两个划分。保留最多 6000 万训练 token、各 100 万 held-out token，保存实际数量、tokenizer 及 token 数据哈希。每个故事追加 EOS；训练窗口允许跨故事边界。

结果默认在 `data/processed/tinystories/`。若需重建，指定新的 `--output`，避免覆盖已有数据。近似重复不在本次去重范围内。

## GPU 短试跑和正式对照

先对真实数据做一次短训练测吞吐与显存：

```powershell
.\start_training.ps1 --mode train --device cuda --steps 100 --optimizer adam --lr 0.001
```

`train` 模式只运行指定优化器，使用 `--seeds` 的第一个种子；真实数据不会在此阶段自动下载。正式启动全部对照：

```powershell
.\start_training.ps1 --mode compare --device cuda
```

Linux 或已激活远程环境：

```bash
python scripts/train_lm.py --mode compare --device cuda
```

正式默认设置：

| 项目 | 默认值 |
|---|---|
| 架构 | 6 层全注意力 Transformer，d=256，4 heads，FFN=688 |
| 序列 / microbatch / 梯度累积 | 256 / 8 / 4；每步 8192 个目标 token |
| 学习率搜索 | 每臂 5 个候选，每候选 611 步，固定 tune seed=7 |
| Adam LR 候选 | 0.0001、0.0003、0.001、0.003、0.01 |
| Adam-S / shuffled LR 候选 | 0.003、0.01、0.03、0.1、0.3 |
| 复验 | 每臂 seed 42、43、44，各 6104 步，约 5000 万 token |
| 验证 | 每 100 步及末步，固定前缀最多 65536 个目标 token |
| 优化 | FP32，恒定 LR，Adam betas=(0.9,0.999)，eps=1e-8，无 weight decay 或梯度裁剪 |
| 权重统计 | step=0 以及每一步；更新幅度以更新前范数为分母 |

每臂按搜索末步的 validation NLL 单独选 LR；各候选使用相同预算。复验 seed 与 tune seed 不重叠；shuffled 排列 seed=`shuffle_seed + training_seed`。记录失败的非有限损失候选，OOM 和其它运行错误直接退出，便于用户统一调整 microbatch。最终 test 只在搜索和训练完成后评估，不用于选择 LR。

每次训练重新设定所有 RNG；独立数据 RNG 保证配对数据顺序相同，初始化 SHA256 可验证三臂初值相同。打乱 RNG 不消耗模型/数据 RNG。使用确定性 PyTorch 算法，同设备环境可复验，但不同设备和 PyTorch 版本不保证逐位一致。

显存不足时保持有效 batch 的一种调整：

```powershell
.\start_training.ps1 --mode compare --device cuda --batch-size 4 --accumulation 8
```

若统计开销过大，可加 `--trace-every 10`：前 20 步仍逐步记录，此后每 10 步记录一次真实的单步更新（不是 10 步累计差）。这会降低差异起始时间的分辨率。`--no-plots` 可关闭绘图；原始 CSV 和损失日志照常保存。当前不含自动早期分离层检测或统计显著性推断。

## 输出文件

- `launch.json`、`data-metadata.json`、`source/`：运行参数、真实数据标识与实际运行的源码快照。
- `search.json`、`selected_lrs.json`：全部搜索结果及每臂选出的 LR；发散候选保留 `failure.json`。
- `repeat/<optimizer>-seed<seed>/config.json`：初始化哈希、原始 RMS、实际 s、尺度来源和随机种子。
- 各运行的 `metrics.jsonl`、`weights.csv`、`last.pt`：逐步损失与验证、逐权重三项指标/RMS、最终模型与优化器状态。
- `summary.json`：全部复验的 validation/test NLL、PPL、实际 token 预算与包含验证/统计的吞吐率。保留每个 seed，不只输出平均值。
- `figures/loss.png` 及每个权重名称的 PNG：损失和逐张量范数、相对初值、单步相对更新曲线。

`norm` 指展平后的欧氏范数，不是矩阵谱范数；step=0 的相对更新以及零分母比值写为 NaN。`.weight`（包括 Norm）参与缩放和非零尺度排列，bias 不缩放；零初始化权重的 Adam-S 学习率为零。模型快照可加载，但完整的中断续跑尚未接入。
