# Linux 服务器：从 clone 到训练

需要 Python 3.11 或 3.12（含 venv/pip）、Git、Bash。GPU 实验需要可被 `nvidia-smi` 识别的 NVIDIA 显卡和兼容驱动；安装脚本不修改驱动、不使用 sudo。

```bash
git clone https://github.com/Unsone/KDA-Hybrid-LM.git
cd KDA-Hybrid-LM
bash scripts/setup_linux.sh
bash start_training.sh smoke
```

安装器创建项目内 `.venv`，安装 PyTorch **2.9.1 / CUDA 12.6** 和固定的 Linux 训练依赖，安装当前项目，并执行依赖检查及 CUDA 矩阵乘法检查。默认不读取 Windows 本机的 `requirements.txt`，项目安装元数据也不再要求该文件中本机的 PyTorch 版本。

`smoke` 使用内置合成句子快速验证三种优化器、搜索、复验、检查点和绘图，不下载语料，也不是正式结果。安装和启动均无需手动激活虚拟环境。

## 安装选项

如果系统默认 Python 版本不合适，指定已有的 Python 3.11/3.12：

```bash
PYTHON=python3.11 bash scripts/setup_linux.sh
```

若服务器没有合适的 Python，也可以先用已有 Conda 安装创建环境，再运行安装器：

```bash
conda create -n kda-python python=3.11 -y
conda activate kda-python
bash scripts/setup_linux.sh
```

对于 RTX 50 系列等需要较新 CUDA wheel 的设备，可以选择 CUDA 12.8：

```bash
TORCH_BACKEND=cu128 bash scripts/setup_linux.sh
```

脚本默认使用官方 PyTorch wheel 源，支持 `TORCH_BACKEND=cpu/cu126/cu128/cu130`。其它 PyTorch 版本或 CUDA 通道需自行确保版本组合与驱动兼容，参考 [PyTorch 官方版本安装命令](https://pytorch.org/get-started/previous-versions/)。wheel 自带运行时依赖，脚本不编译 CUDA 内核。

CPU 服务器只做流程验证：

```bash
TORCH_BACKEND=cpu bash scripts/setup_linux.sh
bash start_training.sh smoke --device cpu
```

若复用已有 Linux 虚拟环境中的 PyTorch，给安装与启动设置相同的 `VENV_DIR`：

```bash
VENV_DIR=/path/to/existing/env TORCH_BACKEND=existing bash scripts/setup_linux.sh
VENV_DIR=/path/to/existing/env bash start_training.sh smoke
```

此模式不重装 PyTorch，但会安装清单中的其它依赖。安装器不共享服务器其它环境；默认 `.venv` 已存在时只更新此环境。安装后的实际依赖保存在 `.venv/server-environment.txt`。

如默认 PyPI 连接不稳定，可给普通依赖指定镜像；PyTorch 仍使用官方源：

```bash
PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple bash scripts/setup_linux.sh
```

## 数据与正式实验

```bash
# 第一次下载原版 TinyStories TXT（约 2GB）并训练 BPE、生成固定数据划分
bash start_training.sh prepare --download

# 先测真实数据上的显存、吞吐和损失
bash start_training.sh train --device cuda --steps 100 --optimizer adam --lr 0.001

# 等预算 LR 搜索 + Adam / Adam-S / shuffled 各三个种子的复验
bash start_training.sh compare --device cuda
```

已有本地数据时，使用以下命令替代下载步骤。数据必须为以 `<|endoftext|>` 分隔故事的原版 TXT 格式：

```bash
bash start_training.sh prepare \
  --train-text /path/to/TinyStories-train.txt \
  --valid-text /path/to/TinyStories-valid.txt
```

默认正式配置为六层约 900 万参数的纯 Transformer，每步 8192 个目标 token；每臂五个 LR 候选、搜索每次约 500 万 token、正式复验每次约 5000 万 token。当前使用 FP32、恒定学习率、无 weight decay。完整口径见 [训练指南](launch_training.md)。

显存不足时可以保持有效 batch 不变：

```bash
bash start_training.sh compare --device cuda --batch-size 4 --accumulation 8
```

## 后台运行与输出

SSH 断开后仍需继续训练时，在后台运行：

```bash
mkdir -p outputs/logs
nohup bash start_training.sh compare --device cuda \
  > outputs/logs/adam-comparison.log 2>&1 < /dev/null &
echo "PID: $!"
tail -f outputs/logs/adam-comparison.log
```

终端或日志会打印具体输出目录。`outputs/metrics/compare-时间戳/` 内包含 `summary.json`、LR 搜索记录、各次运行的 `metrics.jsonl`、`weights.csv`、`last.pt`，以及 `figures/` 的损失和逐权重曲线。首次训练之前先检查 smoke 和真实数据短试跑；当前没有中断续训入口，`last.pt` 在单次运行结束时保存，长实验更适合放在稳定会话中。

验证代码：

```bash
.venv/bin/python -m pytest tests -q
```

本地已验证 CPU 流程、BPE 小样本和 Bash 语法/参数转发；完整 Linux 安装及 GPU 长训练需要在目标服务器验收，不把本机验证等同于服务器验证。
