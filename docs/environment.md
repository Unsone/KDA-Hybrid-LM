# 环境与运行

## 已验证的本机环境

Windows / PowerShell，Python 3.11.15，uv，项目内 `.venv`，PyTorch 2.15.0+cpu。直接依赖版本见 [requirements.txt](../requirements.txt)，pyproject.toml 从该文件读取依赖。2026-11-16 全量组件测试 21 项通过；依赖兼容性也可用下面命令检查。

本机负责开发、CPU 数值测试、绘图和文档，远程 GPU 负责正式训练及高效内核实验。当前没有配置本机 CUDA/ROCm，也没有正式 GPU 训练结果。

## 安装或复建（PowerShell）

```powershell
Set-Location D:\UV\KDA-Hybrid-LM
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe --torch-backend cpu --index-url https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
uv pip install --python .venv\Scripts\python.exe --index-url https://pypi.tuna.tsinghua.edu.cn/simple --no-deps -e .
uv pip check --python .venv\Scripts\python.exe
```

本次配置过程中，直接连接 PyPI 遇到 TLS 握手失败，因此基础依赖使用清华镜像；PyTorch 后端仍通过 uv 的 cpu 选项选择。环境已存在时，无需重复创建。

依赖清单固定直接依赖版本，没有锁定全部传递依赖，因此不是跨平台完整锁文件。后续正式实验还应保存实际环境快照与平台信息。

## 运行与测试

无需激活环境：

```powershell
.\.venv\Scripts\python.exe -c "import torch, kda_hybrid_lm; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe -m pytest tests/unit/test_mixers.py tests/unit/test_softmax_attention.py -q
.\.venv\Scripts\python.exe -m pytest tests/unit/test_norm.py tests/unit/test_modules.py -q
```

CPU 环境 CUDA=False 正常。main.py 目前仅输出 Hello，训练脚本与 YAML 配置为空；不要把执行空脚本成功视为训练成功。编辑器选择 `.venv/Scripts/python.exe`；已安装 ipykernel，可直接选择该环境作为 Notebook 内核。

## 远程 GPU / Kaggle

不要复制本机 .venv。将源代码和固定版本配置传到目标环境，先记录 Python、PyTorch、驱动、GPU 型号、显存、计算能力和实际精度支持，再决定依赖与后端；不要在 Notebook 中无条件覆盖整套预装 PyTorch。

若使用 Kaggle，Notebook 只负责准备环境、路径和启动任务，核心实现继续维护在仓库。代码可由固定 Git commit 或版本化代码包获取。`/kaggle/input` 用作只读输入，输出写入 `/kaggle/working`；长期保留需实际保存 Notebook 输出版本或下载，不能只依赖编辑器草稿自动保存。

若分配到 T4，应采用适合它的精度策略，不能假设原生 BF16 加速或当前所有 Triton/FLA 算子可用。若需要这些内核，优先核对较新 GPU 与 [Triton 官方兼容要求](https://github.com/triton-lang/triton#compatibility)。云平台的 GPU 型号、额度和会话上限以账户页面为准，不把免费资源作为连续长期训练的保证。

## 训练中断与精度（后续必须实现）

Trainer 需支持定时检查点、时间预算退出和续训。完整检查点应保存模型、优化器、调度器、FP16 scaler、step/token 数、随机状态、数据采样进度和最终配置。只保存模型权重不能完整恢复训练过程。

当前组件能在 CPU 做部分低精度输入测试，但不能据此声称 GPU AMP 已验证。递推中的 float() 不能单独阻止外部 autocast 降低 matmul 精度；该策略仍待训练集成时明确处理。推理采用 eval() 与 inference_mode()，无关样本之间清空历史。

## 本项目算力使用顺序

先 CPU 测试 → GPU 小尺寸验证 → MQAR → 约 20M 试跑并实测吞吐 → 决定是否开展 100M/1B token 实验。预算测算与显存口径见 [benchmarks.md](benchmarks.md)。本阶段不需要预先开发多卡训练。
