# Adam-S 理论 RMS 对照与筛选绘图

本次变更只在 `AdamS_Lab` 分支。新增 `adam_s_theory`，实际初始权重不变，仅替换固定学习率倍率。

## 倍率定义

| 参数类型 | `adam_s` | `adam_s_theory` |
|---|---|---|
| Linear | 实测初始 RMS | `1 / sqrt(3 * in_features)` |
| Embedding | 实测初始 RMS | `model.config.embedding_std`，本次为 0.02 |
| RMSNorm | 实测初始 RMS，本次为 1 | 1 |

理论 RMS 指初始化分布的 `sqrt(E[w²])`，不是有限样本 RMS 的期望 `E[sqrt(mean(w²))]`。两者有微小差别。本实验采用前者作为初始化规则所给出的确定性尺度。

两种方法都使用标准 Adam 的一阶、二阶矩、偏差修正、`eps=1e-8`，无 weight decay；不改变梯度、实际初始权重或模型前向。理论尺度不会消耗随机数。参数初始 RMS 和实际使用的 scale 都保存在每次训练的 `config.json/scales` 中。bias 沿用已有实现不缩放，本次模型 `bias=false`。

该规则对应当前 HybridLM 的初始化实现：Linear 保留 PyTorch 默认均匀分布，Embedding 为指定标准差的零均值正态分布，RMSNorm 为全 1。若以后更改初始化规则，应同步更改理论倍率，不能套用到任意自定义初始化上。

## Linux 服务器补跑

建议解压到新的代码目录，复用原来的环境和预处理数据，不覆盖旧代码与实验结果。ZIP 不含数据、虚拟环境、检查点或 Git 历史，无需重新安装依赖。

在本机 Windows PowerShell 上传（复用已配置的 SSH 跳板别名）：

```powershell
scp "D:\UV\KDA-Hybrid-LM\outputs\metrics\packages\KDA-Hybrid-LM-AdamS-Lab-theory.zip" PKU_8x4090_48G:~/lcx/
```

在服务器执行：

```bash
mkdir -p "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab"
unzip "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab-theory.zip" -d "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab"
cd "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab"
export VENV_DIR=/data/yyd/lcx/KDA-Hybrid-LM/.venv

# 先检查四种优化器的完整微型流程，只使用内置合成数据。
CUDA_VISIBLE_DEVICES=4 bash start_training.sh smoke \
  --device cuda --optimizers adam adam_s shuffled adam_s_theory

mkdir -p outputs/logs
LOG="outputs/logs/theory-$(date +%Y%m%d-%H%M%S).log"
CUDA_VISIBLE_DEVICES=4 nohup bash start_training.sh theory \
  --reference /data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM/outputs/metrics/compare-20261117-141000-525408 \
  --data-dir /data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM/data/processed/tinystories \
  --device cuda > "$LOG" 2>&1 < /dev/null &
echo $! > outputs/logs/theory.pid
echo "PID=$(cat outputs/logs/theory.pid) LOG=$LOG"
tail -f "$LOG"
```

`tail -f` 用 Ctrl+C 退出不会停止后台训练。重连后检查：

```bash
cd "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab"
ps -p "$(cat outputs/logs/theory.pid)" -o pid,etime,stat,args
tail -n 25 "$(ls -t outputs/logs/theory-*.log | head -n 1)"
```

`theory` 入口读取旧实验的 launch 配置，核对预处理数据内容 SHA256、模型配置、PyTorch 版本、调参与复验各 seed 的初始权重 SHA256，然后仅补跑新方法：

- 调参种子 7；候选 LR 为 0.003、0.01、0.03、0.1、0.3，每个 611 步。
- 从验证 NLL 选择 LR，复验种子 42、43、44，每个 6104 步。
- 每步 8192 目标 token；每次复验 50,003,968 token。
- batch size 8、梯度累积 4、序列长 256；FP32、恒定 LR；每步记录张量统计。
- 相同评估间隔和 65,536 个评估 token；test 不参与 LR 选择。

总计 21,367 次更新，175,038,464 个目标 token。按 23 秒 / 100 步粗估约 82 分钟，另加启动与绘图时间。后台作业可能更慢，实际以日志为准。

输出在新代码目录的 `outputs/metrics/theory-时间戳/`。完成后除了原有 summary/search/selected_lrs/config/weights/metrics/figures，还会有 `reference_check.json`，记录数据验证、原始配置以及逐种子的初始化和 token 预算一致性。不要重建或替换原数据；若预检报错，先核对路径和原 PyTorch 环境。

通用入口也支持 `bash start_training.sh compare --optimizers adam_s_theory`，但它使用 CLI 默认值，不会自动读取旧实验或做上述预检，因此补跑推荐使用 `theory --reference ...`。

## 仅绘制 Adam 与 Adam-S

从日志重新绘图，不需要加载 `.pt` 检查点或重新训练。在本机 PowerShell：

```powershell
.\.venv\Scripts\python.exe scripts/plot_comparison.py --runs outputs/metrics/server-results/compare-20261117-141000-525408 --optimizers adam adam_s
```

输出为旧结果目录内的 `figures_adam_adam_s/`，原来的 `figures/` 保留。`loss.png` 对比训练 loss 和验证 NLL；每个参数的两张图保留权重 norm、norm_ratio、relative_update 和梯度/动量/二阶矩/Adam 方向 RMS。每个 seed 单独一条线，没有跨 seed 平均或抽样。坐标范围按筛选后的数据重新计算。`plot_manifest.json` 记录使用了哪些运行。

脚本会自动定位本机的 `repeat/`，忽略 summary 中已失效的服务器绝对路径。旧版本只有权重统计的 CSV 也能画权重图。

新实验结束并下载后，可以合并对比（替换示例目录名）：

```powershell
.\.venv\Scripts\python.exe scripts/plot_comparison.py --runs outputs/metrics/server-results/compare-20261117-141000-525408 outputs/metrics/server-results/theory-实际时间戳 --optimizers adam adam_s adam_s_theory --output outputs/figures/adam-s-theory-comparison
```

对比不同目录前应先确认新实验 `reference_check.json` 的配对检查通过。绘图本身不会证明任意两个目录拥有相同训练条件。
