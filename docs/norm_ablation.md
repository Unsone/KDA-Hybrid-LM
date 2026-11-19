# 矩阵与 RMSNorm 步长：四组机制对照

目标是检验：Adam-S 的早期学习行为是否主要来自 Norm 参数较大的更新系数，以及这种改变是否影响后期优化。本次固定更新系数，不做学习率搜索，不把结果解释为四种方法调优后的最佳性能。

## 预先固定的设置与预测

| 组 | 程序名 | 所有矩阵的有效学习率 | 所有 RMSNorm 的有效学习率 |
|---|---|---|---|
| A | `adam` | 0.001 | 0.001 |
| B | `adam_s_matrix` | `0.03 * RMS(W0)` | 0.001 |
| C | `adam_norm` | 0.001 | 0.03 |
| D | `adam_s` | `0.03 * RMS(W0)` | 0.03 |

矩阵包括 Embedding、注意力 Q/K/V/O、MLP gate/up/down 和输出头；Norm 包括每层的两组增益向量，以及最后的 RMSNorm。按模块类型识别 Norm，不按名字猜测。本次模型无 bias。所有组保持 Adam 的相同 betas、eps、一阶矩和二阶矩逻辑，FP32、恒定 LR、无权重衰减或梯度裁剪。

这里的 `RMS(W0)` 采用实测初始 RMS，与旧实验 Adam-S 一致。新代码不修改初始权重。所有有效 LR 保存在 `config.json/scales` 和 `weights.csv/effective_lr`；B/C 还记录参数类型。

预先约定的比较：

- `B-A`、`D-C`：在两种 Norm 步长下，改变矩阵步长的效果。
- `C-A`、`D-B`：在两种矩阵设置下，增大 Norm 步长的效果。
- `D-B-C+A`：两项改变在加性 NLL 尺度上的交互；负值意味着组合的 NLL 改善超过两项单独改变之和。这不是跨尺度通用的交互量。
- 主要观察训练前期第 100 步、过渡期第 1000 步、中后期第 3000 步和终点第 6104 步的配对验证 NLL；同时保留完整曲线。测试集仅作为固定终点的最终报告，不用它挑选实验配置。
- 若 C 重现 D 的早期优势，支持 Norm 步长贡献；若 B 重现而 C 未重现，则支持矩阵分配贡献；若只有 D 出现，则关注交互。若只改变权重范数曲线而未改变 loss，范数变化不足以解释性能差异。
- Norm 的较大 LR 可能伴随更快增益调整及波动，后期可能有代价。这些是待检验预测，不预设一定成立；范数平台不等于参数收敛。

原始三种子均为 42、43、44，固定原数据、模型、评估、数据顺序和 6104 步预算。每组每个 seed 为 50,003,968 个目标 token。程序在开始长训练前核对实际数据内容 SHA256、模型配置、PyTorch 版本、实际初始化 SHA256、原 A/D 的 LR/倍率和预算。运行后还核对四组初始化、第一步训练损失与评估网格。

## 上传与覆盖解压

在本机 Windows PowerShell：

```powershell
scp "D:\UV\KDA-Hybrid-LM\outputs\metrics\packages\KDA-Hybrid-LM-AdamS-Lab-norm-ablation.zip" PKU_8x4090_48G:~/lcx/
```

在服务器：

```bash
cd /data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM
unzip -o "$HOME/lcx/KDA-Hybrid-LM-AdamS-Lab-norm-ablation.zip" -d .
export VENV_DIR="$PWD/.venv"

# 使用已确认可用的项目内部环境；新版不增加依赖，无需重新安装。
"$VENV_DIR/bin/python" -c 'import sys,numpy,torch; print(sys.executable); print(torch.__version__)'
CUDA_VISIBLE_DEVICES=4 bash start_training.sh norm-ablation --smoke --device cuda
```

ZIP 仅包含本分支已提交的代码，不包含虚拟环境、数据、已有结果或 Git 历史。覆盖解压会替换代码文件，保留原有数据、环境和结果。服务器当前不要同时运行其它版本的代码修改任务。

## nohup 正式启动：默认只补跑 B/C

原 A/D 已完成且保存了日志，因此默认复用它们，只运行 B/C；输出仍给出四组对照、配对差值和全部图像。

```bash
cd /data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM
export VENV_DIR="$PWD/.venv"
mkdir -p outputs/logs
LOG="outputs/logs/norm-ablation-$(date +%Y%m%d-%H%M%S).log"

CUDA_VISIBLE_DEVICES=4 nohup bash start_training.sh norm-ablation \
  --reference outputs/metrics/compare-20261117-141000-525408 \
  --data-dir data/processed/tinystories \
  --device cuda > "$LOG" 2>&1 < /dev/null &

echo $! > outputs/logs/norm-ablation.pid
echo "PID=$(cat outputs/logs/norm-ablation.pid) LOG=$LOG"
tail -f "$LOG"
```

Ctrl+C 结束 `tail -f` 不会停止后台训练；确认日志出现进度后可断开 SSH。重连后检查：

```bash
cd /data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM
ps -p "$(cat outputs/logs/norm-ablation.pid)" -o pid,etime,stat,args
tail -n 25 "$(ls -t outputs/logs/norm-ablation-*.log | head -n 1)"
```

默认共 6 次训练、36,624 次更新。按此前 23–25 秒 / 100 步估计约 2.3–2.6 小时，另加绘图时间。需要重新训练全部四组时，在正式命令中增加 `--arms A B C D`，约为两倍时间；仍然固定 LR、不做搜索，仍核对原实验配置。

输出目录为 `outputs/metrics/norm-ablation-时间戳/`。smoke 使用独立的 `norm-ablation-smoke-` 前缀及合成数据，永远跑四组，不能用来判断正式实验结论。

## 结果文件

- `protocol.json`：运行前写入的四组定义、假设、固定系数、预算、观察阶段和差值方向。
- `summary.json`：此次实际训练的结果，默认只有 B/C。
- `comparison_summary.json`：四组结果，包括注明 `reused=true` 的旧 A/D。
- `reference_check.json`：配对检查；`reference-metrics/` 保存小体积旧 loss 日志，不复制旧模型或大 CSV。
- `repeat/`：新组各 seed 的 config、逐步 loss/NLL/PPL、逐张量权重/更新/梯度/动量统计以及最终检查点。与旧实验记录口径一致。
- `validation_curves.csv`：四组逐次验证指标；`validation_contrasts.csv`、`stage_contrasts.csv`、`test_contrasts.csv`：各 seed 的配对差值。
- `contrast_summary.json`：验证 NLL 差值的跨 seed 均值和样本标准差。只有三个 seed，不据此宣称因果贡献已被统计显著验证。
- `validation_overview.png`：四组平均验证曲线、早期和后期放大。阴影是 seed 最小–最大值，不是置信区间。
- `factorial_effects.png`：矩阵、Norm 干预和交互的配对验证 NLL 差值曲线。
- `figures/`：四组的 loss 与逐参数图；新组和旧组都会显示，方便观察 Norm 与矩阵动力学。
- `source/`：实际执行的训练、优化器、模型和实验入口快照。

原 A/D 的大日志目录需要保留到本次绘图完成。`comparison_summary.json` 的 path 可能指向原服务器路径；本机合并重绘可读取两个实验目录，不依赖这些绝对路径：

```powershell
.\.venv\Scripts\python.exe scripts/plot_comparison.py --runs outputs/metrics/server-results/compare-20261117-141000-525408 outputs/metrics/server-results/norm-ablation-实际时间戳 --optimizers adam adam_s_matrix adam_norm adam_s --output outputs/figures/norm-ablation
```

该命令适用于默认只补 B/C 的结果。若新实验选择重训全部四组，只传新目录，避免同名优化器/seed 重复。

## 限速下载分析文件

在服务器选定真实结果目录（替换实际时间戳），排除检查点：

```bash
RUN=norm-ablation-实际时间戳
tar -czf outputs/metrics/norm-ablation-analysis.tar.gz \
  --exclude='*.pt' --exclude='*.pth' --exclude='*.ckpt' --exclude='__pycache__' \
  -C outputs/metrics "$RUN"
```

在本机 Windows PowerShell，以约 1 MB/s 为上限下载并解压：

```powershell
scp -l 8000 "PKU_8x4090_48G:/data/yyd/lcx/KDA-Hybrid-LM/KDA-Hybrid-LM/outputs/metrics/norm-ablation-analysis.tar.gz" "D:\UV\KDA-Hybrid-LM\outputs\metrics\server-results\"
tar -xzf "D:\UV\KDA-Hybrid-LM\outputs\metrics\server-results\norm-ablation-analysis.tar.gz" -C "D:\UV\KDA-Hybrid-LM\outputs\metrics\server-results"
```
