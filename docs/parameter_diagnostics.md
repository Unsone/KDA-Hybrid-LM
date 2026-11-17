# 逐步参数、梯度与 Adam 矩记录

每次运行的 `weights.csv` 为长表，一行对应一个 `(step, parameter)`。覆盖 `named_parameters()` 中所有 `requires_grad=True` 的唯一张量，包括 embedding、投影、MLP、Norm 和 bias；共享参数只记录一次。优化器是否缩放某个参数与是否记录它相互独立，bias 仍不参与 Adam-S 缩放。

默认 `--trace-every 1` 记录初值及每一次 optimizer step。梯度累积的多个 microbatch 对应一次更新，不分别记为多个 step。若显式设置稀疏记录，则前 20 步仍逐步记录，之后按指定间隔和末步记录。

## 时间对齐

step=t 的一行表示：

- `pre_norm`：更新前 `||W_(t-1)||₂`。
- `grad_*`：用于该次更新的 `g_t`。当前训练循环是多个 microbatch loss 除以 accumulation 后累积，故为有效 batch 的平均梯度。Adam 更新后读取未被改变的 `p.grad`，在下一次清零之前记录；当前无裁剪、无 AMP。
- `momentum_*`、`second_moment_*`：此次更新完成后的 Adam 状态 `m_t`、`v_t`，不是更新前的状态。
- `norm` / `rms`：更新后的 `W_t`。
- `update_norm`：实际参数差 `||W_t-W_(t-1)||₂`，含浮点舍入。
- `relative_update`：上述实际差分除以更新前的权重范数。

## 字段

| 字段 | 含义 |
|---|---|
| `parameter`, `numel` | 稳定参数名、张量元素个数 |
| `parameter_step`, `has_grad` | Adam 对此参数的实际状态步数、本次是否有梯度 |
| `norm`, `rms` | 权重展平后的欧氏范数与 RMS |
| `norm_ratio` | 更新后权重范数 / 初始权重范数 |
| `pre_norm`, `update_norm`, `relative_update` | 更新前范数、实际更新范数、单步相对更新 |
| `grad_norm`, `grad_rms` | 累积平均梯度的欧氏范数与 RMS |
| `momentum_norm`, `momentum_rms` | 原始一阶矩 m 的欧氏范数与 RMS |
| `momentum_hat_norm`, `momentum_hat_rms` | 偏差修正一阶矩 m_hat 的欧氏范数与 RMS |
| `second_moment_norm`, `second_moment_rms` | 原始二阶矩 v 的欧氏范数与 RMS；单位是梯度的平方 |
| `second_moment_hat_norm` | 偏差修正二阶矩 v_hat 的欧氏范数 |
| `sqrt_second_moment_hat_rms` | RMS(sqrt(v_hat)) = sqrt(mean(v_hat))，表示典型梯度尺度 |
| `adam_direction_norm`, `adam_direction_rms` | u=m_hat/(sqrt(v_hat)+eps) 的欧氏范数与 RMS，尚未乘 lr 或 s |
| `effective_lr` | 该参数组实际使用的 lr；Adam-S 中为 global_lr × s |

所有欧氏范数都将张量展平计算，矩阵权重对应 Frobenius 范数，不是谱范数。跨尺寸比较时同时使用 RMS。偏差修正使用各参数在优化器状态中的实际步数，而不是无条件使用全局 step。

step=0 的原始 m/v 范数为零，梯度、偏差修正矩、更新方向和实际更新幅度为 NaN。初始权重或更新前权重为零时，对应比值记 NaN。本步 `grad=None` 时 `has_grad=0`，梯度与更新方向记 NaN，已有的矩状态仍可记录；实际更新为零。梯度数值为零和没有梯度是不同情况。

## 用于机制分析

将同一层的 `grad_rms`、`momentum_hat_rms`、`sqrt_second_moment_hat_rms` 与 `adam_direction_rms` 对齐，可以区分原始梯度规模、历史平滑、二阶矩归一化和最终尺度分配。再结合 `effective_lr`、`relative_update` 和损失曲线定位差异阶段。

`u` 与真实更新同时保留，可检查在当前无 weight decay、无裁剪的 Adam 配置下，实际更新是否近似 `-effective_lr × u`。不同运行轨迹上的梯度和矩会随参数变化，日志相关性本身不等同于因果证据。

每个参数生成原有权重三项指标图，以及 `_optimizer.png` 的梯度、原始一阶矩、修正一阶矩、sqrt(v_hat)、自适应方向五项 RMS 曲线。记录的是标量摘要，不是完整权重或梯度张量；后者只在最终检查点保留可用的模型/优化器状态。旧运行不能补齐其历史梯度与矩轨迹。
