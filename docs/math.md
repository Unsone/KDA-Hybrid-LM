# 数学原理与张量约定

以下公式解释当前实现。原论文完整结构与本项目简化版的差别单独注明。

## 统一记号

B 为批量，T 为本次序列长度，C 为隐藏维度，H 为头数，D=C/H。投影后的 Q/K/V 为 `[B,H,T,D]`。公式省略 batch/head 下标，将 q、k、v 写为列向量；本项目 S 的行对应 value、列对应 key，因而 `S k` 是预测 value。

源码保留行向量 `[B,H,1,D]`：读取 `S q` 等价写成 `q_row @ S.T`；外积 `v k.T` 等价写成 `v_row.T @ k_row`。时间切片用 `t:t+1` 保留维度，输出沿已有时间维 cat。

## SoftmaxAttention

$$A=\operatorname{softmax}(QK^T/\sqrt D+M),\qquad O=AV.$$

M 在未来位置为负无穷，其余为零。带缓存时，第 i 个新 query 的绝对位置为 P+i，允许访问 key 位置 j≤P+i；不能直接对长方形分数矩阵使用左上角下三角掩码。

当前实现保存全部历史 K/V；显式构造分数矩阵，不是 FlashAttention。Q/K 不使用线性组件的 L2 归一化，且当前没有接入 RoPE。

## LinearAttention：外积累加基线

当前采用 L2 归一化后的 q/k，不带 softmax 或分母归一化：

$$S_t=S_{t-1}+v_tk_t^T,\qquad o_t=S_tq_t.$$

零初始状态下，输出等价于历史 value 按点积加权：

$$o_t=\sum_{i\le t}v_i(k_i^Tq_t).$$

权重可以为负，也不要求和为 1。这不是所有文献中“线性注意力”的统一定义；当前选择便于与 delta 规则做受控对照。

## DeltaNet：只修正预测误差

$$\hat v_t=S_{t-1}k_t,\quad e_t=v_t-\hat v_t,$$
$$S_t=S_{t-1}+\beta_t e_tk_t^T,\qquad o_t=S_tq_t.$$

β 由每头一个投影经 sigmoid 得到，形状为 `[B,H,T,1]`。它控制误差修正幅度，不是对整个 S 做新旧比例混合。若 k 为单位向量：

$$S_tk_t=(1-\beta_t)\hat v_t+\beta_tv_t.$$

旧记忆已经准确时，误差为零，不会重复累加完整 value。对其他方向 u，变化量为 βe(kᵀu)，因此相似方向可能相互影响。

## GatedDeltaNet：先整体衰减，再修正

$$\widetilde S=\alpha_tS_{t-1},\qquad S_t=\widetilde S+\beta_t(v_t-\widetilde S k_t)k_t^T.$$

α 是每头标量，当前由 sigmoid 生成。必须使用衰减后的状态计算预测误差。α 控制旧记忆通过多少，β 控制当前修正通过多少，故称为软门控。投影权重由训练学习，而每个 token 的门值在前向时由输入产生。

## KDA：逐 key 通道衰减

$$\widetilde S=S_{t-1}\operatorname{Diag}(\boldsymbol\alpha_t),$$
$$S_t=\widetilde S+\beta_t(v_t-\widetilde S k_t)k_t^T.$$

等价展开为：

$$S_t=S_{t-1}\operatorname{Diag}(\boldsymbol\alpha_t)(I-\beta_tk_tk_t^T)+\beta_tv_tk_t^T.$$

α 的形状为 `[B,H,T,D]`，单步 `[B,H,1,D]`。`state * alpha_t` 按列缩放，正好等价于右乘对角矩阵，无需真的构造对角矩阵。论文常使用转置后的状态布局，比较公式时必须先统一方向。

若所有通道 α 相同，KDA 退化为 GDN；当前测试验证了该性质。细化的是 key 特征坐标，不是历史 key 的个数；固定状态仍会混合多条历史关联。

当前 GDN/KDA 使用直接 sigmoid 投影，尚无论文完整的衰减参数化、短卷积、输出门控和分块算法。通道门增加了表达能力，但是否在本项目获得更好质量，必须经训练实验验证。

## RMSNorm

$$y_i=w_i\frac{x_i}{\sqrt{C^{-1}\sum_jx_j^2+\epsilon}}.$$

对每个 token 的最后一维独立处理，不减均值；w 为初始化为 1 的 `[C]` 可学习参数。当前先转 FP32 计算，最终恢复输入 dtype；这是面向常规混合精度的实现，并非完整保留 float64 精度的版本。

## SwiGLU

$$y=W_d\left[\operatorname{SiLU}(W_gx+b_g)\odot(W_ux+b_u)\right]+b_d,$$
$$\operatorname{SiLU}(z)=z\sigma(z).$$

两条分支由 C 映射到中间维度 I，最后回到 C；不跨 token 混合。SiLU 门值可以为负或大于 1，与 α/β 的概率范围不同。无偏置时参数量为 3CI；最终应用输出 Dropout，残差和 Norm 由未来 Block 负责。

## RoPE

对相邻特征对施加旋转：

$$\begin{bmatrix}x'_{2j}\\x'_{2j+1}\end{bmatrix}=\begin{bmatrix}\cos\theta&-\sin\theta\\\sin\theta&\cos\theta\end{bmatrix}\begin{bmatrix}x_{2j}\\x_{2j+1}\end{bmatrix},\quad\theta=p\cdot\mathrm{base}^{-2j/D}.$$

当前采用相邻配对，D 必须为偶数。旋转保持向量长度，同步平移 Q/K 位置时点积不变；offset 支持分段。它尚未接入 SoftmaxAttention。

## 因果短卷积

每个通道独立处理邻近位置，窗口 K 包含当前 token 与前 K-1 个 token，左侧不足补零，再应用 SiLU。PyTorch Conv1d 的核顺序从窗口最旧位置到当前位置；它使用互相关约定，不应手动再次翻转卷积核。

缓存是最近 K-1 个原始输入，固定长度，与线性记忆矩阵不是同一种状态。它补充局部模式，不替代长期记忆。

## 参数、状态与梯度

`nn.Parameter` 和各 Linear 的权重是优化器更新的长期参数。S、KV、卷积历史是当前序列的运行状态。forward 建立计算图，loss.backward 累积梯度，optimizer.step 才修改参数。

当前递推核心显式转 FP32，但外部 autocast 仍可能让 matmul 使用低精度；“最终 S 为 FP32”不证明内部所有乘法都是 FP32。正式 AMP 训练前需明确该策略，当前尚未修正。

参考：[GDN](https://arxiv.org/abs/2412.06464)、[KDA](https://arxiv.org/abs/2510.26692)、[RoPE](https://arxiv.org/abs/2104.09864)、[SwiGLU](https://arxiv.org/abs/2002.05202)。
