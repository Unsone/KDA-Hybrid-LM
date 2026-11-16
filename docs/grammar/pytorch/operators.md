# PyTorch：运算、广播与精度

## dtype 与精度

```python
q.float()                 # -> torch.float32，等价 q.to(torch.float32)
x.to(torch.bfloat16)      # 转半精度
x.dtype                   # 查看当前 dtype
```

- 本项目约定：q / k / v / beta / state 全部在 FP32 里算递推，输出投影后再 `output.to(x.dtype)` 转回原精度
- 原因：状态沿 T 步累加，半精度误差会累积；只在最终输出时降精度省显存
- 混合精度训练里常见组合：参数 bf16 + 状态 fp32

## F.normalize

```python
q = torch.nn.functional.normalize(q.float(), dim=-1)
```

- 公式：`x / max(||x||_2, eps)`，默认 `eps=1e-12`
- 归一化后点积 = 余弦相似度，取值范围锁在 `[-1, 1]`
- 归一化 **k**：使 `kᵀk = 1`，DeltaNet“β=1 时一步精确覆盖”的性质才成立
- 归一化 **q**：输出 `||S q|| ≤ ||S||`，控制读数幅度，训练更稳
- 坑：全零向量会被返回全零（不会报错），该 token 的写入会失效

## 广播

- 从右往左对齐，某一维为 1 或缺失时可自动扩展
- 例：`correction [B,H,D] * beta_t [B,H,1]` → 每个 head 用各自的 β
- 例：`torch.zeros(B, H, D, D)` 与 `state [B,H,D,D]` 逐元素相加

## 常用数值函数

```python
torch.sigmoid(x)              # (0, 1)，门控/写入门
torch.softmax(x, dim=-1)      # 必须给 dim
F.dropout(x, p=0.1, training=self.training)
```

- `softmax` 不写 `dim` 会按最后一维算，行为不明确，显式写
- `nn.Dropout` 在 `eval()` 后自动关闭；`F.dropout` 需要手动传 `training`

## 复合运算链：DeltaNet 的单步更新

原始写法（`state: [B,H,D,D]`，`k_t/v_t: [B,H,D]`，`beta_t: [B,H,1]`）：

```python
prediction = torch.matmul(state, k_t.unsqueeze(3)).squeeze(3)   # 旧答案 [B,H,D]
error      = v_t - prediction                                   # 差距  [B,H,D]
correction = beta_t * error                                     # 打折  [B,H,D]
state      = state + torch.matmul(correction.unsqueeze(3), k_t.unsqueeze(2))  # 写回 [B,H,D,D]
```

形状追踪：

| 步骤 | 形状 | 含义 |
|---|---|---|
| `state` | `[B,H,D,D]` | 每个 head 一张联想记忆矩阵 |
| `k_t.unsqueeze(3)` | `[B,H,D,1]` | 补一列，才能批量矩阵乘 |
| `matmul(...)` | `[B,H,D,1]` | 旧答案 |
| `.squeeze(3)` | `[B,H,D]` | 去掉那一维 |
| `v_t - prediction` | `[B,H,D]` | 预测误差 |
| `beta_t * error` | `[B,H,D]` | `[B,H,1]` 广播到 `[B,H,D]`，逐 head 打折 |
| `correction.unsqueeze(3)` | `[B,H,D,1]` | 外积左因子 |
| `k_t.unsqueeze(2)` | `[B,H,1,D]` | 外积右因子 |
| `matmul(...)` | `[B,H,D,D]` | 秩一修改量 |
| `state + ...` | `[B,H,D,D]` | 新状态 |

要点与坑：

- 不能直接写 `matmul(state, k_t)`：`[B,H,D,D] × [B,H,D]` 的批量维度对不上，`k_t` 会被当成 `[H,D]` 的矩阵；必须先 `unsqueeze` 成 `[B,H,D,1]`
- 统一用 `unsqueeze(-1)` / `squeeze(-1)`，不要写死维号 3
- `state = state + X` 是**新建**张量；`state += X` 是原地修改，递推张量参与反向传播时容易报 in-place 错误，所以这里用 `+` 是对的
- 等价写法（免去 squeeze）：
  ```python
  pred  = torch.einsum('bhde,bhe->bhd', state, k_t)
  delta = torch.einsum('bhd,bhe->bhde', correction, k_t)
  ```

关联：[delta_net.py](../../../src/kda_hybrid_lm/mixers/delta_net.py) 第 53–57 行
