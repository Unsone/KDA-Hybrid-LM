# PyTorch：索引与视图

## 形状约定（本项目）

- 输入输出：`[B, H, T, D]`（batch / head / 序列长度 / head 维度）
- DeltaNet 状态：`[B, H, D, D]`，每个 head 一张联想记忆矩阵
- 写维度参数优先用 `dim=-1`，不要依赖具体维号，避免张量排布一改就错

## 按 token 切片

```python
# q: [B, H, T, D]
q_t = q[:, :, t, :]   # [B, H, D]，所有 batch、所有 head 的第 t 个 token
q_t = q[..., t, :]    # 等价写法
q_t = q[:, :, t]      # 等价：尾部的整维可以省略
```

- 整数索引 → 降维；`q[:, :, t:t+1, :]` → 保留成 `[B, H, 1, D]`
- 沿时间轴遍历的简写：`for q_t in q.unbind(2):`，`unbind` 直接拆成 T 个 `[B,H,D]` 视图

## 视图 vs 拷贝

```python
piece = q[:, :, 0, :]   # 视图：与 q 共享内存
piece += 1              # 原地修改会改到 q 本身！
```

- 切片、`view` 是视图；`reshape`（不连续时）、`clone`、`transpose().contiguous()` 可能产生拷贝
- 只读计算（matmul、normalize 等）用视图没有副作用

## 常见坑

- `dim=3` 是硬编码：换张量排布就错，统一写 `dim=-1`
- `squeeze(3)` 只在该维长度为 1 时删除；`head_dim=1` 时要确认删掉的是想删的那一维
- 整数索引抽帧后做矩阵乘，要 `unsqueeze` 把维补回来（见 [shape.md](shape.md)）
- 视图和原张量共享内存，原地操作（`+=`、`masked_fill_`）前先想清楚

关联：[delta_net.py](../../../src/kda_hybrid_lm/mixers/delta_net.py) 中 `q_t = q[:, :, t, :]` 一带
