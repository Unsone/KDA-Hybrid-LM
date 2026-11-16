# PyTorch：形状变换

## 增删维

```python
k_t = k[:, :, t, :]        # [B, H, D]
k_t.unsqueeze(-1)          # [B, H, D, 1]
k_t.unsqueeze(-2)          # [B, H, 1, D]
k_t.unsqueeze(0)           # 最前面加一维
x.squeeze(-1)              # 删最后一维（仅当长度为 1）
```

- `unsqueeze(-1)` / `squeeze(-1)` 比写死维号安全
- `squeeze()` 不带参数会删掉所有长度为 1 的维，容易误删 batch 维，尽量带参数

## matmul 的形状规则

`torch.matmul(a, b)`：

- 一维 × 一维 → 点积（标量）
- 二维 × 二维 → 矩阵乘
- 更高维 → **批量矩阵乘**：前缀维度按广播对齐，最后两维做矩阵乘
- `[B,H,D,D] @ [B,H,D,1]` → `[B,H,D,1]`，再 `squeeze(-1)` 得到 `[B,H,D]`

## 外积（DeltaNet 写回状态）

```python
correction.unsqueeze(-1) @ k_t.unsqueeze(-2)
# [B,H,D,1] @ [B,H,1,D] -> [B,H,D,D]
```

等价的 einsum 写法（可读性更好，便于对照公式）：

```python
torch.einsum('bhd,bhe->bhde', correction, k_t)
```

## view / reshape / permute / transpose

```python
x.transpose(1, 2)      # 换轴，只改 stride，不一定连续
x.reshape(B, T, C)     # 需要时先连续化再改形状（可能拷贝）
x.view(...)            # 要求内存连续，否则报错
x.contiguous()         # 强制连续
x.permute(0, 2, 1, 3)  # 多轴一起换
```

- 本项目输出：`output.transpose(1, 2).reshape(B, length, dim)`——`transpose` 后不能直接 `view`，所以用 `reshape`
- 记法：**reshape 宽容、view 挑剔、contiguous 兜底**

## 常见坑

- `view` 在 transpose / permute 之后默认会失败
- `reshape` 不保证是视图，原地修改可能改不到原张量
- 广播只在“维度相等或其中一个是 1”时生效，前缀维度要对齐

## stack / cat 与"逐 token 收集再组装"

```python
outputs = []
for t in range(T):
    outputs.append(out_t)          # list 里是 T 个 [B, H, D]
x = torch.stack(outputs, dim=2)    # → [B, H, T, D]：在新维度上堆叠
x = x.transpose(1, 2)              # → [B, T, H, D]
x = x.reshape(B, T, H * D)         # → [B, T, C]，合并 head 维
```

- `torch.stack(list, dim)`：**新增**一维；要求所有张量形状完全一致；`dim` 是插入位置
- `torch.cat(list, dim)`：在**已有**维上拼接；除拼接维外形状须一致，不新增维
- 拿不准时的等价写法：`torch.cat([t.unsqueeze(2) for t in outputs], dim=2)`
- `append` 存的是张量**引用**不是拷贝，之后别再原地改这些张量
- 空 list 喂给 `stack` 会报错
- 逐 token 收集 T 个张量比较占显存（长序列尤其），分块 kernel 正是为了避免这种逐帧收集
