# Python 语法

## 多维索引与切片

**问题**：`q[:, :, t, :]` 是什么语法？

**结论**：方括号里的逗号把索引拆到各个维度上；`:` 表示这一维全取（等价 `slice(None)`），整数索引会把该维度**删掉**（降维）。

```python
import numpy as np

x = np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
x[:, :, 2, :].shape     # (2, 3, 5)  整数索引：第 3 维消失
x[:, :, 2:3, :].shape   # (2, 3, 1, 5) 切片：第 3 维保留，长度 1
```

- 等价写法：`q[:, :, t, :]` 就是 `q[(slice(None), slice(None), t, slice(None))]`，逗号在 Python 里构造的是元组
- 省略号：`q[..., t, :]` 与上面等价，`...` 自动补全前面所有 `:`
- 负索引：`q[:, :, -1, :]` 取最后一个 token；步长：`q[..., ::2, :]` 隔一个取一个
- 坑：Python 原生 list 只能一维索引；numpy / torch 是因为实现了 `__getitem__` 才支持逗号写法

## range / enumerate / zip

```python
for t in range(length): ...        # 0 .. length-1，惰性生成，不建列表
for i, t in enumerate(items): ...  # 同时拿下标
for a, b in zip(xs, ys): ...       # 并行迭代；3.10+ 可加 strict=True 检查长度一致
```

- `range` 只适合整数序列；要遍历张量的某一维用 `unbind`（见 [pytorch/indexing.md](pytorch/indexing.md)）

## 解包

```python
q_t, k_t, v_t = tensor_tuple     # 右侧元素个数必须与左侧一致
```

- 循环里写 `for q_t, k_t in zip(...)` 是同一个机制

## 类型转换

- `x.float()` 是 torch 张量方法，不是 Python 内置：Python 里 `float(x)` 才是字符串 / 整数转浮点
- `int` 做索引、`slice` 做切片，两者在张量上的行为差异见上面第一条

## 类与对象：`__init__`

**问题**：`__init__` 方法是什么？

**结论**：类的"初始化方法"，**创建对象时自动调用**，用来设置属性；不是手动调用的普通方法。

```python
class DeltaNet:
    def __init__(self, hidden_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

model = DeltaNet(512, 8)   # 自动调用 __init__(model, 512, 8)
```

- `self` 是实例本身，调用时不用传；方法第一个参数必须留出 `self`（名字可换，但约定叫 self）
- 严格说创建对象由 `__new__` 负责、`__init__` 只做初始化；日常说"构造函数"也指它
- 不要写 `return`；返回非 `None` 会 `TypeError`
- 继承 `nn.Module` 的类必须先 `super().__init__()`，否则参数注册 / forward 会出问题
- 双下划线包裹的方法叫**魔术方法 / dunder**，由 Python 在特定操作时自动调用：`__repr__`（print）、`__len__`（len）、`__call__`（对象当函数调）、`__getitem__`（方括号索引）

## 包与模块：`__init__.py`

**问题**：`__init__.py` 文件是什么？

**结论**：它把所在目录标记成一个 **Python 包**，目录里的模块才能用 `import 包.模块` 导入；包被第一次导入时它会执行一次。

```text
src/kda_hybrid_lm/
├── __init__.py          # 让 kda_hybrid_lm 成为包
└── mixers/
    ├── __init__.py      # 让 mixers 成为子包
    └── delta_net.py
```

```python
# mixers/__init__.py 里通常会写导出
from .delta_net import DeltaNet   # 点号表示当前包
__all__ = ["DeltaNet"]
```

- 空文件也能用：只起"标记为包"的作用（本项目目前就是空的）
- 填了导出后，外部能写 `from kda_hybrid_lm.mixers import DeltaNet`；不填只能写全路径 `from kda_hybrid_lm.mixers.delta_net import DeltaNet`
- 包**第一次导入**时执行一次，之后不再执行
- 和 `__init__` 方法只是名字像，**毫无关系**
- 尽量只放导出 / 版本号，别放重逻辑：容易循环导入、拖慢启动
