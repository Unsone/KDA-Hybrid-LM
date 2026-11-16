# 语法笔记（grammar）

记录对话中问到的语言 / 库语法点，整理成可快速复习的卡片。只记**怎么用、为什么这样写、哪里会踩坑**；算法理论（如 DeltaNet 推导）不放在这里。

## 组织规则

- **语言自带语法** → 根目录一个文件：`python.md`
- **第三方库** → 每个库一个子目录，按知识域拆分文件，例如 PyTorch：
  - `indexing.md`：索引、切片、维度语义、视图与拷贝
  - `shape.md`：增删维、reshape / permute / 内存布局、matmul 形状规则
  - `operators.md`：运算、广播、dtype 与精度、数值函数
  - 以后按需新增：`modules.md`（nn.Module / 参数 / forward）、`training.md`（autograd / 优化器 / 数据）、`serialization.md`（保存加载）等
- **新出现依赖包**（numpy、transformers、triton…）→ 新建同名子目录 + 至少一个 md 文件，并把条目登记到下面的索引表
- 文件名一律小写英文；每条笔记尽量一屏内能复习完

## 笔记格式

每条笔记按下面结构写，缺项可以省略：

- **问题**：当时怎么问的
- **结论**：一句话说清
- **示例**：最小可运行片段
- **坑**：容易写错的地方
- **关联**：相关源码位置或其它笔记

## 索引

| 文件 | 内容 |
|---|---|
| [python.md](python.md) | 多维索引 / 切片、range / enumerate / zip、解包 |
| [pytorch/indexing.md](pytorch/indexing.md) | 4D 张量约定、按 token 切片、视图与拷贝 |
| [pytorch/shape.md](pytorch/shape.md) | unsqueeze / squeeze、matmul 形状规则、外积、reshape / permute |
| [pytorch/operators.md](pytorch/operators.md) | dtype / 精度、F.normalize、广播、常用数值函数 |
