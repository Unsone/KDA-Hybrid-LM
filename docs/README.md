# 文档索引

本仓库处于“组件实现已完成，端到端训练待建设”阶段。文档将已实现行为、实验计划和待测结果分别说明，状态核对日期为 2026-11-16。

| 文档 | 阅读目的 |
|---|---|
| [项目介绍](../README.md) | 目标、当前状态、快速开始与路线 |
| [architecture.md](architecture.md) | 模块职责、接口、状态与训练依赖 |
| [math.md](math.md) | 五种 Mixer 与基础模块的计算含义 |
| [environment.md](environment.md) | 本机安装、运行测试、远程计算安排 |
| [experiments.md](experiments.md) | MQAR、语言建模、消融与公平性设计 |
| [benchmarks.md](benchmarks.md) | 显存、吞吐、数值误差和预算的测量规范 |
| [report.md](report.md) | 技术报告框架、现有验证与待填结果 |
| [grammar/README.md](grammar/README.md) | 已有 Python/PyTorch 学习笔记 |

建议先读项目介绍与架构，再结合源码看数学文档；准备正式实验时阅读实验和基准规范。grammar 目录记录语言和库的用法，算法解释放在 math.md，现有笔记保持原样。
