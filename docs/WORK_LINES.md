# 毕设与避障：独立工作区，共享成果和数据

更新：2026-10-16。用户要求将毕设迁入 `E:\Graduation`；本工作区和本聊天继续避障项目。两边的成果和数据可以相互复用。

| 工作线 | 工作入口 | 当前内容 |
| --- | --- | --- |
| 避障 | 本仓库的 [项目状态](PROJECT_STATE.md) → [研究决定](CURRENT_DECISION.md) | Android、头戴 ToF/RGB 融合、盲杖互补预警、硬件演示及原实验历史 |
| 毕设 | [Graduation 入口](E:/Graduation/README.md) → [毕设状态](E:/Graduation/docs/PROJECT_STATE.md) → [资料索引](E:/Graduation/docs/RESOURCE_INDEX.md) | 选题筛选、ORBIT 等公开基准、找物／功能部位定位、开题和论文材料 |

`assistive-thesis-screen` 与 `assistive-object-finding` 的完整源码、当前页、结果及运行记录已迁到 Graduation 的 `research/active/`，后续在新工作区继续。旧版本仍可从本仓库迁移前的 Git 历史恢复。

## 共享方式

- 两边的 `artifacts.local` 访问同一份 F 盘数据、模型、实验产物和回执；不重复搬运。毕设的具体访问入口与历史论文材料见新资料索引。
- 成果可以双向引用和复用。引用保留来源、数据划分、消费历史、指标分母、单位和适用范围；迁目录不构成新实验或独立测试。
- 避障的 `hardware-bringup`、`dtr-r0` 及历史协议、失败和冻结证据留在本仓库。旧 CNH 论文文稿在 Graduation 有带来源记录的历史副本。
- 各工作区独立决定优先级、下一步和提交范围；新实验使用独立 run 目录，避免覆盖共享存储中的既有输出。
- 已在副机运行的毕设队列沿用原工作根、请求、作业、恢复状态和累计预算；本次迁移不重启任务。自动跟进改从 Graduation 读取和更新记录。

本仓库不再承载毕设新源码或新论文工作。共享数据的物理入口继续遵循 [本地产物规则](LOCAL_ARTIFACTS.md)。
