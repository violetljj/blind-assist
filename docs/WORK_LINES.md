# 两条工作线：毕设与避障

更新：2026-10-15。用户确认：**“毕设”和“避障”是两条不同的线，只部分重合**。
之后对话、记录和提交里说“毕设”只指毕设线，说“避障”只指避障线；话里没说清是哪条时先问。

| | 毕设线 | 避障线 |
| --- | --- | --- |
| 要回答什么 | 毕业论文做什么题目、拿什么可量化成果撑起来 | BlindAssist 头戴前视避障能否做好、做成可演示的系统 |
| 当前做法 | 在公开数据和公开基线上做小规模筛选，找有可测提升空间的方向 | ToF（CNH）+ RGB 单目深度融合；盲杖互补预警（盲杖先碰到的不报，碰不到的报）；手机 App 与硬件演示 |
| 研究目录 | [`assistive-thesis-screen`](../research/active/assistive-thesis-screen/CURRENT.md)（选题筛选：商品字段读取、无障碍问答、ORBIT/VizWiz 基准等）· [`assistive-object-finding`](../research/active/assistive-object-finding/CURRENT.md)（找物、功能部位定位） | [`hardware-bringup`](../research/active/hardware-bringup/CURRENT.md)（实机融合、ADT/ASE 盲杖互补，`*_CANE_*` 报告）· [`dtr-r0`](../research/active/dtr-r0/CURRENT.md)（近场 ToF/RGB 算法与模拟） |
| 代码与文档 | 各自目录下的脚本；载荷在 `artifacts.local/work/` | Android 模块、[硬件演示](HARDWARE_OBSTACLE_DEMO.md)、[项目状态](PROJECT_STATE.md)、[研究决定](CURRENT_DECISION.md) |
| 状态（10-15） | 选题未定；各方向都只有 Development 小样结论 | 盲杖互补：几何规则可行，瓶颈在单目深度造假支撑，正在试学习式几何 |

## 两条线的关系

- 避障线的成果可以被毕设**引用**（例如盲杖互补预警曾被讨论作为论文题目候选），但是否写进论文、写成哪一章，由毕设线决定。
- 避障线的优先级、停止规则和下一步不受毕设选题影响；毕设线的筛选结果也不改变避障线的决定。
- [项目状态](PROJECT_STATE.md) 与 [研究决定](CURRENT_DECISION.md) 只覆盖避障线；毕设线的进展看上表两个目录的 `CURRENT.md`。
- 新开研究目录时，在 `CURRENT.md` 开头写明归属（毕设 / 避障）；两者都不是的（如已暂停的 [L10](../research/active/l10-r0/CURRENT.md) 等历史路线）不必归入。
