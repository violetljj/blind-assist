# RCLE 研究主线

状态：current

最后核验：2026-07-26

适用范围：BlindAssist 论文研究、毕业设计、院内演示与创新竞赛。

## 结论

RCLE-RF（Rotation-Compensated Local Expansion Risk Field，旋转补偿局部扩张风险场）是 BlindAssist 当前研究主线。

这次切换只改变研究优先级和后续任务入口，不改变正式 App、默认 YOLO 模型、既有 USTRF 实验版或任何安全权限。BlindAssist 仍是研究与演示原型，不面向视障人士独立使用，也不形成真实用户有效性或安全认证结论。

## 两级指导关系

| 层级 | 文档 | 职责 | 执行权限 |
| --- | --- | --- | --- |
| 长期大目标 | `D:\edge\BlindAssist_RCLE-RF_新总纲领与Codex执行手册_R1.0.md` | 定义 RCLE-RF 完整能力地图、论文结构、Risk Field、演示与可能的后期扩展 | 不直接生成任务，不得一次性展开全部阶段 |
| 当前小目标 | `D:\edge\BlindAssist_RCLE_Minimal-First_R1.1.md` | 定义 RCLE-Minimal R0、当前实验、Kill Gate、代码边界与近期交付 | 当前唯一执行协议；冲突时优先于 R1.0 |
| 仓库入口 | 本文 | 固定当前主线、状态、边界和恢复入口 | 后续任务开始前必须先读 |

本次核验的 R1.0 SHA-256 为 `4D138FFD4B3BC5F5DA16B0502F162652540C67FB84C9B88F09604E529DD1D6C2`，R1.1 SHA-256 为 `C6BC9E0D8A1C9665B319CB8141362B098B94B742772E351C24BE33445CC3BCD2`。源文档内容或哈希变化时，必须重新核对本页。

R1.0 回答“项目最终可以长成什么”，R1.1 回答“现在只做什么”。R1.0 中的 Risk Field、深度、bearing、shear、Android 主动提醒、多源扩展和产品化内容，只有在 R1.1 对应门禁通过并被单独授权后才能进入任务。

## 当前唯一研究问题

当前只验证一个可证伪命题：

> 在已知相机旋转和连续图像条件下，旋转补偿后的局部扩张，能否比未补偿光流更少响应纯旋转，同时保留真实接近造成的闭合响应？

RCLE-Minimal R0 的输出是以 `s^-1` 表示的局部 expansion。它不是碰撞概率、危险概率、告警决策或安全保证。第一阶段不输出 Risk Field、TTC、路线相关风险、主动语音或振动提醒。

## 当前执行状态

R1.1 的 **Phase A：Synthetic Signal Audit** 已执行完毕，并在唯一一次
implementation-only coverage revision 后按冻结停止语义关闭。当前没有活跃的
Phase A 修订或 Phase B 后继执行权限。

Phase A 必须使用程序生成的、可复算的连续帧与运动真值，固定旋转补偿、Sparse LK、局部仿射拟合和网格汇总，比较纯旋转泄漏与真实闭合保留。trial-level leakage error 和 closing error 是主判据；RSR/CRR 只作诊断比例。

只有 Phase A 完成、报告可复算且 Kill Gate A 通过后，才可另立任务决定是否进入 Phase B。当前不授权：

- Bonn real-source 审计；
- 单项增强或多算法堆叠；
- Risk Field 与 token/lifecycle；
- Android RCLE 集成或主动告警；
- 真人试验、独立行走或生产晋级；
- 为追求“产品完整”而提前解决全数据权威、硬件闭环或安全认证。

## Phase A 当前结果

[Phase A R0 Synthetic Signal Audit](RCLE_MINIMAL_PHASE_A_SYNTHETIC_SIGNAL_AUDIT_R0_RESULT_2026-07-26.md) 已按结果前冻结的 `2520`-trial 协议完整运行并独立复算，终态为 `REVISE / VALID`。

clean yaw/pitch 旋转泄漏抑制、roll 不增噪、scale up/down 与 rotation+scale closing 保留、15/30/60 FPS 一致性及三类 stress 中可评价 trial 的误差门均通过；但逐 condition coverage 硬门失败：clean worst cell 为 `10/20`，partial-occlusion pitch worst cell 为 `0/20`，因此总体 `2419/2520` 可评价不能构成 Kill Gate A PASS。

[Phase A Coverage Revision R1](RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1_RESULT_2026-07-26.md) 已执行这唯一一次版本化实现修订。它保持 R0 协议、全部 trial/seed/threshold/gate 与输入 hash 不变，使 clean 达到 `1680/1680`、stress 达到 `810/840`，并消除了 affine residual 超门；但 partial-occlusion pitch 四个 cell 仅为 `12/20、12/20、13/20、13/20`，worst `0.60 < 0.70`，coverage 仍失败。R1 receipt `d5edb952…2b15` 独立复算为 `REVISE / VALID`。

由于冻结协议规定单次版本化 REVISE rerun 仍失败即停止，当前研究层终态为 `STOP_CURRENT_IMPLEMENTATION / VALID`。不得再做第二次 coverage revision、降门或改矩阵；Phase B 与 Replay Demo 仍未开放。未来若获明确新授权，只能另立结果前冻结的新信号/观测模型假设，不能继续在本轮 2520 trials 上选择实现。

## 与既有工作的关系

### Route-conditioned USTRF

Route-conditioned USTRF 已按 [program closure R1](../ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) 收口。它是历史研究与负结果，不再是研究主线，也不具有 pilot、blocked-waiting、阶段选择或自动后继权限。其 detector、route、lifecycle 和 evidence receipts 继续保留，不回写、不伪装成 RCLE 结果。

### Egomotion-compensated looming

`scripts/research/egomotion_compensated_looming/` 及对应 Looming R0/R1 文档、Bonn/Farneback/oracle traces 是 RCLE 的前序探索和来源权威证据。它们证明了已有机制、数据缺口和错误对齐风险，但不等于 R1.1 Phase A 已完成，也不能绕过 Synthetic Signal Audit 或 Kill Gate A。

用户已中止本项目的其他工作。当前工作树中的 looming 文件已按 [前序现场冻结](RCLE_PRECURSOR_FREEZE_2026-07-25.md) 保留，原 Looming R1 不再续跑；这些文件不删除、不重写、不冒充 RCLE 结果。RCLE Phase A 可以直接沿用这个 canonical Module，增加独立最小入口；不得另建平行的 `research/rcle/` 算法实现目录。

## 实现与产物边界

| 类型 | Canonical 位置 |
| --- | --- |
| 研究代码 | `scripts/research/egomotion_compensated_looming/` |
| Phase A R0 / coverage R1 实现 | 上述 Module 内版本隔离的 `rcle_minimal/`、`rcle_minimal_r1/` 与对应 runner |
| 程序生成数据 | `artifacts.local/datasets/rcle_minimal_{r0,r1}/` |
| 报告、表格、图与 receipts | `artifacts.local/evidence/rcle_minimal_{r0,r1}/` |
| 当前主线与权限 | `docs/research/rcle/README.md` |
| 历史 USTRF/Looming 证据 | `docs/research/ustrf-sc/` |

不得把生成数据、回放结果、模型审阅或演示脚本表述为真人效果。不得为了展示而改写实验数据；演示层可以增强可视化，但必须与研究结果和产品权限分离。

## 近期交付顺序

1. 已冻结 looming 前序现场并完成 Phase A R0。
2. 已按 R0 `REVISE` 只执行一次 implementation-only coverage R1。
3. R1 coverage 仍失败，当前实现已按冻结语义停止。
4. 不规划 Phase B 或 Replay Demo；未来只有明确授权的新预注册假设可另立任务。

## 主线变更规则

后续 Codex 任务默认服从以下优先级：

1. 用户最新明确指令；
2. 本 current 入口和 R1.1 当前执行协议；
3. R1.0 长期能力地图；
4. 日期化 USTRF/Looming 目标、结果和历史 handoff。

任何旧文档中的“当前主线”“独立新算法入口”或“下一阶段”，若与本页冲突，均降级为历史上下文，不构成执行授权。主线再次变化时，必须同步更新本页、`AGENTS.md`、根 `README.md`、文档索引和 `DEVELOPMENT_LOG.md`。
