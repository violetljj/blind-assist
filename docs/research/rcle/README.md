# RCLE 研究主线

状态：current

最后核验：2026-07-27

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
implementation-only coverage revision 后按冻结停止语义关闭；随后独立冻结的
Observable Support Recovery 已通过 development 与 sealed validation。用户随后
单独激活 **Phase B**；B0 R1 archive/member/CRC/timestamp inventory 已
`PASS / VALID`。唯一 B1A source-native geometry execution 已消费 claim，但
独立 replay 因 blank-grid ledger key-set mismatch 判
`INVALID_EXECUTION_CLOSE_B1`。旧 terminal 保留；按新的
[closure-scope overlay](RCLE_PHASE_B_BONN_B1A_CLOSURE_SCOPE_2026-07-26.json)，
它实际关闭 B1A R5 evidence instance、已消费 protocol version 与明确依赖它的
B1B R5，不关闭 RCLE 科学问题。

项目现已采用[渐进式研究治理](../../RESEARCH_GOVERNANCE.md)，并明确开放
[Phase B Progressive Discovery](RCLE_PHASE_B_PROGRESSIVE_PROTOCOL_2026-07-26.md)。
当前只做来源画像、连续几何分布、约束质疑、失败资产复用和高信息增益假设；
不读取 RCLE RGB algorithm outcome，不产生机制确认或产品结论。
[动态数据候选池](RCLE_PHASE_B_DYNAMIC_DATA_CANDIDATE_POOL_2026-07-26.md) 已记录
TUM、ETH3D、ICL-NUIM 与 EVIMO2 的当前证据和成本，但不是硬准入或批量下载队列。

首个 Discovery 假设
[PB-H1 role proxy R0](RCLE_PHASE_B_PB_H1_ROLE_PROXY_R0_RESULT_2026-07-26.md)
已完成 `SUPPORT / VALID`：受控 fixture 证明相同 raw speed 的横移与前向接近可由
signed radial coherence + time-normalized parallax 区分；固定 burned Bonn
首窗呈横向/混合 translation 结构。旧 raw-speed gate 因果错位，但 absolute
radial aggregate 单独也不能充当 approach gate。后续单来源
[TUM `fr2/rpy` source-native geometry audit R0](RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_RESULT_2026-07-26.md)
已 `PASS / VALID`：9/10 固定窗口可评价，窗 0/3/6 提供 rotation-dominant
真实几何；但本 source 没有出现旧 `0.02 m/s` 的窗口级错杀。后继
[小型 real-data geometry canary R0](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_PREREGISTRATION_2026-07-26.md)
已完成独立 F1 预注册：固定窗 `0/3/4/6` 和 `1196` 个 pair record，只检验
producer/independent-validator 的身份、schema、弃权与 float64 parity。版本隔离
实现及[独立实现审查](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-26.md)
已 `PASS`，implementation lock `0d833b83…e2387`。随后绑定该 lock 的
[唯一正式执行](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_RESULT_2026-07-27.md)
已完成：producer 与独立 validator 各输出/复算 `1196` pair，全部 identity、
schema、abstention、branch 与 float64 parity mismatch 为 `0`，终态
`VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`。这只调通 geometry
interface；RCLE RGB algorithm outcome 未读取。[RGB algorithm canary R0
预注册](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_PREREGISTRATION_2026-07-27.md)
现已冻结最小 rotation-leakage comparator、数据 firewall、独立 validator 与性能
准入设计，并已通过[独立设计审查](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DESIGN_REVIEW_RESULT_2026-07-27.md)。
TUM 没有建立真实 positive approach role，因此当前终态为
`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`；设计审查最多使设计包
有效，最大权限仍为 `EXECUTION_NOT_AUTHORIZED`。随后另立的
[真实 positive approach 数据角色准入 R0](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R0_RESULT_2026-07-27.md)
在 source access 前完成 exclusive claim、source descriptor hash 和几何门冻结，
只审计唯一 `EVIMO2 v2 / Flea3 / sanity_ll` 的 metadata/pose/depth。13 个固定
首窗、`3895` pair 经独立全量重算 mismatch 为 `0`，但 `0/13` 窗同时通过
coverage、signed radial 与 positive-fraction 门，终态仍为
`HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`。没有读取 RGB algorithm
outcome、换源/换窗或降门；在该 evidence version 当时，algorithm
implementation/performance qualification 与正式 execution 决策继续关闭。

Phase A 必须使用程序生成的、可复算的连续帧与运动真值，固定旋转补偿、Sparse LK、局部仿射拟合和网格汇总，比较纯旋转泄漏与真实闭合保留。trial-level leakage error 和 closing error 是主判据；RSR/CRR 只作诊断比例。

只有独立 sealed synthetic validation 通过且 Phase B 被单独激活后，才可进入
B0。当前 Progressive Discovery 不授权：

- B1 R5 同版本重跑、B1B R5 或任何 RCLE RGB algorithm metric；
- patch、替换或重跑已消费 claim 的 B1A；
- 把 Bonn B1A diagnostic 或用过的 sequence 重新包装为 unseen confirmation；
- 单项增强或多算法堆叠；
- Risk Field 与 token/lifecycle；
- Android RCLE 集成或主动告警；
- 真人试验、独立行走或生产晋级；
- 为追求“产品完整”而提前解决全数据权威、硬件闭环或安全认证。

## Phase A 当前结果

[Phase A R0 Synthetic Signal Audit](RCLE_MINIMAL_PHASE_A_SYNTHETIC_SIGNAL_AUDIT_R0_RESULT_2026-07-26.md) 已按结果前冻结的 `2520`-trial 协议完整运行并独立复算，终态为 `REVISE / VALID`。

clean yaw/pitch 旋转泄漏抑制、roll 不增噪、scale up/down 与 rotation+scale closing 保留、15/30/60 FPS 一致性及三类 stress 中可评价 trial 的误差门均通过；但逐 condition coverage 硬门失败：clean worst cell 为 `10/20`，partial-occlusion pitch worst cell 为 `0/20`，因此总体 `2419/2520` 可评价不能构成 Kill Gate A PASS。

[Phase A Coverage Revision R1](RCLE_MINIMAL_PHASE_A_COVERAGE_REVISION_R1_RESULT_2026-07-26.md) 已执行这唯一一次版本化实现修订。它保持 R0 协议、全部 trial/seed/threshold/gate 与输入 hash 不变，使 clean 达到 `1680/1680`、stress 达到 `810/840`，并消除了 affine residual 超门；但 partial-occlusion pitch 四个 cell 仅为 `12/20、12/20、13/20、13/20`，worst `0.60 < 0.70`，coverage 仍失败。R1 receipt `d5edb952…2b15` 独立复算为 `REVISE / VALID`。

由于冻结协议规定单次版本化 REVISE rerun 仍失败即停止，该实现终态为 `STOP_CURRENT_IMPLEMENTATION / VALID`。不得再做第二次 coverage revision、降门或改矩阵。当前 Phase B 权限来自后续 Observable Support Recovery sealed PASS 与用户单独激活，不回写或改写本轮 coverage 失败；不能继续在旧 2520 trials 上选择实现。

## Observable Support Recovery R0 设计边界

[Observable Support Recovery R0 设计预注册](RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_PREREGISTRATION_2026-07-26.md)
已冻结一个、且仅一个新观测模型候选：
`OBSERVABLE_THREE_FRAME_SUPPORT_MANAGER_R0`。它只研究如何用连续三帧中的
forward-backward consistency、photometric consistency 和 track lifecycle
识别可观测遮挡，并在原 3×3 cell 内做确定性空间补点；不得读取 generator
occlusion mask，不得降低或绕过任何原门。

本边界当前状态为
`DESIGN_FROZEN / DESIGN_REVIEW_PASS / EXECUTION_NOT_AUTHORIZED`。最终
设计锁 `3fcc21e2…52bac` 已通过
[独立只读审查](RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_DESIGN_REVIEW_RESULT_2026-07-26.md)，
但审查 PASS 不构成实现授权。旧 seeds
`1000–1019` 与 R0/R1 结果永久只作 discovery；新的 development
`2000–2019` 和 sealed validation `3000–3019` 已在机器设计锁中结果前固定，
但当前禁止实现、物化 formal trials、运行结果或抓取真实数据。只有设计锁
通过后续独立任务另获明确权限，才可“只实现一个候选”；未来 development 或
validation 任一原 clean、error、FPS、stress、coverage 门失败即关闭候选。
即使独立 validation 全门 `PASS`，Phase B 也仍需另行决定，不自动开放。

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
4. Observable Support Recovery R0 的唯一 implementation + development gate 已按明确授权完成；development `2000–2019` 原完整 2520-trial matrix 为 `PASS / VALID`，receipt `93b4c924…214e3c`。
5. 独立 sealed validation `3000–3019` 已一次性完整运行，2520/2520、六组件全 PASS、receipt `d10afb25…6365c` 独立复算 `VALID`；详见 [Sealed Validation 结果](RCLE_OBSERVABLE_SUPPORT_RECOVERY_R0_SEALED_VALIDATION_RESULT_2026-07-26.md)。
6. 当前终态为 `INDEPENDENT_SYNTHETIC_PASS_ONLY_PHASE_B_REMAINS_CLOSED_PENDING_SEPARATE_DECISION`。真实数据、Phase B、Replay Demo、Android、人体、安全和生产权限仍关闭；任何后继必须另立决策。
7. 已另立 [Phase B Bonn 入口预注册](RCLE_PHASE_B_BONN_ENTRY_PREREGISTRATION_2026-07-26.md)，唯一下一候选为 `BONN_METADATA_BLIND_AUTHORITY_AND_COHORT_FREEZE_R0`。最终锁 `e49d1f88…31c9e9` 已通过[独立只读审查](RCLE_PHASE_B_BONN_ENTRY_DESIGN_REVIEW_RESULT_2026-07-26.md)，当前 `DESIGN_REVIEW_PASS / EXECUTION_NOT_AUTHORIZED`；旧 Bonn 历史排除 manifest 为 9 条，payload 与正式 Phase B 仍禁止访问或运行。
8. 经明确授权后，唯一 [Bonn metadata gate R0](RCLE_PHASE_B_BONN_METADATA_GATE_R0_RESULT_2026-07-26.md) 的 26-row 内容与 6 条 cohort 可复算，但独立审查发现 runner override 绕过 one-run/canonical-output 合同，终态为 `CONTENT_VALID / EXECUTION_CONTRACT_FAIL`。R0 receipt 和 cohort 降为 non-authoritative diagnostic，候选关闭；payload、window inventory 和所有 Phase B 指标仍未授权。
9. R1/R2 继续暴露 preclaim 项目读取，均保留为 diagnostic；最终 [R3 canonical metadata authority](RCLE_PHASE_B_BONN_METADATA_AUTHORITY_R1_R3_RESULT_2026-07-26.md) 以最小 hash-bound bootstrap 在任何项目读取前 exclusive-create claim，唯一正式门和独立复算均 PASS。权威 receipt 为 `05a283b8…489b`，固定 26/9/6 与 cohort `513b770d…ae86e`。
10. [Phase B Bonn B0 预注册](RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_PREREGISTRATION_2026-07-26.md) 已通过[独立设计审查](RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_DESIGN_REVIEW_RESULT_2026-07-26.md)，design lock `a0b04ac5…c757`，当前 `EXECUTION_AUTHORIZED / NOT_STARTED`。这表示已推进到可进入正式 Phase B B0 acquisition/timestamp-inventory 的边界；本轮没有下载 2.26 GB payload，也没有运行 Phase B 指标。
11. B0 R0 启动前的数据可行性审计对六个 official URL 做了无 body `HEAD`，违反 network-before-claim 顺序；已按 [R0 合同结果](RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R0_EXECUTION_CONTRACT_RESULT_2026-07-26.md) 公开关闭，未伪装成 formal run。
12. [B0 R1 recovery 预注册](RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1_PREREGISTRATION_2026-07-26.md) 保持同一 cohort、URL、分母和门；design 与 implementation review 均 PASS 后，唯一 canonical run 已完成并由独立 validator 复算。结果为 [B0 R1 `PASS / VALID`](RCLE_PHASE_B_BONN_FORMAL_ENTRY_B0_R1_RESULT_2026-07-26.md)：6/6 archive/timestamp 可评价，固定 `10` 个窗口，receipt `dc0ffe9a…1f86`。当前只允许冻结 B1 metric protocol；image/pose metric 读取与 B1 execution 仍关闭。
13. B1 R5 protocol 与 B1A implementation review 通过后启动唯一 source-native geometry run。producer 完成十窗输出，但 independent full replay 在 24 个 abstaining pair 的 216 个 grid 上发现 blank-grid key-set mismatch，并连带触发 ledger identity mismatch；[正式结果](RCLE_PHASE_B_BONN_B1A_RESULT_2026-07-26.md)为 `INVALID_EXECUTION_CLOSE_B1`。诊断 receipt 同时显示 rotation/approach 合格 sequence 均为 `0`，但 INVALID 内容不构成正式数据结论。B1B 未运行且关闭。
14. 用户基于多轮工作明确要求研究治理本身能够学习、质疑并进化。仓库已采用 Discovery/Canary/Development/Confirmation/Deployment 渐进冻结；每次失败必须形成 learning record，候选按因果差异、信息增益、可证伪性和成本排序，失败资产可降级为 regression/canary/counterexample/source characterization。B1A closure overlay 保持科学问题 `OPEN`，新的 [Progressive Discovery R0 contract](RCLE_PHASE_B_PROGRESSIVE_DISCOVERY_R0_CONTRACT_2026-07-26.json) 已成为当前 Phase B 边界。
15. [PB-H1 role proxy R0](RCLE_PHASE_B_PB_H1_ROLE_PROXY_R0_RESULT_2026-07-26.md) 已用一个受控 fixture 与 deterministic burned Bonn 首窗完成。公式通过纯旋转零响应和前向解析标定；旧 raw-speed proxy 错位。下一步只审计 TUM `fr2/rpy` 的 source-native geometry，不下载其他来源或读取算法输出。
16. [TUM `fr2/rpy` source-native geometry audit R0](RCLE_TUM_FR2_RPY_SOURCE_NATIVE_GEOMETRY_AUDIT_R0_RESULT_2026-07-26.md) 已按下载前冻结的 10 s 全窗规则完成：9/10 窗、2852/2990 formula-level pair 可评价；window 4 因 source-depth coverage `<0.50` 弃权，rotation-dominant 窗为 0/3/6，终态 `PASS / VALID`。旧 `0.02 m/s` 在本 source 未出现窗口级错杀；没有换序列或运行 RCLE RGB。下一步可以单独设计小型 geometry canary。
17. [小型 real-data geometry canary R0](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_PREREGISTRATION_2026-07-26.md) 已冻结为 F1 implementation canary，机器合同 SHA `48f8b901…453c`。版本隔离 producer、独立 validator、18 项 fixture/mutation tests、exact output schema 和 one-shot claim runner 已通过[独立实现审查](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_IMPLEMENTATION_REVIEW_RESULT_2026-07-26.md)，implementation lock 为 `0d833b83…e2387`。绑定该 lock 的[唯一正式 R0](RCLE_PHASE_B_REAL_DATA_GEOMETRY_CANARY_R0_RESULT_2026-07-27.md) 已消费 claim：窗 0/3/4/6 共 `1196` pair，producer/validator 全量一致，所有 gating mismatch 为 `0`，failure receipt 不存在，终态 `VALID_IMPLEMENTATION_DEBUGGED_GEOMETRY_INTERFACE_ONLY`。正式运行约 2 h 07 min，gzip TGZ 逐 pair 回扫造成约 `2156.646 GiB` 逻辑读取；这不改变 parity PASS，但下一 evidence version 必须做真实 archive mechanics 性能预检。RGB algorithm outcome 未读取，algorithm canary 仍需另立任务和权限。
18. [RGB algorithm canary R0 F1 设计包](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_PREREGISTRATION_2026-07-27.md) 已经三轮独立只读审查，前两轮 `FAIL` 留痕，第三轮 [DESIGN_REVIEW_PASS](RCLE_PHASE_B_RGB_ALGORITHM_CANARY_R0_DESIGN_REVIEW_RESULT_2026-07-27.md)。数据角色、exact pair identity、唯一 algorithm specification、独立 validator 和 performance/preflight 合同已机器闭合；真实 positive approach role 仍为 `0`，performance qualification 为 `NOT_RUN`，终态保持 `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID / EXECUTION_NOT_AUTHORIZED`。
19. [Real positive-approach role admission R1](RCLE_PHASE_B_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R1_RESULT_2026-07-27.md) 在 payload 前冻结完整 burned/exclusion manifest、唯一 ETH3D `sofa_3` source authority、contract 与 implementation lock。唯一 claim 后只执行 `1 GET / 0 retry / 0 replacement`；response artifact 为 `84,363,952` bytes、SHA `9ea6fcc8…3c5d1`，但 official container identity 不满足冻结 required-member 合同，独立 source-access validator 为 `VALID`，终态 `HOLD_ALGORITHM_CANARY_APPROACH_ROLE_INCOMPLETE / VALID`。geometry 未运行、role 未准入，performance qualification 不存在且关闭；整个 ETH3D sofa scene family 已烧掉，禁止换 `sofa_4` 或作为 confirmation。
20. [Real positive-approach source-authority Discovery R0](RCLE_REAL_POSITIVE_APPROACH_SOURCE_AUTHORITY_DISCOVERY_R0_RESULT_2026-07-27.md) 在任何候选 payload 前比较 CID-SIMS、CoRBS 与 KITTI-360。CID-SIMS ScienceDB V6 的 `Floor3` 闭合为固定 `floor3_1/2/3` 三-run family：许可 `CC BY-NC-ND 4.0`，每条 archive 均有 file ID、精确 bytes、官方 MD5 和稳定 HEAD 入口，坐标链闭合；因此唯一下一数据族候选为 CID-SIMS Floor3。CoRBS E3 当前官方对象均 `403` 且缺 bytes/checksum，KITTI-360 只保留车载压力角色。本版未下载 payload、未读 geometry/RGB outcome、未运行算法；独立 R2 必须在 GET 前结果盲选择一个 exact Floor3 archive，失败后禁止换 run 或换源。
21. [CID-SIMS `floor3_1` RGB development canary R0](RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_CID_SIMS_FLOOR3_1_RESULT_2026-07-27.md) 复用本地缓存并绑定相邻两窗；冻结算法输出在 window 1 为 `299/299` 触发、window 0 为 `148/299`。原 validator 因浮点 `dt` 精确比较缺陷将 R0 判 `INVALID_R0_EVIDENCE / INVALID`，该终态永久保留。
22. [Posthoc validator R1](RCLE_RGB_ALGORITHM_DEVELOPMENT_CANARY_R0_POSTHOC_VALIDATOR_R1_RESULT_2026-07-27.md) 不重跑算法，只对 immutable ledger/cache/result 做 Decimal identity 与 aggregate 复算，终态为 `POSTHOC_OUTPUT_AUDIT_VALID / VALID`；它只证明输出一致性，不回写 R0 或形成独立确认。
23. [CID-SIMS pairwise geometry alignment R0](RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_PAIRWISE_GEOMETRY_ALIGNMENT_R0_RESULT_2026-07-27.md) 对 immutable RGB ledger 的 598 对逐对重算 source-native depth+pose geometry，coverage `598/598`、abstention `0`。三档合并触发为 `0/139`、`5/17`、`442/442`；window 0 的 148-pair 触发段覆盖其后 143-pair positive-geometry 段，说明该窗后半段实际进入 approach。独立 validator 重算全部 geometry/aggregate，`errors=[] / VALID`。结论仅为 outcome-aware posthoc mechanism alignment；不调阈值、不构成 performance qualification，下一阶段应使用本地缓存中的互斥窗口做 RGB-outcome-blind geometry-stratified holdout。
24. [CID-SIMS floor3_1 disjoint geometry-stratified holdout R0](RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_1_DISJOINT_GEOMETRY_STRATIFIED_HOLDOUT_R0_RESULT_2026-07-27.md) 已在 W0–W1 加 10 秒 W2 guard 后固定 W3–W11，并要求精确 `2 positive + 2 below-reference`、选中窗起点至少间隔 20 秒。七个身份合格窗均为 `299/299` geometry 可评价、positive fraction `1.0`、持续约 `9.96–9.98 s`；W5/W9 因 299/298 帧未过冻结 300-frame identity，低参考窗为 `0`。终态 `GEOMETRY_STRATIFIED_WINDOWS_NOT_EVALUABLE / VALID`；selected RGB identity/cache/ledger 均不存在，RGB bytes `0`、算法未运行、validator `errors=[]`。这只证伪 floor3_1 剩余固定网格能同时提供两类角色的假设，不是 RGB 失败、cross-source confirmation 或性能资格；无自动救场后继。
25. [CID-SIMS floor3_2 cross-sequence development holdout R0](RCLE_RGB_ALGORITHM_CID_SIMS_FLOOR3_2_CROSS_SEQUENCE_HOLDOUT_R0_RESULT_2026-07-27.md) 在 transport/science/implementation 预冻结后，以 8-worker 对不同 official run 的全部 18 个完整 10 秒窗执行 geometry-only。结果为 `17 positive / 0 below-reference / 1 ambiguous`，selection `NOT_EVALUABLE`，RGB identity/cache/ledger 均不存在、RGB bytes `0`、算法未运行。Formal validator 因 W3 偶数样本 median 的 Decimal/float 表示级精确比较保留 `CROSS_SEQUENCE_HOLDOUT_INVALID / INVALID`；另立 post-hoc R1 仅用 `rel_tol=1e-12, abs_tol=1e-15` 修复数值等价，冻结 evidence 全量复核 `errors=[] / VALID`，不回写 formal terminal。结论是 Floor3 两个 run 均不能提供冻结 2+2 低参考对照；下一步须另立包含静止/横移/远离角色的 source-authority discovery，不自动使用 floor3_3。

## 主线变更规则

后续 Codex 任务默认服从以下优先级：

1. 用户最新明确指令；
2. 本 current 入口和 R1.1 当前执行协议；
3. 仓库级渐进式研究治理；
4. R1.0 长期能力地图；
5. 日期化 USTRF/Looming 目标、结果和历史 handoff。

任何旧文档中的“当前主线”“独立新算法入口”或“下一阶段”，若与本页冲突，均降级为历史上下文，不构成执行授权。主线再次变化时，必须同步更新本页、`AGENTS.md`、根 `README.md`、文档索引和 `DEVELOPMENT_LOG.md`。

任何 current、`AGENTS.md`、R1.0/R1.1 或数值门都允许被 evidence-backed
`RULE_CHALLENGE` 质疑；在版本化 amendment 生效前不得静默绕过。历史结果和
receipt 不回写，但其数据、实现与失败可按新治理显式降级复用。
