# USTRF-SC 研究文档索引

状态：research index
当前生产授权：无；正式 App 状态仍以 `docs/SANPO_CURRENT_STATUS.md`、当前协议和可复现门禁报告为准。

## 首要入口

- [前沿论文研究指导（2026-07-22）](USTRF_FRONTIER_PAPER_GUIDE_2026-07-22.md)：13 篇本地论文的证据—论点映射，以及 tracker/TTC、时序深度、显式路线 dense risk、开放词汇/VLM 和反馈研究的固定协议与停止条件；仅为日期化研究指导。
- [项目工作记录与恢复入口（2026-07-20）](USTRF_SC_PROJECT_RECORD_AND_RESUME_2026-07-20.md)：本轮总体思路、实验结果、GPU 中断边界和下一步。
- [新窗口交接（2026-07-20）](USTRF_SC_WINDOW_HANDOFF_2026-07-20.md)：当前暂停点、证据入口、GPU 边界、参考文档清单和可复制续接提示词。
- [双环实施状态与证据边界](USTRF_SC_IMPLEMENTATION_STATUS.md)：逐模块实现、证据、缺口和授权。
- [研究型离线量化基线（2026-07-20）](USTRF_SC_RESEARCH_METRICS_2026-07-20.md)：SANPO 表征、公开 source-native 几何/轨迹、REveL detector、range/radial-motion 分层与 V13 门禁。
- [安全内核实验方案（2026-07-20）](USTRF_SC_SAFETY_KERNEL_EXPERIMENT_2026-07-20.md)：本轮详细实现与实验演化。

## 协议与专项记录

- [causal route-relative intrusion signal R0（2026-07-24）](USTRF_CAUSAL_ROUTE_INTRUSION_SIGNAL_R0_RESULT_2026-07-24.md)：首次加入 track/relation/timing 之外的 route-relative radial/lateral trend 与 bbox-scale expansion，冻结 5-frame `2-of-3` 后一次性跑完整 41 序列；仅覆盖 `7/11=21/33`，负暴露 `43/4.956min=8.6759/min`，终态 `SIGNAL_REJECT / VALID`。这是新信息信号的定量淘汰，不是提醒能力提升，也不允许事后调阈值回救。
- [current-input policy feasibility bound R0（2026-07-24）](USTRF_CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0_RESULT_2026-07-24.md)：只对 track scope、active route relation、route validity、reset 与相对 timestamp 构成的共享单调 lease family 求经验上界；保留 two-frame、fail-closed、one-token/reset、no-renewal，并以忽略 nominal TTL 的乐观 relaxation 穷尽完整 duration frontier，不输出候选 policy/threshold。最大 coverage 仅 `8/11=24/33`，经验风险 `<=2 token` 时仅 `2/11=6/33`，终态 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`；不再调时间参数或接 opener。
- [candidate-independent policy failure attribution R1（2026-07-24）](USTRF_CANDIDATE_INDEPENDENT_POLICY_FAILURE_ATTRIBUTION_R1_RESULT_2026-07-24.md)：只读复算冻结 policy 的 24 个 supported-cell miss 与 34 个负暴露 token。以 96 次 oracle qualification opportunity 为互斥原子，得到资格不足 `39`、TTL 后 oracle `39`、relation gap 提前失效 `12`、route unknown 提前失效 `6`、track `0`；24 个 cell 中 6 个保留混合原因。34 个负 token 全部联结失效原因，终态 `POLICY_FAILURE_ATTRIBUTION_CLOSED / VALID`，但 policy coverage/risk reject 不变。
- [candidate-independent causal token policy/risk gate R1（2026-07-24）](USTRF_CANDIDATE_INDEPENDENT_CAUSAL_TOKEN_POLICY_RISK_GATE_R1_RESULT_2026-07-24.md)：先在 41 条 full-sequence / 62,229 帧上冻结 500ms 持续资格、500ms TTL、fail-closed 失效与 one-token-per-track/reset 抑制，inventory 冻结后才联结 oracle/负暴露。有效期内 supported coverage 仅 `9/33`；负暴露为 `34/4.956min=6.86/min`，95% Poisson UCB `9.13/min`，cluster 支持也不足，故终态 `POLICY_COVERAGE_REJECT / VALID`，不得接 opener。
- [truth-blind causal per-track attribution-token producer 可行性与 extra-token 审计 R0（2026-07-24）](USTRF_TRUTH_BLIND_CAUSAL_PER_TRACK_ATTRIBUTION_TOKEN_PRODUCER_AUDIT_R0_RESULT_2026-07-24.md)：先在独立进程冻结候选无关的 41 条 full-sequence ledger / 62,229 帧，truth/event/oracle 解码均为 0；之后 post-hoc 得到 33/33 oracle-supported coverage，3 个无 active relation 单元继续 fail closed，unknown/reset/duplicate token 为 0。但 5,126 枚 token 中有 5,113 枚 extra，4.956 分钟负暴露内 153 枚（30.87/min），并完整记录 6,328 次被抑制重复激活，故终态仅 `HOLD_FOR_POLICY_GATE / VALID`，不能进入 opener 集成。
- [eligible target attribution → isolated one-shot opening 机制修复诊断 R1（2026-07-24）](USTRF_ELIGIBLE_TARGET_ATTRIBUTION_ORDERED_ISOLATED_OPENING_R1_RESULT_2026-07-24.md)：父 candidate-blind fact pack 冻结后才以 proxy/model truth 生成 36 个 reset-scope oracle token ledger；opener 明确消费 truth-derived token/window scope，但拒绝 raw truth。父 formed-delivery 的 6 个单元仍 token-qualified，27 个 pre-open/quarantine 单元反事实恢复；36 次 background namespace 强制变异均不改变目标 opening，且 27/27 在 qualification 前或当帧有 background activity。三候选均为 `11/12`，唯一无 active relation 的三个单元继续 fail closed。仓库仍无 causal candidate-blind token producer，因此不比较、selection 或开放 Android/H2/人体/生产。
- [known-route eligible-delivery formation 候选盲失败归因 R1（2026-07-24）](USTRF_KNOWN_ROUTE_ELIGIBLE_DELIVERY_FAILURE_ATTRIBUTION_R1_RESULT_2026-07-24.md)：冻结 123 个 candidate-blind formation trace receipt 后才联结 truth，对 36 个 C1–C3 × clearance-event 单元做互斥归因。6 个已形成 delivery；30 个失败中 3 个从未形成 active relation、9 个 window 前 pre-open、13 个 window 内但早于 target attribution pre-open、5 个 pre-invalid baseline latch/guard quarantine；reset split、streak 不足与 unexplained 均为 0。归因门通过但机制未修复，未比较、selection 或开放更高权限。
- [route-invalid + reset-scoped lifecycle 机制诊断 R1（2026-07-24）](USTRF_ROUTE_INVALID_RESET_LIFECYCLE_DIAGNOSTIC_R1_RESULT_2026-07-24.md)：只在冻结 123 条 C1–C3 trace 外包一层 lifecycle guard；三个候选的 unknown/stale active 帧从 `12,621 / 7,165 / 12,759` 降为 0，`1,235 / 801 / 1,238` 个 known→invalid-active 转换同帧关闭，且 reset key 无跨段泄漏。但 relation-based truth clearance 仍为 `0/12 / 1/12 / 0/12`，总体机制门失败；route-invalid/reset terminalization 不计 truth clearance，未比较、selection 或开放更高权限。
- [route-target R2-L1 metric profile R1 结果（2026-07-24）](USTRF_ROUTE_TARGET_R2_L1_METRIC_PROFILE_R1_RESULT_2026-07-24.md)：只读采用已闭合的 `123/123` 条 C1–C3 trace，输出三个逐指标 profile 并由 validator 重算 `186,687` candidate-frame；critical miss 均为 `0/8` 但 bound 不足，三个 profile 均有 unknown/stale active-alert 硬 veto 且 clearance 点估计失败，repeat underpowered、evidence age 因 `0/62,229` consume timestamp 为 `not_evaluable`。未比较、排名、selection 或开放更高权限。
- [route-target L1 candidate replay R2 结果（2026-07-24）](USTRF_ROUTE_TARGET_L1_CANDIDATE_REPLAY_R2_RESULT_2026-07-24.md)：在 41/41 canonical ledger 上完成 C1–C3 共 `123/123` 条权威 trace、`186,687` candidate-frame 和 `45` 次 reset；用户明确将本次 replay 内存门修订为 4 GiB，A1 的 10 条完整 trace 按父哈希继承、A2 新运行 113 条。A2 独立逐帧状态重放与 A3 strict-schema finalization 均为 `VALID`；A4 又在每条确定性复演前真实检查 4 GiB 门，123 次观测最小 `7,592,321,024` bytes，回执 `PASS`，且不创建新权威 trace。未生成 profile、比较、selection、shadow/H2 或生产权限。
- [route-target L1E materialization recovery R3 结果（2026-07-24）](USTRF_ROUTE_TARGET_L1E_MATERIALIZATION_RECOVERY_R3_RESULT_2026-07-24.md)：以目标 App 私有目录替代跨 UID external-files，并以严格串行“一进程一分片”完成剩余 38 条。A1 保留原 6 GiB 门；用户明确授权后，A2/A3 将主机可用内存门修订为 4 GiB，并分别用短控制路径与 Windows extended-path 写入闭合路径故障。最终 canonical input 为 `41/41` ledger、`62,229/62,229` 帧、`15/15 reset`；C1–C3、trace/profile 仍为 0。
- [route-target R2-L1X-L2P 结果（2026-07-24）](USTRF_ROUTE_TARGET_R2_L1X_L2P_RESULT_2026-07-24.md)：在任何新候选输出前冻结 L2 fresh-selection 执行级预注册与 non-executable L3 lockbox 模板；R2 和 outcome-unseen A1 均保留独立失败收据。最终因设备 manifest materialization 与 6 GiB 内存门耗尽，以 `FAIL_CLOSED_EXECUTION_ABORTED` 闭合；C1–C3、trace/profile、selection、L3、shadow/H2/人体/生产均未运行或开放。
- [route-target R2-L1E 单次探索 profile 结果（2026-07-24）](USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_R1_RESULT_2026-07-24.md)：合法终态为 `FAIL_CLOSED_EXECUTION_ABORTED`；6 GiB 可用内存门在初始尝试和两次有界重试中均触发，故首个 CrowdBot device raw attempt 未创建、候选未运行。逐 ledger 审计验证 2/41 条、4,594/62,229 帧 canonical raw，精确记录其余 39 条、57,635 帧缺口；selection、Android shadow、H2 与生产权限保持关闭。
- [route-target R2-L1E 单次探索 profile 通宵目标（2026-07-24）](USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_OVERNIGHT_GOAL_2026-07-24.md)：绑定 R2-L1 收据与冻结 C1–C3，让三个候选各对 41 条 masked sequence ledger 单次 replay，并在 15 个冻结观测断点重置状态；canonical raw 逐 ledger 分片闭环，不选胜者、不新增数据、不开放 Android/H2/生产。
- [route-target R2-L1 指标资格物化结果（2026-07-23）](USTRF_ROUTE_TARGET_METRIC_ELIGIBILITY_R2_L1_RESULT_2026-07-23.md)：候选盲覆盖 6,369 个事件/提案 × 8 指标；critical miss、clearance、unknown/stale 具备 L1 探索资格，repeat/evidence age 为条件资格，event recall/regeneration/false alerts 仍为 L0。mask、分母、来源贡献和删失原因已哈希绑定并通过完整复算；C1–C3 仍未运行。
- [route-target 证据成熟度分层标准 V2](USTRF_ROUTE_TARGET_EVIDENCE_MATURITY_STANDARD_V2.md)：当前协议真源；把全或无来源准入拆成指标级可评估性、右删失与 L0–L4 授权，保留 R1 性能/真实性门并冻结轮间版本化调整和来源预算。当前仅 L0，R2 候选未运行。
- [route-target evidence closure R1 阶段结果（2026-07-23）](USTRF_ROUTE_TARGET_EVIDENCE_CLOSURE_R1_RESULT_2026-07-23.md)：历史结果保持 `DATA_BLOCKED / STOP_SOURCE_SEARCH`；C1–C3 未运行，现有数据只保留诊断/回归或由 V2 重新冻结后的最高 L1 探索权限，Android shadow/H2 关闭。
- [route-target evidence closure R1 预注册（2026-07-23）](USTRF_ROUTE_TARGET_EVIDENCE_CLOSURE_R1_PREREG_2026-07-23.md)：固定 detector/tracker 与 H2 关闭边界，冻结五态逐人 route role、三条单接缝 oracle 和最多三个结构候选；当前 15+15 只做归因，胜者只能由全新两来源、含共现 person 的 sealed holdout 按逐来源/worst-source 门选出。
- [detector target attribution R1 与 association-only H1（2026-07-23）](USTRF_DETECTOR_TARGET_ATTRIBUTION_R1_RESULT_2026-07-23.md)：Android canonical G1b 语义 parity `4594/4594`；隐藏输出后冻结 15 个 target person 事件与 2,297 帧负窗 all-person/absent truth，baseline 两来源 coverage `3/3 + 12/12`、critical miss `0`，因此停止换 detector；T0–T3 均仅 `14/15` 且负窗误提醒 `8.620/min`，shadow 门失败，H2 继续关闭。
- [detector taxonomy coverage v1（2026-07-22）](USTRF_DETECTOR_TAXONOMY_COVERAGE_V1_RESULT_2026-07-22.md)：定位 tracker/TTC R1 的 host `[1,84,2100]` 解码轴错误；正确 host/SM-S9280 分别在 2639/2617 帧产生 person proposal，15/15 正事件区间均有 proposal，但 Android Canvas 与 PIL input/raw exact parity 为 0/4594 且缺目标 person bbox truth，因此 G3–G5、T0–T3 与 H2 继续关闭。
- [tracker/TTC 固定协议消融 R1（2026-07-22）](USTRF_TRACKER_TTC_ABLATION_R1_RESULT_2026-07-22.md)：两条 LILocBench 准入来源按 15 正/15 同源等长负窗口生成 4594 帧 host detector ledger；冻结 App 模型在 PIL letterbox 几何下输出 0 个阈值内 person 框，T0 event recall `0`、critical miss `1`，clearance/ID/TTC 不可评，按停止门跳过 T1–T3；不声称 Android 像素 parity。
- [R3 第三来源有界检索结论（2026-07-22）](USTRF_SENSOR_REPLAY_R3_THIRD_SOURCE_BOUNDED_OUTCOME_2026-07-22.md)：fail-closed 检索 IDSIA MSMPT `s9/s12/s13/s14`；s9 在 GT-only 门拒绝，s12/s13 在完整适配后双模型 AND 门拒绝，s14 同时失败 `.50` 深度几何门并被双模型拒绝。正式冻结 `2/3` 数据局限，不降门、不拿 Bonn 负样本补数、evaluator 未运行。
- [LILocBench 动态来源闭环（2026-07-22）](USTRF_SENSOR_REPLAY_R3_LILOCBENCH_GT_PRESCREEN_2026-07-22.md)：冻结 24/12/0.03/0.50，完整下载并适配 `dynamics_0` 2397 帧与 `lt_changes_dynamics_0` 8377 帧；双 reviewer + 独立裁决分别冻结 3/12 个事件，累计计 `2/3`。Bonn 两条双审拒绝、一条稀疏门拒绝；满三条前 evaluator 不运行。
- [OpenLORIS 来源替换结果（2026-07-22）](USTRF_SENSOR_REPLAY_R3_OPENLORIS_SOURCE_REPLACEMENT_2026-07-22.md)：只替换来源、不调整 R3 冻结门；7 条 office 稀疏预筛拒绝，2 条 cafe 经完整连续片段双模型一致拒绝，故准入仍为 `0/3`，冻结 evaluator 未运行。
- [连续 RGB-D+pose closed-loop R3（2026-07-22）](USTRF_SENSOR_REPLAY_R3_RESULT_2026-07-22.md)：独立 RGB-D VO、route truth/causal prediction、审核前 candidate trace、完整序列双模型 review 与逐来源五项门已实现；三条 TUM dynamic 序列均因非身体绑定前向路线被一致拒绝，五项保持不可评，`DO_NOT_SELECT_HARDWARE`。
- [多来源 RGB-D+pose sensor replay R2（2026-07-22）](USTRF_SENSOR_REPLAY_R2_RESULT_2026-07-22.md)：ETH3D/ICL-NUIM/TartanAir 各 120 帧的同步与时序重投影运输层通过，但隔离双模型一致拒绝 route/event admission，故 pose drift、路线投影和事件闭环仍不可评；保持 `DO_NOT_SELECT_HARDWARE`、120/U0 关闭。
- [跨相机 Codex 代理评测 R0（2026-07-21）](USTRF_CROSSCAM_CODEX_PROXY_R0_2026-07-21.md)：公开 360/POV 输入适配、三轮 Codex 共识与真实 Android bbox-route 对比；首个负样本暴露路线边界假阳性，只授权后续 proxy 扩样。
- [跨相机路线投影与走廊几何 R1（2026-07-21）](USTRF_CROSSCAM_ROUTE_PROJECTION_CORRIDOR_R1_2026-07-21.md)：显式投影收据、polygon + bbox bottom-center + 三档不确定性；Pexels 边缘车辆由确定侵入降为弃权，并完成 SM-S9280 等价实现测试。
- [跨相机目标归因诊断 R1.1（2026-07-21）](USTRF_CROSSCAM_TARGET_ATTRIBUTION_R11_2026-07-21.md)：六来源唯一目标账本、oracle 几何、Edmonton 稳定短窗投影与 SM-S9280 target-aware v2 已完成；Japan 首因改判路线合同，其余五来源首先受 detector taxonomy 覆盖阻塞。
- [跨相机 marker held-out R1.2 预注册（2026-07-21）](USTRF_CROSSCAM_HELDOUT_R12_PREREG_2026-07-21.md)：先在已见 R1.1 诊断集确认 YOLOE 静态三类 taxonomy，再冻结全新 3 正/3 负来源；新来源尚未运行 detector，Android export/parser 门仍关闭。
- [跨相机 marker held-out R1.2 结果（2026-07-21）](USTRF_CROSSCAM_HELDOUT_R12_RESULT_2026-07-21.md)：oracle 6/6；离线与 SM-S9280 真机均为正例召回 3/3、负例目标假告警 0/3、实例匹配 5/6；parser canary 与 host/device 事件结论一致性已闭合，但仍无生产授权。
- [跨相机连续事件工程 R1.2a（2026-07-21）](USTRF_CROSSCAM_CONTINUOUS_R12A_RESULT_2026-07-21.md)：12 个已见来源连续重放为正例 `4/6`、负例假告警/重复交付/共现接管均为 `0`；SM-S9280 600 秒无推理解码失败且温升通过，但 inference p50/p95 `762/978ms` 使设备门失败。R1.3 仅预注册 12 个未打开槽位。
- [跨相机移动端连续事件 R1.2b（2026-07-21）](USTRF_CROSSCAM_MOBILE_R12B_RESULT_2026-07-21.md)：同一 FP16-640 模型的 benchmark-only GPU 路线在 SM-S9280 达到 inference p50/p95 `40/54ms`、600 秒 0 失败、温升 `4.0°C`；但正事件仅 `4/6`，Japan 暴露事件 truth/路线代理冲突，London 仍为 detector 连续漏检，总体门失败且 R1.3 继续锁定。
- [跨相机 truth—路线几何一致性 R1.2c（2026-07-21）](USTRF_CROSSCAM_TRUTH_GEOMETRY_R12C_RESULT_2026-07-21.md)：独立正事件 oracle 为 `5/6` 一致；双模型复核加第三模型仲裁后，Japan 被裁为 `unknown/exclude` 而非正例或计分负例。当前缺第六个合格正事件，London FP16-768 GPU 候选仍禁止执行，R1.3 继续封存。
- [R1.2c 非 R1.3 seen positive 预注册（2026-07-22）](USTRF_CROSSCAM_SEEN_POSITIVE_R12C_PREREG_2026-07-22.md)：从已打开的 Bangkok Modern Center 来源冻结红白交通锥路线侵入事件；333s/336s 为 robust inside、339s 为 clear proxy，资格计数补到 6，但须先物化 R1.2c v2 并重跑全六例 oracle，768 仍未授权。
- [Bangkok 替换与 R1.2c v2 结果（2026-07-22）](USTRF_CROSSCAM_R12C_V2_RESULT_2026-07-22.md)：六正例 oracle `6/6` 后执行唯一 FP16-768 GPU 候选；真机事件召回 `5/6`，London 仍连续漏检，故跳过 soak、锁住 R1.3，并前瞻冻结 stride-4/P2 小目标 detector 假设。
- [stride-4/P2 小目标 detector R1.2d 受控研究（2026-07-22）](USTRF_CROSSCAM_SMALL_TARGET_R12D_RESULT_2026-07-22.md)：三 seed 配对 P2/P3 均只命中 `4/6`，London 均为 `0/22`，P2 虽提高离线小框 recall 但未转化为事件收益且稳定性/假检测更差；假设不支持，停止候选并继续锁住 R1.3。
- [R1.2d 后续工作计划（2026-07-22）](USTRF_POST_R12D_NEXT_WORK_PLAN_2026-07-22.md)：停止 detector 架构竞赛；按首个真实 matched pair、10-episode 采集链、独占 ARCore frame-bound 几何 canary、正式 truth/geometry 双门和 U0 的顺序推进，并为每阶段冻结停止条件。
- [模型代理 pilot 与 ARCore frame-bound R1 结果（2026-07-22）](USTRF_MODEL_PROXY_FRAMEBOUND_R1_RESULT_2026-07-22.md)：10 个模型生成 episode 经两次隔离模型 review 与 1000 帧重算后只开放正式代理矩阵扩展；SM-S9280 自动 canary 因 raw depth/tracking/稳定 Anchor 均为 0 触发冻结，故不扩 120 集、不运行 U0。
- [路线条件化无类别风险场主线](../../ROUTE_CONDITIONED_OBJECT_AGNOSTIC_RISK_FIELD_PLAN_2026-07-20.md)：当前优先研究路线；typed route-risk seam 已建立，真实事件与设备米制几何硬门仍阻塞，未授权训练或接入 App。
- [REveL YOLO11n 8/32 帧 crop/tiling 配对实验](USTRF_SC_REVEL_CROP_TILING_PAIRED_2026-07-20.md)：8 帧 canary 恢复 4/8 small miss，但 FP 从 4 增至 14，按预注册停止并跳过 32 帧。
- [设备阶段策略](USTRF_SC_DEVICE_PHASE_POLICY.md)
- [设备几何校准与证据协议](USTRF_SC_CALIBRATION_PROTOCOL.md)
- [临时手持刚体标定执行单](USTRF_SC_PROVISIONAL_HANDHELD_CALIBRATION_RUNBOOK.md)
- [离线安全仿真与自动验证](USTRF_SC_OFFLINE_SAFETY_SIMULATION.md)
- [SANPO 数据回放接入](USTRF_SC_SANPO_REPLAY_INTEGRATION.md)

## 外部与历史参考

- [GPT USTFR 完整算法框架与 Codex 交接指引（2026-07-21）](archive/2026-07-21-gpt-guidance/README.md)：保存用户提供的 Markdown/Word 原件；仅作外部历史参考，不覆盖当前 USTRF-SC 实施状态、门禁或生产授权。

## 阅读规则

- 想知道“现在做到哪里、下一步从哪里接”：先读项目工作记录，再读实施状态。
- 想复核具体数字：读取对应 `artifacts.local/evidence/` JSON/HTML receipt，不从概述反推。
- 想改变设备、训练或生产权限：以仓库 current 协议和 promotion gate 为准，日期化研究文档不能单独授权。
