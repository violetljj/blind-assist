# USTRF-SC 研究文档索引

状态：research index
当前生产授权：无；正式 App 状态仍以 `docs/SANPO_CURRENT_STATUS.md`、当前协议和可复现门禁报告为准。

## 首要入口

- [项目工作记录与恢复入口（2026-07-20）](USTRF_SC_PROJECT_RECORD_AND_RESUME_2026-07-20.md)：本轮总体思路、实验结果、GPU 中断边界和下一步。
- [新窗口交接（2026-07-20）](USTRF_SC_WINDOW_HANDOFF_2026-07-20.md)：当前暂停点、证据入口、GPU 边界、参考文档清单和可复制续接提示词。
- [双环实施状态与证据边界](USTRF_SC_IMPLEMENTATION_STATUS.md)：逐模块实现、证据、缺口和授权。
- [研究型离线量化基线（2026-07-20）](USTRF_SC_RESEARCH_METRICS_2026-07-20.md)：SANPO 表征、公开 source-native 几何/轨迹、REveL detector、range/radial-motion 分层与 V13 门禁。
- [安全内核实验方案（2026-07-20）](USTRF_SC_SAFETY_KERNEL_EXPERIMENT_2026-07-20.md)：本轮详细实现与实验演化。

## 协议与专项记录

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
