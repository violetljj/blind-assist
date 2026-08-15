# 算法研究入口

状态：`current / ALGORITHM_FOCUS=VITG_G0 / ACTIVE_REVERSIBLE_LANES=1 / VITG_G0_PREOUTCOME_PROTOCOL_FROZEN / SCHEDULE=EVIDENCE_GATED_NO_DAY_QUOTA`

Assistive Geometry 的 factor-wise、obstacle 与 Q-Plane 表示支线均已按停止条件关闭；TARO R38 也已在
parent-disjoint ARKitScenes Validation confirmation 上有效失败并关闭。用户已明确新开 SATOM-A，
它通过主动稀疏 metric range 与 causal task-space memory 改变输入信息、时间结构、表示和动作，
不是旧 selector 的 rescue continuation。SATOM-R0 的 Bonn Real E0 在任何 arm metric 产生前，
因 frozen DepthART prior 无法稳定提供 ground-height observability 而 `NOT_EVALUABLE` 并关闭。
GA-SATOM 的 physical-ToF G0 协议保留但按用户新的无 ToF 选择暂停。当前主线改为独立的
VI-Task Geometry：利用现有 AtomS3R-M12 同刚体 OV3660 + 板载 BMI270，先检验 metric-frame
observability；当前缺同步 capture implementation、calibration、具体 VIO lock 与 exact fresh roster。
DepthART D3R6 仍保持暂停；其 bounded deferral contract 与 fresh gate 保留，但 post-hoc
same-domain random audit 不支持 risk ranking 的增量收益，不恢复 D3R6 执行权限。

表中“下一动作”就是唯一 successor；完整历史、数值和所有禁止项留在路线真源。`无` 表示
路线保持 closed、paused 或 diagnostic，不能从旧文档推导隐含下一步。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作（唯一 successor） | 禁止动作 | 影响默认 App |
|---|---|---|---|---|---|---|
| VI-Task Geometry | 同刚体 RGB+IMU 自校准 metric pose/ground，再为未来 task geometry 分配 computation/parallax budget | `WILD_LAB / VITG_G0_PREOUTCOME_PROTOCOL_FROZEN / ATOMS3R_M12_ONBOARD_BMI270_CONFIRMED / BLOCKED_ON_SYNCHRONIZED_CAPTURE_CALIBRATION_IMPLEMENTATION_LOCK_AND_FRESH_ROSTER / REAL_G0_NOT_RUN / NO_TOF / NO_TRAINING` | [VI-Task Geometry current](vi-task-geometry/README.md) | `VITG_G0_ATOMS3R_RGB_IMU_CAPTURE_PREFLIGHT`：实现并编译验证 ToF-disabled 同钟 writer，绑定 calibration、reference、A0-A3 implementation 与 exact roster | 用手机 IMU配眼镜视频；重跑 ARCore D45；打开 outcome 后改 VIO/parent/gate；提前训练、主动分配或接 Android | 否 |
| GA-SATOM | 固定总稀疏测距信息预算内，先建立 metric ground frame，再保留剩余预算给未来 task sensing | `PAUSED_BY_NO_EXTERNAL_TOF_SELECTION / G0_PROTOCOL_RETAINED / REAL_G0_NOT_RUN / NO_PROCUREMENT` | [GA-SATOM current](ga-satom/README.md) | 无；只有用户明确恢复 external-ToF 路线才可重开 preflight | 采购/采集/运行 physical G0；用单区/模拟/旧输出替代；提前运行 G1/arm/训练 | 否 |
| SATOM-A | metric pose + active sparse ToF + frozen dense prior → causal task-space occupancy memory | `WILD_LAB / SATOM_R0_REAL_E0_NOT_EVALUABLE / DEPTHART_GROUND_HEIGHT_OBSERVABILITY_FAIL / NO_ARM_METRIC / CLOSED_NO_TUNING` | [SATOM-A current](satom/README.md) | 无；新的 pre-outcome 协议须先提供独立 ground-height observability 或 materially different representation | 用 truth height/scale 修补 candidate；在已打开 Bonn/DepthART 输出上放宽高度门、改 winner rule、训练或接 Android | 否 |
| BlindAssist Assistive Geometry | 连续因子 → selective no-regret → body-swept geometry | `CORRECTION_GAIN_LOPO_FAIL_STOP / ANGULAR_BOUNDARY_FAIL_CLOSED_SAFE_BUT_TASK_INERT / SUPPORT_VALIDITY_FAIL_OPEN / OBSTACLE_RGB_INTERACTION_FAIL_STOP / POSE_ANALYTIC_FAIL_STOP / QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING / CURRENT_OBSTACLE_TASK_ROUTE_CLOSED / SCIENTIFIC_NOT_RUN` | [Assistive Geometry current](assistive-geometry/README.md) | 无 active successor；需新增 source-native obstacle supervision，或 Q-Plane 家族之外的 materially different representation 后另立协议 | 调参/重跑 Q-Plane 或进入 O0-B/训练；继续调旧 obstacle 候选；让 support 创建 validity；重开 correction/boundary/fresh3；把 consumed negative 写成 task success | 否 |
| TARO / Task-directed Active Risk Observability | 同预算观测使 body/path query 更早可识别 | `R31_V6_LEAK_INVALIDATED / R36_FRESH_PARENT_FAIL / R37_CLOSED / R38_REFERENCE_PREFLIGHT_PASS / R38_R32_FRESH_PARENT_CONFIRMATION_FAIL / TARO_CLOSED_NO_RESCUE` | [TARO current](taro/README.md) | 无；R38 已有效消费，论文主线盘点前不建立 R39/R40/R41 | 覆盖/重跑 R38；回调 R31/R32/R35/R36/R37；以 R38 outcome rescue tuning；接默认 App/risk/guidance；夸大为 fresh-source、broad breakthrough、安全或产品成功 | 否 |
| AG-QSF / Queryable Survival Geometry | profile-queryable q-contact 生存分布统一 clearance 与 horizon occupancy | `CLOSED_DATA_SUPPORT_INSUFFICIENT / H1_NOT_EVALUABLE` | [AG-QSF current](assistive-geometry-qsf/README.md) | 无；只有新的 pre-outcome target/data contract 可重开 | 伪造 parent-disjoint censor support、实现未授权 H2 或声称数学假设被反证 | 否 |
| AG-CBF / Corridor Bottleneck Field | 检验 ground-aligned corridor bottleneck 是否保留三带摘要丢失的信息 | `R0_CLOSED_DATA_SUPPORT_NOT_EVALUABLE / ORACLE_NEVER_AUTHORIZED` | [AG-CBF current](assistive-geometry-cbf/README.md) | 无；新版本必须从 source-geometry/data support 重开 | 填补 UNKNOWN、事后改 gate、实现 oracle/model 或复用旧权限 | 否 |
| DepthART-S | parent-relative risk + fixed-budget UNKNOWN deferral，降低危险 false-clear 且封顶 coverage/false-block 代价 | `R1_RESEARCH_MAINLINE / D3R6_FROZEN_GATE_PASS / BOUNDED_DEFERRAL_CONTRACT_PRESERVED / POSTHOC_SAME_DOMAIN_RANDOM_AUDIT_NO_INCREMENTAL_RANKING_SUPPORT / USER_PAUSED / R2_CANDIDATE_NOT_AUTHORIZED` | [DepthART current](hftf/README.md) | 无；按用户要求暂停 | 在已打开 parent 上重调 2% budget/checkpoint；把 fresh gate 写成 ranking 增量证据；把 UNKNOWN 当 negative；读 sealed R2 或写成默认 App/安全结论 | 否 |
| DA2 | teacher、baseline、reference、fallback | `CLOSED / FROZEN_DEVELOPMENT_REFERENCE` | [DA2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 无；新问题必须建立新版本和独立数据路线 | 把 reference 写成 active candidate 或沿用旧下一步 | 否 |
| YOLO + 语义分割双环 | 论文次线：事件级风险/可通行性增量 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE` | [dual-loop current](dual-loop/README.md) | `RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR` | truth 未冻结时训练、选模或晋级 | 否 |
| RCLE-RF | 暂停的历史风险场研究 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE current](rcle/README.md) | 无；只有用户明确重开后才能建立新 scoped successor | 消费暂停前授权或把旧 snapshot 的下一步当当前权限 | 否 |
| USTRF-SC route-conditioned | 已收口的历史路线代理 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF closure](ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md) | 无；实质不同的假设必须登记为新路线 | 把实验 App、旧协议或诊断结果写成当前主线 | 否 |

## 算法边界

数据、链路、性能和部署证据可以支撑算法，但不能替代 admission；算法结果也不能反向证明
产品安全、默认 App 可用性或真实用户效果。
