# USTRF-SC 双环实施状态与证据边界

状态日期：2026-07-22。
总原则：本表区分“接口/单元测试存在”“设备候选观测存在”“独立验证完成”和“生产授权”。除非明确写为生产授权，否则默认均为 production-isolated experiment。

## 2026-07-22 独立体验版边界

- `ustrfExperiment` 是独立 application ID 的体验变体，可与正式 App 并存；只有该变体会绕过旧 `RiskAnalyzer`，将 `UstrfImageRouteProxy` 输出通过 object-agnostic risk-evidence seam 送入共享 `AssistDecisionKernel`。正式 debug/release 路径不变。
- 当前输入是同帧 CameraX/YOLO 检测框和固定画面中心假设路线。候选路线的最低代理风险只作诊断，绝不输出“向左/向右安全”；没有代理侵入时明确提示“不代表安全”，帧、时钟或尺寸不可信时 fail closed 为 HIGH/停下重扫。
- 这是为了提前体验 USTRF 的“路线条件化、对象无关风险、共享稳定/反馈内核”交互，不是完整 USTRF-SC。米制深度、稳定 pose、ground、physical TTC、真实 route truth 和事件验证仍缺失，所以授权仅为 `EXPERIMENTAL_APP_EXPERIENCE_ONLY`，不构成训练、生产或独立行走授权。

| 双环模块 | 当前实现 | 已有证据 | 未满足的门 | 当前授权 |
| --- | --- | --- | --- | --- |
| 传感输入与时间戳 | 隔离 CameraX timestamp/rotation/latest-only shadow | SM-S9280 上 receipt、rotation bracket 和 fresh sidecar 通过 | 跨传感器时钟与长期热/延迟 | benchmark-only |
| 位姿与同步 | raw ARCore pose bridge、VIO candidate/promoter 合同 | ARCore 在移动窗口可观测 TRACKING | ARCore world 为 `EPHEMERAL_PER_FRAME`；无跨帧稳定世界、无独立 mount 外参 | 不可进入 pose buffer/risk field |
| 深度与地面 | raw depth/confidence metadata gate、metric geometry promoter 合同；pure-Kotlin `UstrfMetricDepthGeometryAdapter` 只接收已注册 metric-depth raster、完整外参和已验证 body-local ground plane，产出 traversable/lower-body/head；`UstrfGroundVisibilityDropProposer` 只对观测深度相对地面交点的远距跳变提出 `DROP` | 861 个 moving depth candidate 中仅 1 个严格同帧；短窗内参 observation；JVM synthetic raster 回归覆盖 ground/body/head、错 registration、未验证 ground 与 stale depth；drop proposer 覆盖远距跳变、zero-depth 不误报与错 registration 拒绝；解析 temporal benchmark 的 hash、array 与 CUDA 重投影审计通过；公开 TartanAir JapaneseAlley P000/P002 分别因候选平面距离不合理/不稳定被同一门槛拒绝；CARLA 行人 RGB-D slice 的 20 帧同步 depth/pose/timestamp CUDA 审计通过，但含 image-reflected 坐标且未绑定人体/地面；真实 Bonn Dynamic `moving_obstructing_box` 的 590 RGB/592 depth/593 OptiTrack pose 同步与 CUDA 重投影通过，仍无 body/ground/event truth | 真实 depth-to-camera 登记、完整 SE(3) 外参、精度/置信/地面真值；drop 的真实 precision/recall 与遮挡反例 | Adapter/proposer 均为 offline/theory-only；不接入设备 |
| 动态运动/TTC | pure Kotlin TTC estimator、motion assembly gate、`UstrfEgoCompensatedMotionPromoter`：仅以相邻 frame 的已验证 pose delta 将同一目标轨迹转换为相对速度/TTC evidence；`UstrfSyntheticDynamicTtcReplay` 批量消费带真值的对向/横穿/故障轨迹 manifest；GPU `audit_vkitti2_dynamic_tracks.py` 对公开连续 track 真值作 source-native ego compensation 审计；GPU `audit_argoverse_av1_timestamped_ttc.py` 对真实 timestamped AV/actor 轨迹计算 source-native 秒级 TTC；GPU `audit_revel_dynamic_rgb_labels.py`、`audit_revel_dynamic_vicon_trajectories.py` 与 `audit_revel_rgb_vicon_reprojection.py` 审计公开动态场景的 2D 人体标注、Vicon helmet/sensor-suite 米制轨迹及其官方标定重投影；CPU `align_revel_detector_failures_with_vicon.py` v2 以原生 Vicon 严格包围 image timestamp，生成 source radial range-rate、approach/recede 和离线非因果 TTC-proxy 分层 | JVM 覆盖 user-motion compensation、未验证/错绑 pose 拒绝、栅格边界拒绝与 TTC；RTX 5060 审计并由 Kotlin 回放同一 9 条动态轨迹：7/9 准入、2/9 拒绝、6/6 TTC 最大误差 1ms、4/4 碰撞标签一致；VKITTI2 Scene01/clone 6,767 对连续轨迹中，363 个 source-moving 对获得 precision 1.000 / recall 0.9972，但无 timestamp/body receipt；Argoverse AV1 sample 1,282 个前向对具 0.10017s 中位时间间隔，得到 5 条三秒内 source-native TTC 候选；REveL Dynamic 的 8,580 RGB/label 帧全配对、13,018 个 2D helmet-colour boxes 全部几何有效，以约 23.073Hz 提供两类连续目标片段；其 371.805s ROS bag 提供 22,644/22,465 个 continuity-filtered helmet—sensor 米制相对运动对，20ms 同步率 94.61%/97.31%，并用官方 calibration 将 Vicon helmet 投回同类 RGB box（命中 89.61%/97.04%）；detector 770 框中 488 个 motion 可用，approaching/quasi-static/receding recall 为 0.93137/0.90291/0.90608，TTC-proxy<3s 仅 10 个且 10/10，逐框 receipt 精确复跑一致 | 无真机 optical flow/tracker Adapter、同源真实连续 RGB-D/VIO 的动态 assistive-event 真值及按 physical TTC 分段召回；REveL 是 helmet/sensor-suite 的 arbitrary Vicon world，TTC-proxy 不含人体包络、closest approach 或事件标签 | 仅 `source-motion-stratification-only`；不能声明动态风险感知可用 |
| 时空风险场 | pure Kotlin risk field、uncertainty、TTL、field reset、已验证位姿增量的离线 warp；warp 已接入 `UstrfSafetySession`，非法 receipt 会 reset field 并 STOP；`UstrfDocumentFiveMeterProfile` 提供 0–5 m/0.5 m 的共享 geometry-motion-risk BEV 配置与五候选人体包络；`UstrfTemporalGeometryConsistency` 新增为相邻 metric packet 的非授权审计层；`UstrfSyntheticTemporalGeometryReplay` 可批量读取 dependency-free CSV/TSV 基准并调用 drop proposer | deterministic replay、受控故障回归、8 个 labelled offline scenarios、risk-field 与 session 的位姿平移/转向 warp frame-bound fail-closed 回归；五米 profile 对 5m hazard STOP、geometry/motion cell 对齐的回归；temporal geometry 覆盖已验证 ego motion 的静态匹配、错 frame/未验证 pose 拒绝和 `DROP` 不可由缺口确认；14 双帧解析 benchmark 的 CUDA 审计为 8 个静态目标重投影 RMSE=0、4/4 `DROP` 检出，Kotlin batch replay 14/14 admitted、8/8 static match、2/2 gap 无 false `DROP`、2/2 actual drop detected；TartanAir P000/P002 各 19 对在原生相机坐标的 GPU 重投影覆盖约 97%、中位残差 5.56/3.27mm，但仍被锁在 geometry input 之外；SANPO-Synthetic 25 帧 raw metric-depth 结构审计通过 | 真实 metric geometry/dynamic adapter；SANPO pose 和设备 pose 均尚未获准进入 warp；无相邻帧 ground-plane/event truth | shadow-only |
| 安全走廊与监督器 | discrete capsule corridor、hard constraint、fail-closed supervisor；离线五候选包络扫掠；`UstrfSyntheticCorridorSafetyReplay` 从 CUDA truth TSV 重放同一 SafetySession | JVM 场景覆盖 stale/lost/unknown/occupancy/drop/head/dynamic risk；256 个 body-local 解析场景中 action 256/256、应 STOP 59/59、clear 32 个零误 STOP、非故障走廊选择 240/240、16 个 injected fault 均 STOP；GPU 生成器与 Kotlin replay 曾捕获并修复胶囊右边界 slice 不一致 | 真机场景事件、视觉感知误差、时延、用户实际包络与可用性 | 仅 offline-theory-only 的 STOP/SLOW trace |
| 事件与关键帧 | event-driven slow-loop receipt 合同、reference-free event gate | JVM frame/TTL/event mismatch 回归；SM-S9280 静止 CameraX 30 帧中 3 个 frame-bound event、27 个周期抑制 | 非米制语义源、真实用户查询/目标变化和自动多模型事件证据 | benchmark-only，不驱动生产链 |
| 语义模块 | semantic receipt/hint 合同、privacy-minimized trace | JVM admission/digest 回归 | 未引入 OCR/VLM Android Adapter；无语义真值 | 仅 typed fixture |
| 持久场景图 | scene candidate/persistent fact 合同 | `EPHEMERAL_PER_FRAME` 延迟写入回归 | inter-frame stable spatial receipt、场景一致性真值 | 禁止空间持久化 |
| 任务与目标 | task/goal proposal 合同 | event binding、TTL/low confidence 回归 | 用户目标输入、route provider、自动协议验收 | 不生成 corridor/action |
| 反馈与日志 | `UstrfStructuredSafetyOutput`：CONTINUE/ADJUST_LEFT/ADJUST_RIGHT/SLOW_DOWN/STOP/SCAN，包含 heading、speed scale、risk、confidence、走廊宽度、TTL、reason；由 `UstrfSafetyDecision` 单向派生并写入 session trace digest v2 | JVM 映射、降级不放行、session/digest binding 回归 | 用户反馈、音频/震动策略及隐私评估；设备左右坐标约定与走廊实体宽度的独立核验 | 仅 shadow/audit artifact，不驱动用户指令 |

## 2026-07-20 route-conditioned 主线补充

- 主线 seam：`UstrfRouteConditionedRiskInteractor` 只消费同 frame/coordinate system 的显式 route receipt 与 object-agnostic `UstrfRiskField`，输出 route intrusion evidence；不读取类别、box 或 detector AP，不输出用户动作。任何 invalid/stale/future/risk-model-inferred route 都拒绝且不回退中心走廊。
- 事件硬门：`configs/ustrf_sc_route_conditioned_event_collection_v1.json` 冻结 120 episode / 60 matched pair。正式 full-matrix 与 10-episode pilot 已拆成独立 scope/contract/schema；逐帧 ledger 会重算 capture ns、video PTS、clock summary、route 因果/投影绑定，双审必须是两份互不可见的 GPT/Codex review 文件并由独立哈希 adjudication 绑定，分歧自动进入第三模型。当前只有空模板，eligible model-reference 为 0；pilot 成功也只表示自主采集链完整。
- 几何硬门：`scripts/validate_ustrf_sc_device_metric_geometry.py` 要求五类同设备 typed artifact、设备/mount/calibration 绑定、分项 metrics 精确一致，并让每项收据继续 hash-bind 至少一个 raw/gate source artifact；`blocked/in_progress` 也审计已有收据。SM-S9280 已建立红灯 bundle，绑定 r3 `1/861`、r5 `0/843` source-aligned depth 与 `EPHEMERAL_PER_FRAME` pose，blocker 为 `BLOCKED_ON_SOURCE_ALIGNED_METRIC_DEPTH_AND_INTER_FRAME_STABLE_POSE`。`device_metric_geometry_admission=false`；未来单独通过也只允许 geometry shadow。
- detector 边界：crop/tiling r1 已冻结。后续 detector 变量只能是独立的 crop-view FP 抑制实验（如跨 view 一致性门），不得继续扫描 r1 的 NMS、overlap 或 score。

因此当前授权仍是：研究主线已切换且实验扩展已冻结，主动队列只保留真实路线事件 pilot 与同设备米制几何 evidence pack；真实事件、设备几何、正式 App 与生产反馈均未放行。

## 2026-07-21 E0 安全加固

- route-conditioned 事件 manifest 现强制绑定合同 ID、benchmark-only、生产权限、route 父来源与 episode source receipt；训练资格服从 config authority。当前合同即使未来收齐 120 episode，也只授权 teacher upper-bound evaluation，不自动授权学生训练。
- route-risk seam 现拒绝 future/stale risk field，并把 evidence TTL 截断到 500ms 风险场新鲜度窗；不再允许“路线还新、风险场已旧”的组合返回 Available。
- r816c 在原环境和全部冻结参数下完成 identity-bound 复跑：216 个 example ID 与 route rows 逐项一致，global/route/exact 预测、指标、fold 和系数 SHA 与旧 r816 精确一致。within-image wrong-route 正式 gate 为 `PASS_IDENTITY_BOUND_SYNTHETIC_ROUTE_SPECIFICITY`：正确路线 BA `.91555`，两种错路线 `.72492/.79515`，每个父来源均同方向下降。但绑定的 r818 稳定门仍失败（mean BA `.87737 < .90`，worst no-alert recall `.79710 < .80`），所以组合结论是 `BLOCKED_ON_R818_STABILITY`；真实事件、设备、学生训练与生产权限均不变。

## 2026-07-21 P0 shared decision parity

- 生产 `AssistSessionCoordinator` 与 Android device benchmark 已共用 `AssistDecisionKernel`，统一 temporal、stabilization、event、低置信侧向行人确认、反馈 receipt 和 trace 顺序；正式 App 的 lifecycle `commitIfCurrent` 边界不变。
- 四帧分割风险黄金矩阵与 unavailable-retry 回归锁定旧生产语义；benchmark v2 同时输出 raw/stable risk，旧 `model_risk` alias 保持 raw，事件/提醒聚合采用 production stable-risk 语义并明确与旧报告不可直接比较。
- device-event extractor 只接受 v2 + `blindassist_shared_decision_kernel_v1` + 已知反馈 adapter；模拟 planner 接受不冒充物理设备反馈送达。
- 当前授权：`P0_HOST_VERIFIED`。这只关闭生产/benchmark 决策实现漂移；本轮没有新增真机 benchmark、U0 teacher、120-episode 自动多模型事件证据、设备米制几何或生产模型授权。
- P0 真机旁证已补齐：SM-S9280/API 36、90 帧/3 序列、shared-kernel v2 报告 SHA256 `6b2d39b...b96b4a25`；candidate P95 `57.674ms`、event recall `1.0`、critical miss `0`、repeat delivery `0`、clearance `1.0`、false alerts/min `0`，49 次 duplicate attempt 被抑制。仍有 2 次 event ID regeneration；反馈 adapter 是 deterministic planner acceptance，不是物理投递，且该 historical benchmark 不提供冻结的自动多模型 U0 事件参考，因此授权只提升为 `P0_DEVICE_TRACE_VERIFIED`，不触发 App/模型晋级。

## 2026-07-21 U0 evaluator readiness

- 已新增 eval-only、dependency-free 的 U0 六臂 evaluator：四个正式臂加 uniform/shuffled route 负控，共用 frozen frame ledger 与 `blindassist_shared_decision_kernel_v1`，并绑定 truth/config/manifest/implementation/artifact/threshold/trace SHA。
- evaluator 内部重算 route-conditioned model-consensus event-reference validator，并钉死官方 config 与 validator bundle SHA；要求正式 120 episode / 60 pair 分母、LOSO、唯一身份、critical fold 分母和逐 session/scene/matched-pair 指标。pair 共享 route plan/provider policy/route choice，各 episode 投影分别绑定自身 video/frame ledger；绝对门、route-control 增益、unknown-low-obstacle 跨 session 增益和 causal 不退化门均已预注册。
- U0 prediction schema 已升级为 v2 unified-runner 合同：实际启动 preregistered Python subprocess wrapper，但正式 registry 必须声明并绑定 `android_kotlin_assist_decision_kernel_v1` 与 frozen `AssistDecisionKernel` 实现；synthetic protocol fixture 使用独立 backend，不能冒充 Android。每臂只获得不含 review/adjudication/event label 的 inference manifest；决策 cadence 冻结为与采集 ledger 一致的 500ms exact grid，kernel 原生事件状态与 feedback reason 有显式映射，YOLO/bbox 允许 kernel-native optional event ID，dense 臂仍强制原生 event identity。
- LOSO 不再是自报 held-out 字段：每个 arm 的每个 session 都必须提交 exact train-session/episode inventory、fold artifact 与 training receipt；fixed baseline 声明 `fixed_no_fit_v1`。uniform control 由 runner 生成 full-frame constant field；shuffled control 使用 held-out session 内按 episode ID 排序的 cyclic shift-one，禁止 seed、标签与 refit。正确 route、control route 与 truth-route binding 分离，不能继续给 control 假报原 route hash。
- 统一 dependency-free suite 现为 11 files / 54 tests。valid synthetic process proof 实际执行 6 arms / 12 subprocess / 252 frame traces；LOSO 泄漏、漏臂/重复臂、跨 JSON identity、非零退出、漏帧、标签字段注入、手改摘要、文件/registry/kernel/dependency 漂移、旧事件状态、feedback 映射、bbox-route receipt 因果/门控漂移，以及 dense teacher provenance/normalization 漂移全部拒绝。空正式 template 仍 exit 2/零报告，synthetic bundle 在正式 authority 下仍不能授权 U0/S0。
- shared `AssistDecisionKernel` 新增严格 object-agnostic risk-evidence input：只接收当前帧、有限期、无 bbox/检测距离/预置 trend/event/feedback 的归一化 `RiskResult`，并继续走同一 temporal、stabilizer、event、confirmation 与 feedback 顺序。NONE 语义矛盾、score-breakdown 漂移、越界分数和乱序时间全部 fail closed；共享内核 facade 加 7 个直接依赖文件均由 U0 bundle 独立哈希。`UstrfU0DenseRiskEvidenceAdapter` 冻结 route-intrusion/local-peak 到 NONE/LOW/MEDIUM/HIGH 的映射，并在 SM-S9280/API 36 通过 3 个 instrumentation tests。U0 admission 另冻结四个 dense/control 臂必须提交的 teacher 模型身份/来源、权重哈希、实现、LOSO fold、route、逐帧 field/evidence/unknown/归一化算术 receipt；许可证元数据有则记录、无则不阻止隔离研究。这只是第三臂前置 seam，不是 teacher 实现。
- dense teacher WIP 已升级为 v2 可复算合同：`uint32-le / 1e6` fixed-point field 将路线无关 source hash 与 route-interaction hash 分开；runner 只执行 materialized hash-bound adapter/threshold 副本并在执行前后复验，admission 从 receipt 内序列化 cell 重算 field inventory、hash、intrusion/peak/unknown，LOSO artifact 不再写入不确定运行耗时且逐训练帧绑定 RGB/depth SHA。该变化关闭实现/阈值 TOCTOU 与自报摘要信任洞，但没有产生四个真实 dense/control adapter、正式 teacher evidence 或任何晋级权限。
- `baseline_yolo_geometry` 已有真实 Android adapter：host 只负责去标签输入的 hash/ADB 搬运，Android instrumentation 重算 video/ledger/artifact/config，以 `MediaExtractor` 选择 20ms 内最近编码 sample，同时哈希选中的压缩 sample 字节与解码后 canonical RGBA8888，运行 shipped YOLO11n CPU-4-thread TFLite 与 shared `AssistDecisionKernel`，最终由设备生成 adapter output。receipt 绑定 app/test APK、build fingerprint、模型/标签、host/device 源码、逐帧 PTS/encoded-sample/RGBA/detector timing；admission 独立重验。方法级 instrumentation selector 回归已防止同类其他 adapter 测试被误执行。最终内核哈希 `d28ea341...d7ac04d` 的 SM-S9280/API 36 公开视频 3 帧 r4 smoke 连续两次稳定字段完全相同；首次/repeat output SHA256 为 `ad061060...a57cfa` / `4a99c43f...be12d5`，receipt SHA256 `06a214ab...d5ac0`。该 smoke 无人类事件真值，不是安全精度评估。
- `detector_bbox_explicit_route` 也已有真实 Android adapter：对每个 500ms ledger frame 只选择 `timestamp <= frame <= valid_until` 的最新显式路线 sample，固定使用 bottom-centre anchor + 1/2/3 秒 waypoints、0.08 frame-width 半宽走廊和 bbox 底部 25% footprint。门控只保留或排除原始 detection，不改 bbox/置信度/kernel 阈值；future/stale/低置信/invalid route 送空 detection 列表。设备逐帧回执记录 sample、waypoints、每个 bbox/footprint、最短距离与 keep，host 和 admission 独立重算因果与门控算术。最终内核 APK 的 r3 负控仍在同一 encoded sample/RGBA/APK 下使中心路线排除左侧 person（669.07px，raw `NONE`）、左侧路线保留它（75.75px，raw `MEDIUM`），左侧路线复跑的 backend/gate/decision 稳定字段一致；三份 output SHA256 为 `dca5a025...9b52f3`、`b206e2d2...f8cca9`、`9db2cff5...15bf9`，receipt SHA256 `c64a5eff...cd353`。证据位于 ignored `artifacts.local/evidence/ustrf-u0-bbox-route-device-smoke-20260721-r3/`，没有人类事件真值或 U0 权限。
- 当前状态提升为 `U0_TWO_ANDROID_ADAPTERS_AND_DENSE_KERNEL_SEAM_DEVICE_VERIFIED_BLOCKED_ON_HUMAN_TRUTH_AND_FOUR_REAL_ADAPTERS`，不是 U0 PASS；teacher field generator/LOSO artifact 与四个 dense/control 真 adapter 仍缺失并继续 fail closed。

## 2026-07-21 跨相机 Codex proxy R0

- 新增独立 `scripts/research/ustrf_crosscam_codex/`：公开来源/许可/SHA 收据、250ms full-context Codex provisional teacher、500ms causal Codex、三轮 2/3 共识、相机投影、assumed route 和 Android bbox-route candidate 均 hash-bound；正式 U0、自动多模型事件门、设备米制几何、训练和生产权限仍分别受控。
- MuSoHu 360° 样本因全景直用风险分歧及 forward-axis/遮挡不确定而 fail closed。Pexels 3874684 前 6 秒负样本获 6/6 Codex `none`；真实 SM-S9280 Android bbox-route 臂产生 1 次边界假阳性（车辆 route distance `48.4636px` < corridor half-width `51.2px`），代理折算 false alerts/min `10.0`。该短窗只定位路线投影/走廊问题，不能评价召回或长期告警率。
- 下一有效动作是 source-held-out 的小规模正负跨相机扩样与预注册几何敏感性，不在该单样本上调阈值，不微调模型。完整记录见 [R0 报告](USTRF_CROSSCAM_CODEX_PROXY_R0_2026-07-21.md)。

## 2026-07-21 路线投影与走廊几何 R1

- 新增显式 `route-projection-receipt`：绑定投影模式、forward axis、路线来源/权限/置信度来源、当前相机帧凸 polygon，以及 world route / camera pose / dynamic projection 是否真实存在。MuSoHu 缺可信 forward axis 继续 fail closed；Pexels 只承认 manual current-frame proxy。
- 新 gate 使用 bbox bottom-center 地面接触代理，不再用 bottom-25% footprint 与固定中心折线相交；按画面宽度 1%/2%/3% 三档投影误差输出 inside/outside/uncertain，三档不一致即 abstain。Pexels 同一 8 detections 从旧 gate `kept=1` 变为 robust `inside=0 / uncertain=1 / outside=7`，边缘车辆不再获得确定路线内断言。
- Kotlin 等价实现仅在 `device-benchmark`；SM-S9280/API 36 instrumentation 3/3 通过。正式 App、旧 U0 v1 gate、模型和生产权限未改。六来源 held-out 清单与停止门已冻结，正样本召回仍待验证。完整记录见 [R1 报告](USTRF_CROSSCAM_ROUTE_PROJECTION_CORRIDOR_R1_2026-07-21.md)。

## 已通过的验证层级

1. `:core:ustrf:test`：纯 Kotlin 的安全、校准、慢环和 digest 合同。
2. `:ustrf-shadow-benchmark:compileDebugAndroidTestKotlin`：隔离 Android benchmark 编译。
3. SM-S9280/Android 16：CameraX 时间 receipt、reference-free event gate、ARCore capability/raw pose/raw depth freshness/image intrinsics metadata 的受控观测。
4. `scripts/check_docs_index.ps1`：文档顶级索引完整性。
5. `scripts/report_ustrf_sc_research_benchmark.py`：汇总解析 geometry、解析 dynamic TTC 与公开 source-native temporal/RGB-D/轨迹 receipt，生成 JSON/HTML；V13 为 15 gate / 14 pass 的 `CONDITIONAL_RESEARCH_GO`，唯一 device metric-geometry admission 继续 BLOCKED。
6. REveL YOLO11n bounded detector：512 个 uniform frame / 770 boxes 上 AP50 0.92747、precision/recall/F1 0.83313/0.88831/0.85984；small/medium/large recall 为 0.24324/0.87571/0.96306。r2 对同一 index 精确复跑并保存 512 行逐帧 receipt，守护最高 46°C、22.38W、0 相关系统事件。128 帧 320px 配对没有 small-recall 增益且 F1 下降，故不扩载；Vicon source-range 对齐显示 0–5m recall 0.9375、5m 以上 0.7222；CPU source-radial motion 对齐 488/770，三类 recall 为 0.93137/0.90291/0.90608，TTC-proxy<3s 仅 10 个。该项只提供 bounded-public-rgb/source-range/source-motion baseline，不提供用户距离、physical TTC、body/event 或设备准入。

这些层级不能互相替代：JVM 通过不证明设备数据有效；设备候选出现不证明标定或事件性能；独立标定通过后仍需几何/动态事件 shadow；所有这些完成前都不能接入默认 App。

## 当前路线与下一次可执行证据

当前手机路线是 reference-free shadow，不再以物理棋盘格或 calibration manifest 作为前置。CameraX event gate 已完成静止设备的帧绑定和节流证据；离线安全内核已增加 8 个带标签合成场景与五通道人体包络回归，见 [离线安全仿真](USTRF_SC_OFFLINE_SAFETY_SIMULATION.md)。风险场现在可在 `UstrfVerifiedPoseDelta` 精确绑定前后 frame、并由独立 offline/geometry gate 标记后执行静态证据的平移与转向重投影；任何缺失、未验证或错帧 receipt 都会拒绝 warp。session record 现同步生成影子结构化输出：只有 supervisor 仅保留 `SHADOW_ONLY`、且已选走廊严格匹配时，才会把安全候选标为 CONTINUE 或左右微调；任何其他原因继续映射为 SLOW_DOWN、STOP 或 SCAN。它不是用户可执行指令。新的 metric-depth Adapter 已覆盖“受准入深度 + ground plane → geometry packet”这段离线理论链；`UstrfTemporalGeometryConsistency` 随后以精确 frame-bound pose delta 对相邻 geometry packet 做匹配 receipt。二者均没有设备证据时仍不被调用，且明确不把缺失深度转换为 `DROP`。几何/SANPO 模型层的下一变量仍应是相邻帧可见性/一致性与 metric-depth 平面边缘；detector 层的 source radial-motion CPU 分层已经完成，不能混称为设备 TTC 或事件验证。

未来眼镜路线需按 [设备阶段策略](USTRF_SC_DEVICE_PHASE_POLICY.md) 新建 frame、时钟、标定和设备证据，不能迁移手机观测或 manifest。

若以后选择开启任一设备的“受控几何/事件 shadow”，需要一个经独立自动验证的 calibration manifest，按 [校准协议](USTRF_SC_CALIBRATION_PROTOCOL.md) 填充，至少包含：固定 `bodyFrame`、外部 artifact SHA-256、相互隔离的采集 Agent 与验证 Agent、样本/姿态覆盖、内参 reprojection、depth registration 与完整 mount repeatability。没有该包，`UstrfIndependentCalibrationEvidenceVerifier` 必须返回 unavailable。

在此之后，仍需要单独的几何和动态事件集，以及同机时延、热、连续自动多模型事件评测；任何单项绿灯都不允许替代完整 promotion gate。
