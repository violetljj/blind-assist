# USTRF-SC 安全内核实验方案（2026-07-20）

## 状态与授权

- 方案编号：`ustrf_sc_safety_kernel_r1`
- 状态：`experiment_in_progress / production_isolated`
- 文档类型：日期化研究方案与实现快照；不是当前生产或 SANPO 协议真源。
- 默认 App：保持 `yolo11n_fp16_320.tflite` 和现有提醒链。
- 授权：不训练、不导出 TFLite、不读 blind、不接入 Android shadow、不替换默认模型。

本方案保存“不确定性感知时空风险场与目标条件安全走廊（USTRF-SC）”的独立实现路线。它复用项目已有的 latest-only、事件闭环和真机测量经验，但不把现有检测框风险规则误称为 USTRF-SC，也不改变 SANPO、Corridor-Causal 或 RC-OARF 的门禁结论。

## 1. 要解决的问题

现有 App 主要从检测框、类别、图像位置和短时目标趋势产生提醒；它不能证明以下安全命题：

1. 多个来源的几何、下坠、头部、动态接近和未知区域如何在同一局部表示中随时间保留、衰减和冲突；
2. 走廊是人体包络而非单一画面中心线时，哪一个短时动作仍然满足硬安全约束；
3. 位姿、感知、时间或质量失效时，系统能否明确拒绝继续或方向指令。

USTRF-SC 的目标不是替代任何单模型，而是给快环增加一个可独立审计的决策内核：

```text
带单调时间/TTL的观测 + 位姿/质量状态
  -> 风险格（占用、可通行、下坠、头部、动态TTC、不确定性、年龄）
  -> 离散人体包络候选
  -> 硬约束优先的风险评分
  -> 安全监督器
  -> 实验性结构化命令 / replay 证据
```

## 2. 与既有路线的关系

| 路线 | 主要问题 | 本实验的关系 |
| --- | --- | --- |
| 正式 YOLO App | 已知类别目标的提醒 | 保持生产基线；不调用 USTRF 代码。 |
| SANPO 分割 | 可通行、边界与连续场景候选 | 未来可以成为 `RiskObservation` 的一个 Adapter；当前不接入。 |
| Corridor-Causal Student | 走廊相对关系与事件生命周期 | 未来可提供 motion/semantic 证据；当前独立，且不使用其未闭合真值。 |
| RC-OARF | 显式路线条件化的密集风险学习 | 未来可能成为感知 Adapter；本轮不设计其模型接口。 |

USTRF-SC 是安全内核实验，不是第四个待训练视觉模型。感知 Adapter 必须在进入内核前给出来源、坐标、单调时间、TTL、质量和不确定性；没有这些合同的输出一律不能成为安全行动依据。

## 3. 不可变隔离规则

- 不修改 `RiskAnalyzer`、`RiskEventTracker`、`FeedbackPlanner`、`AssistSessionCoordinator` 或 `feature:assist` 默认链。
- 不写入 `app/src/main/assets`，不新增模型、云端调用、硬件连接或权限。
- 不读取训练集、SANPO canonical、公开银标、blind 数据、用户视频或现有未提交实验产物作为本轮输入。
- 首轮只使用合成、纯 Kotlin 的 `RiskObservation` fixture，验证内核的时序与 fail-closed 语义。
- 产物将来只写入 `artifacts.local/experiments/ustrf-sc/<run-id>/`；源代码只提供实验内核与测试。
- 任意未来的 Android 接入都必须是默认关闭的 shadow Adapter，并重新通过数据、离线、INT8、同机事件和独立 GPT/Codex 自动发布准入。

## 4. P0 实现合同

### 4.1 时间、身份与新鲜度

所有 tick 使用单调 `timestampNs` 和不可复用 `frameId`。每个观测包括 `sourceFrameId`、`sourceTimestampNs`、`producedTimestampNs`、`validUntilNs` 和来源。内核拒绝：

- 时间倒退或 frame ID 倒退；
- source frame 与当前 tick 不匹配；
- `validUntilNs < producedTimestampNs`；
- 已过 TTL 的观测；
- 不能证明来自当前局部坐标系的观测。

### 4.2 健康状态与 fail-closed

最小健康向量为 `capture / pose / geometry / motion / thermal`。首轮只实现状态与监督逻辑，不假装拥有 VIO、深度或温度 Adapter。

| 条件 | 命令上限 | 原因 |
| --- | --- | --- |
| pose 非 `TRACKING` | `STOP` | 局部风险格不得跨无效位姿继续使用。 |
| capture/geometry/motion 缺失或过期 | `STOP` | 不能用空观测推断安全。 |
| 中央人体走廊未知 | `STOP` | unknown 不是 traversable。 |
| 质量 degraded 且存在风险 | `SLOW_DOWN` 或 `STOP` | 不允许输出方向建议。 |
| 状态完整且候选满足硬阈值 | 实验候选 | 不代表可发布方向指令。 |

### 4.3 风险格

首轮使用用户局部二维整数格和高度带，而非稠密 3D 占用网络。每个 cell 维护：

```text
occupancy, traversability, dropRisk, headRisk,
dynamicTtcMs, uncertainty, ageMs, sources
```

更新是确定性的：同一输入序列得到逐 cell、逐命令一致输出。静态风险按寿命衰减；动态 TTC/未知风险以更短寿命衰减。没有 VIO Adapter 时不实现或声称实现真实 world warp；该能力保留为后续 `PoseAdapter` seam 的门。

### 4.4 候选走廊与评分

首轮只生成离散候选：停止、直行、左右小偏移。候选以包含反应裕度的二维人体 capsule 采样格；不得只检查中心线。评分包含碰撞、下坠、头部、TTC、不确定性和动作切换成本，但硬约束优先于加权总分。任一 candidate 若越过碰撞、下坠、头部、TTC 或 unknown 阈值即不可选。

在无真实 route provider、VIO 和足够事件真值前，候选的 `ADJUST_LEFT/RIGHT` 只写入 replay 输出，安全监督器不会授权它成为 App 指令。

## 5. 模块与 seam

首轮代码位于独立的 `:core:ustrf` JVM module（包名 `com.linnan.blindassist.ustrf`），保持 pure Kotlin、无 Android 依赖，也不反向依赖现有风险/反馈域：

| Module | Interface 的职责 | 不承担的职责 |
| --- | --- | --- |
| `UstrfContracts` | 单调时间、健康、观测、命令与拒绝原因的不变量 | 相机/VIO/模型 Adapter |
| `UstrfRiskField` | 时序融合、年龄、来源和 cell 查询 | 像素投影或世界坐标变换 |
| `UstrfCorridorPlanner` | capsule 候选、硬约束、确定性最低风险选择 | 用户路线或真实转向控制 |
| `UstrfSafetySupervisor` | 统一 fail-closed 覆盖与命令有效期 | TTS、震动、UI |
| `UstrfReplayRunner` | 排序、故障注入、确定性事件记录 | Android CameraX 录制 |

这些 module 的 leverage 是让未来至少两个 Adapter（例如合成 replay 与 Android shadow）在相同 Interface 下使用同一安全实现；如果第二个 Adapter 未出现，seam 仍保持小且不提前泛化。

## 6. 最小验证矩阵

| 用例 | 预期 |
| --- | --- |
| 观测 TTL 过期 | `STOP / OBSERVATION_STALE` |
| pose lost | `STOP / POSE_NOT_TRACKING` |
| 中央未知区 | `STOP / CENTRAL_CORRIDOR_UNKNOWN` |
| 直行碰撞、左侧安全 | planner 选择左侧实验候选；supervisor 不授权给生产链 |
| 下坠/头部/TTC 硬风险 | 候选被拒绝，产生 `STOP` 或 `SLOW_DOWN` |
| 静态 cell 老化 | 只按预注册衰减；不跨 timestamp 倒退 |
| 相同 replay 两次 | 输出逐 tick 完全一致 |
| timestamp/frame ID 回退 | replay 明确拒绝，不使用旧状态 |

完成上述单元验证仅证明纯逻辑合同，不证明真实感知、VIO、米制距离、TTC、用户可用性或助盲安全。

## 7. 后续门与停止条件

进入 Android shadow 前，必须先完成受控 replay 的上述矩阵、真实 `PoseAdapter`/深度 Adapter Spike、数据来源与自动多模型连续事件参考门。任何一项不足时，输出只能是研究日志，不能是 App 行为。

立即停止或保持 `proposal_only` 的条件：真实 Adapter 无法提供统一时钟/TTL；未知区被默认成可通行；方向建议需绕过监督器；通过大量 abstain 掩盖关键漏报；或同机 P95/新鲜度不满足当前项目门。

## 8. 本轮验收

本轮验收标准仅为：独立设计已保存；核心纯逻辑可通过单元测试；测试证明 stale、pose lost、未知和硬风险均 fail closed；默认 App/SANPO 路线没有代码或资产变化。任何更高层结论必须等待单独的 evidence 和 gate。

### R1 实现记录

- 已新增 production-isolated 的 `:core:ustrf` JVM module；它不被 `app`、`feature:assist` 或既有风险/反馈 module 引用。
- 已实现：`UstrfContracts`、`UstrfRiskFieldBuilder`、`UstrfCorridorPlanner`、`UstrfSafetySupervisor` 与 `UstrfReplayRunner`。
- 当前 replay 仅消费合成 typed fixture；其严格拒绝 frame ID/time/coordinate frame 回退，且 perception 必须绑定源 frame。它不是现有 `ReplayFrameSource` 的替换。
- 已验证：`gradlew.bat :core:ustrf:test --offline --no-daemon --console=plain` 于 2026-07-20 通过（27 tests，0 failures/errors）。
- 已新增 `UstrfUncertaintyFusion`：固定 model / geometry / age / OOD 的显式组合公式；它不等于已完成 ECE、Brier、coverage 或阈值校准。
- 已新增 `UstrfTraceDigest`：对不含原始图像的结构化 safety trace 生成版本化 canonical text 与 SHA-256。回归证明未来 tick 的输入变化不能改写既有 trace prefix。
- 已新增 `UstrfPoseBuffer` seam：以图像 capture time 插值 mock pose；拒绝未包围、过期、低置信、失跟和相机坐标错配的 pose receipt。它不是 ARCore/VIO Adapter，也不证明真实重投影精度。
- 已新增 `UstrfGeometryProjector` 与 `UstrfTtcEstimator`：只允许 metric geometry 进入 BEV risk field，并按当前时刻外推类别无关的最接近时间；relative/unknown depth、静止运动和过期 receipt 不产生安全几何结论。
- 已新增 `UstrfPerceptionAssembler`：只有同一 `UstrfFrameStamp` 的 metric geometry 与动态 motion receipt 才能原子地装配为 `UstrfPerceptionPacket`；任一错帧或无效动态回执会整体拒绝，输出 TTL 取所有 receipt 的最短值。空动态列表只表示 Adapter 没有报告动态证据，绝不等于环境已安全。
- 安全修正：`capture`、`geometry` 或 `motion` 非 `VALID` 现在与 pose 丢失、错帧、过期感知同级，监督器必须 `STOP_AND_REASSESS`；不再仅记录 reason 后退化为 warning。
- 已新增 `UstrfSafetySession`：它是 Adapter/replay 的唯一纯 Kotlin 组合点，只有可用且在 `decisionAtNs` 时仍 fresh 的装配包才会更新风险格和规划。任一装配失败、手写错帧包或“采集时仍有效但排队到决策时已过期”的包，均不更新 field、记录精确 failure，并以 `PERCEPTION_ASSEMBLY_UNAVAILABLE` 进入 shadow-only `STOP_AND_REASSESS`。
- 已新增 `UstrfSessionTraceDigest`：session replay 的版本化 canonical digest 同时包含 safety decision 和精确的 assembly failure code；它不含图像、位姿或用户数据。
- 已新增 `UstrfCaptureReceiptValidator`：为未来 CameraX Adapter 固化 `ImageProxy` 硬件时间戳、映射后的单调 capture time、接收时间、坐标与标定版本的 receipt；它拒绝帧/硬件时间回退、超帧龄、时钟域或标定静默变化。它不声称已经完成 CameraX/IMU 跨时钟同步。
- 已新增 `UstrfRouteReceiptResolver`：慢环 route/goal 必须绑定快环 `queryFrame`、决策时刻、坐标系和 TTL，且只能提出受限的实验 offset；错帧、过期、未来签发、低置信或过大偏移全部拒绝。它只生成候选 `UstrfRouteIntent`，安全动作仍只由监督器签发。
- benchmark 调研：仓库已有 `CameraXAnalysisStreamGeometryAuditTest` 的 `image.imageInfo.timestamp` 单调审计、`CameraRotationTimestampPairingAuditTest` 的 rotation-vector 前后夹逼，以及 `LatestOnlySidecar`/`RgbaLumaSidecar` 的真机 shadow 路径；它们仍为 benchmark-only。经用户授权，`device-benchmark` 已加入 `implementation(project(":core:ustrf"))`，从而可新增独立的 `benchmark/ustrf` 时间戳 shadow test。现阶段不得把 analyzer 入场 `elapsedRealtimeNanos` 误作源 capture timestamp，也不得接入默认 App。
- 已释放 benchmark 依赖并新增 `UstrfCameraTimestampShadowDeviceTest`：它将 `ImageProxy.imageInfo.timestamp` 原样作为 `UstrfFrameStamp` capture time，验证 receipt、rotation-vector bracket 和 copied latest-only sidecar 的聚合帧龄；不保存帧、不调用 YOLO/反馈、不执行 pose/VIO/走廊规划。`gradlew.bat :device-benchmark:compileDebugKotlin --offline --no-daemon --console=plain` 已通过。2026-07-20 尝试在已连接 `R5CX10M8Y8X` 部署时，因设备已有 `com.linnan.blindassist` 与本地 debug APK 签名不一致而收到 `INSTALL_FAILED_UPDATE_INCOMPATIBLE`。获授权后执行精确 `adb uninstall com.linnan.blindassist`，设备返回 `DELETE_FAILED_INTERNAL_ERROR`；设备含受限 Secure Folder 用户（user 150），ADB 无权枚举其包，未删除或覆盖任何 App。因此尚无真机执行证据，需在 Secure Folder 中手动移除冲突包或改用隔离设备。
- 已新增独立 Android application module `:ustrf-shadow-benchmark`（包名 `com.linnan.blindassist.ustrfbenchmark`），以避免 Secure Folder 中同包名签名冲突；它不依赖或改动默认 App。真机 `SM-S9280 / Android 16` 已通过唯一 CameraX timestamp shadow test：30/30 receipt 有效、rotation bracket 覆盖 1.0 且最大 19.754271 ms、37 个 latest-only shadow 新鲜结果最大帧龄 125.665622 ms、worker failure 为 0。rotation-vector 的 orientation-only Adapter 也对 30/30 个 bracketed receipt 强制产生 `NOT_TRACKING`；它绝不被授权为 VIO。聚合证据写入 `artifacts.local/evidence/ustrf-sc/2026-07-20-isolated-camera-timestamp-shadow-sm-s9280-r1/summary.json`。这只证明本机时间 receipt/rotation bracket/shadow 新鲜度与 orientation-only fail-closed；不证明 pose/VIO、相机外参、米制几何、TTC、用户可用性或生产授权。
- 已新增独立 ARCore capability audit：`ustrf-shadow-benchmark` 以 `com.google.ar:core:1.33.0` 的 optional manifest 和最小 `GLSurfaceView` texture host 创建/关闭 session；所有 session configure/resume/update 均在 GL 线程，避免把普通 instrumentation 线程误当作 ARCore 上下文。`SM-S9280 / Android 16` 真机通过审计：`SUPPORTED_INSTALLED`、`DepthMode.AUTOMATIC` supported、60 次 update 中 42 次 timestamp 严格前进。该次设备静置短测为 `TRACKING=0`、raw depth=0；因此 `vio_candidate_observed=false`、`vio_gate_open=false`，不生成 `UstrfPoseState.TRACKING` receipt，也不允许 metric geometry 进入安全风险格。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-arcore-capability-audit-sm-s9280-r1/summary.json`。要打开该门，仍需受控移动场景中的连续 TRACKING、相机到身体外参与时间绑定、原始深度/地面尺度校准、失跟/遮挡/热降级回放和独立事件门。
- 已新增 `UstrfVioPoseReceiptPromoter`：ARCore/VIO 的世界位姿先以 `UstrfVioPoseCandidate` 形式存在，不能直接伪装为安全 `UstrfPoseSample`。promoter 要求候选与当前 image frame 完全一致、处于 `TRACKING`、decision-time 仍 fresh、候选置信度达标、roll/pitch 不超平面假设，并要求 camera-frame 一致且独立验证、未过期、高置信的 `UstrfCameraBodyExtrinsicsReceipt`。成功时只输出 camera-frame pose；外参仅作为被验证的元数据，当前明确不执行未经完整 3D 合同验证的 camera-to-body 变换或风险格 world warp。33 个纯 Kotlin 回归覆盖了有效 admission、未验证/过期/错相机外参、错图像帧和超 tilt 的 fail-closed 分支。
- ARCore world-frame 修正：官方 `Pose` 文档说明其环境理解变化可调整 world model，frame 取得的数值 world coordinate 不能跨 rendering frame 持久使用。因此 `UstrfVioPoseCandidate` 新增 `worldFrameStability`；`EPHEMERAL_PER_FRAME`（ARCore raw frame pose 的当前默认状态）即使 `TRACKING`、fresh、外参齐全也被 `UstrfVioPoseReceiptPromoter` 以 `WORLD_FRAME_NOT_INTERFRAME_STABLE` 拒绝。只有另行证明的 `INTER_FRAME_STABLE` receipt 才可进入现有 `UstrfPoseBuffer`。参考：[ARCore Pose](https://developers.google.com/ar/reference/java/com/google/ar/core/Pose)。
- 已新增 Android `UstrfArCoreRawPoseReceiptAdapter`：它只在 `Frame.timestamp == UstrfFrameStamp.capturedAtNs` 时记录 ARCore camera translation、quaternion 和 `TrackingState`，并固定标记 `EPHEMERAL_PER_FRAME`。真机短审计（SM-S9280 / Android 16）记录 60 个 raw receipt，状态均为 `PAUSED`，41 次 frame timestamp 严格前进，`TRACKING=0`、raw depth=0。该证据证明 bridge 的时间绑定和原始状态记录，不证明 tracking/VIO/几何；聚合 receipt 在 `artifacts.local/evidence/ustrf-sc/2026-07-20-arcore-raw-pose-receipt-sm-s9280-r1/summary.json`。
- 受控移动真机证据：独立 benchmark 的第二次 900-update window 在 SM-S9280 / Android 16 取得 `TRACKING=813`、`PAUSED=87`、最长连续 TRACKING state observation=813、state transition=1、raw depth candidate=813、strictly advancing timestamp=874。该段 state 观察说明设备在移动环境中可进入并维持 tracking，但 900 次 update 并非 900 张不同图像，连续 observation 不能被扩大解释为 813 个独立图像；同时所有 raw pose 仍为 `EPHEMERAL_PER_FRAME`。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-arcore-moving-capability-sm-s9280-r2/summary.json`。这打开“候选采集”门，不打开跨帧 stable world、外参、metric geometry、事件或生产门。
- ARCore 审计测试支持 `ustrfArcoreFrameAttempts` instrumentation 参数（默认 60、上限 1800）；独立 Activity 会保持屏幕常亮并显示采集提示。该参数只延长同一 benchmark-only 观测窗口，不能改变任何 gate，也不能把 `TRACKING` 自动变成生产可用 pose。
- 回放组合点安全修正：`UstrfSafetySession` 现在将 `health.pose / capture / geometry / motion` 作为感知包入场的原子前置条件。以前即使 supervisor 会 STOP，一个有效感知包仍可能在上述任一 health receipt 缺失时更新风险格；现已改为产生 `POSE_UNAVAILABLE`、`CAPTURE_UNAVAILABLE`、`GEOMETRY_UNAVAILABLE` 或 `MOTION_EVIDENCE_UNAVAILABLE` 失败码，field 保持 `null`，并经回归确认不会写入部分证据。该规则是后续多传感器 replay 的最小安全不变量。
- 已新增 `UstrfControlledFaultReplay`：同一 `UstrfSafetySession` 组合点可对按帧单调的输入显式注入 `POSE_LOST`、capture/geometry/motion unavailable、perception source-frame mismatch 与 perception expired。replay 不提供“自动恢复”捷径，下一帧必须由调用方显式提供新的 receipt。回归证明无故障基线可更新 field；六类故障均为 `field=null + STOP_AND_REASSESS + PERCEPTION_ASSEMBLY_UNAVAILABLE`，且两次相同 replay 的 session trace canonical text 完全一致。它目前消费 typed fixture，不是设备日志或人工事件真值。
- 恢复边界修正：任一被拒绝的 atomic fast-loop frame 现在都会 reset 内部 `UstrfRiskFieldBuilder`。因此 pose/capture/geometry/motion 或 frame/timing 失败之后，即便下一帧恢复健康，也只能从新 receipt 重建 field，不能衰减复用失效区间前的障碍/可通行证据。`失跟 → 健康恢复` 回归确认旧中心障碍不会重新出现。
- 结构化影子输出与 pose-warp 闭环：`UstrfSafetySession` 现将经过严格相邻 frame 绑定且显式 `verifiedForOfflineReplay` 的 `UstrfVerifiedPoseDelta` 传给 risk field；错误或未验证的 receipt 触发 `POSE_DELTA_INVALID`、reset field 与 shadow STOP。每个 session record 同时携带由 supervisor 单向映射的 `UstrfStructuredSafetyOutput`（CONTINUE/ADJUST_LEFT/ADJUST_RIGHT/SLOW_DOWN/STOP/SCAN、heading、speed、risk、confidence、走廊宽度、TTL、reason），trace digest 升级为 v2 并绑定该输出。它只用于 offline/shadow replay；任何非 nominal reason 都不可输出 CONTINUE/ADJUST，更不可成为 App 用户指令。
- 动态轨迹自运动补偿：新增 `UstrfEgoCompensatedMotionPromoter`。它仅接受严格相邻 frame、同一 track、未过期且达到置信阈值的局部坐标观测，并以已验证 `UstrfVerifiedPoseDelta` 把上一帧目标位置转换至当前 body frame 后才计算相对速度；结果可作为既有 TTC estimator/assembly 的 `UstrfMotionGridEvidence`。未验证或错绑 pose、低置信/过期轨迹、身后或局部栅格外目标一律不产生 TTC evidence。它仍不是 Android optical-flow/tracker Adapter，也没有动态人因事件真值。
- 文档空间范围 profile：新增 `UstrfDocumentFiveMeterProfile`，将 geometry projector、ego-motion promoter、risk field、corridor planner、supervisor 与结构化输出配置到同一 `0.5 m × 0–5 m` 的本地 BEV 网格，且 corridor 使用 `[-2,-1,0,1,2]` 五候选与一格半宽人体包络。回归覆盖完整五米安全通道的 shadow CONTINUE、5m 中心高风险的 STOP，以及 metric geometry 与 ego-motion 目标在同一 0.5m 栅格坐标上的对齐。该 profile 仍只接收 typed offline evidence，不能被扩大解释为真实设备几何或用户安全证明。
- 尚未实现：CameraX 时间戳 Adapter、VIO/pose 插值和 world warp、深度/地面/动态 Adapter、录制日志格式、Android shadow、真机延迟/热/功耗 Spike、连续自动多模型事件评测。
- 未修改 `DEVELOPMENT_LOG.md` 与 `docs/README.md`：两者在本轮开始前已有其他任务的未提交改动；为避免跨任务覆盖，本日期化研究快照承担本轮持久记录，待该文档现场可安全合并时再补开发日志和索引入口。

### R1.1 原始深度新鲜度审计

- 已新增 `UstrfArCoreRawDepthReceiptAdapter`：它只记录 raw-depth/匹配 confidence 的尺寸、stride、时间戳和 freshness metadata；若任一时间戳不等于当前 `UstrfFrameStamp.capturedAtNs`，统一标记为 `REPROJECTED_OR_STALE`。Adapter 不读取或保存像素，不生成 `UstrfGeometryPacket`，也不把 depth candidate 视作可通行/障碍结论。
- 受控移动真机审计（SM-S9280 / Android 16，900 update）通过：`TRACKING=861`、`PAUSED=39`、最长连续 tracking observation=861、raw-depth candidate=861；但只有 **1** 个 depth+confidence pair 同时与当前 source frame 对齐，另 **860** 个明确为 reprojected/stale。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-arcore-moving-freshness-sm-s9280-r3/summary.json`。因此“有 raw depth”不能等价于“当前帧米制几何”，当前 metric-geometry gate 继续 closed。
- benchmark GL host 修正为 `RENDERMODE_CONTINUOUSLY`，使 `Session.update` 与 `Session.close` 的 GL queue action 在同一存活线程执行；此前按需渲染的收尾 timeout 产生的窗口没有被记录为证据。该修正仅在隔离 benchmark Activity 内，不触及默认 App。

### R1.2 米制几何投影授权合同

- 已新增 pure Kotlin `UstrfMetricGeometryReceiptPromoter`。它将 raw-depth receipt、capture receipt、相机内参、深度到相机坐标登记和完整 `SE(3)` 相机—身体外参作为原子输入；要求 depth 与 confidence 均严格绑定当前 source frame、单位显式为 millimeters、所有版本/坐标系相符且在决策时仍 fresh，并要求内参、登记和完整外参各自独立验证且达到置信阈值。
- 原先 planar pose seam 的 yaw-only `UstrfCameraBodyExtrinsicsReceipt` 不能用于此门；新 `UstrfCameraBodyFullExtrinsicsReceipt` 要求归一化 quaternion，专门避免把平面导航假设错误扩大为地面/高度的 3D 投影能力。
- promoter 成功仅返回 `UstrfMetricGeometryProjectionAdmission.Available`（可由未来 Adapter 消费的输入合同）；它明确不生成 `UstrfGeometryPacket`、不采样像素、不拟合地面、不声明障碍/下坠/可通行，也不更新风险格。回归覆盖 reprojected depth、confidence 错帧、未验证内参/登记/外参、版本与坐标帧不匹配及非归一化 quaternion 的 fail-closed 分支。
- `gradlew.bat :core:ustrf:test --offline --no-daemon --console=plain` 于 2026-07-20 通过。当前仍没有独立量测的设备内参验证、深度登记、完整 mount 外参或逐像素置信度/地面真值；故真机 metric-geometry gate 仍为 closed。

### R1.3 ARCore image-intrinsics 观察与 benchmark 生命周期修正

- 独立 benchmark 新增 `UstrfArCoreRawCameraIntrinsicsAdapter`，按同一个 `Frame.timestamp == UstrfFrameStamp.capturedAtNs` 合同读取 image intrinsics；只输出 image size、focal length 与 principal point 原始 observation，不提供 calibration version、confidence 或 `independentlyVerified=true` 的捷径。
- 为消除设备上“循环完成后第二次 GL queue action 超时”的生命周期竞态，ARCore `Session` 的 create/configure/resume、update loop、pause/close 现置于同一受限的 GL task；timeout 按观测窗口上界计算。真机短测通过，证明的是 benchmark 宿主可完成受控收尾，不是 ARCore 安全能力。
- SM-S9280 / Android 16 的 60-update 短窗口获得 60 个 raw image-intrinsics observation、distinct signature=1：`640x480, fx=415.65775, fy=415.78708, cx=321.46466, cy=239.0193`。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-arcore-intrinsics-observation-sm-s9280-r4/summary.json`。它仅表明该窗口 API 输出未漂移；不验证 depth-to-camera registration、full SE(3) mount、深度精度/置信校准、地面或安全几何。

### R1.4 独立校准证据门与现场协议

- 已新增 `UstrfIndependentCalibrationEvidenceVerifier` 和 `UstrfCalibrationTrialEvidence`。它要求 source artifact SHA-256、不同 collector/reviewer、显式 review approval、最小样本与姿态覆盖、内参 reprojection、depth registration、mount 平移/旋转 repeatability 以及有效期。它拒绝 self-review、未批准复核、未来/过期 evidence 与各项阈值越界；成功仍只表示实验 calibration manifest 可用，绝不生成安全几何或生产授权。
- 已保存现场协议：`docs/research/ustrf-sc/USTRF_SC_CALIBRATION_PROTOCOL.md`。该协议将实际采集、隐私边界、可复现 metric、独立复核、回填字段和停止条件与上述代码合同一一对应。当前没有任何实测 calibration manifest 被填入，故 `independentlyVerified`/metric-geometry gate 仍为 closed。

### R1.5 慢环事件、语义、场景与任务合同

- 已新增 pure Kotlin `UstrfSlowLoopReceiptResolver`，实现图中“事件与关键帧选择 → 语义模块 → 持久场景图 → 任务与目标管理”的实验性数据合同。输入由 event、语义 receipt、场景候选和 task/goal proposal 构成；事件与语义必须绑定同一 query frame、在 decision-time 不过期且达到阈值。task proposal 还必须绑定同一 event。
- 输出的语义只有既有 `UstrfSemanticHint`，可写入 fast-loop trace 以解释当前帧，但不包含 action、speed、heading 或 corridor 字段；因此类型上不能签发 `UstrfSafetyAction`。现有 supervisor 继续忽略 semantic hint 对快环 safety 决策的影响。
- 场景候选显式携带 `UstrfWorldFrameStability`。`EPHEMERAL_PER_FRAME` 时会保留当前帧语义但设置 `sceneMemoryDeferredForEphemeralWorldFrame`，绝不写入 persistent scene fact；即便 `INTER_FRAME_STABLE`，也必须同源帧、足够置信且其自身 TTL 未过期。回归覆盖错帧、未来/过期/低置信语义、错 event/过期 task、过期场景和 ephemeral world 的 fail-closed 行为。
- `gradlew.bat :core:ustrf:test --offline --no-daemon --console=plain` 于 2026-07-20 通过。慢环目前无 OCR/VLM/scene retrieval Android Adapter、无用户目标真值，也没有生产接入；这是可回放的独立接口实现而非已完成语义导航。

### R1.6 慢环隐私最小化 trace

- 已新增 `UstrfSlowLoopTraceDigest`。canonical trace 仅包含 source frame、触发类型、语义是否接受、场景是否持久化/因 ephemeral world 延迟，以及 task proposal 是否接受；它明确不含 semantic label、OCR 文本、用户 goal 文本、图像或 pose payload。
- 回归证明同一 admission 状态的不同语义/目标文本得到相同 digest，而 available/unavailable 状态变化会改变 digest。因此 trace 可用于核对慢环门禁与确定性回放，不成为语义内容或用户查询的旁路存储。

### R1.7 实施状态审计

- 已新增 `docs/research/ustrf-sc/USTRF_SC_IMPLEMENTATION_STATUS.md`，按双环图的每个模块记录代码、JVM/真机证据、未满足门和当前授权。它将 CameraX/ARCore 的“候选可观测”与 metric geometry/跨帧世界/生产可用明确隔离，避免任何单项成功被扩大解释。

### R1.8 临时手持刚体标定准备

- 当前可用真机为 SM-S9280；为准备不改变默认 App 的后续标定，已保存 `USTRF_SC_PROVISIONAL_HANDHELD_CALIBRATION_RUNBOOK.md`。其 `bodyFrame` 仅定义为手机刚体 `handheld-device-body-v1`，明确禁止将其扩大为人体/胸前/眼镜外参。该执行单规定 source freshness、30 样本/5 桶、拆装重复、artifact digest 与独立复核的实际采集顺序。
- 这是一项透明的临时硬件假设：基于此前已运行的手机真机 benchmark，目的是使下一次现场采集可重复；尚未生成 manifest、未填 `independentlyVerified`，所有 metric geometry 与生产门继续 closed。

### R1.9 无参考物手持基线

- 已按“手持、无标定物、无独立复核”的 fail-closed 配置运行新一轮 SM-S9280 900-update isolated audit：`TRACKING=844`、最长连续 tracking observation=844、raw image-intrinsics signature count=1，但 843 个 raw-depth candidate 中 source-aligned=0、reprojected/stale=843。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-handheld-reference-free-baseline-sm-s9280-r5/summary.json`。
- 该结果是有价值的负证据：手持移动可提供 tracking candidate 与稳定的 API intrinsics observation，却不能在无参考物条件下得到当前帧 raw depth；也不能构造 depth registration、完整外参或独立 calibration manifest。因此它不打开任何 metric geometry/production gate。

### R1.10 可打印参考靶

- 因现场没有现成标定物、但可安排独立复核，已生成 A4 100% 打印的 8x10、20 mm checkerboard（7x9 inner corners）：`artifacts.local/calibration/ustrf-sc/ustrf_sc_checkerboard_a4_20mm_r1.pdf`。渲染视觉检查确认棋盘、100 mm 校验线和页边完整；配套 manifest 固化 SHA-256 `178b58254d4e4f9fb06e59b443f3c5a78c7cc38d6ae597d62511f8faaaa23812`、方格定义和打印拒绝条件。
- 此靶只提供可追溯参考物，必须先实测 100 mm line；它本身不生成 calibration manifest、不填 independent verification，亦不改变任何 metric geometry/production gate。

### R1.11 设备阶段决策

- 用户明确当前以手持手机试验、未来可能采用眼镜设备，且不希望当前承担物理标定负担。已将手机路线改为 reference-free shadow：保留快环/慢环合同、时间/候选观测、非米制语义和事件回放，不以打印靶或 calibration manifest 作为手机阶段前置；metric geometry 与空间持久化继续 closed。
- 已新增 `USTRF_SC_DEVICE_PHASE_POLICY.md`。眼镜定型后必须通过 `GlassesFrameSource` 等既有 seam 建立新的 camera/body frame、clock、calibration manifest 和 device benchmark；明确禁止把 SM-S9280 的 ARCore、内参、外参、时间统计或 manifest 迁移到眼镜。

### R1.12 无参考物慢环事件门

- 已新增 pure Kotlin `UstrfSlowLoopEventGate`：周期关键帧、用户查询、目标变化、高不确定性和安全复核都只能产生绑定 source frame、坐标系、签发时刻与 TTL 的 `UstrfSlowLoopEvent`。它不持有图像、语义文本或任何 action/corridor/speed 字段。
- gate 对 source frame 回退、坐标系切换、低于熵阈、最小事件间隔和周期未到逐项抑制；回归测试覆盖每项 fail-closed 行为。其作用是把手机 reference-free 路线限制在可审计的“候选事件”，而不是让慢环绕过快环作出走向决策。
- 独立 `:ustrf-shadow-benchmark` 的 CameraX 影子测试已在静止 SM-S9280/Android 16 通过：30 个手机帧中接受 3 个 event，3/3 与 source frame 绑定，27 个按周期阈值抑制。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-reference-free-event-shadow-sm-s9280-r1/summary.json`。不需要标定物或移动，不接 OCR/VLM、不写 persistent scene，也不改变默认 App。

### R1.13 无人工采集的离线安全仿真

- 为不依赖人工采集推进 G0，新增 `UstrfOfflineSafetyScenarioRunner`。它只接受显式 `synthetic-fixture-v1` / `synthetic-motion` 的带标签局部几何与相对运动 evidence，贯通 metric geometry projector、TTC、risk field、五候选人体包络 corridor 和 fail-closed supervisor；没有图像、模型预测、ARCore、设备标定或用户数据。
- 自动场景覆盖 clear corridor、中央占用、全宽下坠、全宽头部障碍、动态交汇、中央未知、过期几何和 pose lost。下坠/头部/动态/未知/过期/失跟均应 STOP；中央单格占用会令中心与相邻窄通道因人体包络被拒绝，只留下侧向 trace candidate。所有动作仍带 `SHADOW_ONLY`，不生成生产导航命令。
- `:core:ustrf:test --offline --no-daemon --console=plain` 于 2026-07-20 通过：59 tests、0 failures、0 errors，其中 `UstrfOfflineSafetySimulationTest` 3 tests 全过。证据：`artifacts.local/evidence/ustrf-sc/2026-07-20-offline-safety-simulation-r1/summary.json`。该结果只满足逻辑 fixture 的 G0 一部分，不证明真实感知、真机性能或用户安全。
