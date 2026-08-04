# hftf

状态：`development / candidate-side-lane /
R3.1-reference-opportunity-not-evaluable /
D3-Q0-screening-effect-contract-frozen /
D6-real-veto-transfer-not-supported /
D6-real-calibration-increment-not-supported /
D6-real-phase-early-pair-not-supported /
D6-motion-alignment-separability-mixed /
D6-raft-residual-flow-not-stable /
D8-thor-magni-local-route-supervision-materialized /
D8-high-dimensional-coarse-actionability-signal-observed /
D8-equal-capacity-temporal-actionability-increment-not-stable /
D8-temporal-spatial-corridor-signal-weak /
D8-equal-capacity-temporal-spatial-increment-not-stable /
D8-full-local-field-history-increment-not-supported /
D9-jrdb-corridor-replication-not-supported /
frozen-feature-history-route-stop /
D10-trainable-tail-temporal-increment-not-supported-stop /
D11-causal-kinematic-information-not-supported /
D12-future-onset-target-five-fold-ready /
D13-future-onset-temporal-spatial-increment-supported /
D14-direction-preserving-raft-features-materialized /
D14-explicit-motion-future-onset-increment-not-supported /
D15-jrdb-corridor-future-onset-two-fold-ready /
D15-jrdb-future-onset-history-replication-not-supported /
D16-tartanground-future-onset-three-fold-ready /
D16-tartanground-history-increment-not-supported /
A1-consumed-motion-increment-not-supported /
A0.1-android-probability-head-parity-runtime-supported /
metric-depth-android-dualarm-runtime-supported-no-realtime /
dual-rate-metric-depth-observer-r1-development-not-supported /
dense-metric-depth-propagation-r0-development-not-supported /
metric-depth-calibration-head-r0-development-not-supported /
spatial-calibration-head-r1-development-not-supported-stop /
spatial-tof-e-arm-activated-capture-protocol-pending /
scale-free-r1-bonn-source-support-not-evaluable /
scale-free-r2-arkitscenes-source-support-not-evaluable /
frozen-single-frame-posthoc-temporal-residual-family-stop`

Spatial Calibration Head R1 已按预冻结合同完成 fresh、visit-disjoint ARKitScenes
开发评价。3,000 帧开发缓存与 3,600 帧 RGB 身份审计通过数据边界，但 9,423 参数
spatial head 在 `0/4` 折联合优于常数，固定 validation 的 coverage/MAE/agreement/
false-clear/temporal/ECE 六门全失败；sealed 米制真值未打开，手机 shadow 未授权，
纯 RGB 尺度扩展停止。协议与结果权威见
[protocol](../../../docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_PROTOCOL_2026-08-04.json)、
[development result](../../../docs/research/hftf/SPATIAL_CALIBRATION_HEAD_R1_DEVELOPMENT_RESULT_2026-08-04.json)。
预冻结 ToF 切换条件已触发；当前只激活同摄像头/session 的多区 ToF E 臂准备，尚未
采购、采集或产生性能证据，见
[E-arm activation](../../../docs/research/hftf/SPATIAL_TOF_E_ARM_ACTIVATION_2026-08-04.json)。

最新深度观测器同屏终态：
`METRIC3D_FP16_BALANCE_DAV2_DUAL_FREQUENCY_DIAGNOSTIC_FAIL`。这里的目标
是质量/开销 Pareto，不要求轻模型超过最重模型的绝对精度。在同一 120 帧已消费 TUM
屏幕上，Metric3D FP16 保持五项全过，稳态中位数 `142.33 ms`、CUDA 峰值
`573 MiB`；DA V2 Small Metric FP16 为 `54.27 ms / 328 MiB`，但 false-clear
仍为 `24.29%`，只保留为“高频相对结构 + 低频米制锚点”候选，不得独立驱动米制
clearance。MoGe-2 仅快约 9%、显存约 2 倍且任务质量更差，不是当前平衡点。
三者都只是观测器；Metric3D 是当前单模型 balance/teacher，不默认成为最终部署模型。
固定每五帧一次 Metric3D 锚点的因果回放把 DA 的 clearance MAE 降到 `0.16506 m`，
稳态顺序均值为 `79.41 ms`，但包络一致率 `88.62%`、false-clear `6.97%`，仍有两项
gate 未过，而且同步锚点 P95 为 `198.59 ms`。因此只支持下一步研究异步锚点与共驻
内存，不允许在已消费 cohort 搜索锚点周期救援。
完整结果见 `DEPTH_OBSERVER_CLEARANCE_A0_CONSUMED_RESULT.md`。

冻结的 `DUAL_RATE_METRIC_DEPTH_OBSERVER_R1` 已执行。D 臂用独立 CUDA worker、
completion-time 因果调度、最近三个已完成锚点的 Theil-Sen 全局仿射校正和 1 秒
source-age `UNKNOWN`，没有搜索 cadence、TTL 或拟合器。已知帧 MAE 为
`0.15241 m`，但只有 `58/120` 帧保持已知，包络一致率 `86.21%`、false-clear
`8.43%`，终态 `R1_DEVELOPMENT_TASK_GATES_NOT_SUPPORTED`。按已测
`1500.794 ms` Metric3D HTP service time 做的共享加速器资源审计又得到 DA 中断
`112/120`、anchor completion age `1.500794 s`，终态
`R1_PHONE_SHARED_HTP_FEASIBILITY_NOT_SUPPORTED`。不得在该 consumed cohort 调参
救援；fresh final-camera、共驻内存、温度仍未评价，因此不把负结果升级为 ToF 唯一性
或采购授权。协议和结果见
`../../../docs/research/hftf/DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_PROTOCOL_2026-08-03.md`
与
`../../../docs/research/hftf/DUAL_RATE_METRIC_DEPTH_OBSERVER_R1_DEVELOPMENT_RESULT_2026-08-03.md`。

两个冻结后继也已执行并判负。稠密 residual + 双向 RAFT 传播得到 `81/120` paired、
MAE `0.17694 m`、一致率 `89.81%`、false-clear `7.02%`，终态
`DENSE_PROPAGATION_CONSUMED_DEVELOPMENT_NOT_SUPPORTED`。离线 Metric3D 教师的
770 参数 DA CLS calibration head 得到 MAE `0.19980 m`，4/4 折优于 raw DA、仅
2/4 折优于训练折常数 affine，一致率 `89.67%`、false-clear `8.92%`，终态
`CALIBRATION_HEAD_CONSUMED_DEVELOPMENT_NOT_SUPPORTED`。三条当前候选均停止 consumed
调参，但这不授权 ToF 采购；综合决策见
`../../../docs/research/hftf/METRIC_DEPTH_THREE_ROUTE_DECISION_2026-08-03.md`。

最新 NPU 性能终态：
`HTP_EXECUTION_SUPPORTED_HIGH_FREQUENCY_NOT_SUPPORTED_RELATIVE_ONLY`。Qualcomm
DA V2 Small 的现成相对深度 DLC 已在 SM8650 / HTP V75 上确认无 CPU fallback；六次固定
输入中，cached float 默认档平均执行 `174.32 ms`，W8A16 为 `451.63 ms`。显式
`burst` 档为 `177.19 ms`，未形成性能救援。这个结果只证明本机 HTP 执行与成本，不能
把相对深度提升为米制 clearance，也不证明真实图像质量、功耗或持续热性能。当前端侧
性能问题仍未解决；后继必须是 metric checkpoint 的独立转换、输入/decoder 降本或更小
的专用 metric observer，而不是在本次已消费 timing 上挑性能档。完整记录见
`DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.md`。

最新米制 NPU/稀疏锚点终态：
`DAV2_METRIC_392X518_HTP_DEPLOYMENT_PARITY_SUPPORTED /`
`PER_SEGMENT_SPARSE_SCALE_ANCHOR_DEVELOPMENT_SIGNAL_5_OF_5 /`
`GLOBAL_ONE_SHOT_CAMERA_SCALE_FALSE_CLEAR_FAIL / PERIODIC_METRIC_SCALE_ANCHOR_REQUIRED`。
同一 Hypersim 米制 ViT-S 的 `392x518` HTP cached burst 平均执行 `123.19 ms`，相对
ORT 的平均米制差 `0.0145 m`；但未标定 clearance 只过 2/5 门。每段前 10 帧三带
稀疏米制尺度锚点、后 20 帧盲评时，MAE/一致率/false-clear 为
`0.0981 m / 93.77% / 4.95%`，5/5 门通过；只用第一段一次性全局尺度时 false-clear
回升至 `6.99%`，4/5 门。因此当前唯一保留的端侧架构是“低分辨率 DA Metric HTP +
周期性多区 ToF/稀疏米制尺度锚点 + 过期 UNKNOWN”，不是纯 RGB 或一次标定方案。详见
`DAV2_METRIC_QAIRT_SPARSE_SCALE_R0_RESULT.md`。

最新稀疏尺度侧车回放终态：
`SPARSE_SCALE_CLEARANCE_SIDECAR_REPLAY_SUPPORTED_CONSUMED_PROXY /`
`CLOCK_DOMAIN_BINDING_REQUIRED_AND_VERIFIED / REAL_TOF_REGISTRATION_NOT_EVALUATED`。
新的 class-free 侧车从 120 帧已消费 RGB 重新执行 `392x518` 观测器、三带 clearance、
时间戳锚点和 UNKNOWN 门控；每段前 10 帧锚定、后 20 帧评价仍得到
`0.0981 m / 93.77% / 4.95%`，五项任务门全过，且锚点前没有提前 VALID。首次把绝对
图像时钟误接到相对 manifest 时钟时，119 帧保持 UNKNOWN，证明错误时钟域 fail
closed；修复后以 manifest timestamp 为唯一锚点时钟。`5000 ms` 只用于短序列回放，
不是部署 TTL。该结果验证接口因果性与已消费质量，不是实机 ToF、端到端 HTP 或性能
问题已解决。详见 `SPARSE_SCALE_CLEARANCE_SIDECAR_R0_RESULT.md`。

最新真实硬件接口终态：
`MULTIZONE_TOF_RGB_INTERFACE_READY_SYNTHETIC_ONLY /`
`CURRENT_PHONE_PUBLIC_DEPTH_SOURCE_ABSENT / REAL_TOF_HARDWARE_NOT_PRESENT`。
多区 ToF 不再被当作与 RGB 共光心的三带距离：每个 zone 的径向量程先沿标定射线恢复
ToF 3D 点，经刚体 `T_rgb_from_tof` 投影到矫正 RGB，再与同像素 DA optical-z 求稳健
尺度。新增 PnP 外参标定、同 clock-domain 因果队列、range/sigma/zone/band/scale-MAD
质量门，并接入现有 clearance sidecar。5 项新合成几何测试和共 14 项相关测试通过。
当前 SM-S9280 的只读 ADB 探测未发现公开 Camera2 depth output 或多区 ToF，Windows
也未连接 ToF 硬件，所以这一步只完成硬件无关接口；详见
`MULTIZONE_TOF_RGB_ADAPTER_R0.md` 和 `MULTIZONE_TOF_RGB_ADAPTER_R0_RESULT.json`。

本轮硬件选型终态：
`VL53L8CX_DEFAULT_CANARY_VL53L5CX_AVAILABILITY_FALLBACK`。默认传感器选
ST `VL53L8CX`：8×8、4 m、65°、最高 60 Hz，支持 1 MHz I²C 与 3 MHz SPI，连续
模式官方示例约 215 mW；它只补绝对尺度和逐区质量，不替代 RGB clearance。第一套
实验台优先 `X-NUCLEO-53L8A1 + STM32 Nucleo USB bridge`，最终小型化改用
`SATEL-VL53L8 + A568 direct bus` 或小型 USB bridge。`VL53L5CX` 仅作为供货备选；
单点 ToF、超声和手机 proximity 不进入该支线。尚未采购，价格、库存和载板电气仍需
下单时确认。详见 `MULTIZONE_TOF_HARDWARE_SELECTION_R0.md`。

## 研究问题与版本

本 Module 服务 `HFTF_CANDIDATE_LANE_R0`：检验历史 RGB 能否预测面向行人身体包络的
短时未来可通行/碰撞风险场，而不是继续给 YOLO 增加后处理规则。当前只执行
`HFTF_H0_SOURCE_FEASIBILITY_R0`，允许的 claim 是来源与教师接口可行性，不是模型效果、
创新性、用户效果或安全性。

当前章程与终态见 `docs/research/hftf/README.md`。通用 H0 的 partial terminal 仍保留；
source-specific H0.1/H0.2 已准入下一阶段的 geometry proxy canary。

### M3D-CF 当前碰撞占用支线（2026-08-03）

最新独立支线决策见 `M3D_CF_ROUTE_STATUS_2026-08-03.md`。当前冻结结论是
`KEEP_CANDIDATE_SIDE_LANE_CURRENT_OCCUPANCY_ONLY`：普通标定 RGB 上的
UniDepthV2-S + 3D clearance + RAFT + 低容量概率头已获得 TUM same-family fresh
Development 支持，并已有 candidate-only JSONL/MP4 CLI；0.5 秒未来占用未通过
`sitting_halfsphere` fresh transfer，Bonn 跨数据集代理因 source pose/reference
不兼容而 `NOT_EVALUABLE`。该支线不依赖 ARCore，但在最终外接摄像头和目标设备验证前
不得晋级主线、提醒或 safety claim。

A1 冻结六臂比较保留终态 `COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL`：完整概率场在
1,716 个已消费机会中仍是非 oracle 臂的 Brier、F1、recall 最优者（`0.08936 /`
`0.89841 / 88.43%`），但 MCC `0.75579` 未超过 2D corridor 的 `0.76141`。
后续只用同一已消费 cohort、相同 LOSO/逻辑回归/阈值执行的固定双臂增量消融得到
`A1_CONSUMED_MOTION_INCREMENT_NOT_SUPPORTED`：加入十个运动特征使 recall 增加
`0.59` 个百分点且 4/7 窗口 Brier 改善，但 pooled Brier 恶化 `3.94%`、F1
下降 `0.00678`、MCC 下降 `0.02292`。因此保留概率质量、F1 与召回的描述性支持，
但不主张独立 causal-motion increment，也不搜索特征子集或阈值救援。

SM-S9280 / Android 16 上的隔离 instrumentation canary 又对 1,716 个绑定特征行
执行 20 次冻结 A0.1 概率头：概率和 0.50 决策 mismatch 都为 `0`，最大绝对误差
`2.22e-16`，P95 `0.001615 ms`，终态
`A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_SUPPORTED`。这只清除了 Kotlin
标准化/逻辑头的数值与时延风险；`heavy_inference_covered=false`，因此 UniDepth、
clearance geometry、RAFT、内存和持续热行为仍未获得 Android 证据。

随后冻结的 `METRIC_DEPTH_ANDROID_DUALARM_R0` 首次把 Metric3Dv2-S 与
UniDepthV2-S 放入同一 `onnxruntime-android:1.26.0` instrumentation。两臂在
SM-S9280 的 CPU 与 NNAPI 会话均能加载并完成推理，但当前全精度直跑不具备实时性：
CPU P95 分别为 `5367.79 ms` 与 `1721.97 ms`，Metric3D 是 UniDepth 的
`3.1172x`；对应顺序进程 PSS 增量约 `832.13 MiB` 与 `260.54 MiB`。注册 NNAPI
反而使 P95 分别恶化 `33.77%` 与 `14.10%`，且不能证明 accelerator-only coverage。
因此保留 Metric3D 为精度 teacher/部署优化首选，保留 UniDepth 为延迟参考；两者当前
均不得进入实时 App。完整协议与结果见
`METRIC_DEPTH_ANDROID_DUALARM_R0_PROTOCOL.md` 和
`METRIC_DEPTH_ANDROID_DUALARM_R0_RESULT.md`。

### D6 SANPO real veto transfer

当前可逆 Development 接口把 D7 三人 model-blind RGB review 中 3/3 `REJECT`
的 SANPO 区间物化为去重 media manifest，再运行 frozen veto ranker：

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf materialize_stage_c_d6_sanpo_blind_negative_media.py `
  --candidate-index <sanpo-candidate-index.jsonl> `
  --rgb-review <review-a.jsonl> `
  --rgb-review <review-b.jsonl> `
  --rgb-review <review-c.jsonl> `
  --staging-root <review-bundle-staging-root> `
  --output-root artifacts.local/evidence/hftf/<blind-negative-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf export_stage_c_d6_veto_review_candidates.py `
  --media-manifest artifacts.local/evidence/hftf/<blind-negative-run>/media_manifest.jsonl `
  --threshold-report artifacts.local/evidence/hftf/stage-c-d6-veto-eligibility-confidence-residual-canary-v0/conservative-veto-execution-summary.json `
  --output-root artifacts.local/evidence/hftf/<veto-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_real_veto_ranking.py `
  --output artifacts.local/evidence/hftf/<real-ranking-run>/report.json

E:\codex-tools\projects\blindassist\toolchain\venvs\learned-component-validator-py311\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_real_veto_calibration.py `
  --ranking-report artifacts.local/evidence/hftf/<real-ranking-run>/report.json `
  --output artifacts.local/evidence/hftf/<real-calibration-run>/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf run_stage_c_d6_sanpo_real_phase_early_pair_canary.py `
  --output-root artifacts.local/evidence/hftf/<real-phase-early-pair-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_motion_alignment_separability.py `
  --heldout-fold 0 `
  --output artifacts.local/evidence/hftf/<motion-alignment-run>/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf evaluate_stage_c_d6_sanpo_raft_motion_representation.py `
  --raft-weights artifacts.local/models/hftf/torch/optical-flow/raft_small_C_T_V2-01064c6d.pth `
  --output artifacts.local/evidence/hftf/<raft-motion-run>/report.json
```

物化器逐帧验证 review-bundle SHA，重叠 observation 必须 byte-identical，且每个输出
五帧窗口必须完整落入至少一个 3/3 reject 区间。该 label 只支持 clip/window
actionable-negative，不产生 cell localization、positive-event safety 或 system-event
truth。ranker 的 `--threshold-report` 只应用已存在的 zero-training-true-alert
threshold，不搜索 threshold。30-event evaluator 固定比较 candidate 与
`1 - baseline risk` 的 cell/event-phase ranking。calibrator 固定 5-fold
source-session-held-out split、`StandardScaler + L2 LogisticRegression(C=1,
liblinear, class_weight=balanced)`，只比较 baseline-only 与增加 candidate
mean/p95/max 的 candidate-aware arm；不搜索 feature、C、model、fold 或 threshold。
real-phase early-pair runner 固定 `seed17/model-fold0/heldout-fold0`、20 epochs 和
final-epoch evaluation；训练/held-out source sessions 隔离，只有 event-phase p95
AUROC 与 AP delta 同时为正才允许扩展。
motion-alignment evaluator 用 sparse-LK + RANSAC partial affine 比较 raw/aligned
相邻帧残差；两臂固定相同 54 维 grid features 与 L2 projection。配准 coverage 在
监督投影前判定，失败只产生 `NOT_EVALUABLE`。
RAFT evaluator 固定校验官方 small weights SHA-256，并在一次运行中输出五折 raw
pixel / raw flow / global-motion-residual flow 对照；不按 fold 选择 representation。

### D8 THOR-MAGNI local route supervision

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d8_thor_magni_local_route_supervision.py `
  --output-root artifacts.local/evidence/hftf/<local-route-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  evaluate_stage_c_d8_thor_magni_rgb_history_screen.py `
  --samples artifacts.local/evidence/hftf/<local-route-run>/samples.jsonl `
  --output-root artifacts.local/evidence/hftf/<rgb-history-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d8_thor_magni_equal_capacity_temporal_head.py `
  --samples artifacts.local/evidence/hftf/<local-route-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<rgb-history-run>/features.npz `
  --output artifacts.local/evidence/hftf/<temporal-head-run>/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  extract_stage_c_d8_thor_magni_spatial_features.py `
  --samples artifacts.local/evidence/hftf/<local-route-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<spatial-run>/features.npz

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d8_thor_magni_equal_capacity_temporal_head.py `
  --samples artifacts.local/evidence/hftf/<local-route-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<spatial-run>/features.npz `
  --output artifacts.local/evidence/hftf/<spatial-head-run>/report.json
```

物化器从 19 个 THOR-MAGNI Pupil/QTM sessions 派生 wearer-motion-relative
`2×6×4` future occupancy、未来最小同步距离、1.25 m 近距和前向走廊侵入代理。
这些只属于 source-native geometric Development supervision。筛查器固定 pretrained
MobileNetV3-small 和五折 source-session isolation，比较 current-only 与
history-residual linear readout。D8 结果只在近距/走廊 coarse actionability 层支持
较高维 history separability signal；完整 48-cell field 与连续距离排序不支持继续
扩展。相同 4,610 参数、相同训练预算的 current/history temporal head 随后没有复制
coarse 增量，因此不把前述 signal 升级为 history 独立增量。最后的 13,586 参数
spatial-map 对照只在 corridor AUROC/AP 上得到 5/5 fold 的小增量
(`+.0040/+.0038`)，近距不支持，AP 也只有 9/15 units 为正；记录 weak signal 后
停止当前 THOR frozen-backbone 模型搜索。

### D9 JRDB independent-dataset corridor replication

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d9_jrdb_local_route_replication.py `
  --output-root artifacts.local/evidence/hftf/<jrdb-route-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  extract_stage_c_d9_jrdb_spatial_features.py `
  --samples artifacts.local/evidence/hftf/<jrdb-route-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<jrdb-feature-run>/features.npz

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d8_thor_magni_equal_capacity_temporal_head.py `
  --experiment d9-jrdb `
  --samples artifacts.local/evidence/hftf/<jrdb-route-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<jrdb-feature-run>/features.npz `
  --output artifacts.local/evidence/hftf/<jrdb-replication-run>/report.json
```

D9 从四个本地 JRDB sequences 的连续 RGB360 与同 frame-stem `labels_3d`
物化 104 个 samples。geometry-only census 后固定两个完整 source-pair folds，
主检验只复现 D8 corridor weak signal。corridor AUROC/AP delta 在 0/2 folds 和
1/6、0/6 units 为正，终态为
`D9_JRDB_TEMPORAL_SPATIAL_CORRIDOR_REPLICATION_NOT_SUPPORTED`。近距负对照的小正值
不触发 target 切换；当前 frozen-feature history route 停止。

### D10 THOR-MAGNI trainable-tail temporal canary

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  cache_stage_c_d10_thor_magni_rgb_history.py `
  --output artifacts.local/evidence/hftf/<d10-cache-run>/history_rgb_uint8.npy

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d10_thor_magni_trainable_tail_canary.py `
  --cache artifacts.local/evidence/hftf/<d10-cache-run>/history_rgb_uint8.npy `
  --output artifacts.local/evidence/hftf/<d10-run>/report.json
```

cache 先写 `.partial.npy`，完整填充并计算摘要后才 atomic replace；工程中断可删除
partial 后按同一输入重建，不烧毁 source。canary 冻结 MobileNet blocks `0..8`，
训练 blocks `9..12`，current/history 两臂共享完全相同的 765,386 个 trainable
parameters。固定 seed17、五折、8 epochs 的四项 AUROC/AP gate 全部失败，终态为
`D10_TRAINABLE_TAIL_TEMPORAL_INCREMENT_NOT_SUPPORTED_STOP`。按预定规则不扩 seeds、
不运行 JRDB zero-shot，也不调整解冻边界、学习率、epoch 或 head。

### D11–D13 true future-onset repair

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  evaluate_stage_c_d11_thor_magni_kinematic_information_ceiling.py `
  --output artifacts.local/evidence/hftf/<d11-run>/report.json

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d12_thor_magni_future_onset.py `
  --output-root artifacts.local/evidence/hftf/<d12-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d13_thor_magni_future_onset_temporal_baseline.py `
  --samples artifacts.local/evidence/hftf/<d12-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<d13-run>/report.json
```

D11 发现原 future-ever target 被 `t=0` current risk 主导：current-static QTM
geometry 已有约 `.89–.97` AUROC，causal history kinematics 未稳定改善。D12 因而
只在当前安全样本中定义未来 onset，得到近距 `157/530`、走廊 `148/616` 正例/eligible，
五折均含正负。D13 相同 frozen-spatial 等容量 head 在修正 target 上通过预定
median/正折门，但增量只有约 `+.0008–+.0020`，且走廊 AP mean 仍为负。该弱
representation signal 只授权显式 motion representation，不授权系统或主线晋级。

### D14 explicit-motion future-onset canary

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  extract_stage_c_d14_thor_magni_explicit_motion_features.py `
  --output artifacts.local/evidence/hftf/<d14-feature-run>/features.npz

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d14_thor_magni_explicit_motion_onset_canary.py `
  --motion-features artifacts.local/evidence/hftf/<d14-feature-run>/features.npz `
  --output artifacts.local/evidence/hftf/<d14-canary-run>/report.json
```

feature extractor 对五帧中的四个相邻 pair 运行固定 RAFT-small，保留 3×6 grid 的
raw/residual x、y、mean/p90 magnitude，不做表示选择。4,312 pairs 全部完成。
等容量 canary 的 current arm 输入全零 motion，candidate 输入真实 flow。走廊
AUROC/AP mean delta 为 `+.0219/+.0240`，但 AP median 为负且仅 2/5 folds 正；
近距 AP mean `-.0103`。终态
`D14_EXPLICIT_MOTION_FUTURE_ONSET_INCREMENT_NOT_SUPPORTED`；不调 RAFT/grid/head，
保留 source-local corridor signal 与 D12 onset task。

### D15 JRDB true-future-onset replication

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d15_jrdb_future_onset.py `
  --output-root artifacts.local/evidence/hftf/<d15-onset-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d15_jrdb_future_onset_replication.py `
  --samples artifacts.local/evidence/hftf/<d15-onset-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<d15-replication-run>/report.json
```

D15 从 JRDB anchor-frame 3D-person geometry 计算 current state，物化 proximity
14/102、corridor 10/71 onset-positive/eligible；两个固定 source-pair folds 都有
正负例。等容量 frozen-spatial replication 的 corridor AUROC/AP 两折 seed-mean
均为负，aggregate `-.00618/-.03098`，终态
`D15_JRDB_FUTURE_ONSET_HISTORY_REPLICATION_NOT_SUPPORTED`。D13 只保留为 THOR
source-local weak signal；不继续当前 frozen representation search。

### D16 TartanGround true-future-onset baseline

```powershell
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  materialize_stage_c_d16_tartanground_future_onset.py `
  --output-root artifacts.local/evidence/hftf/<d16-onset-run>

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  extract_stage_c_d16_tartanground_spatial_features.py `
  --samples artifacts.local/evidence/hftf/<d16-onset-run>/samples.jsonl `
  --output artifacts.local/evidence/hftf/<d16-feature-run>/features.npz

E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe `
  scripts/run_research_tool.py hftf `
  run_stage_c_d16_tartanground_future_onset_temporal_baseline.py `
  --samples artifacts.local/evidence/hftf/<d16-onset-run>/samples.jsonl `
  --features artifacts.local/evidence/hftf/<d16-feature-run>/features.npz `
  --output artifacts.local/evidence/hftf/<d16-baseline-run>/report.json
```

D16 继承 D5 三折 environment assignments，物化 19,478 eligible cells、1,652 onset
cells；near/far × body/head 四个 targets 在每折都有正负。等容量 frozen-spatial
history 的 near 增量仅约 `+.0005–+.0012`，far body/head AUROC 为负，未达到
预定 effect floor。终态关闭 frozen single-frame feature + post-hoc temporal
residual family；下一候选必须在 representation pretraining 阶段共同编码五帧。

## 稳定 Interface

从仓库根目录运行：

```powershell
$runId = 'h0-source-feasibility-r0-REPLACE_WITH_NEW_RUN_ID'
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_source_feasibility.py `
  --replay-root artifacts.local/evidence/datasets/sanpo-synthetic-replay-25frames-20260720 `
  --output "artifacts.local/evidence/hftf/$runId/source_feasibility.json"
```

输入必须包含 hash-bound RGB、panoptic mask、metric depth、相机内参、pose CSV、
`dataset_spec.json`、`manifest.replay.jsonl` 和既有 source-integrity QA。相对路径必须
保持在 replay root 内；报告路径必须位于 `artifacts.local/` 且已存在时拒绝覆盖。
静态 projection 资格由脚本独立复算全部文件 hash、完整 PNG decode/dimensions、depth
header/shape 与 finite-positive samples；QA 还必须以 schema、`ok`、frame count 和逐
depth path 与 manifest 一致。输入中的 `SANPO-Synthetic`/official split 字段只作为
内部一致性声明；本 H0 不把本地 manifest 自报内容当作来源身份的密码学认证。
重复 canonical asset path 或完整 RGB/mask/depth hash triplet 会 fail closed；QA
布尔字段必须是精确 JSON boolean，字符串 `"false"` 不视为 false declaration；
frame count、fraction 与相机内参拒绝 bool 或字符串伪数值。

multi-height/future 的**结构准备度**不能由普通 CSV 列名或非空占位字段获得。通用
H0 可检查下列精确合同：

- `hftf_body_frame_contract`：精确 schema、frame/axis/unit/direction、有限且归一化
  SE(3)、ground reference 和 provenance；
- `hftf_pose_binding`：hash-bound JSONL，把每个 manifest row 一一映射到唯一 raw pose
  row，并核对 session/sequence/frame/time、admitted tracking state、有限 position 与
  归一化 quaternion。

即使上述结构检查全部通过，本工具仍把 multi-height/future 判为 `NOT_EVALUABLE`。
真实准入还必须由 source-specific verifier 分别复算标定 receipt 与原始
pose-frame/time mapping；hash-bound sidecar 不能给自己签发权威。

### SANPO source-specific H0.1/H0.2

H0.1 discovery：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/verify_sanpo_pose_geometry_authority.py `
  --evaluation-mode discovery `
  --replay-root <single-session-replay-root> `
  --official-repo artifacts.local/downloads/sanpo_dataset_official_repo `
  --output artifacts.local/evidence/hftf/<run-id>/authority.json
```

H0.2 replication 对 H0.1 已冻结的
`p_world = R_xyzw @ p_opencv_camera + translation_m` 做跨 session 检验：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/verify_sanpo_pose_geometry_authority.py `
  --evaluation-mode frozen_canonical_replication `
  --replay-root <independent-single-session-replay-root> `
  --official-repo artifacts.local/downloads/sanpo_dataset_official_repo `
  --output artifacts.local/evidence/hftf/<run-id>/authority.json
```

verifier 固定 official repository commit/common.py hash，在线复核 GCS object
generation/size/MD5/CRC32C，再验证本地 MD5、official pose-row/frame-index 规则、48 个
pose/basis hypothesis、metric-depth reprojection 和 semantic-ground local plane。
`frame_num / session fps` 只表示 nominal relative time。

三个或更多独立 frozen-replication reports 用以下命令聚合：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/aggregate_sanpo_proxy_replication.py `
  --report <session-a-authority.json> `
  --report <session-b-authority.json> `
  --report <session-c-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort.json
```

聚合器拒绝重复 source session，并保持 physical calibration、student/effect、主线和
产品层为 `NOT_EVALUABLE`。

### H1 geometry teacher canary

H1 必须使用已提交的 frozen protocol 和其中四个精确 authority/report hashes。R0
360° evidence version 已执行并 burned；后续正式运行使用 R1 forward-sector protocol
与四个全新 sessions：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_geometry_teacher_canary.py `
  --protocol docs/research/hftf/HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1_2026-08-01.json `
  --session <fresh-replay-a> <fresh-authority-a.json> `
  --session <fresh-replay-b> <fresh-authority-b.json> `
  --session <fresh-replay-c> <fresh-authority-c.json> `
  --session <fresh-replay-d> <fresh-authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_canary.json
```

runner 会重算 protocol、authority、manifest、dataset spec、pose 与每个实际消费
depth/mask 文件 hash；theta edges 同时约束 cell probes 与 obstacle binning，partial
sector 外 points 不 wrap；R0/R1 future field 保持 anchor-centric，所有版本的 UNKNOWN
都留在冻结 denominator。
输出终点只可能是 H1 的 `NOT_EVALUABLE`、multi-height/future stop 或
`GEOMETRY_PROXY_MECHANISM_SUPPORTED`，后者也不会自动授权 H2。

R2 protocol 额外绑定 source-preparation contract。runner 对每个 anchor 选择冻结
lookback/tolerance 下的严格历史 frame，仅用 history-to-anchor pose 计算
ground-tangent velocity，并为 `.4/.8 s` 分别生成 horizon-specific rolling origin、
probes 与 obstacle bins。future pose 只作为 observation，不定义 origin；predicted 与
observed ground-origin error 仅作 diagnostic，不进入 gate。

### Stage B swept-envelope label mechanics D0

R2 只关闭 angular-cell point-support proxy；它没有实现原始 Stage B 所需的人体横向
包络、候选轨迹 swept collision 与足部 ground continuity。D0 因而只在已烧毁的 R2
sources 上检查这套标签 mechanics，不是 fresh evidence，也不评价未来轴或 student：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_swept_envelope_label_mechanics.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --r2-protocol docs/research/hftf/HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.json `
  --session <burned-r2-replay-a> <authority-a.json> `
  --session <burned-r2-replay-b> <authority-b.json> `
  --session <burned-r2-replay-c> <authority-c.json> `
  --session <burned-r2-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/mechanics.json
```

实现使用冻结的 synthetic effective half-width、9 个 swept-prism probes 和 5-section
ground support；只有 known 且数值 risk 为零的 cell 才编码为 SAFE，缺失 ground
support 保持 UNKNOWN。输出准入 fresh R3 也只代表 mechanics 可执行且非退化，不代表
风险真值、Stage B 增益、H2 或主线替换。

D1 在同一 burned cohort 上比较 candidate、baseline 与 disjoint dense reference：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/pilot_swept_envelope_reference_metrics.py `
  --pilot docs/research/hftf/HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --r2-protocol docs/research/hftf/HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.json `
  --session <burned-r2-replay-a> <authority-a.json> `
  --session <burned-r2-replay-b> <authority-b.json> `
  --session <burned-r2-replay-c> <authority-c.json> `
  --session <burned-r2-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/reference_metrics.json
```

candidate/reference pixel grids 不相交；四个 reference count thresholds 必须全部报告。
D1 只设计 R3 gate，不选择 fresh outcome。

formal R3 必须使用已绑定四 source hashes 的 protocol：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_b_reference_comparison.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_PROTOCOL_R3_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-preparation docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_SOURCE_PREPARATION_R3_2026-08-01.json `
  --session <fresh-r3-replay-a> <authority-a.json> `
  --session <fresh-r3-replay-b> <authority-b.json> `
  --session <fresh-r3-replay-c> <authority-c.json> `
  --session <fresh-r3-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/stage_b_r3.json
```

runner 先裁决 source/reference/known readiness，再裁决 obstacle 增益，最后单列 ground
opportunity 与 agreement。full terminal 也只允许冻结下一 Stage C protocol，不直接授权
future execution 或 student。

R3.1 单 source qualification 只能运行 reference arm：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/qualify_stage_b_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan artifacts.local/evidence/hftf/r3-1-inventory-plan-20260801/inventory_plan.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --replay-root <candidate-replay> `
  --authority <candidate-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

runner 固定 D0 mechanics hash，复核实际消费的 depth/mask 与 authority bindings，并拒绝
16 个 burned sessions。报告不包含 candidate、baseline、confusion 或 arm delta。

40-session bounded inventory plan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_r3_1_inventory_candidates.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

planner 验证 official split generation/hash，只读 description 与 RGB/mask/depth 对象清单，
记录 burned/ineligible 跳过原因及前 40 个 eligible sessions 的确定性 frame indices。

完成全部 source reports 后，使用 cohort aggregator 验证冻结顺序与 reference-only
firewall：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/aggregate_r3_1_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan artifacts.local/evidence/hftf/r3-1-inventory-plan-20260801/inventory_plan.json `
  --report <rank-001-qualification.json> `
  --report <...in exact contiguous inventory order...> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort_result.json
```

若先得到 4 个 qualified source，报告数必须精确停在第 4 个 qualified rank；若不足 4
个，则必须提供全部 40 个报告才能得到 budget-exhausted terminal。R3.1 实际终态为
`R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`，不得在同一队列继续扫描或降门。

### Stage B split-source R4

R4 obstacle source 先按 56-session burn ledger 生成最多 12 个的新 inventory plan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_r4_obstacle_inventory_candidates.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

每个 source 的 qualification 只可计算 obstacle dense reference：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/qualify_r4_obstacle_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --replay-root <candidate-replay> `
  --authority <candidate-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

terrain component 完全由冻结解析 profiles 生成，不读取 SANPO outcome：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_r4_analytic_terrain.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/terrain_result.json
```

terrain pass 只代表 controlled mechanics component 通过，不能独自签发 joint R4
terminal 或 Stage C 权限。

前四个 qualification 通过后，先锁定 source hashes，再运行 obstacle arm 和 joint
aggregation：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_r4_obstacle_opportunity_cohort.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --report <contiguous-rank-qualification.json> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort_lock.json

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_r4_obstacle_reference_comparison.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --cohort-lock <cohort-lock.json> `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --session <replay-a> <authority-a.json> `
  --session <replay-b> <authority-b.json> `
  --session <replay-c> <authority-c.json> `
  --session <replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/obstacle_result.json
```

### Stage C SANPO body/head temporal-student F0

F0 的 source planner 只读取 official split、description、intrinsics、pose object
receipt 与 RGB/mask/depth object inventory；不下载媒体，不计算 geometry/student
outcome：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_f0_sanpo_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

planner 必须排除 effective 60-session burn union，按完整 ID 字典序固定 12 个 source，
并按 rank 固定 `6 train / 3 dev / 3 heldout`。任何 geometry outcome 打开后不得重新
规划或替换。

F0.1 在任何 media/geometry/student outcome 前把 heldout 加强为 official test
split。它复用 F0 metadata plan 的前九个 train/dev candidates，只对 official test
文件顺序做 heldout metadata scan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_f0_1_sanpo_cross_split_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --f0-plan artifacts.local/evidence/hftf/<f0-run-id>/inventory_plan.json `
  --output artifacts.local/evidence/hftf/<f0-1-run-id>/inventory_plan.json
```

输出必须固定 `6 train / 3 dev / 3 official-test heldout`，且所有 outcome firewall
保持 false，才授权 exact media acquisition。

在下载前用 source-lock validator 固化 exact sessions、split、物理 timeline 与
description/pose GCS receipts：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_f0_1_sanpo_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --f0-plan artifacts.local/evidence/hftf/<f0-run-id>/inventory_plan.json `
  --cross-split-plan artifacts.local/evidence/hftf/<f0-1-run-id>/inventory_plan.json `
  --output artifacts.local/evidence/hftf/<lock-run-id>/source_lock.json
```

`aggregate_r4_split_source_result.py` 是唯一可签发 joint R4 terminal 的工具；单个
component 不得提前开放 Stage C。

F0.1 exact media 获取后，先用
`audit_stage_c_f0_1_sanpo_acquisition.py` 对 12 个包的 300 组 RGB/mask/depth、
split、物理索引、GCS/local hash 与 pose 文件做统一审计；再逐 source 运行
`verify_sanpo_pose_geometry_authority.py --evaluation-mode
frozen_canonical_replication`，最后由
`aggregate_stage_c_f0_1_sanpo_authority.py` 封口 exact authority cohort。

teacher opportunity 必须使用 outcome 前冻结的
`HFTF_STAGE_C_SANPO_TEACHER_OPPORTUNITY_EXECUTION_CONTRACT_F0_1_2026-08-01.json`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_f0_1_teacher_opportunity.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_SANPO_TEACHER_OPPORTUNITY_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --f0-protocol docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --f0-1-protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-lock <source_lock.json> `
  --acquisition-audit <acquisition_audit.json> `
  --authority-cohort <authority_cohort.json> `
  --datasets-root artifacts.local/evidence/datasets `
  --authority-root artifacts.local/evidence/hftf/<authority-run-id> `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

该工具只输出 source/role/horizon/height 汇总，不物化 cell corpus。只有终态
`F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS` 才可物化 train candidate
corpus 与 dev reference targets；official-test heldout targets 必须继续封闭到
checkpoint 冻结后的 ordered evaluation。

Stage C C0 在任何 EgoWalk RGB/depth media 内容打开前，先用 exact dataset revision
和四个 metadata hashes 复算 239 条 trajectory 的健康门与冻结 cohort：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_c0_egowalk_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json `
  --metadata-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/inventory.json
```

planner 只读 parquet/meta 与远端 LFS size/hash，不下载或打开 RGB/depth，也不读取
annotation、teacher label 或 student output。必须精确复现冻结的两个不同日期 source，
否则在 media acquisition 前 fail closed。

cohort lock 后只下载其绑定的两组 media，再以官方 `gray16le mm -> m / zero ->
UNKNOWN` 规则完整解码，并运行冻结的 32-frame transport/surface-support audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_c0_egowalk_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json `
  --inventory <locked-inventory.json> `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport_audit.json
```

audit 会完整解码全部 RGB/depth 帧并核对 LFS/local SHA、帧数、5 Hz rate 与 PTS；
32-frame canary 只检查正有限 depth 和 bottom-half/common support，不读取 semantic
class、annotation 或 hazard/safe truth。

C0 的 container nominal-rate 门失败后，C0.1 只允许在 hash-bound consumed replay 上
用 parquet frame/timestamp + meta fps 修复 timebase authority：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_c0_1_egowalk_timebase_repair.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/timebase_repair.json
```

runner 要求 predecessor 的唯一 failures 精确为 RGB/depth nominal-rate mismatch；
若存在任何其他 C0 failure，禁止用 C0.1 越过。

Stage C D0 在 consumed calibration sources 上运行冻结的 depth-only ground-plane /
horizontal-support reader、七个 structural canaries 与第二遍 determinism：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_d0_semantic_independent_label_readiness.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/label_readiness.json
```

formal runner 不读取 semantic class、annotation 或 RGB outcome。`<1.2 m` 与缺失
support 永远 UNKNOWN；即使 D0 full pass，也只允许冻结 fresh-source label/student
canary protocol。

Stage C D1 在同一 consumed cohort 上检验 history-origin-causal 的 future observation
label increment：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_d1_causal_future_label_mechanics.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/future_label_mechanics.json
```

runner 同时计算 current-observation-only baseline 与 current+future candidate；future
pose 只能重投影 observation，不能决定 causal origin/grid orientation。D1 不训练
student。

Stage C E0 source lock 必须在读取六条 fresh RGB/depth 前复算：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_fresh_student_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --inventory artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-lock-r1-20260801/inventory.json `
  --pretrained-weight artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 只读已消费的 metadata inventory、parent hashes 与通用预训练权重，不打开
fresh RGB/depth。只有 `E0_FRESH_SOURCE_LOCK_VALIDATED` 才授权获取机器合同中精确
绑定的六条媒体；仍不授权 teacher corpus 或 student training。

正式 source lock 通过后，只允许按协议 exact allow-list 获取 E0 媒体：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/acquire_stage_c_e0_fresh_media.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-fresh-source-lock-20260801/source_lock.json `
  --output-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --manifest artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801/acquisition_manifest.json
```

首次打开 RGB/depth 后六条 source 全部 burned。acquisition 只验证 exact bytes 并授权
transport decode audit；它不读取 geometry label outcome，也不授权 teacher corpus 或
student training。

E0 fresh transport audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_fresh_media_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --acquisition-manifest artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801/acquisition_manifest.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport.json
```

audit 完整 decode pose/RGB/depth、核对 PTS 与 parquet 物理 timebase，但不计算
geometry labels。只有 `E0_FRESH_MEDIA_TRANSPORT_SUPPORTED` 才授权 teacher mechanics
和 role-opportunity audit。

E0 teacher mechanics + role opportunity audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_teacher_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --transport artifacts.local/evidence/hftf/stage-c-e0-fresh-transport-20260801/transport.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

audit 复用 hash-bound D0/D1 mechanics，输出 source/role aggregates 而不持久化完整
teacher corpus。只有 `E0_FRESH_TEACHER_AND_ROLE_OPPORTUNITY_SUPPORTED` 才授权后续
corpus generation；任何 dev/heldout opportunity failure 都不得换样。

E0.1 fresh evaluation source lock：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_1_fresh_evaluation_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --pretrained-weight artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 先复核 E0 负终态与八条 consumed exclusion，再从原 inventory 重算新的
dev/heldout。它不打开新媒体；只有
`E0_1_FRESH_EVALUATION_SOURCE_LOCK_VALIDATED` 才授权 exact acquisition。

E0.1 exact fresh evaluation media acquisition：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/acquire_stage_c_e0_1_fresh_evaluation_media.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-1-source-lock-20260801/source_lock.json `
  --output-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --manifest artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801/acquisition_manifest.json
```

acquisition allow-list 只含新 dev/heldout 与公共 metadata。首次打开后两条永久 burned；
仍不计算 label 或 student。

E0.1 fresh transport audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_1_fresh_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --acquisition-manifest artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801/acquisition_manifest.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport.json
```

transport 只完整 decode 与核对 timebase；通过后只授权 `.4 s` teacher opportunity，
不重开 `.8 s`。

E0.1 `.4 s` fresh teacher/opportunity audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_1_teacher_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --transport artifacts.local/evidence/hftf/stage-c-e0-1-fresh-transport-20260801/transport.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

runner 只解码 anchor 与 `anchor+2` teacher depth，报告中明确
`zero_point_eight_second_output_computed=false`。新 dev/heldout mechanics 与 opportunity
全过后才授权 corpus/training。

E0.2 fixed multi-source batch lock：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_2_fixed_batch.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 同时排除 consumed trajectory 与 recording date，复算唯一固定的 3 dev +
3 heldout；只有 `E0_2_FIXED_BATCH_SOURCE_LOCK_VALIDATED` 才允许获取该 batch。

E0.2 fixed-batch ordered qualification：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_e0_2_fixed_batch_qualification.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-2-source-lock-20260801/source_lock.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-2-fixed-batch-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

runner 内部仍严格按 acquisition → transport → `.4 s` teacher → role opportunity
顺序执行；前门失败不运行后门。它不计算 `.8 s` 或 student。固定 batch 无 successor
expansion。

F0.1 SANPO official-test heldout one-shot 必须按下列顺序、canonical 路径逐步执行；
任一步失败即停止，尤其 consumption ledger、truth-join receipt 或 terminal-validation
receipt 出现后不得重跑对应阶段：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/materialize_stage_c_f0_1_heldout_package.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --f0 docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --mechanics docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --authority-cohort artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-authority-cohort-20260801/authority_cohort.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --training-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-student-training-validation-20260801/validation.json `
  --datasets-root artifacts.local/evidence/datasets `
  --authority-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-authority-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_f0_1_heldout_package.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --datasets-root artifacts.local/evidence/datasets `
  --package-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/predict_stage_c_f0_1_heldout.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --inference-inputs artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/inference_inputs.jsonl `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --datasets-root artifacts.local/evidence/datasets `
  --checkpoints-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-student-training-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_f0_1_heldout.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --truth artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/heldout_truth.jsonl `
  --prediction-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-effect-result-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_f0_1_heldout_result.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --truth artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/heldout_truth.jsonl `
  --prediction-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801 `
  --result artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-effect-result-20260801/result.json `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-terminal-validation-20260801
```

T0 short-path transport 只允许在合同固定的 outcome-open Development source 上执行。
合同及实现必须先提交推送并确认远端一致；不得用该 CLI 打开 fresh/reserved source：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_t0_sanpo_short_path_transport.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json `
  --transport-root artifacts.local/evidence/hftf/t0-short-path-transport-20260801 `
  --session-id 12b65d2c76d7ad0c17d7ac791089b8cae0bb059c9b02a6f23129044192bc93bb `
  --official-split train --start-frame 0 --target-fps 10 --frame-count 25 `
  --report-output artifacts.local/evidence/hftf/stage-c-t0-short-path-acquisition-20260801/acquisition.json

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_t0_sanpo_short_path_equivalence.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json `
  --candidate-root artifacts.local/evidence/hftf/t0-short-path-transport-20260801/r/50bce40f5469ad75 `
  --output artifacts.local/evidence/hftf/stage-c-t0-short-path-equivalence-20260801/equivalence.json
```

acquirer 在首个网络请求前验证 exact contract/source/root/config、自身 hash、G0
outcome-open role 与 canonical consumed package。validator 完全离线，逐帧验证 remote
object identity、本地 SHA/MD5、metadata、transport receipt 以及 final/`.tmp` 路径
`<240`。candidate manifest/spec hash 是 post-open transport receipt，不允许在合同中
预填。失败不重跑、不补 partial、不换源。

D2 metadata qualification 只读取 generation/SHA 绑定的 official-train split、
candidate `description.json` 与对象 receipts/listings；不读取 RGB/mask/depth bytes，
也不读取 `camera_poses.csv` 内容。合同、planner 与 planner test 必须先提交推送，且
CLI 在首个网络请求前验证三者 tracked、clean、hash-bound，并确认
`HEAD == origin/master`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d2_official_train_metadata.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_OFFICIAL_TRAIN_METADATA_QUALIFICATION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --output artifacts.local/evidence/hftf/stage-c-d2-official-train-metadata-qualification-20260802/qualification.json
```

planner 固定排除 78 个 burned/consumed/closed/reserved parents，按 official-train
`session_id` 升序选择前 6 个 metadata-eligible 新 parents。candidate 级请求经三次
内部 retry 后仍 404 或 metadata 不合法时写入 ineligible ledger 并继续；完整 split
不足 6 个即 `STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。扫描只执行一次，不追加或
替换 sources。合同、planner 与 planner test 必须来自同一 clean remote HEAD，
`--retries` 必须精确为 3；CLI 在首个网络请求前写入不可覆盖的 durable attempt
marker，失败或中断后也不允许重扫。成功只允许冻结下一份 media/mechanics 合同，
不直接授权媒体、pose 内容、teacher、student 或 D2 mechanics。

2026-08-02 的唯一 metadata scan 已以
`D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED` 锁定 6 条升序 official-train
parents。durable qualification SHA-256 为
`63a217c3e658bbe4fee9e351c5c9abf68379ec2ccb89a6c3449f1581e385ee47`；
独立审计重算 13 项 bindings、900 个媒体对象 receipts 与 18 个 modality receipt
hashes 后 `CLEAR`。这些 source 只在 metadata 层被打开并锁定；媒体与 pose 内容仍未
读取。不得重扫、追加或替换，下一步必须先冻结另一个 hash-bound one-shot
media/mechanics contract。

D2 mechanics 实现还必须绑定 D2.1 definition clarification。exact G0 不允许预先
过滤全局 theta/distance domain 外点；全部 admitted obstacle points 都对每个 cell
产生 signed proxy，nonmember 以正 closed-box SDF 参与 second-smallest。ground-aligned
rotation 使用 history/current forward 在 current ground tangent plane 上的最短有符号
角，并以 Rodrigues 绕 current up 延拓；predicted right 固定为
`cross(predicted_forward,current_up)`。每个 anchor 只读自身 history/current inputs，
其 0.4/0.8 s records 必须在处理后续 anchor 前 durable 写入。D2.1 JSON SHA-256 为
`51ed1c0bc2a98481b4991f237d44979cf0c455624031c2c0ee41715ec0d6a8f0`。

D2 六源媒体获取合同只物化 metadata scan 已锁定的 6 个 official-train Development
parents。正式 CLI 必须从 tracked、clean、pushed 的 exact contract/acquirer/test
及 SANPO network transport dependency 启动，并在首网前再次确认
`HEAD == origin/master`、固定 `--retries 3`、canonical root 不存在且 durable
attempt 可独占创建、完成 `flush + fsync`。source-blind 路径预检命令为：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_d2_six_source_media.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_SIX_SOURCE_MEDIA_ACQUISITION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --preflight-only `
  --preflight-output artifacts.local/evidence/hftf/stage-c-d2-six-source-media-path-preflight-20260802/preflight.json
```

已封存的 preflight 覆盖 1510 个 final/staging/downloader `.tmp` 内容路径，最大长度
173；它不联网、不读取媒体，也不创建 acquisition root。正式一次性命令为：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_d2_six_source_media.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_SIX_SOURCE_MEDIA_ACQUISITION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3
```

acquirer 逐项绑定 frozen generation/size/MD5；完整 pose CSV 校验后只把 13 个 selected
rows 写成独立 hash-bound pose slices，供后续 future-blind preprocessor 按 anchor
最小读取。RGB/mask/depth 不在获取阶段解码。任何 source 失败都只产生
`D2_MEDIA_ACQUISITION_NOT_EVALUABLE_NO_RETRY_NO_SOURCE_REPLACEMENT`，不得重跑、换源、
追加或 partial fill；成功也只允许另冻 mechanics execution contract，不直接授权
preprocessor、future truth、effect 或 student。

唯一一次正式获取已达到
`D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED`：254/254 下载请求均在 attempt 1
成功，6/6 source 原子发布。独立离线复算闭合 378 个 files、234 个媒体对象、
6 个 pose CSV 与 78 个 pose slices；媒体只做 hash/size/MD5，未解码，future truth
未打开。per-frame acquisition index SHA-256 为
`60e63e2df8b2813519e90a287b841dbcfa2b2c9a9b0765b1f10ebcf7c9c8b2a8`。
下一步只能先冻结并推送 mechanics execution contract，再运行 future-blind
preprocessor；不得直接打开 future truth。

D2 mechanics execution contract 绑定 D2/D2.1、完整 metadata qualification、已封存
六源媒体结果、per-frame acquisition index、G0/swept mechanics，以及 exact
common/preprocessor/evaluator/tests bytes。preprocessor 的 prediction root 必须原先
不存在，并在首次 current pose/media read 前写入 durable attempt；每个 anchor 的
points/prediction 必须在处理下一 anchor 前落盘。它只读 history/current pose 与 current
depth/mask，不读取 future truth：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/preprocess_stage_c_d2_future_blind.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D2_MECHANICS_EXECUTION_CONTRACT_2026-08-02.json `
  --output-root artifacts.local/evidence/hftf/stage-c-d2-future-blind-predictions-20260802
```

只有 completion 闭合 exact 42 anchor predictions、42 points 与 84 horizon records 后，
才允许启动一次 evaluator。evaluator 必须在首个 future pose/depth/mask read 前以
exclusive create + `flush + fsync` 写出 truth-join receipt；receipt 或既有 canonical
failure 任一存在都禁止第二次 truth join：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_d2_transport_effect.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D2_MECHANICS_EXECUTION_CONTRACT_2026-08-02.json `
  --output artifacts.local/evidence/hftf/stage-c-d2-transport-effect-result-20260802/result.json
```

任何 failure/中断均不重跑、不换源、不追加或同 cohort 调参。正终态也只授权另冻 RGB
student protocol，不授权训练或执行；研究主线、默认 App、Android、生产与 safety
权限保持关闭。

唯一一次 D2 mechanics 已以
`D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT` 封存。42 个
future-blind predictions 在 truth receipt 前全部 durable，truth join 精确执行一次并
产生 84 个 records；24 个 opportunity strata 只有 8 个通过，16 个失败 strata 全部
缺少冻结门要求的 5 个 known-risk cells，其中 3 个还同时低于 coverage 与 known-safe
门。UNKNOWN→SAFE 为 0。effect gates 因 opportunity 不足未获判定权限，不能从该
result 声称 transport 支持或不支持。该 cohort 不重跑、不换源、不追加、不调参；
RGB student 与全部主线/App/生产/safety 权限保持关闭。

D3-Q0 不重跑或替换 D2，而先构造 conditional challenge cohort。第一执行级仅允许
metadata-only roster：从 exact D2 exclusion union 继承 78 个 parents，再排除完整 D2
六源 cohort，按 official-train session ID 升序锁定前 40 个 metadata-eligible slots。
它不读取 RGB/mask/depth bytes、pose 内容、support 或 truth：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d3_q0_metadata_roster.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_METADATA_ROSTER_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --output artifacts.local/evidence/hftf/stage-c-d3-q0-metadata-roster-20260802/roster.json
```

contract/planner/test/D2 metadata helper 必须先 tracked、clean、hash-bound、推送并确认
`HEAD == origin/master`。durable attempt 在首网前 exclusive create + `fsync`；失败、
中断或 roster 不足均不重跑、不追加。成功也只允许另冻完整
reference-and-support qualifier、sealed-truth firewall 与 effect skeleton，不能直接
打开 40-slot media/truth。

metadata-only roster 已一次锁定为
`D3_Q0_METADATA_ROSTER_40_SLOTS_LOCKED`。后续筛选与 effect execution contract 现已在
任何 D3 slot media/support/truth 前冻结：40 slots 按原顺序逐个消费，failure 也消耗
slot，首 6 个四 strata 合格 source 立即停止；不允许 replacement、reorder、budget
expansion 或同 outcome 调门。

每次只执行唯一 next slot；runner 在首网前 durable 写入 attempt，只下载 1 个 pose
CSV 与 normalized `2..12` 的 depth/mask，RGB 为 0。它只计算两臂 support known 与
future truth，不计算两臂 clearance/effect。`aggregate_required=false` 时继续唯一
next slot；只有变为 true 时才运行 aggregator：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/run_stage_c_d3_q0_next_slot.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/aggregate_stage_c_d3_q0_screening.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json
```

aggregator 只读 closed selector/failure receipts，不读 sealed payload。只有 exact
first-six selection durable 后，才允许先生成 future-blind predictions，再一次性打开
selected-six sealed truth：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/preprocess_stage_c_d3_q0_selected_future_blind.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_d3_q0_sealed_effect.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json
```

preprocessor 只读 pose `0..8` 与 current/history depth/mask `2..8`，不读
future-only `9..12` 或 sealed payload。evaluator 必须在全部 42 predictions durable
后写入 open-once receipt，重算资格与 exact support equality，再复用未改变的 D2 effect
estimand/gates。任何 failure/interruption 均保留 partial artifacts，禁止 rerun 或换源。
selector/failure 必须反向绑定 durable slot attempt；aggregator 在首个 receipt read
前写 attempt。global screening、slot、aggregator、preprocessor 或 effect 的孤儿
`.tmp/.orphan/attempt` 只允许直接封存 no-rerun failure，不得重开已消费输入。

原 D3-Q0 正式 slot 1 已封存为 `D3_QUALIFICATION_INVALID_STOP`。runner 写出的
selector 把 `slot_attempt_sha256` 同时放在允许的 authority hashes 与禁止的顶层，
closed-schema 自检因此拒绝 admission。没有改 validator、重写 receipt 或重启媒体；
slot 1 永久 burned，forensic qualification terminal 不得进入 cohort。

当前所有上述 D3-Q0 命令均已终止授权。唯一后继是先冻结并推送全新的 schema-only
Q0.1 contract：只删除重复顶层字段，从原 slot 2 开始，最多使用剩余 39 slots；所有
资格门、effect gates、顺序、no-replacement/no-expansion 与 UNKNOWN 规则不变。新
contract、canonical root、tests 与独立审计完成前不得运行 slot 2。

Q0.1 冻结后使用同一受审计入口但必须传入新合同。第一次调用只写新 root 的 global
attempt 与 slot-1 carry-forward burn receipt，返回 `screening_initialized=true`，
不得访问媒体；第二次调用才执行原 slot 2：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/run_stage_c_d3_q0_next_slot.py `
  --contract docs/research/hftf/HFTF_STAGE_C_D3_Q0_1_SCHEMA_REPAIR_SCREENING_EFFECT_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3
```

carry-forward receipt 只绑定 Q0 protocol/roster/contract、invalid result 和
screening-invalid 的 opaque SHA-256；不得包含或读取旧 sealed payload/selector
outcome。后续 aggregator/preprocessor/evaluator 也必须使用同一 Q0.1 合同。

Q0.1 已按原顺序各一次消费完原 40-slot 预算：slot 1 仅 carry-forward burned，
slots 2–40 新开 39 个来源；qualified 为原 slots `3/14/20/29/37`，原 slots
`2/28` 因 current ground sample 不足封存为 execution failure，其余 32 个为合法
not-qualified selector。只有 5 个 qualified，未达到冻结的 first-six，aggregator
因此一次性关闭为
`D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_BUDGET_EXHAUSTED_NO_EXPANSION`。
`selection.json` 与 `formal/` 不存在；preprocessor/effect 未获授权、未运行。

这只表示当前冻结 roster/order/budget/gates 未形成 formal effect cohort，不表示
transport/HFTF 获支持或被否定。该 cohort 禁止重跑、补槽、替换、扩容或同 outcome
调门；后续只能在新的独立 protocol/data-role 边界下提出候选。

D4 改问 fresh source population 的 opportunity ecology/recruitability。M0 planner
不复用 Q0 的 first-40 控制流，而完整保序 ledger official-train 1560 IDs：全局 124
exclusion 中 118 个在 train 且零请求跳过，另 6 个属于 official test；剩余 1442 个
candidate 各做一次 description/pose-receipt/mask-depth-listing metadata 判定。RGB
listing、pose/media bytes、support/truth/effect 均关闭。

正式 one-shot 命令只可在 contract/planner/tests 提交推送、独立审计 `CLEAR` 且
canonical root 不存在后运行：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d4_m0_metadata_census.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D4_M0_METADATA_CENSUS_EXECUTION_CONTRACT_2026-08-02.json `
  --output-root artifacts.local/evidence/hftf/stage-c-d4-m0-metadata-census-20260802 `
  --retries 3
```

5 Hz pool manifest durable 后才允许 allocation attempt；其后只生成一次 32-byte OS
CSPRNG seed。`N<64` 在 seed 前停止；否则按冻结 rank 机械分配 ecology/effect/
unassigned。任何 partial root 后续只冻结 INVALID，不联网、不 resume、不重抽。

## 输出

只写入显式的 `artifacts.local/evidence/hftf/<run-id>/source_feasibility.json`。报告分别
裁决静态 metric projection、多高度身体包络教师、短时未来教师和独立 student-effect
评价，不把上一级可用性自动传递给下一级。本通用 H0 只可能准入静态 projection；
multi-height/future 需要后续 source-specific admission，student-effect 必须由 H2/H3
的独立 hash-bound parent-event ledger validator 裁决。

## 安全边界

这是 host-only `DEVELOPMENT_STANDARD` 审计。不训练模型，不读取 fresh/blind，不修改
Android、提醒或默认 App。合成深度/位姿派生结果只能叫 geometry-derived proxy；
没有独立人类事件真值时不得称为风险真值。

## 停止条件

报告产生下列一个终态即停止：

- `HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE`
- `HFTF_H0_SOURCE_FEASIBILITY_PARTIAL`
- `HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE`
- `HFTF_H0_1_POSE_MAPPING_ONLY`
- `HFTF_H0_1_SANPO_PROXY_FRAME_ADMITTED`
- `HFTF_H0_2_CANONICAL_PROXY_NOT_REPLICATED`
- `HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`
- `HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_NOT_EVALUABLE`
- `HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`

任何 blocker 只关闭相应 evidence instance。修复来源合同必须生成新输出路径，不覆盖
旧报告；不得靠默认行号、跨 session 时间差、自报事件数量或 session 改名补出
pose-frame binding、future span 或 effect eligibility。

## 假设与规则质疑

方向、距离、高度、人体包络、dynamic 与 uncertainty 都是历史 USTRF 的继承
primitive；不构成新颖性。唯一待证表示增量是 action-agnostic、history-only RGB 对
显式 short-future layered cells 的预测。falsifier 是：多高度或未来轴相对
single-height/current-field 没有独立增量，或 student 在相同事件账本与算力约束下不能
优于 incumbent。

## 失败资产复用

失败报告可作为数据来源缺口、pose/body-frame 合同、teacher leakage 与 evaluation
readiness 的 regression fixture；不能重包装为 HFTF 模型负结果、创新性结论或
unseen Confirmation。
