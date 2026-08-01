# BlindAssist 脚本索引

`scripts/` 根目录只保留跨领域稳定 Interface、当前共享 Implementation 和兼容入口；完成或冻结的研究 campaign 下沉到 `scripts/research/<domain>/`。调用方不应依赖研究子目录的内部布局。

## 稳定入口

- `run_host_research.ps1`：电脑端 CPU 进程池研究启动器；按本机实测解析 interactive/balanced/throughput 的 8/12/16 worker，当前 16 GiB 主机默认保留 4 GiB 系统内存并限制嵌套数值线程。只调度 host research，不改变科学参数，也不适用于 Android/边缘端。
- `run_guarded_host_research.ps1`：超过 3 分钟或正式 one-shot 的统一电脑端入口；先校验 hash 绑定性能收据和当前 RAM/VRAM，再启动 runner、注入已标定 worker、附加监控，并拒绝既存、缺字段、非完成或计数未闭合的 progress 终态；需要 isolated/no-bytecode 等解释器约束时，用 `-PythonArguments @('-I','-B')` 把参数放在 script 前。
- `run_dual_loop_d0_egomotion_error_attribution_r1.py`：burned single-capture D0-R1 的稳定 Adapter；只暴露依赖冻结、无权限 implementation lock、独立复核绑定的 activation、一次性 producer/analysis 与完全独立 execution validator，禁止读取 production A/B trace、旧 F-1B、Confirmation 或自动启动任何后继 canary。
- `run_dual_loop_d0_egomotion_error_attribution_r2.py`：D0-R1 consumed runtime failure 的新身份恢复 Adapter；固定 isolated Python、完整 dependency tree 与单消息 operational probe，只写全新 `run-r2`，科学算法、统计出口和 claim ceiling 不变。
- `run_dual_loop_d0_egomotion_error_attribution_r3.py`：D0-R2 consumed PyYAML failure 的最终 runtime-recovery Adapter；绑定独立 8-package environment、AST import closure、合成 producer/validator calibration parser smoke 与新 `run-r3`，科学算法、统计出口和 claim ceiling 不变。
- `run_dual_loop_target_local_background_warp_residual_r0.py`：目标局部背景 warp residual R0 的 B Development 离线入口；只暴露 truth-blind producer、truth-late evaluator、实现锁与合成回归测试，不接 C1/C2、Android 或产品路径。
- `run_dual_loop_radial_geometry_lite_r1.py`：target/track-conditioned causal radial geometry LITE Development R1 的稳定 Adapter；只暴露尺寸审计、无真值生产、实现锁校验及生产成功后的条件评估，旧 F-1B decision、Confirmation、Android 与产品路径不在输入或授权面。
- `run_dual_loop_radial_geometry_lite_r2.py`：R1 guarded-progress UTC 类型失败后的前向 R2 Adapter；科学实现与 R1 相同，但使用新 protocol/implementation/output identity，R1 output 不在输入面。
- `run_dual_loop_production_temporal_ab_input_preflight.ps1`：生产 `TemporalRiskTracker` 因子 A/B R0 的 outcome-blind 输入身份入口；逐一复核两个冻结 CrowdBot session 的 `4422` 个 PNG 哈希、时间顺序、尺寸与 canonical inventory，不读取 truth、不运行 detector 或候选 A/B。
- `run_dual_loop_production_temporal_ab.py`：同一路线的稳定 truth-blind producer validator 与条件 evaluator 入口；validator 逐帧绑定冻结 frame ledger，并发布包含 trace/producer/lock/activation/validation 哈希的 seal；`evaluate` 只接受该 seal 与 lock-bound truth-membership receipt。
- `run_dual_loop_production_temporal_ab_device.ps1`：指定 `SM-S9280 / SM8650` 的 build/install/input/prestart/formal/collect 真机入口；formal 需要 `ACTIVATED` receipt，候选 namespace 或远端 one-shot marker 已存在时 fail closed。
- `monitor_host_research_process.ps1`：电脑端长任务外部监控器；汇总根进程及递归子进程的 CPU/I/O/RSS、子进程数和可用的 NVIDIA GPU 利用率/显存/温度/功耗，把阶段、瓶颈提示、建议动作、疑似停滞和终态写入独立 JSONL/最新状态文件。若 runner 本身不发布完成单元数，它会明确将百分比与 ETA 留空。
- `validate_host_research_preflight.py`：校验长任务性能准入收据；缺少真实访问机制 pilot、有界耗时、结果等价、进度字段、资源预算或 formal one-shot 合同时返回 `PERFORMANCE_NOT_QUALIFIED`。
- `validate_information_ceiling_three_arm_d0.py`：信息上限三臂 D0 的稳定独立 validator 入口；从 canonical SANPO manifest、source masks/source regions 与设备逐帧 risk-input 账本复算三臂 identity、parent-event 指标和冻结终态。
- `run_research_contract_tests.py`：CI 与本地共用的无 GPU、无设备研究合同回归。
- `validate_research_protocol.py`：渐进式研究协议和 closure-scope overlay validator；当前默认 R4 采用论文优先的可修复/可重跑 Development、synthetic mapping/decoder canary 和算法选模/平台工程双类 benchmark，只有显式激活才进入最终 Confirmation，并按合同 ID 保留 R1/R2/R3 历史解析；同时校验 policy 最低诚信内核、identity/evidence、GATE、pending outcome、closure 引用/依赖和 question retirement。
- `evaluate_ustrf_sc_u0_teacher_upper_bound.py`：USTRF U0 六臂（四个正式比较臂 + uniform/shuffled route 负控）LOSO 事件评价；会重算完整 route-conditioned GPT/Codex 共识真值门并拒绝 blind/future/hash 漂移。
- `run_ustrf_sc_u0_candidate_bundle.py`：U0 六臂统一 subprocess runner；只向 adapter 暴露去标签 inference manifest，冻结 500ms cadence、真实 route control、逐 fold LOSO artifact/训练收据及 kernel backend，再组装可 admission 的 v2 bundle。
- `run_ustrf_sc_u0_android_baseline_adapter.py`：`baseline_yolo_geometry` 真实 Android adapter；只搬运 hash-bound 去标签 request/video/ledger/artifact/config，最终 output 由真机 instrumentation 内的 shipped TFLite YOLO + shared Kotlin kernel 生成。需先用稳定 `.android-home` debug key 安装 app 与 `device-benchmark` APK，且只能作 U0 评测 adapter，不授权 App/模型替换。
- `run_ustrf_sc_u0_android_bbox_route_adapter.py`：`detector_bbox_explicit_route` 真实 Android adapter；设备按 500ms frame 因果选取最新显式路线 sample，用冻结路线走廊筛选未改写的 YOLO bbox 后调用同一 shared kernel，并输出逐 sample/bbox 门控回执；host 独立重算几何。unknown route 空 gate，不能用于 App 运行时或模型替换。
- `generate_ustrf_sc_u0_dense_teacher_loso_artifact.py`：第三臂 WIP 的 label-free、fold-local dense teacher 校准入口；只接受训练 fold 的许可视频与帧清单并输出 deterministic auxiliary-only artifact/receipt，不读取 GPT/Codex 共识事件真值、blind、future 或 held-out 输入，也不授权 Android/App/生产使用。dense field v2 使用百万分之一 fixed-point、分离 source/route-interaction hash，并由 admission 从序列化 cell 重算摘要；四个真实 dense/control adapter 与正式 evidence 仍缺失，不代表第三臂完成。
- `research/ustrf_sc/four_arm_signal_probe.py`：冻结 15 正/15 等长负窗口、4594 帧的研究级连续分数探针；比较 bbox+matched route、同一 metric-depth dense field+matched/uniform/shuffled route 的配对排序，不选报警阈值、不运行 tracker/TTC/lifecycle，并可用 `--validate-existing` 在独立进程重算全部结果。
- `research/ustrf_sc/bbox_route_attribution_probe.py`：dense 分支停止后的最小 bbox-route 归因探针；严格复用父 A arm 的 4108 个 common-eligible frame，将同一冻结 bbox confidence field 接到 matched/uniform/within-source shuffled/bbox-only，只比较 q50/q90/q95 正负配对排序与 matched 直接增量，并可用 `--validate-existing` 全量复算。
- `generate_sanpo_counterfactual_capture_plan.py --pilot`：生成固定 1 session × 5 scene × 1 matched pair 的 10-episode 空试采槽位；只描述采集责任，不生成证据或标签。
- `validate_ustrf_sc_route_conditioned_event_pilot.py`：试采链路审计；重算 video/clock/frame ledger/route 投影因果绑定及独立双审/裁决，只能输出 pipeline audit，永不产生 truth、U0、训练或运行时授权。
- `validate_ustrf_sc_model_proxy_event_pilot.py`：大模型生成的 5 场景/10 episode 代理 pilot 稳定审计入口；重算 contact sheet、1000 个解码帧、生成 lineage、两次隔离模型 review 及原始输出绑定。通过只允许扩正式代理矩阵，`human_truth/U0/training/runtime/production` 始终为 false。
- `validate_ustrf_sc_arcore_frame_bound_canary.py`：独占 ARCore `Session` 单 `Frame` 设备 canary 的 host 重算入口；从逐帧 JSONL 重算 Camera2 时间戳、camera image、raw depth/confidence、tracking 与持久 Anchor，门关闭以 exit 2 和 `FREEZE_FRAME_BOUND_METRIC_GEOMETRY` 收口。
- `validate_ustrf_sc_u0_prediction_bundle.py`：U0 预测证据 admission；重算 runner/registry、去标签输入、逐 fold LOSO provenance、route control、adapter request/output 与逐帧 shared-kernel trace，只允许从 feedback receipt 派生提醒；Android dense/control 臂另强制 teacher 许可证/权重/实现、fold、field、unknown 与归一化算术 receipt，当前缺真实 adapter 时 fail closed。
- `validate_ustrf_sc_capture_frame_ledger.py`、`validate_ai_review_receipt.py`：正式 full-matrix truth 与 pilot 共用的帧证据和 GPT/Codex receipt 验证 Implementation；由上述稳定入口调用。
- `validate_ustrf_sc_device_metric_geometry.py`：同设备米制几何总门；完整包要求五类 typed artifact 与设备/mount/calibration/metrics 精确绑定并继续 hash-bind raw/gate source，`blocked/in_progress` 包也会审计已有收据；通过只授权 isolated geometry shadow。
- `run_research_tool.py <domain> <tool.py> [args...]`：统一研究 Adapter；当前支持 `egomotion-compensated-looming` RCLE 几何准入/transport、`public-video` 历史归档、`ustrf-crosscam-codex` 代理评测、`ustrf-sensor-replay` 多来源 RGB-D+pose 回放和 `ustrf-route-target-evidence-closure` route-target 证据闭环域。RCLE 外部合同只绑定这个 root adapter，不直接绑定研究内部脚本路径。
- `run_public_video_campaign_tests.py`：发现并运行 `scripts/research/public_video/` 的完整测试集。
- `run_public_video_edge_inference.ps1`：已冻结 campaign 真机闭环的稳定 Adapter；调用方不依赖研究目录内部路径。
- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_project_structure.ps1` / `test_check_project_structure.ps1`：脚本根 allowlist、开发日志预算、研究 Module 合同、内部路径和跨 Module import 门禁；仓库卫生检查会自动调用它。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档、research domain README/index 与本地链接门禁。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。
- `run_npu_candidate_acceptance.ps1`：SM-S9280/SM8650 上的独立 NPU 候选安装、QNN HTP runtime marker、正式包/数据不变式与候选专属卸载回滚门；不清除正式 App 数据。
- `generate_qnn_preprocess_candidate.py`：生成并自校验隔离的 QNN 预处理候选；只写入 `artifacts.local/experiments/qnn-preprocess-fusion-v1/`，不修改 App assets，也不构成发布、默认路由或生产授权。

## 领域模块

- [`research/public_video/`](research/public_video/)：已冻结的公开视频 / public-silver 历史 campaign。细粒度语义索引和迁移说明保留在该目录，不再向根目录增加实验轮次脚本。
- [`research/ustrf_crosscam_codex/`](research/ustrf_crosscam_codex/)：公开头戴视角视频上的 Codex provisional silver / causal comparator，以及显式 route-projection receipt、polygon bottom-center 三档不确定性审计；不产生客观传感器事实、真人用户效果、设备米制几何或 U0/生产授权。
- [`research/ustrf_detector_taxonomy_coverage/`](research/ustrf_detector_taxonomy_coverage/)：已关闭的 detector taxonomy / target-attribution campaign，以及历史 REveL detector 失败分类与灵敏度诊断；仅保留 benchmark 复核价值，不重开 detector、shadow、H2 或生产授权。
- [`research/ustrf_route_target_evidence_closure/`](research/ustrf_route_target_evidence_closure/)：route-target 候选盲真值、指标资格、receipt-aware replay、逐指标 L1 profile、observability/JRDB source audit 与单变量 lifecycle 机制诊断。JRDB cross-sequence replication 已冻结 3 个新 sequence × 120 帧并以原 PCD/oriented-box/四类 kernel 复算，pooled object/pair support 为 `83.08% / 80.81%`，但远距仅 1/3 可评，仍无 selection、route/event、shadow/H2 或生产权限。
- [`research/egomotion_compensated_looming/`](research/egomotion_compensated_looming/)：RCLE canonical Module；当前阶段、终态、执行权限与下一步只以 [RCLE current 入口](../docs/research/rcle/README.md) 为准，本索引不复制动态结论。
- [`research/dual_loop_radial_geometry_lite_r0/`](research/dual_loop_radial_geometry_lite_r0/)：双环 successor 的 REveL Development 输入冻结、truth-only natural-event ledger 与后续 causal replay Module；旧 F-1B decision 输出不在输入面。
- [`research/dual_loop_radial_geometry_lite_r1/`](research/dual_loop_radial_geometry_lite_r1/)：R0 首次尺寸漂移失败后的单变量 Development 修复；原生解码尺寸不同时两臂统一 `FRAME_SHAPE_CHANGE` 弃权，不进行任何重采样或跨帧桥接。
- [`research/dual_loop_radial_geometry_lite_r2/`](research/dual_loop_radial_geometry_lite_r2/)：R1 guarded-progress UTC `Z` 解析失败后的 execution-envelope successor；只重置 evidence identity/namespace，不改变双臂或科学门。
- [`research/dual_loop_production_temporal_ab/`](research/dual_loop_production_temporal_ab/)：既有生产 object-detector temporal geometry 的因子 A/B Module；当前只完成 outcome-blind 设备输入身份预检，候选 producer/evaluator、正式执行和 Confirmation 仍由冻结合同逐级授权。
- [`research/dual_loop_jrdb_shadow_r0/`](research/dual_loop_jrdb_shadow_r0/)：标注条件化真实 LiDAR 质心到现有 `DualLoopShadowAdmitter`/`AssistDecisionKernel` 的 host-only 工程闭环；行为源保持 `OBJECT_DETECTOR`，标注条件化仅由 `REPLAY_TIMELINE + DualLoopTargetProvenance.REPLAY_ANNOTATION` 留痕，生产 allowlist 保持为空。
- [`research/dual_loop_depth_geometry_r0/`](research/dual_loop_depth_geometry_r0/)：burned REveL 固定子集上的 Depth Anything V2 target-depth Discovery；静态距离排序与 temporal direction 分开评价，不授权 runtime/Android。
- [`research/dual_loop_global_motion_compensated_flow_r0/`](research/dual_loop_global_motion_compensated_flow_r0/)：LITE R2 后的 background-homography residual target-flow Development；固定质量门和原 469-event readiness，不做结果后阈值搜索。
- [`research/dual_loop_target_local_background_warp_residual_r0/`](research/dual_loop_target_local_background_warp_residual_r0/)：目标局部背景环 similarity warp residual 的 B Development offline producer/evaluator；固定 R1–R4 选择、truth-late join 和实现锁，不接 Android 或产品路径。
- [`research/dual_loop_causal_track_tristate_r0/`](research/dual_loop_causal_track_tristate_r0/)：最小因果框尺度三态 source；7 帧 log-box-height 严格同号才确认/否定接近，否则弃权。独立 JRDB annotation-track Confirmation 已通过，并移植到 Android 非干预 shadow；不授权 active 提醒。
- [`research/dual_loop_multitrack_counterfactual_r0/`](research/dual_loop_multitrack_counterfactual_r0/)：使用完整 production detections 检验多目标历史能否为误提醒提供反证；R0 已被零负例触发点命中否决，并产出最小 scene-scale R1 与设备逐帧 parity evaluator。
- [`research/dual_loop_scene_scale_cross_source_r1/`](research/dual_loop_scene_scale_cross_source_r1/)：冻结 scene-scale contradiction 在 Matoaka 10,724 帧上的 truth-late 跨来源 Development 回放；只复现行级抑制，不产生事件级、Confirmation、默认生产或安全结论。
- [`research/dual_loop_r1_event_failure_decomposition_r0/`](research/dual_loop_r1_event_failure_decomposition_r0/)：只消费已关闭 Development trace、truth ledger 与 receipt 的 R1 post-terminal 逐窗口失败分解和 Development-only upper-bound audit；不重跑、不调阈值、不实现 R2。
- [`research/dual_loop_unseen_natural_event_r0/`](research/dual_loop_unseen_natural_event_r0/)：按输出盲 metadata registry、冻结事件窗、baseline adequacy 与同事件延迟门，评价固定 `039757b` 在全新自然视频上的事件级 canary；单来源不作总体外推。
- [`research/dual_loop_semantic_refresh_q0/`](research/dual_loop_semantic_refresh_q0/)：固定模型全频参考下的事件保持型语义刷新调度 Q0 R0/R0.1 离线评测；R0.1 补齐 risk-episode 对齐、signed delay 和约束型 operating point，每个 arm 隔离 cache/feedback/event state，不接 Android、能效、产品或安全路径。
- [`research/dual_loop_segmentation_technical_smoke/`](research/dual_loop_segmentation_technical_smoke/)：YOLO + semantic segmentation 双环的单一 reference technical smoke；只验证 RGB 输入/输出合同、有限值、类别分布、主机耗时和可视化，不产生模型比较、融合、A/B/C 或 Android 权限。
- [`research/dual_loop_segmentation_complementarity/`](research/dual_loop_segmentation_complementarity/)：固定 image-space YOLO/segmentation A/B/C Development mechanism diagnostic；支持同模型 host YOLO trace 适配、按 session-first 汇总 class-wise uncovered fraction、时序稳定性和 host cost，独立 validator 复算。Shiraz/Shanghai 已观察到跨来源图像空间信号，但不产生风险、事件、设备 parity、Android 或生产结论。
- [`research/dual_loop_segmentation_candidate_utility/`](research/dual_loop_segmentation_candidate_utility/)：冻结并执行 source-native pixel/component utility R0；用 SANPO v4 dev/blind truth 比较 YOLO-only、segmentation-only、union 三臂，独立复算 candidate recall、false activation、raw/motion-warped temporal stability 和 host cost；不授权 Android、QNN、风险事件或主动提醒。
- [`research/dual_loop_segmentation_model_selection/`](research/dual_loop_segmentation_model_selection/)：已封存的 DDRNet-23-Slim / SegFormer-B0 R1 模型选择 Module；终态为 `SEGMENTATION_MODEL_SELECTION_R1_BLOCKED / MODEL_SELECTION_NOT_EVALUABLE`，禁止修复或重跑，已消费 fresh 仅可作 regression/rehearsal/validator。
- [`research/dual_loop_segmentation_r2_p0/`](research/dual_loop_segmentation_r2_p0/)：R1 blocked 后的 candidate-qualification readiness Module；冻结完整 native→canonical decoder/mapping、SHA-closed materialized view、synthetic/consumed rehearsal、逐帧逐阶段 runtime rows 和一次 36 点 DDRNet refinement。当前终态为 `R2_NOT_WORTH_BURNING_FRESH_HOLDOUT`，未选择或读取新 fresh truth，不授权 R2、device、Android、风险事件或提醒。
- [`research/dual_loop_segmentation_failure_atlas/`](research/dual_loop_segmentation_failure_atlas/)：当前科学主线的 200-frame pilot 与固定 320-frame dev/consumed 定向扩展；输出组件级失败机制、session/role 覆盖、pilot-vs-expansion 排序、非组合式空间/因果时序/置信 gating、三态 residual 可标注性及固定成功/失败案例图。host-only visual sidecar 另以 `VISUAL_CANDIDATE_ONLY / drives_alerts=false` 展示 YOLO boxes、raw heatmap、候选、通过/拒绝/abstain 与原因。当前终态 `GATING_PARTIAL`，不训练模型、不访问 fresh holdout、不接 Android、risk 或默认 App。
- [`research/dual_loop_segmentation_conditional_gating/`](research/dual_loop_segmentation_conditional_gating/)：Atlas `GATING_PARTIAL` 的独立 Development successor；R0 primary 一次执行与 85,235 项检查已 `VALID`，但 FP reduction `0.092572`、最低 session recall retention `0.774580` 未过门。历史 machine terminal 不变，scope 纠正为 primary-only；前向 R0.1 两个 diagnostic-only shadows 又通过 167,327 项检查，但均无 material 或 session heterogeneity，故只关闭这个精确三臂静态手工门家族，并把主边界转入后续 FP-aware DDRNet R0；held-out 只作 burned Development stress，不接 Android、risk、alerts 或默认 App。
- [`research/dual_loop_segmentation_residual_aware_ddrnet/`](research/dual_loop_segmentation_residual_aware_ddrnet/)：静态手工门家族关闭后的单一 FP-aware DDRNet Development successor；保持 backbone、四类真值、loss、70% hazard-guided crop、三 seed 和训练预算不变，只把 30% unguided full-frame 分支改为同 seed train-only baseline FP-pixel 加权的 full-frame draw。三组 same-seed pair 均未通过九门，validator 复推六个 checkpoints 后终态为 `FP_WEIGHTED_SAMPLING_NOT_SUPPORTED`；停止该 sampler，不接 INT8、runtime、Android、risk、alerts 或默认 App。
- [`research/dual_loop_segmentation_learned_component_validator/`](research/dual_loop_segmentation_learned_component_validator/)：Failure Atlas 后的轻量 learned-component Development validator；固定 21 列 current/past 因果特征、唯一 Logistic Regression、10 个 source-session nested LOSO、训练上下文内 operating-point 选择、九项 utility 门和 host latency/有界内存门。11,757 个 held-out predictions 与 terminal 已由 validator 复算为 `VALID`；候选只过 4/9 utility 门，host P95 `9.376145 ms` 也失败，终态 `NOT_SUPPORTED_AND_GATING_STOP`。关闭当前 active learned gating，不访问 fresh、不换分类器/feature subset，不产生训练、Confirmation、Android、risk/feedback 或提醒权限。
- [`research/riskseg_r0_event_eval/`](research/riskseg_r0_event_eval/)：RISKSEG-R0 的 output-blind event-eval 数据门 Module；生成全窗口 RGB contact sheets，绑定 RGB/source-mask hashes，并在两路隔离 review 与必要裁决后验证 `8/8/7/7` 四桶、session-disjoint、窗口不重叠及 source-session 配额。当前数据门为 `HOLD_EVENT_EVAL_DATA`，未冻结 truth，不启动 PIDNet 预检或训练。
- [`research/dual_loop_dg_srf_structural_complementarity_f0/`](research/dual_loop_dg_srf_structural_complementarity_f0/)：已完成的新因果变量 DG-SRF F0；固定 Depth Anything V2 Small 单帧相对逆深度、逐帧尺度、输出健康门、`N/E/R+/R-`、surface trend、YOLO residualization、LOSO maximin 九门和独立 reference validator。520 帧输出全部健康，29,031 项独立复算有效，但 D1-D4 均无跨组 stable signal、D4 只在 1/10 组优于最佳单信号、九门只过 4/9，终态 `STRUCTURAL_SIGNAL_NOT_SUPPORTED_STOP`。关闭该精确定义的 F0，不产生 F1、未知障碍、米制深度、路线、事件、Android/QNN/A568、产品或安全权限。
- [`research/information_ceiling_three_arm_d0/`](research/information_ceiling_three_arm_d0/)：同一 90-frame / 3-parent-event SANPO consumed Development cohort 上的 current YOLO、mask-derived truth boxes 与 source-mask current-adapter 三臂设备审计；独立 validator 复算逐帧输入和事件终态。结果只支持当前链的路线诊断，不把 mask adapter/source-policy 增益冒充纯 bbox 上限。
  - [`research/hftf/`](research/hftf/)：独立 HFTF 候选支线的 source/teacher feasibility Module；通用 H0 只准入静态 metric projection，source-specific verifier 进一步认证 official loader/GCS pose-frame binding、metric-depth transform 与 per-frame local-ground proxy，并由 cohort aggregator 要求至少三个独立 session 复现；当前只授权 H1 geometry teacher canary，不训练 student、不修改主线或默认 App。
- [`research/central_obstruction_agent_label_readiness_d0a/`](research/central_obstruction_agent_label_readiness_d0a/)：中央图像阻塞 D0-A0 reuse-first 输入宇宙、D0-A1 排除式 calibration lock，以及 D0-A successor 的 observation-only 双 pass fixed-clip 转换与 validator；successor 终态 `CENTRAL_OBSTRUCTION_AUXILIARY_FEATURE_ONLY`，D0-A2/D0-A3/A4/模型效果关闭。
- [`research/common/`](research/common/)：至少两个研究域真实复用的共享 Implementation；领域规则和授权不得进入该 Module。
- [研究 Module 模板](research/README.md)：新路线必须声明稳定 Interface、输出、安全边界与停止条件。
- 根目录 USTRF-SC、SANPO 数据治理、训练与 benchmark 脚本：当前仍共享少量 SANPO 模型/证据 helper，待形成独立稳定 Interface 后再按域下沉，禁止为追求目录外观一次性拆断依赖网。

## 模型、设备与数据约定

- 模型导出/检查：`export_yolo11n_tflite.py`、`inspect_tflite.py`、`export_depth_anything_v2_tflite.py`、`inspect_depth_model.py`。
- detector/device benchmark：`detector_lab.py`、`benchmark_tflite_detectors.py`、`run_yolo26n_device_benchmark.ps1`、`run_detector_ab_device_benchmark.ps1`、`run_device_regression.ps1`。
- SANPO 训练与门禁：`train_sanpo_segmentation_keras_torch.py`、`train_export_sanpo_segmentation.py`、`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`。
- 当前论文研究主线以 [双环 current 入口](../docs/research/dual-loop/README.md) 为准，并
  前向服从 `THESIS_FIRST_RESEARCH_GOVERNANCE_R4`；暂停的 RCLE 与已关闭 USTRF 保留
  历史终态，但未来新 Development 不自动继承旧 one-shot/formal 门。

## 运行约定

- 从仓库根目录执行；通用 Python 优先使用 `E:\codex-tools\bin\blindassist-python.cmd`。
- 电脑端多进程算法研究优先通过 `run_host_research.ps1` 选择 `interactive`、`balanced` 或 `throughput`；具体规则见 [HOST_RESEARCH_COMPUTE.md](../docs/HOST_RESEARCH_COMPUTE.md)。
- 上述直接入口用于可逆短开发循环；正式 one-shot/不可逆 claim、预计超过 15 分钟、
  高 I/O/内存/设备风险，或轻量 pilot 无法给出运行上界的新任务必须通过
  `run_guarded_host_research.ps1`。3–15 分钟的可逆任务只需轻量 timeout、进度和
  scoped-output 合同。
- 超过数分钟的电脑端研究任务必须有可独立读取的进度状态；对当前不支持进度协议的冻结 runner，可用 `monitor_host_research_process.ps1` 监测既有 PID，不得用重启来换取可见性。
- 下载、数据集、benchmark、训练输出和临时文件进入忽略的 `artifacts.local/`。
- 需要联网、GPU 或 ADB 的脚本必须显式说明；设备脚本运行前确认目标设备与 module。
- 新增 runnable script 必须归入明确领域 Module，并同时更新本索引或该领域 README；不要重新把研究轮次平铺回根目录。
- `scripts/policy/root-files.txt` 是根目录精确清单；只有真正稳定的 Interface 或跨域共享 Implementation 才能经评审加入。
- `DEVELOPMENT_LOG.md` 上限为 6000 行、1200000 bytes、最老条目 28 天；超限时按月原文归档并在根日志保留链接。
- SANPO 普通论文训练先按 [SANPO 训练协议](../docs/SANPO_TRAINING_PROTOCOL.md)选择
  `THESIS_DEVELOPMENT`；只有改变 production canonical 数据、读取 blind 或启动默认模型
  晋级时，才进入 `PRODUCTION_PROMOTION` 的完整隔离与门禁。
- 文档变更完成前运行 `scripts/check_docs_index.ps1`。
