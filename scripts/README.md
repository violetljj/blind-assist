# BlindAssist 脚本索引

`scripts/` 根目录只保留跨领域稳定 Interface、当前共享 Implementation 和兼容入口；完成或冻结的研究 campaign 下沉到 `scripts/research/<domain>/`。调用方不应依赖研究子目录的内部布局。

## 稳定入口

- `run_host_research.ps1`：电脑端 CPU 进程池研究启动器；按本机实测解析 interactive/balanced/throughput 的 8/12/16 worker，当前 16 GiB 主机默认保留 4 GiB 系统内存并限制嵌套数值线程。只调度 host research，不改变科学参数，也不适用于 Android/边缘端。
- `run_guarded_host_research.ps1`：超过 3 分钟或正式 one-shot 的统一电脑端入口；先校验 hash 绑定性能收据和当前 RAM/VRAM，再启动 runner、注入已标定 worker、附加监控，并拒绝既存、缺字段、非完成或计数未闭合的 progress 终态。
- `run_dual_loop_radial_geometry_lite_r1.py`：target/track-conditioned causal radial geometry LITE Development R1 的稳定 Adapter；只暴露尺寸审计、无真值生产、实现锁校验及生产成功后的条件评估，旧 F-1B decision、Confirmation、Android 与产品路径不在输入或授权面。
- `monitor_host_research_process.ps1`：电脑端长任务外部监控器；汇总根进程及递归子进程的 CPU/I/O/RSS、子进程数和可用的 NVIDIA GPU 利用率/显存/温度/功耗，把阶段、瓶颈提示、建议动作、疑似停滞和终态写入独立 JSONL/最新状态文件。若 runner 本身不发布完成单元数，它会明确将百分比与 ETA 留空。
- `validate_host_research_preflight.py`：校验长任务性能准入收据；缺少真实访问机制 pilot、有界耗时、结果等价、进度字段、资源预算或 formal one-shot 合同时返回 `PERFORMANCE_NOT_QUALIFIED`。
- `run_research_contract_tests.py`：CI 与本地共用的无 GPU、无设备研究合同回归。
- `validate_research_protocol.py`：渐进式研究协议和 closure-scope overlay validator；区分 discovery warning 与 confirmation error，校验 policy 最低诚信内核、可复算的仓库 identity/evidence JSON、可执行 GATE、pending outcome、closure 引用/依赖和 question retirement，并拒绝结果后原地改门、INVALID 扩大关闭科学问题及无理由 Cartesian sweep。
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
- [`research/common/`](research/common/)：至少两个研究域真实复用的共享 Implementation；领域规则和授权不得进入该 Module。
- [研究 Module 模板](research/README.md)：新路线必须声明稳定 Interface、输出、安全边界与停止条件。
- 根目录 USTRF-SC、SANPO 数据治理、训练与 benchmark 脚本：当前仍共享少量 SANPO 模型/证据 helper，待形成独立稳定 Interface 后再按域下沉，禁止为追求目录外观一次性拆断依赖网。

## 模型、设备与数据约定

- 模型导出/检查：`export_yolo11n_tflite.py`、`inspect_tflite.py`、`export_depth_anything_v2_tflite.py`、`inspect_depth_model.py`。
- detector/device benchmark：`detector_lab.py`、`benchmark_tflite_detectors.py`、`run_yolo26n_device_benchmark.ps1`、`run_detector_ab_device_benchmark.ps1`、`run_device_regression.ps1`。
- SANPO 训练与门禁：`train_sanpo_segmentation_keras_torch.py`、`train_export_sanpo_segmentation.py`、`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`。
- 当前研究主线以 [RCLE current 入口](../docs/research/rcle/README.md) 为准，并服从 [渐进式研究治理](../docs/RESEARCH_GOVERNANCE.md)。旧 route-conditioned USTRF 和已关闭 RCLE 版本均保留为历史、反例、回归或前序证据，不自动产生当前 authority。

## 运行约定

- 从仓库根目录执行；通用 Python 优先使用 `E:\codex-tools\bin\blindassist-python.cmd`。
- 电脑端多进程算法研究优先通过 `run_host_research.ps1` 选择 `interactive`、`balanced` 或 `throughput`；具体规则见 [HOST_RESEARCH_COMPUTE.md](../docs/HOST_RESEARCH_COMPUTE.md)。
- 上述直接入口仅用于可逆短开发循环；预计超过 3 分钟或正式 one-shot 的新任务必须通过 `run_guarded_host_research.ps1`。
- 超过数分钟的电脑端研究任务必须有可独立读取的进度状态；对当前不支持进度协议的冻结 runner，可用 `monitor_host_research_process.ps1` 监测既有 PID，不得用重启来换取可见性。
- 下载、数据集、benchmark、训练输出和临时文件进入忽略的 `artifacts.local/`。
- 需要联网、GPU 或 ADB 的脚本必须显式说明；设备脚本运行前确认目标设备与 module。
- 新增 runnable script 必须归入明确领域 Module，并同时更新本索引或该领域 README；不要重新把研究轮次平铺回根目录。
- `scripts/policy/root-files.txt` 是根目录精确清单；只有真正稳定的 Interface 或跨域共享 Implementation 才能经评审加入。
- `DEVELOPMENT_LOG.md` 上限为 6000 行、1200000 bytes、最老条目 28 天；超限时按月原文归档并在根日志保留链接。
- 改变 canonical 数据、冻结回归集或读取 blind 数据时遵守 [SANPO 训练协议](../docs/SANPO_TRAINING_PROTOCOL.md)。
- 文档变更完成前运行 `scripts/check_docs_index.ps1`。
