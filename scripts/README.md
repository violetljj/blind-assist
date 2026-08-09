# BlindAssist 脚本索引

`scripts/` 根目录只保留跨领域稳定 Interface、当前共享 Implementation 和兼容入口；完成或冻结的研究 campaign 下沉到 `scripts/research/<domain>/`。调用方不应依赖研究子目录的内部布局。

研究脚本冷启动先读 [`research/REGISTRY.md`](research/REGISTRY.md)；HFTF/DepthART 再读
[`research/hftf/INDEX.md`](research/hftf/INDEX.md)。这些索引只描述职责和路径，不复制动态状态。

## 稳定入口

- `run_host_research.ps1`：电脑端 CPU 进程池研究启动器；按本机实测解析 interactive/balanced/throughput 的 8/12/16 worker，当前 16 GiB 主机默认保留 4 GiB 系统内存并限制嵌套数值线程。只调度 host research，不改变科学参数，也不适用于 Android/边缘端。
- `run_model_matrix.py`：manifest 驱动的历史模型/基线离线矩阵入口；统一逐帧 trace、model/config/dataset identity、artifact 引用、进度和 resume。RISKSEG-R0 已完成设备 trace 优先以 replay 接入，不自动重跑设备实验；详细合同见 `research/model_matrix/README.md`。
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
- `run_research_tool.py candidate-event-mining <tool.py> [args...]`：长视频/公开数据候选事件自动挖掘的稳定 Adapter；只做 canonical frame trace、窗口发现、去重/聚类、candidate-blind Luna review bundle 和 discovery candidate pool，不产生事件真值或生产权限。
- `run_research_tool.py candidate-event-mining run_real_video_batch.py`：真实公开视频的 bounded host batch adapter；按固定 cadence 物化 review frames，运行 YOLO11n + Depth Anything V2，并把 segmentation/HFTF 的可选 sidecar 或明确标注的 image-space proxy 接到 canonical trace。它只产生 `THESIS_DEVELOPMENT` discovery 输入，不把 proxy 当作 segmentation 模型，也不授予事件/安全/生产权限。
- `run_research_tool.py candidate-event-mining acquire_wikimedia_candidates.py`：从公开 Wikimedia Commons API/原文件 URL 下载并登记候选源，写入 `F:\ba-data\blindassist-candidate-event-mining\` 的 source records 与 project index；不把许可证 receipt 等同于事件真值。
- `run_research_tool.py candidate-event-mining run_segmentation_sidecar.py` / `run_hftf_sidecar.py` / `attach_sidecars.py`：分别运行真实 SegFormer、现有 HFTF student，以及按 `source_id × session_id × frame_index` 做 hash-bound post-inference join；缺失通道不补零，模型输出仍只作为 discovery signal。
- `run_research_tool.py hftf <tool.py>`：HFTF Module 的稳定仓库入口；当前 D6 SANPO real-veto materialization、frozen-threshold export 与 30-event ranking evaluator 均通过该 Adapter 运行，输出仅写入 `artifacts.local/evidence/hftf/`。
- `run_research_tool.py candidate-event-mining select_review_queue.py`：全量候选报告的确定性 review-budget 选择器；按 source×taxonomy 与 cluster 覆盖选取有限窗口，未选候选保持 `not_reviewed`，不会被静默当作负例。
- `run_research_tool.py candidate-event-mining merge_candidate_pools.py`：合并不重叠的已复核 batch，拒绝 pool/queue ID 漂移、批次重叠和未覆盖母报告候选。
- `run_research_tool.py candidate-event-mining register_run_index.py`：把 adapter、全量候选报告、review queue、bundle、Luna receipts 和 candidate pool 的路径/hash 追加到 `F:\ba-data\blindassist-candidate-event-mining\run_index.json`。
- `run_public_video_campaign_tests.py`：发现并运行 `scripts/research/public_video/` 的完整测试集。
- `run_public_video_edge_inference.ps1`：已冻结 campaign 真机闭环的稳定 Adapter；调用方不依赖研究目录内部路径。
- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试；默认只检查卫生，使用 `-IncludeStructure` 才串联项目结构门。
- `check_project_structure.ps1` / `test_check_project_structure.ps1`：脚本根 allowlist、开发日志预算、研究 Module 合同、内部路径和跨 Module import 门禁；需要结构/政策验证时单独运行，避免与卫生门重复执行。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档、research domain README/index 与本地链接门禁。
- `audit_research_structure.ps1`：只读输出研究 Module 合同、HFTF 角色计数和 support 迁移清单。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。
- `run_npu_candidate_acceptance.ps1`：SM-S9280/SM8650 上的独立 NPU 候选安装、QNN HTP runtime marker、正式包/数据不变式与候选专属卸载回滚门；不清除正式 App 数据。
- `generate_qnn_preprocess_candidate.py`：生成并自校验隔离的 QNN 预处理候选；只写入 `artifacts.local/experiments/qnn-preprocess-fusion-v1/`，不修改 App assets，也不构成发布、默认路由或生产授权。

## 研究模块

研究模块不在本页重复列出实验轮次、动态终态或指标。按以下入口定位：

- [研究职责总表](research/REGISTRY.md)：当前入口、部署、诊断、archive 与 support 分区。
- [全部 Module 索引](research/MODULE_INDEX.md)：67 个研究 Module 的逐项链接和稳定研究族分类。
- [HFTF / DepthART 角色索引](research/hftf/INDEX.md)：`roles.json` 机器匹配规则和迁移合同。
- [HFTF support 迁移队列](research/hftf/SUPPORT_MIGRATION_QUEUE.md)：按主题簇安全下沉历史文件。
- [候选事件挖掘](research/candidate_event_mining/README.md)：discovery-only 候选池。
- [公共实现](research/common/README.md)：跨域复用的共享 implementation。
- [双环与其他历史模块](research/README.md)：只提供模块合同，不复制动态研究结论。

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
