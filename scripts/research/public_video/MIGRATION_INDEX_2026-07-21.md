# Public-video campaign 迁移前语义索引

> 这是 2026-07-21 目录迁移前的完整语义快照，用于保留各实验脚本的职责与安全边界。campaign Implementation 现位于本目录；稳定调用入口见 [README](README.md)。下文中未带路径的 public-video/public-silver 文件名均指本目录。

## 仓库、发布与环境

- `check_repo_hygiene.ps1` / `test_repo_hygiene.ps1`：仓库卫生门禁与测试。
- `check_docs_index.ps1` / `test_check_docs_index.ps1`：顶层文档索引与本地链接门禁。
- `run_research_contract_tests.py`：本地与 CI 共用的无 GPU、无设备、无可选科学依赖研究合同回归入口。
- `archive_apk.ps1`、`verify_release_apk.ps1`、`verify_apk_16kb.ps1`：APK 校验与归档。
- `restore_codex_skills.ps1`：恢复仓库内 skills 快照。

## 模型导出与静态检查

- `export_yolo11n_tflite.py`、`inspect_tflite.py`
- `export_depth_anything_v2_tflite.py`、`inspect_depth_model.py`
- `smoke_depth_model.py`、`smoke_depth_anything_v2_pytorch.py`
- `fetch_remote_zip_members.py`

模型源文件、下载和导出结果应写入 `artifacts.local/models/` 或 `artifacts.local/downloads/`。

## 检测器与设备 benchmark

- `prepare_coco100.py`、`detector_lab.py`、`benchmark_tflite_detectors.py`
- `run_yolo26n_device_benchmark.ps1`
- `run_detector_ab_device_benchmark.ps1`
- `run_depth_fusion_benchmark.ps1`
- `run_device_regression.ps1`

设备脚本可能调用 Gradle、ADB 并覆盖手机上的 debug 包；运行前确认设备和目标 module。证据统一写入 `artifacts.local/evidence/`。

## USTRF-SC 独立研究与回放

- 合成与 SANPO-Synthetic 回放：`acquire_sanpo_synthetic_replay.py`、`audit_sanpo_synthetic_replay.py`、`audit_sanpo_synthetic_metric_replay.py`、`generate_ustrf_synthetic_temporal_geometry_benchmark.py`、`generate_ustrf_synthetic_dynamic_ttc_benchmark.py`、`generate_ustrf_synthetic_corridor_safety_benchmark.py`
- 公开几何/动态来源审计：`audit_tartanair_slice.py`、`estimate_tartanair_ground_plane.py`、`audit_tartanair_temporal_reprojection.py`、`audit_carla_ped_rgbd_slice.py`、`audit_bonn_rgbd_dynamic_source.py`、`audit_bonn_rgbd_dynamic_reprojection.py`、`audit_vkitti2_dynamic_tracks.py`、`audit_argoverse_av1_timestamped_ttc.py`
- REveL 来源与 detector 诊断：`audit_revel_dynamic_rgb_labels.py`、`audit_revel_dynamic_bag_inventory.py`、`audit_revel_dynamic_vicon_trajectories.py`、`audit_revel_rgb_vicon_reprojection.py`、`benchmark_revel_yolo_person_detector.py`、`run_guarded_revel_yolo_smoke.ps1`、`analyze_revel_detector_failures.py`、`compare_revel_detector_sensitivity.py`、`compare_revel_detector_tiling.py`、`align_revel_detector_failures_with_vicon.py`
- 路线事件与设备硬门：`validate_sanpo_counterfactual_episodes.py` 在 route-conditioned 配置下额外验证 capture clock、非未来 route trace、criticality 与 hashed adjudication；`validate_ustrf_sc_device_metric_geometry.py` 验证独立标定、稳定 pose、source-aligned depth、body-local ground、完整路线事件真值和同机性能收据。
- RC-OARF E0 路线特异性负控：`evaluate_ustrf_sc_rc_oarf_route_specificity_control.py` 对冻结 r816 预测做同图 LEFT/STRAIGHT/RIGHT 循环错配，不重跑特征或重拟合；只生成 synthetic-mechanism 收据，不提供真实 provider/event/device 权限。
- 显式路线端侧几何一致性：`audit_explicit_route_geometry_conformance.py` 把冻结 route anchor 与 detector bbox 重放为纵横比感知的 normalized-device 几何；逐 anchor/逐帧必须零不一致。它只审计 benchmark 语义，不验证真实 route provider，也不授权 App/default runtime。
- 汇总门禁：`report_ustrf_sc_research_benchmark.py`；默认继续关闭 device gate，只有原始设备 evidence bundle 经上述 validator 完整通过才可标为 `device-geometry-shadow-only`。对应 `test_*.py` 与实现同名配对。

这些脚本只产出隔离研究证据；公开来源的 source-native 距离、轨迹或 TTC proxy 不能替代 body-local assistive event truth，也不授权默认 App 或模型替换。GPU 调度按 [USTRF-SC 窗口交接](../../../docs/research/ustrf-sc/USTRF_SC_WINDOW_HANDOFF_2026-07-20.md) 的风险分级边界执行。

## SANPO 数据集构建与治理

- 发现与构建：`discover_sanpo_sequence_candidates.py`、`build_sanpo_sequence_evalset.py`、`build_blindassist_evalset.py`
- 克隆与合并：`clone_sanpo_sequence_evalset.py`、`clone_sanpo_event_phase_evalset.py`、`merge_sanpo_sequence_evalsets.py`
- 选择与定稿：`select_sanpo_sequence_by_geometry.py`、`finalize_sanpo_sequence_evalset.py`
- 复核：`create_sanpo_v2_review_decisions.py`、`apply_sanpo_review_decisions.py`、`review_sanpo_sequence_with_model.py`、`screen_sanpo_mask_windows.py`
- v3 治理：`build_sanpo_v3_annotation_queue.py`、`prepare_sanpo_v3_dataset_views.py`、`validate_sanpo_v3_dataset.py`、`freeze_sanpo_v3_regression.py`
- 公开与程序化数据：`build_public_v3_canonical_dataset.py`、`build_procedural_tactile_sequences.py`
- session 拆分：`plan_sanpo_p3_session_split.py`（经同意手机来源只接受 `blindassist_4class_mask_v1` 的人工像素 mask，并逐帧重验 RGB/mask 哈希、原始尺寸、camera/lens 与同意回执）
- 全量候选发现：`run_sanpo_p3_discovery_batches.py`（按参数与官方 session 顺序绑定 checkpoint，支持 fail-closed 断点续跑；完整发现不等于候选接纳、人工真值或训练授权）
- 反事实事件：`validate_sanpo_counterfactual_episodes.py`（字段模板：`configs/sanpo_counterfactual_episode_manifest_template_v1.json`；只验证人工复核的本地 manifest；每条均需两名独立人工复核的本地 SHA256 证据，不下载、不生成标签，且永不授权替换生产模型）
- 风险轮廓/生命周期 target：`build_sanpo_risk_lifecycle_targets.py`（只从已完整验证的 episode manifest 生成主监督 target；像素监督标注为 `auxiliary_only`，不执行训练或模型升级）
- 风险轮廓/生命周期原型：`sanpo_risk_lifecycle_prototype.py`（读取 hash-attested 的人工双审 target，或显式授权的 `hash_bound_model_silver_provisional` 暂定监督；将半开生命周期区间映射为时序标签，像素监督保持 `auxiliary_only`，不授权校准、blind 评测或默认模型替换）
- 走廊熟悉度诊断：`run_sanpo_corridor_anomaly_probe.py`（冻结 DINO 仅拟合 canonical-train 的 source-semantic walkable 特征，并在 dev 报告 source-class outlier AUROC；高分只能表示 `unknown_motion_or_surface` 候选，永不等价于提醒）
- unknown 蒸馏可行性：`run_sanpo_mobile_unknown_distill_probe.py`（以冻结 DINO 熟悉度分数为 teacher，测试冻结 MobileNet 特征能否在 dev 上重现该非安全分数；不训练学生网络、不保存权重）
- 反事实采集清单：`generate_sanpo_counterfactual_capture_plan.py`（从冻结合同生成空的 96-slot 采集计划；不生成证据、标签或训练许可）
- 公开连续视频候选：`acquire_public_gnd_candidate.py`（先验证 CC0 GND 文件元数据；只有带 `--download` 才下载一个受限大小的公共包，并始终写为模型筛选候选，不能进入训练、校准或生产替换）
- 稀疏 SANPO 长时间线：`acquire_sanpo_rgb_timeline_candidate.py`（只下载公开 CC-BY RGB 帧，默认 1 FPS × 10 秒；不下载 mask、不生成事件标签，只供大模型决定是否值得后续审查）
- SANPO 边界辅助候选：`plan_sanpo_boundary_aux_candidates.py`、`redact_sanpo_auxiliary_candidate.py`（从完整公开 discovery 记录中排除 canonical session，只规划 `auxiliary_pixel_geometry_only` 的 source-mask 候选；SANPO RGB 草稿必须先整人/车脱敏且仍需隐私审核，永不从此路径构造事件或风险真值。）
- 公开视频 GPT/VLM 银标：`validate_public_video_silver_labels.py`（校验多帧时序银标与源帧 SHA256 的绑定；v1 保持只对照，普通公开渠道可下载且哈希完整的 v2 可显式授权隔离的暂定模型监督，许可/隐私元数据缺失只作 limitation。银标永不表述为人工真值，且不授权校准、blind 评测或默认模型替换。）
- 银标—端侧事件对照：`compare_public_silver_to_edge_events.py`（把 hash-bound 的端侧推理事件报告与银标逐 episode 比对；弃权单列、不给一致率分母，输出只称为候选一致性）
- 银标暂定训练迁移：`promote_public_silver_to_provisional_training.py`（从不可变的 v1 CC-BY 或有回执的 CC0 银标生成独立 v2 暂定训练证据包；绑定 image root、逐帧复验文件 SHA，保留原始清单 SHA 和机器隐私标记，且不授权校准、盲测或默认模型替换）
- 银标冻结特征诊断：`run_public_silver_frozen_feature_probe.py`、`run_public_silver_depth_feature_probe.py`（分别冻结现有 MobileNetV3 OS8+OS32 与 Depth Anything V2/DINO 特征，按 `source_id` 分组留一做确定性 class-balanced ridge；MobileNet 路径还可单次测试固定的 segmentation-corridor-relative 池化或全局平移消除后的 residual-motion 统计，并报告显式反事实对的特征增量方向是否足以支持 prototype 初始化）
- 银标对象轨迹诊断：`run_public_silver_object_trajectory_probe.py`（冻结 COCO detector 只产生 proposals，再确定性构造对象轨迹、相对尺度、下方走廊重叠和持续性特征；标签仍只来自 hash-bound provisional episode，按 `source_id` 留一）
- 银标显式深度走廊诊断：`run_public_silver_depth_corridor_profile_probe.py`（冻结 Depth Anything 相对深度输出，只提取固定梯形走廊、逐行相对凸起、左右不对称、下方阻塞比例和时间变化；不把深度当事件真值，也不读取独立模型方向）
- 银标自由空间拓扑诊断：`run_public_silver_free_space_topology_probe.py`（只从当前主线冻结 SANPO 分割 logits 计算自适应路径、可走宽度、瓶颈、横向偏移和四类非可走概率，并单独报告 topology 与 trajectory+topology 的 source-isolated 结果；显式拒绝独立模型方向路径，分割输出仍不是事件真值）
- r7.19 静态表征诊断：`run_public_silver_background_normalized_static_probe.py`、`run_public_silver_segformer_free_space_probe.py`（分别测试注册后近场残差相对两侧背景，以及冻结 ADE20K SegFormer 的软可走支持/无阈值自适应路径；均为看过 r7.17 失败后的 retrospective diagnosis，不能回救 r7.17 或授权训练、校准、blind、Android 与生产）
- r7.19 多通道风险轮廓诊断：`run_public_silver_multichannel_risk_profile_probe.py`（固定拼接 4 个注册局部变化、4 个绝对净空、4 个自适应路径占用和 4 个绕行偏移通道，再分别评测 profile-only 与 trajectory+profile 的 `source_id` 留一闭式 ridge；Rice 只作外部压力，不进入训练折。只有融合严格改善轨迹基线才授权后续五组短跑，报告本身不授权训练、校准、blind、Android 或生产）
- r7.20 机制专家诊断：`run_public_silver_mechanism_routed_expert_probe.py`（在每个外层 source holdout 内训练 observable mechanism router 和两个专家，并把 oracle mechanism 单列；任何 oracle/routed 结果都只诊断表示与标签冲突，不授权保存权重或运行时路由）
- r7.20 合成反事实来源/响应审计：`audit_public_silver_synthetic_counterfactual_response.py`（逐项复验 train-only spec、真实父 episode/frame SHA、合成图 SHA、无 bbox/mask、父来源后代隔离、每障碍族来源数和 GPT/VLM 复核；冻结 teacher 不响应只标为 hard counterexample，不可据此删样本）
- r7.20 父来源隔离 pair-delta：`run_public_silver_synthetic_pair_delta_probe.py`（固定四个语义响应通道，以真实 pair 留出父 source 时排除其全部合成后代；只比较真实排序与 Rice 压力，不保存模型）
- r7.20 完整语义 adapter 诊断：`run_public_silver_full_semantic_pair_adapter_probe.py`（冻结 ADE20K SegFormer，在固定 lower/core/core-minus-peripheral 区域汇总全部类别概率，做 source-isolated 线性 pair adapter；不做类别筛选、阈值搜索或权重保存）
- r7.20 非线性语义表示短跑：`run_public_silver_nonlinear_semantic_adapter_probe.py`（冻结 SegFormer，以固定 4 个 positive-evidence unit 和不可训练等权 readout 做单 seed pair-ranking；父来源留出时排除全部合成后代。只有真实 pair 与 Rice open/close 全过才授权五组稳定性短跑）
- r7.21 逆向反事实审计与 DINO 区域特征门：`audit_public_silver_inverse_counterfactuals.py`、`run_public_silver_dinov2_regional_pair_probe.py`、`run_public_silver_dinov2_bootstrap_short_runs.py`（真实 risk/GPT 移除障碍 clear 只作 train-only pair 监督；DINO 固定最后一层与五个区域，不搜层/区域/阈值；五组短跑只验证 source-bootstrap 稳定性，不保存或部署权重）
- r7.22 DINO 前瞻合同：`build_public_video_dinov2_prospective_contract.py`、`extract_public_video_dinov2_prospective_features.py`、`evaluate_public_video_dinov2_prospective_pair.py`、`configs/public_video_dinov2_pair_contract_r722.json`（完整视频逐秒特征必须在视觉审阅前冻结；新来源、窗口顺序、最少样本和 open/close 均 fail closed；事后不得改方向或窗口）
- r7.23 多专家风险轮廓/生命周期：`run_public_video_multiexpert_risk_profile_prototype.py`、`public_video_multiexpert_risk_profile_contract.py`、`configs/public_video_multiexpert_risk_profile_contract_r723.json`（独立正证据通道 OR 开事件，所有曾打开的通道必须分别确认 close；冲突保持 present/uncertain，absence 不能单独 clear；像素分割与距离场仅 auxiliary，训练、Android 和生产授权保持 false）
- r7.24 多专家前瞻负控：`evaluate_public_video_multiexpert_negative_control.py`（哈希绑定 r7.23/r7.22/r7.11 合同、两路全片特征和视觉负控；在无行人走廊风险窗口中，任一通道开事件即 fail。该失败用于否定无条件专家 OR，不允许事后改窗口或阈值回救）
- r7.25 径向接近诊断/合同：`run_public_video_marker_radial_approach_probe.py`、`public_video_marker_radial_approach_contract.py`、`configs/public_video_marker_radial_approach_contract_r725.json`（对原 r7.11 接受框固定计算纵向进展、水平扫动和面积增长；Japan/Matoaka 只作派生，冻结后 DINO 仅 support-only，训练与端侧授权保持 false）
- r7.26 合同后候选/复核：`scan_public_video_marker_radial_approach_candidates.py`、`evaluate_public_video_marker_radial_approach_review.py`（完整逐秒特征与候选必须在视觉复核前冻结；negative 窗口有任一候选即 fail，context-only 来源不计门禁信用；大模型复核仅为 provisional silver，不是人工真值或生产授权）
- r7.29 正生命周期失败固化：`evaluate_public_video_marker_radial_lifecycle_positive.py`（从冻结 r7.11 特征重建基础生命周期，绑定 r7.25 候选与前后密集接触表；入口正确但在视觉风险结束前 clear 即保留为 false-clear，不能事后合并成 pass）
- r7.30 gap-bridge 诊断/合同：`run_public_video_radial_lifecycle_gap_bridge_probe.py`、`public_video_radial_lifecycle_gap_bridge_contract.py`、`configs/public_video_radial_lifecycle_gap_bridge_contract_r730.json`（径向只负责 entry；开后颜色证据跨短 gap 续同一事件、不再提醒；回顾性最小 9 秒缺失才 clear，必须再过独立真入口/快净空负压力；训练与 Android 授权全 false）
- r7.34 局部接近诊断：`run_public_video_marker_local_approach_probe.py`（在颜色事件内固定扫描 5/7/9/12 接受样本短窗，窗内完全复用 r7.25 门；用于证明短窗会误开路径边界负控，只作诊断，不选择阈值、不冻结合同、不授权训练或 Android）
- r7.35 路径关系反事实：`build_public_video_path_intrusion_counterfactuals.py`、`build_public_video_equal_count_path_relation_pairs.py`、`run_public_video_path_relation_dinov2_probe.py`（生成式模型只产透明锥桶资产；确定性合成保证同数量/尺度/纵深且掩码外父像素不变，DINO 探针做合成留一、镜像及 Japan/Edmonton/Jakarta/Cape Town 压力；所有样本 train-only，失败不冻结合同）
- r7.36–r7.37 前瞻来源筛选沿用 `extract_public_video_traffic_cone_detection_features.py`、`scan_public_video_marker_radial_approach_candidates.py`、`build_public_video_overview_contact_sheets.py`（必须先登记、下载、冻结全片特征与候选，最后才看概览；元数据命中但没有连续行人第一视角时在事件评分前拒绝，不计正负门信用）
- r7.38–r7.41 障碍感知净宽/距离场：`run_public_video_obstacle_aware_route_width_probe.py`（冻结 SegFormer 可走支持与 r7.11 标志框，比较 hard argmax 最大瓶颈、soft 概率净宽和自适应中心线距离场；严格门失败后距离场只作 auxiliary，禁止继续搜索膨胀尺度或把结果写入 Android）
- r7.42 DINO 正负 prototype：`run_public_video_path_relation_positive_negative_prototype_probe.py`（fold 内 unit 正 prototype 减 unit 负 prototype，0 阈值；每个合成后代继承自己的真实父 source 并联动留出。只有 source-isolated balanced accuracy 1.0 且镜像全过才授权五组 bootstrap）
- r7.43 clear-drift nuisance OFAT：同一 prototype 脚本的 `--nuisance-mode project_out_clear_drift`（只投影掉 real delta 与自身 clear 前后漂移平行的分量；不改合成样本、窗口、prototype 或阈值。失败后禁止继续搜索投影变体）
- r7.44 多来源等数量反事实：`build_public_video_multisource_equal_count_pairs.py`（JSON 冻结父图 SHA、来源和每组 clear/risk placement；强制同数量、同纵深尺度、仅横移、掩码外像素不变、train-only 和授权全 false）
- r7.45–r7.46 跨来源 DINO 方向压力：同一 prototype 脚本支持多个 dataset/report，并用 `--feature-mode spatial_grid_4x4` 做唯一位置敏感 OFAT；两种冻结 DINO 表征均失败后，禁止继续搜 pooling/阈值，转向显式障碍位置×预测 ego 路线表征。
- r7.47 显式 ego-route 关系：`run_public_video_explicit_ego_route_relation_probe.py`（只在 marker 掩码内恢复潜在可走支持，再沿冻结 adaptive route 测 q10 障碍距离；单帧门失败后禁止继续搜 inpaint 半径、路线宽度或阈值）
- r7.48 事件级风险轮廓/生命周期：`run_public_video_event_risk_profile_lifecycle_gate.py`（径向入口与路线关系共同开事件，颜色连续性和 9 秒缺失负责保持/清除；分割只作 auxiliary。现有指标满分但缺真实 true-radial safe-lateral 负控时 full closure 必须保持 false）
- r7.49 Japan 因果重放：`evaluate_public_video_japan_causal_lifecycle_replay.py`（每帧只看 prefix，pre-risk clear 建路线基线，径向与路线关系联合开事件，r7.30 管理退出；提前或延迟提醒都按冻结视觉窗口 fail closed，禁止事后改 onset/阈值）
- r7.50 前瞻提醒时序合同：`public_video_event_timing_contract.py`、`configs/public_video_event_timing_contract_r750.json`、`configs/public_video_event_timing_review_template_r750.json`（固定最多提前 3 秒的提醒带并披露其来自 r7.49 Japan 失败后的假设；Japan 不得追溯转为通过。完整门还强制独立真实 true-radial safe-lateral 负控由冻结路线关系 veto，合成/GPT-only 不计门信用）
- r7.51 前瞻正例验收：`evaluate_public_video_event_timing_positive.py`（绑定 r7.50 合同、完整逐秒特征、冻结 r7.25 候选、候选前复核计划与候选后大模型银标；复算冻结 r7.47 路线关系和 r7.30 九秒退出，只在提醒带、路线支持、持续、clear 和单次提醒全过时接受正角色。正例通过仍不能替代不同来源的 true-radial safe-lateral 负控）
- r7.52 真实径向侧向否决：`evaluate_public_video_true_radial_route_veto.py`（绑定 r7.50 合同、完整特征、冻结候选、候选前复核计划和 provisional 大模型负例角色；以候选前窗口为 clear、径向窗口为 marker 复算冻结 r7.47，只有 delta 非正才算 veto。失败也原样落盘，禁止据此调阈值、窗口、inpaint 或安全边距）
- r7.53 未来帧 ego-trace：`run_public_video_future_ego_trace_probe.py`（离线用固定未来帧单应性把底部中心锚点反投影为实际未来路线，再测冻结 marker 命中；只作 retrospective 辅助教师诊断，使用未来信息、失败后换聚合器或调 ORB/RANSAC 均不得获得门禁信用）
- r7.54–r7.56 多来源/稠密/因果路线教师：`run_public_video_future_ego_trace_multisource_probe.py`、`run_public_video_dense_future_ego_trace_probe.py`、`run_public_video_causal_past_ego_trace_probe.py`（先验证未来路线变量能否跨来源分离，再用固定 DIS dense flow 修复稀疏覆盖，最后单独检验仅过去帧的因果外推；未来教师通过不等价于端侧可用，因果失败不得被未来结果遮盖）
- r7.57–r7.60 路线辅助蒸馏：`run_public_video_ego_trace_distillation_probe.py`、`run_public_video_ego_trace_dinov2_probe.py`、`run_public_video_ego_route_distance_field_probe.py`、`run_public_video_multi_horizon_route_field_probe.py`（依次测试手工因果特征、冻结 DINO 全局特征、合并路线距离场和三 horizon 空间 heatmap；全部按 source 留一。空间 pixel AUROC 通过但事件 readout 失败时，禁止继续换 head/阈值，转向扩大自动时序辅助数据）
- r7.61 自动路线辅助集：`build_public_video_temporal_route_auxiliary_dataset.py`、`configs/public_video_temporal_route_auxiliary_dataset_contract_r761.json`（从许可/特征已绑定的连续视频中按 source 限额、确定性均匀选择有 marker 的因果时间点，以固定 DIS future teacher 生成三个训练期路线锚点；manifest 不含事件标签，所有后代必须与 parent source 联动留出）
- r7.62–r7.64 时序路线 head：`extract_public_video_temporal_route_features.py`、`train_public_video_temporal_route_head.py`（固定 43 通道当前视觉+完整空间 past-flow 输入和整来源留出；marker-only 与全连续帧都只拟合 future route auxiliary，事件定位未闭合时禁止启动五组 seed）
- r7.65–r7.66 相对峰值风险轮廓：`audit_public_video_temporal_route_uncertainty.py`、`freeze_public_video_temporal_risk_profile_candidate.py`、`evaluate_public_video_temporal_risk_profile_prospective.py`（披露事后 `.68` 阈值后冻结权重；新来源评估严格离线并拒绝旧 source ID/视频哈希，正例同时要求 r7.30 固定时序与生命周期；仅 diagnostic，不授权训练、Android 或生产）
- r7.66 Bangkok 同源压力对：`audit_public_video_temporal_risk_profile_pair_error.py`（对候选前冻结的 true-radial safe-lateral 与路线侵入事件并排核对 r7.66 head 和 r7.55 future teacher；Bangkok 是事后同源诊断，不得获得独立前瞻或校准信用）
- r7.67a marker-conditioned 线性 probe：`run_public_video_marker_relation_linear_probe.py`（在冻结 marker 掩码内汇总 43 通道因果空间特征，整来源留一、确定性 ridge；亚 patch 检测使用合同冻结的最近 patch 回退，不删 marker 帧）
- r7.68a prototype/source-class bootstrap：`run_public_video_marker_relation_bootstrap_short_runs.py`（每类 `.5` 质量、完整 `(source,class)` 块抽样、bootstrap 内标准化/prototype、五固定 seed；source-macro 为授权指标，Bangkok 只旁路，不沿用 `.68`）
- r7.69 距离场辅助 A/B：`run_public_video_marker_relation_distance_aux_ablation.py`（相同 seed/fold/bootstrap/初始化的 binary baseline 与三 horizon marker-route distance 辅助 treatment 配对；只在预注册主指标改善时保留，不搜索辅助权重）
- r7.70 Bangkok 同源 matched contrast：`build_public_video_bangkok_matched_counterfactual_pair.py`（从合同绑定原视频和两份 review 提取 24 个逐秒哈希帧；两 episode 永远随同一 parent source 隔离，只作 representation-training candidate）
- r7.71–r7.72 同源 pair-ranking/prototype bootstrap：`run_public_video_marker_relation_pair_ranking_probe.py`、`run_public_video_marker_relation_pair_bootstrap_short_runs.py`（active 帧只与最近时刻的同来源 inactive 帧比较；五组优化必须同时不显著落后于零训练 bootstrap prototype，不能以单跑通过掩盖总体退化）
- r7.73 因果 prototype 生命周期：`evaluate_public_video_pair_prototype_risk_profile_lifecycle.py`（五个 source-bootstrap 零训练 prototype 取中位相对分数，逐 episode 用最初三帧因果基线和固定 r7.30 clear 重放；safe 假开或提醒晚于窗口即失败）
- r7.74–r7.75 几何捷径压力：`run_public_video_marker_relation_geometry_matched_probe.py`、`run_public_video_marker_relation_geometry_residual_probe.py`（分别检验按 marker 面积/质心匹配和训练折内 nuisance 投影；时间间隔膨胀或语义判别下降时停止，不继续搜匹配权重）
- r7.76–r7.77 多数时域 target：`run_public_video_marker_relation_majority_horizon_probe.py`、`evaluate_public_video_majority_pair_prototype_lifecycle.py`（只把至少 `2/3` future-route horizon 命中视为强正；离线 source-heldout 可分仍必须再过因果事件提醒/清除，不能替代生命周期门）
- r7.78–r7.78a 独立负例：`evaluate_public_video_majority_prototype_external_negative.py`、`configs/public_video_majority_prototype_external_negative_contract_r778a.json`（对下载/候选/复核后冻结的 Düsseldorf safe-lateral 事件重放 r7.77 prototype；marker gap 只按 r7.30 同事件桥接跳过，并强制接受样本数绑定，失败不能触发阈值或 baseline 搜索）
- r7.79 因果 waypoint readout：`run_public_video_causal_waypoint_linear_probe.py`（对同一 `43×16×16` 因果缓存做固定 block pooling 和多输出 ridge；只判断 dense heatmap readout 是否是定位瓶颈，不保存可部署权重）
- r7.80 分层因果告警：`evaluate_public_video_tiered_causal_alerts.py`（r7.25 只开无方向环境注意，过去 ego trace 连续满足才升级路线阻挡；生命周期读取冻结的 `confirmed_clear_timestamp_ms`，安全事件升级即失败）
- r7.81 事件角色 probe：`run_public_video_event_route_role_linear_probe.py`（按完整来源留出，用 marker-conditioned 因果特征预测事件级 path role；缺 marker 的帧/事件 fail closed 或按合同显式排除，不允许事后翻转 score）
- r7.84 局部高召回 acquisition audit：`run_public_video_marker_local_approach_probe.py`（在长 chromatic event 内扫描固定 5/7/9/12 样本局部窗口；只产生待复核 proposal，不能替代 canonical r7.25、事件真值、校准或 blind 证据）
- r7.85 因果可行动性状态：`evaluate_public_video_causal_actionability.py`、`configs/public_video_causal_actionability_contract_r785.json`（把最终安全结果从早期告警 target 中剥离；只用 current/past trace 输出 context、intervention、route-clear 状态，保留原 review role 并报告因果标签矛盾）
- v3 银标因果门：`validate_public_video_silver_labels.py`（v3 强制 `silver_actionability` 与 `causal_evidence_basis=past_or_current_only`；最终 safe-pass/route-change 只作响应属性，不能把已需注意/干预的 episode 反写为 no-alert）
- actionability 重标与独立 probe：`audit_public_video_actionability_relabels.py`、`evaluate_public_video_cardiff_causal_actionability.py`、`evaluate_public_video_ulm_route_turn_actionability.py`、`build_public_video_actionability_manifest.py`、`run_public_video_actionability_linear_probe.py`、`run_public_video_actionability_profile_lifecycle_probe.py`（统一 current/past-only 状态，保持完整 parent-source 隔离；事件均值与帧级生命周期结果都不得授权训练、校准、Android 或生产）
- ADVIO 视觉惯性路线意图诊断：`audit_advio_visual_inertial_source.py`、`run_advio_turn_intent_probe.py`、`run_advio_turn_intent_invariant_probe.py`、`run_advio_turn_confirmation_probe.py`（先校验官方 MD5、模态哈希、同步与采样率，再用连续时间块留出分别诊断未来路线意图与 current-only 转向确认；future pose 只在前两项生成辅助 target，确认 probe 不访问未来。三个 probe 均不提供风险/actionability 真值；ADVIO 为 CC BY-NC 4.0，严格 research-only）
- 显式 route-intent 上限与融合：`evaluate_public_video_explicit_route_intent_oracle.py`、`explicit_route_intent_fusion.py`（冻结 future-route teacher 只作为外部导航/用户路线的离线 oracle 代理；风险交叠和 open/clear 无学习参数。runtime/eval 禁止未来视频输入，route 缺失或过期只能 context，不允许补猜方向）
- 显式三态选择与覆盖审计：`evaluate_public_video_explicit_choice_route_template.py`、`audit_public_video_explicit_choice_direction_coverage.py`（固定 LEFT/STRAIGHT/RIGHT 模板只作接口诊断；每个方向缺少 intervention 时不得宣称完整 provider）
- 转向候选与连续复核：`search_public_video_explicit_turn_candidates.py`、`search_public_video_background_yaw_turn_candidates.py`、`evaluate_public_video_background_yaw_template_intersection.py`、`extract_public_video_explicit_turn_review_material.py`（future-anchor x 不是 turn class；背景 yaw 只生成候选，固定模板相交后仍需大模型连续复核。平行边界、不同因果障碍和 detector 假框一律不给方向覆盖信用）
- route-intent 输入门：`validate_explicit_route_intent_episode.py`（校验 provider 独立性、投影/姿态对齐回执、单调时间戳、一秒内有效期、1/2/3 秒归一化 waypoint 与 unknown fallback；runtime 明确拒绝 future-video oracle 和风险模型自生成意图）
- 银标运动补偿占用诊断：`run_public_silver_motion_compensated_occupancy_probe.py`（用相邻帧 homography 注册后的下方走廊残差、匹配可靠性和时间汇总做固定 OFAT，并与对象轨迹分别评测；注册失败本身只作为诊断信号，不等价于风险真值）
- 银标机制时间范围诊断：`run_public_silver_mechanism_temporal_range_probe.py`（对置信度合格的动态/静态 matched pair 做 source 内时间范围、端点排序和 leave-one-pair-out 审计，用于判断统一绝对阈值是否成立；不训练或保存模型）
- 银标 pair-relative 生命周期诊断：`run_public_silver_pair_relative_lifecycle_probe.py`（绑定机制覆盖与 temporal-range 报告，只用 SHA 绑定帧序和机制分数变化符号判断 `open/close/abstain`；报告 margin 敏感性但不把事后压力线当校准阈值，要求近期可信参考状态，不授权 Android 或生产）
- 银标 retrospective 关事件压力验证：`evaluate_public_silver_retrospective_close_stress.py`（要求大模型原时序 review 在 detector scoring 前以 SHA 冻结，复算已发布风险窗分数后对比同 source 后段未用净空帧；只能补机制诊断，不能冒充新来源、前瞻、人工真值或训练/Android 授权）
- 银标双证据生命周期融合：`run_public_silver_dual_evidence_lifecycle_fusion.py`（可信相对基线上的强变化直接开/关，弱下降必须与冻结因果语义退出的同 source/同边界证据一致；冷启动、缺少互证和冲突全部 `uncertain`。强 margin 来自事后 stress grid，故 prospective acceptance 固定为 false）
- 双证据生命周期前瞻合同：`public_video_dual_evidence_lifecycle_contract.py`、`configs/public_video_dual_evidence_lifecycle_contract_r717.json`、`extract_public_video_dual_evidence_features.py`、`evaluate_public_video_dual_evidence_prospective_lifecycle.py`（完整视频 1 秒特征必须先冻结，大模型后选三段窗口，验收器再计算 open/close/stability；提取器不接收 review/label/hazard verdict，合同替换、缺帧、硬剪辑、来源污染、弱下降缺互证和 post-clear 重开全部 fail closed）
- 银标训练就绪门禁：`audit_public_silver_training_readiness.py`（逐包和逐帧复验 v2 证据，要求每类独立 source 数、显式匹配反事实对和跨 source 帧隔离；数据门禁与冻结 probe 未同时通过时拒绝五组 head 短跑）
- 银标机制覆盖门禁：`audit_public_silver_mechanism_coverage.py`（把 matched pair 分为动态目标接近与静态通道收窄，默认要求每种机制至少 3 个独立 pair/source，且 pair 内每条 episode 置信度至少 `.65`；禁止用总体 pair 数或低置信度 pair 掩盖机制覆盖不足）
- 银标主线扩展构建：`build_public_silver_r4_counterfactual_expansion.py`、`build_public_silver_r5_static_mechanism_expansion.py`、`build_public_silver_r6_dynamic_pair_candidate.py`、`build_public_silver_r7_wikimedia_counterfactuals.py`（从不可变父包生成独立 r4/r5/r6/r7；r6 只把 `JtMY` 低置信度动态 pair 用于结构诊断；r7 从 SHA256 固定的 Wikimedia 视频提取预注册时间点，增加车道口横穿与砂堆占道两组高置信 pair；所有输入/输出若落入独立模型方向目录即 fail closed）
- 银标受控合成静态反事实：`build_public_silver_synthetic_static_counterfactuals.py`（从精确绑定的真实 no-alert 父 episode 复制 clear 帧，再确定性合成逐帧接近的静态障碍，写出 mask/bbox、manifest、YOLO/COCO 可转换记录和独立 provisional package；产物严格 `train_only`，必须绑定 `parent_source_id`，父真实 source 留出时不得参与训练，且永不计入评测、校准、blind 或生产结论）
- 路线条件合成诊断：`build_public_video_route_conditioned_synthetic_dataset.py`、`audit_public_video_route_conditioned_synthetic_dataset.py`（在已审阅真实父帧上确定性放置左右/直行障碍，输出精确 mask/bbox、三态 route 样本、YOLO/COCO 与全图/路线 overlay QA；所有后代随 `parent_source_id` 联动留出。通用障碍类名或未来合同时间戳版本必须拒绝并新目录重执行）
- 路线条件风险场 probe：`run_public_video_route_conditioned_synthetic_probe.py`（同一冻结 DINO patch map 比较全局 readout、沿路线 readout 和 exact-field head；支持独立 OFAT 的 `binary_patch` 与 `bbox_distance` teacher target，后者固定连续距离场，不搜索 sigma 或 gate）
- 真实迁移与生命周期：`run_public_video_route_conditioned_real_transfer_probe.py` 将 train-only 合成距离场带到真实 provisional source-LOSO；`run_public_video_route_background_real_transfer_probe.py` 是已拒绝的路线+全场背景 OFAT；`run_public_video_route_risk_lifecycle_real_probe.py` 用冻结转移点重建逐帧状态并固定两帧 open/clear。三者都把 future route anchors 限定为离线 oracle proxy，不授权 provider/runtime/production。
- 真实 marker 距离负控：`run_public_video_real_marker_distance_transfer_probe.py` 只在每个训练来源折内用 provisional bbox 拟合连续距离场，held-out detection 完全禁止进入 teacher。该脚本诊断 object localization 是否足够，不把 detector box 当人工真值或 provider/production 证据。
- 路线 patch 交互负控：`run_public_video_route_patch_interaction_probe.py` 用固定投影和连续 route polyline 场直接池化冻结 DINO patch，并以均匀场作同维对照。future anchors 仍只作离线 oracle proxy；失败结果不授权搜索投影、sigma、ridge 或训练非线性 head。
- 合成资产 QA：`prepare_synthetic_chroma_asset.py` 把经视觉复核的 chroma-key RGB 转成确定性二值 alpha，并记录源/输出哈希；无 alpha、可见色边或未审计资产必须 fail closed。
- 路线条件生命周期与短跑：`evaluate_public_video_route_conditioned_synthetic_lifecycle.py`、`run_public_video_route_conditioned_bootstrap_short_runs.py`（前者只解码固定两帧 open；后者在每个训练来源×类别内 bootstrap，prototype 初始化后固定 80 step、五 seed，不保存权重。失败不得事后改学习率、步数或门槛）
- 合成 mask 静态 teacher 诊断：`run_public_silver_synthetic_mask_teacher_probe.py`（冻结 Depth Anything V2 的 DINO 稠密 patch token，只用 train-only composite mask patch 与同位置 exact-clear patch 拟合静态障碍 teacher；每个真实 LOSO fold 排除父 source 相同的合成记录，teacher 输出仍只是诊断表示）
- prompt-free 静态语义诊断：`run_public_silver_prompt_free_semantic_probe.py`（冻结 YOLOE prompt-free 内置词表，不输入文本 prompt、银标、source mask 或合成图；将预注册 surface-material/barrier 类转为走廊时序特征，并分别报告 semantic-only 与 trajectory+semantic 的 source-isolated 结果）
- 因果静态事件退出路由：`run_public_silver_semantic_exit_router.py`（绑定 MIL、trajectory、prompt-free semantic 三份带 sidecar 报告；只有“前段 surface-material 存在、当前消失、时间连续、当前轨迹无动态危险”同时满足才关闭陈旧静态事件。路由无学习参数、不读当前标签、不按 episode ID 特判，并支持缩短 gap 的负控）
- 风险轮廓退出路由：`run_public_silver_risk_profile_exit_router.py`（不修改 v1 router，在隔离实验中要求选定的 surface/barrier 风险类全部消失且轨迹清空；保留同样的 source/gap/sidecar 门禁和负控）
- 公开视频退出外部挑战：`evaluate_public_video_exit_router_external_challenge.py`（先从冻结 prompt-free 扫描产生 surface-only、barrier-only 和风险轮廓并集候选，再打开 GPT 多帧时间边界评分；同时报告风险存在段的语义覆盖率/最长漏检和稳定净空段假激活，避免把连续未检出误写成清空证据。输出只是 discovery challenge，不是人工真值、校准、blind 或生产证据）
- 公开视频 prompt-free 退出发现：`scan_public_video_prompt_free_exit_candidates.py`（registry v1 继续接受 Commons 字段并规范化为通用来源字段；v2 接受 Vimeo 等非 Commons 平台。所有来源先做路径隔离与许可登记，输出仍只是 proposal。默认保持原预注册词表；`--include-workzone-markers` 是默认关闭的探索开关，仅追加冻结内置类 `barricade/cone/construction worker/traffic cone`，不能反写既有基线或直接授权训练）
- Vimeo 候选台账：`search_vimeo_ccby_public_video_candidates.py`、`search_vimeo_ccby_public_video_candidates.ps1`（包装器保持不分页、不登录并保留原始 HTML 哈希；普通公开视频 URL 足以下载和进入隔离研究，条目许可缺失只作 limitation，连续性和画面审阅决定任务有效性。）
- Wikimedia Commons 视频候选台账：`search_wikimedia_public_video_candidates.py`、`search_wikimedia_public_video_candidates.ps1`（MediaWiki 查询必须来自冻结合同；Python TLS 未产生响应时，只有单独 transport erratum 才能允许 Windows TLS 对原查询各执行一次并离线解析原始 JSON。普通公开文件可先下载；连续 POV 和事件因果复核决定事件 authority。）
- Internet Archive 视频候选台账：`search_internet_archive_public_video_candidates.py`、`search_internet_archive_public_video_candidates.ps1`（每条冻结查询只取高级搜索第一页，并要求候选元数据包含 license URL；解析器离线读取原始响应并哈希台账。全文关键词高分不能替代条目页、可下载视频、连续 POV 和因果事件复核）
- 冻结 DINO 退出检索负实验：`retrieve_public_video_exit_windows_with_frozen_dino.py`（用一个已审阅风险/清空短片构造零训练参数的冻结 DINO-S 原型方向，按来源内 robust-z 持续下降排序长视频窗口；原始投影不是跨来源概率，候选必须经过独立多帧连续性审阅，且不得直接授权训练、校准、blind、Android runtime 或生产替换）
- 三态生命周期外部挑战：`evaluate_public_video_tristate_lifecycle_external_challenge.py`（零学习参数的 `present/uncertain/clear` 状态机；固定 `2-of-3` 才进入风险、连续 3 帧缺失才确认净空、clear 后单帧假激活不重开事件。候选状态在读取 GPT 时间参考前冻结，输出只支持外部机制诊断，不授权行人真值、训练、校准、blind、Android runtime 或生产）
- 前瞻生命周期合同：`public_video_tristate_contract.py`、`configs/public_video_tristate_lifecycle_contract_r79.json`（哈希绑定模型、扫描、marker、生命周期、来源资格和授权全 false；带合同运行时拒绝命令行参数漂移）
- marker 失败/计数诊断：`evaluate_public_video_tristate_negative_controls.py`、`audit_public_video_workzone_marker_class_ablation.py`、`audit_public_video_traffic_cone_count_threshold.py`（先固定 r7.9 失败，再做 leave-one-out 与 `traffic cone` 计数阈值诊断；诊断通过不能追溯改写前瞻失败）
- 多锥桶机制专家：`public_video_multicone_policy.py`、`evaluate_public_video_multicone_negative_controls.py`、`evaluate_public_video_multicone_persistent_entry_control.py`（每帧至少两个 `traffic cone`，只服务 dense cone corridor；通过普通场景负控但对稀疏施工桩必须报告 abstain/失败）
- 彩色施工标志特征：`extract_public_video_traffic_cone_detection_features.py`、`public_video_chromatic_marker_policy.py`、`audit_public_video_chromatic_cone_lifecycle.py`、`evaluate_public_video_chromatic_marker_negative_controls.py`（冻结 `traffic cone+barricade`，以检测框 `high_saturation_fraction > dark_fraction` 过滤黑色固定设施；r7.11 使用 `2-of-3 / 5-clear`，只产生机制证据，不是事件真值或 Android 授权）
- 公开视频来源谱系门：`audit_public_video_prospective_source_inventory.py`、`configs/public_video_local_source_inventory_r712.json`（按视频 SHA 去重并核对全部本地注册表；固定机位、剪辑、无正退出、已参与 r7.8/r7.10/r7.11 派生或非行人视角分别显式记录，禁止换别名或重复文件冒充冻结后独立正样本）
- 前瞻正退出验收：`evaluate_public_video_chromatic_marker_prospective_positive.py`、`configs/public_video_prospective_positive_review_template_r713.json`（合同、feature report、来源谱系审计和大模型原时序多帧复核四方 SHA 绑定；要求唯一事件覆盖、风险覆盖率、稳定净空假激活和终态 clear 全部通过，硬剪辑/派生污染/报告替换直接 fail closed）
- prototype/bootstrap 短跑：`run_public_silver_prototype_bootstrap_short_runs.py`（fold 内 prototype 初始化、每类 source bootstrap、五个固定 seed 的 80-step 线性 head；不保存权重，稳定性门禁通过也不授权校准、blind 或生产替换）
- 风险轮廓/潜生命周期 MIL：`run_public_silver_risk_lifecycle_mil_head.py`（逐帧对象/走廊风险 profile 经 smooth-max 聚合接受 episode 级弱监督；生命周期仅为风险曲线解码的潜变量诊断，不能当逐帧真值或生命周期准确率；像素分割不进入主 head。可选 motion residual、首帧时序基线、完整 pair ranking loss、lower-corridor appearance OFAT、严格 terminal pooling 负实验，以及带 `parent_source_id` 的 train-only augmentation；后者只进入训练折并在父真实 source 留出时自动排除，所有指标始终只统计真实 provisional episode）
- 银标影响敏感性：`run_public_silver_label_sensitivity.py`（冻结相同轨迹特征，逐 episode 做 quarantine 后重跑 source-isolated ridge；只把高影响、语义含混样本路由到独立复核，禁止据此删标、翻标或抬高正式分数）
- 公开视频真机推理集：`build_public_video_edge_inference_set.py`（只从银标引用的源帧构建真机资产，逐帧复验 SHA256；不把银标 verdict 放进 Android 输入，也不授权训练或默认模型替换）
- RGB 时间线来源清单：`materialize_public_rgb_timeline_source_manifest.py`（将公开 SANPO RGB-only 时间线变为银标可验证的帧哈希清单；不读取或导出 mask、几何或事件标签）
- 公开视频真机—银标闭环：`run_public_video_edge_inference.ps1`（构建推理集、运行默认端侧模型、拉回事件报告并自动对照；同包名签名冲突时 fail closed，只有显式 `-RemoveConflictingInstall` 才会卸载手机上的现有 BlindAssist）
- Dataverse 连续第一视角候选：`acquire_public_dataverse_candidate.py`、`extract_public_dataverse_rgb_candidate.py`、`machine_redact_public_rgb_candidate.py`（普通公开文件先按大小和 MD5 下载，许可元数据仅记录；脱敏用人脸、车牌和整个人/车辆检测。输出保留 `privacy_audit_required=true` 与无事件真值边界，可进入隔离内部研究/银标训练，但不授权校准、blind 或默认模型替换。）

改变 canonical 数据、冻结回归集或读取 blind 数据的脚本必须遵守 [SANPO 训练协议](../../../docs/SANPO_TRAINING_PROTOCOL.md)。

## SANPO 训练、导出与门禁

- 模型与训练：`sanpo_segmentation_model.py`、`train_sanpo_segmentation_keras_torch.py`
- 导出：`train_export_sanpo_segmentation.py`
- 门禁：`sanpo_training_gate.py`、`sanpo_candidate_quality_gate.py`、`extract_sanpo_device_event_report.py`（仅将模型 SHA256、事件分母和序列时长完整的真机 `benchmark.json` 转为设备事件门输入；任一绑定缺失即拒绝）
- 后端一致性：`sanpo_backend_equivalence.py`
- 辅助损失与诊断：`sanpo_boundary_distance_aux.py`、`run_sanpo_balanced_distance_ablation.py`、`sanpo_deterministic_linear_probe.py`、`sanpo_depth_anything_linear_probe.py`。`run_sanpo_balanced_distance_ablation.py` 只从 canonical train 中按完整 session 隔离评测，并要求 train/eval 的边界像素覆盖比落在预设区间；它不读取 canonical dev/blind、不保存权重，且无论结果均不授权事件真值、训练升级或默认模型替换。后者以冻结官方 Depth Anything V2 特征做 train/dev-only ridge 诊断；可选拼接既有 raw OS8/OS32，但不训练、不读取 blind、不产生可部署权重。

对应的 `test_*.py` 与实现保持同名配对。候选只有在离线、INT8 和设备门全部通过后才允许进入正式 App。

## SANPO benchmark

- `benchmark_sanpo_traversability.py`
- `benchmark_sanpo_gpu_throughput.py`

## 运行约定

- 从仓库根目录执行脚本。
- 本机工具来自 `E:\codex-tools`；通用 Python 检查优先使用 `E:\codex-tools\bin\blindassist-python.cmd`，旧 `.jdk/.android-sdk/.venv*` 入口仅用于迁移兼容。
- 所有下载、数据集、benchmark、训练输出和临时文件进入 `artifacts.local/`。
- 脚本若需要联网、GPU 或 ADB，应在命令或文档中显式说明。
- 新增脚本时同时补充配套测试、文档链接和默认输出目录。
- 文档治理脚本不产生项目输出；文档变更完成前运行 `scripts/check_docs_index.ps1`。
