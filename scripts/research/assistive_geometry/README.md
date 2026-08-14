# Assistive Geometry research scripts

状态：`B1_A0_PERMANENT_NEGATIVE_TERMINAL / R2_F0_SYNTHETIC_REDUCER_PASS / F1_SUPERVISION_FRONTDOOR_SATISFIED / AG_ST_DIRECT_TEACHER_TO_AG_REAL_SEAM_PASS / F1_STUDENT_ATTEMPT17_FAIL_NO_PROMOTION / AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / AG_R2_CROSS_SENSOR_CALIBRATION_CONTROL_R0_AND_R1_FAIL_CLOSED_CONSUMED / R1_INDEPENDENT_REPLAY_CONFIRMED_PRODUCER_FAILURE / CORRECTION_GAIN_LOPO_FAIL_STOP / ANGULAR_BOUNDARY_FAIL_CLOSED_SAFE_BUT_TASK_INERT / SUPPORT_VALIDITY_FAIL_OPEN / OBSTACLE_RGB_INTERACTION_FAIL_STOP / POSE_ANALYTIC_FAIL_STOP / QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING / CURRENT_OBSTACLE_TASK_ROUTE_CLOSED / SCIENTIFIC_NOT_RUN / CONFIRMATION_OUTCOMES_UNOPENED`

本目录包含 BlindAssist Assistive Geometry B0 的冻结合同、shape/export、metadata roster、
可恢复媒体物化与 label-blind integrity 工具：

2026-08-12 当前交付分两层：`run_ag_st_direct_teacher_to_ag_real_seam.py` 先把 source-anchored
SuperTeacher factors 直接接入冻结 adapter/reducer；随后 learned metric/factor recipe 加冻结的
session-height scale anchor，经 `FactorTensorAdapterV2` 在 checkpoint-unseen `sitting_rpy` 上完成最终
12-frame seam。最终为 12/12 valid、12/12 deterministic、11/11 gates、`CLEAR=18 / UNKNOWN=90`；
推理 `targets_loaded=false`，UNKNOWN 未转 negative，也没有任意 baseline 胜负门。Attempt17 与无锚
walking_xyz 负结果保持冻结；当前只支持 research-pipeline mechanics，不是跨传感器或移动部署结论。

`ag_r2_cross_sensor_confirmation/` 的 R0/R1 calibration-control one-shot 均已 fail closed consumed。R1 producer
读取 2 个 YAML 后以 `F2_R1_KALIBR_ROSTOPIC` 停止；matrix discovery 与 target-match count 保持
`null/UNKNOWN`。producer-free validator 先消费独占 replay receipt，再一次性复现同一失败并封存完整 chain；
离线验证 PASS。session archive、checkpoint、source truth、factor scoring 和 Confirmation 均未运行，R1 不得
rerun/resume/replace；该 Formal calibration 路径没有 active successor。当前算法 successor 仅为下述隔离的
factor-wise no-regret Development。

## 稳定 Interface

- `validate_b0_task_contract.py`：对 B0 JSON 合同执行 fail-closed schema/不变量检查；
- `test_validate_b0_task_contract.py`：覆盖有效合同和关键违规合同；
- `preflight_depthart_rectangular_shape.py`：用真实 DepthART-S metric checkpoint 验证
  `1×3×608×448` PyTorch shape、dynamic camera prompt 与 ONNX graph/checker。
- `audit_b0_data_capability.py`：只读 master ledger，区分结构候选与研究角色 authority；
- `plan_b0_arkitscenes_rosters.py`：按冻结 identity 排除快照生成 visit/video-disjoint `16/8/8` roster；
- `preflight_b0_arkitscenes_assets.py`：对五类冻结源资产执行 label-blind HEAD preflight；
- `download_b0_arkitscenes_assets.py`：历史 earliest-common materializer；其 Attempt 3 因 pose 覆盖失败，禁止复用；
- `audit_b0_arkitscenes_pose_coverage.py`：重算冻结窗口与 trajectory 时间域关系；
- `download_b0_arkitscenes_pose_covered_assets.py`：可恢复地物化 trajectory 域内连续 300 帧；
- `audit_b0_arkitscenes_integrity.py`：逐文件 SHA、实际图像解码、内参和 pose 包络审计。
- `arkitscenes_truth_reader.py`：按官方 inverse trajectory convention 将注册模态旋转到逐帧
  upright metric frame，并派生 gravity ground、三通道 body-swept clearance 与 UNKNOWN；
- `materialize_b0_arkitscenes_upsampling_train.py`：仅物化冻结 TRAIN role 的 exact-timestamp
  AppleDepth/FARO/RGB/confidence/intrinsics 对照；
- `validate_b0_arkitscenes_truth_reader.py`：运行 TRAIN-only scale/registration/ground/clearance
  双层门，并写入逐帧 evidence receipt。
- `validate_b1_training_protocol.py`：冻结并校验 B1 target/loss/confidence、A0–A4 additive arms、
  optimizer、数据角色和 implementation-before-training 防火墙。
- `audit_b1_orientation_geometry.py`：只读 pose/identity，审计 full-FOV portrait/landscape
  frame capacity，不打开 image/depth/task outcome；
- `validate_b1_training_protocol_attempt_02.py`：校验当前 dual-orientation overlay、orientation
  buckets、full-FOV K 传播、Development split 与 portrait claim ceiling。
- `materialize_b1_train_targets.py`：只为冻结 TRAIN identity 写入 compact source-upright target，
  不物化 prediction-dependent confidence truth；
- `validate_b1_train_targets.py`：逐 SHA 和 NPZ 语义验证 4,800 个 TRAIN target，并 fail-closed
  检查 UNKNOWN、方向、K、ground、clearance 与 occupancy；
- `assistive_geometry_model.py`：复用 DepthART-S shared decoder feature，提供 Ground、Clearance、
  Occupancy、Confidence heads 与 A0–A4 frozen losses；
- `depthart_training_scan.py`：训练时直接进入部署包内显式 custom Autograd Function，绕过没有
  Autograd-key registration 的外层 inference/export dispatcher；
- `smoke_b1_dual_orientation_training_model.py`：用冻结 checkpoint 在 portrait/landscape 全尺寸上
  验证 forward、loss、encoder/head backward 与 SelectiveScan dispatch boundary。
- `assistive_geometry_training.py`：提供 deterministic parent-balanced/orientation-bucket loader、
  same-orientation carry、augmentation、A0 cosine scheduler 与 collate 合同；
- `smoke_b1_a0_train_execution.py`：以真实 TRAIN 数据执行受限 optimizer step，写出并精确恢复
  model/optimizer/scheduler/scaler/sampler/RNG checkpoint；
- `smoke_b1_a0_train_execution_attempt_02.py`：保留 Attempt 1 RNG-device negative 后，将 checkpoint
  首次加载固定在 CPU；必须以 `-m scripts.research.assistive_geometry.smoke_b1_a0_train_execution_attempt_02`
  运行。
- `train_b1_a0_formal.py`：运行冻结的 A0 TRAIN-only 性能 pilot 与三 seed 正式训练；发布
  guarded progress，按 epoch 原子保存可恢复状态，并保留 `5/10/15/20` checkpoint。
- `train_b1_additive_arm.py`：A1–A4 共用的 outcome-blind 训练 mechanics；所有 arm 从同一
  DepthART 初始化独立训练，只逐项开放冻结 head/loss，等待 A0 Development 结果后另立协议激活。
- `evaluate_b1_a0_synthetic.py`：验证三 seed × 四 retained checkpoint 的 bytes/SHA、内部状态、
  协议与步数完整性，并计算 pooled、九格、parent 与 orientation task metrics；不选择 seed。
- `run_b1_a0_evaluation_dry_run.py`：只用合成 fixture 演练通过路径与 checkpoint 缺失、协议漂移、
  缺 horizon、全局零分母、coverage 塌缩、best-seed 企图等失败终态，并生成 JSON、短报告和
  failure-adjacent log。
- `materialize_b1_development_targets.py`：只有三 seed 正式结果完整时才物化冻结的四个
  `DEVELOPMENT_SELECTION` parent；Calibration 与 Confirmation fail closed。
- `observe_b1_a0_development.py`：用各 seed epoch-20 dense-depth checkpoint 与冻结 gravity/geometry
  后处理生成独立 truth/pred validity 和三态 observation；不读取未训练 task heads。
- `evaluate_b1_a0_development.py`：执行三 seed 无选择聚合，并同时检查 coverage、ground、clearance、
  false-clear/false-block、temporal delta 与 geometry transition 门。
- `analyze_b1_a0_failure_anatomy.py`：只读已消费、SHA-bound 的 A0 Development observations，分解
  tri-state 分布、clearance residual、false-block 阈值一致性、transition failure 和跨 seed
  failure-mask similarity；结果永久 `NOT_ELIGIBLE_FOR_PROMOTION`。
- `geometry_r2_reducer.py`：F0 冻结的零参数 interval reducer；只有 positive lower-bound evidence、
  guaranteed lateral overlap 与 horizon 内 upper-bound distance 同时成立才输出 occupied，歧义或缺失为 UNKNOWN。
- `fixtures/geometry_r2_f0_cases.json`：23 个 SHA-bound analytic factor case，覆盖 depth/scale、support、
  boundary、orientation、uncertainty monotonicity、反 A0 场景和 final-task shortcut 负控。
- `run_geometry_r2_f0_canary.py`：校验协议/实现/fixture SHA 后执行 10 项 conjunctive F0 kill gate，
  只写新 evidence root；不训练、不读真实数据、不自动授予 F1。
- `validate_geometry_r2_f1_protocol.py`：只做 F1-P schema、DCA capability、loss/checkpoint、Kill Gate、
  successor 与 execution-authority 的静态 SHA/语义校验，并断言 F1 trainer/model/materializer 路径不存在。
- `test_validate_geometry_r2_f1_protocol.py`：9 个 mutation tests，覆盖执行扩权、final-task shortcut、
  UNKNOWN-as-negative、能力计数漂移、aggregate checkpoint loss、reducer rescue 与 parent-role overlap。
- `audit_geometry_r2_f1_adapter_gap.py`：静态核对 byte-frozen F1 factor schema 与 F0 reducer input，
  显式列出 scale/support uncertainty、dense→obstacle list 和 camera/frame binding 的 17 个 adapter 操作；
  不实现 adapter、不运行 reducer/canary、不授予执行权限。
- `test_audit_geometry_r2_f1_adapter_gap.py`：7 个静态/mutation tests，验证缺 adapter 必须 fail closed，
  完整静态合同也最多到 `CANARY_NOT_RUN`，并拒绝 learned-graph、可训练参数或 execution 扩权。
- `validate_geometry_r2_f1_adapter_protocol.py`：验证 `14/14` F1 field consumers、全部 F0 field
  producers、17 个 operation、8-case fixture、A01–A10、authority/successor 与 exact SHA bindings；
  不实现或执行 adapter。
- `test_validate_geometry_r2_f1_adapter_protocol.py`：13 个 mutation tests，拒绝字段/operation 缺失、
  task shortcut、receipt/support/missing-depth fail-open、uncertainty strengthening、扩权与 binding drift。
- `factor_tensor_adapter.py`：按冻结 17-operation 合同将完整 F1 factor tensors 确定性转换为 F0 frame；
  零参数、learned-graph 外、无 task outcome，receipt 或局部证据无效时 fail closed。
- `test_factor_tensor_adapter.py`：10 个 focused tests，覆盖 8-case fixture、独立 UNKNOWN 语义、
  orientation parity、component split/merge、uncertainty monotonicity 与 final-task shortcut 拒绝。
- `run_factor_tensor_adapter_canary.py`：只接受 SHA-bound implementation lock，用两个独立进程重放
  每个 frozen case，并只写未存在的版本化 synthetic evidence root。
- `factor_tensor_adapter_v2.py`：把 global metric-scale sigma 与 local depth-shape sigma 分开传递，
  保留旧 adapter 的 fail-closed reducer ABI，不让任一 uncertainty 分量被另一分量覆盖。
- `ag_r2_cross_sensor_confirmation/`：冻结 ETH3D opaque binding/preflight、12+12 roster、RGB+K
  model-only prediction、两次 seal/reload firewall、session source geometry、27-gate factor-only metrics、
  exclusive evidence 与独立重算 validator；包导入不访问 archive/checkpoint，也不创建 evidence root。
- `run_ag_r2_cross_sensor_factor_accuracy_confirmation.py`：只接受另行冻结的 exact one-shot execution
  lock；当前 implementation lock 的全部 execution authority 为 false，不能据此启动真实执行。
- `ag_r2_cross_sensor_confirmation/validate_implementation_lock.py`：只读 tracked control files，复核
  历史 executor implementation predecessor/code/test byte+SHA、45-test receipt 与零 payload access。
- `ag_r2_cross_sensor_confirmation/control_format.py`：只接受 exact camera-node-scoped Kalibr camchain YAML
  nested `T_cam_imu`，拒绝 inline matrix、非正交旋转、重复 path 与无法唯一绑定的 camera node。
- `ag_r2_cross_sensor_confirmation/calibration_control.py`：未来只在独立 hash-bound one-shot 下消费独立
  control root，并且只有 camera-IMU calibration archive 一个 payload 输入；session archives 与模型没有 API。
- `ag_r2_cross_sensor_confirmation/validate_calibration_control.py`：不导入 producer/source adapter/control
  parser，独立重哈希 calibration archive、枚举 YAML、重算 matrix selection 与 evidence manifest。
- `ag_r2_cross_sensor_confirmation/control_format_r1.py`：纯解析每个 Kalibr camera node 的同节点
  `rostopic + T_cam_imu`，不以 `cam0/cam1` 顺序推断目标相机。
- `ag_r2_cross_sensor_confirmation/calibration_control_r1.py`：未来只在另行 hash-bound R1 one-shot 下按
  `/uvc_camera/cam_2` namespace 唯一匹配；失败时保存已知/未知计数、摘要与零 first/best selection receipt。
- `ag_r2_cross_sensor_confirmation/validate_calibration_control_r1.py`：不导入 R1 producer/source/parser，
  先写独占 start receipt，再对 PASS 或任意 fail-closed evidence 独立重放一次 archive、rostopic namespace
  selection 与完整计数；完成后可纯本地验签且不会重开 archive。
- `ag_r2_cross_sensor_confirmation/validate_calibration_control_r1_repair_lock.py`：复核 R0 sealed audit、
  preserved legacy runtime、official selection evidence、R1 amendment、9 个 implementation binding、
  69-test receipt 与零 archive access。
- `ag_r2_cross_sensor_confirmation/validate_depthart_source_manifest.py`：独立复核 29 个实际可导入的
  metric/selective-scan Python 文件及 bytes/SHA，不加载 checkpoint 或模型。
- `ag_r2_cross_sensor_confirmation/validate_repair_implementation_lock.py`：复核 schema v2、official control、
  source manifest、18 个 implementation binding、51-test receipt、零真实 payload 与不可自授权 successor。
- `train_ag_r2_multisource_metric_depth_student.py` 与 `train_ag_r2_f1_attempt18_consumed_cross_domain_adaptation.py`：
  只从分级 factor supervision 学 metric depth 与 support/obstacle/boundary/validity，不训练 final task state。
- `calibrate_ag_r2_session_metric_scale_anchor.py`：只在已消费 factor depth 上从六个预声明物理候选选择
  camera-height quantile anchor；选择过程不读取 CLEAR/OCCUPIED/UNKNOWN 或 reducer output。
- `run_ag_r2_hybrid_factor_student_to_ag_seam.py`：组合冻结 metric/factor checkpoints、可选 session anchor、
  分解 uncertainty 与 deterministic adapter/reducer；factor-only inference 可显式禁止加载目标。
- `materialize_ag_r2_tum_sitting_rpy_final_confirmation_labels.py` 与
  `run_ag_r2_tum_walking_xyz_final_v2_seam.py`：前者物化 source-native/geometry-anchored 12-frame labels，
  后者是参数化的一次性最终 seam runner；文件名保留 walking_xyz 历史，但 final parent/receipt 由参数锁定。
- `run_ag_st_stage0a.py`：独立 `WILD_LAB` factor-only runner；source role 参数化，可从 B0 raw manifest
  或 scoped-media manifest 恢复 RGB/K/pose/partial depth，以确定性连续块隐藏 reference，执行
  source-anchored MapAnything，并输出 baseline、校准前后 depth residual 与 confidence risk-coverage。
  它不读取 clearance/occupancy，不物化 canonical label，不改变 F1 execution authority。
- `test_run_ag_st_stage0a.py`：7 个 focused tests，覆盖隐藏 reference 不回流 Teacher input、mask 可重放、
  source-role 选择、scoped trajectory receipt、observed-anchor scale 与 confidence selective metrics。
- `build_ag_st_factor_labels.py`：把 Stage 0A source-first metric depth、anchor residual 与 multi-view
  reprojection residual 变成 A/B/C/UNKNOWN 分级 pseudo-label；输出 per-factor validity/provenance、
  support-plane、physical-boundary distance 和 uncertainty proxy，可直接供 masked student 读取。
- `test_build_ag_st_factor_labels.py`：8 个 focused tests，覆盖 hidden-reference 隔离、reprojection、
  source priority、uncertainty、派生 provenance、连续 80° 斜面负控与真实 depth-step 正控。
- `build_ag_st_multiteacher_factor_labels.py`：保留 source-native depth 与 MapAnything primary geometry，
  只用独立 Depth Anything V2 的 source-anchored 分歧重标 quality/uncertainty/UNKNOWN；分歧不会被当成
  truth，也不要求第二 Teacher 的全局误差低于主 Teacher。
- `test_build_ag_st_multiteacher_factor_labels.py`：3 个 CPU focused tests，覆盖 observed-anchor scale、
  pair disagreement 对称性、source tier 保留及强分歧转 UNKNOWN。
- `ag_st_tum_rgbd.py`：从 7 个 TUM RGB-D sequence 的目录或 TGZ 精确恢复 RGB、registered depth、K，
  并把官方 groundtruth trajectory 插值为 RGB 时刻的 camera-to-world pose。
- `run_ag_st_tum_cross_source.py`：在 4 FIT / 3 held-out TUM sequence 上原样复用 R0 multi-Teacher
  quality threshold；只在推理后打开隐藏 source depth，PASS 后物化 depth/uncertainty/UNKNOWN，
  未验证 gravity 前 support/boundary 恒为 UNKNOWN。
- `test_ag_st_tum_rgbd.py` 与 `test_run_ag_st_tum_cross_source.py`：6 个 focused tests，覆盖 cohort
  disjointness、RGB-depth pairing、pose interpolation、真实 payload receipts、跨源门与 source-first 标签不变量。
- `plan_ag_st_tum_third_teacher_cohort.py` 与 `run_ag_st_tum_third_teacher.py`：冻结另 7 个未引用 TUM
  parent，在 4 FIT 上选择 DepthART union/consensus witness，随后一次性评 3 held-out；若第三 Teacher
  无 no-regret 增益则保留原两教师配方，不回调阈值。
- `ag_st_depthart_teacher.py` 与 `test_run_ag_st_tum_third_teacher.py`：只读 RGB+K 的冻结 DepthART-S
  metric Teacher，以及 4 个 focused gate/role tests；DepthART 只作独立 witness，不要求击败主 Teacher。
- `diagnose_ag_st_tum_gravity.py`：把 TUM accelerometer 与 mocap pose 联合到 24 个 proper signed-axis
  candidates，验证 Freiburg1/2 的 IMU→RGB optical 映射与 world `+Z` gravity；无 accelerometer 的
  Freiburg3/Xtion 不作推断。
- `materialize_ag_st_tum_gravity_factors.py` 与对应 test：对 gravity-eligible TUM 标签物化 continuous
  normal/support/boundary/obstacle evidence；不具 gravity 的 parent fail-closed 为 UNKNOWN。dominant
  gravity-aligned plane 仍可能是桌面，不冒充 walkable-ground truth。
- `diagnose_ag_st_tum_support_identity.py`：把 source-native depth 通过 pose 投到 parent world frame，
  恢复跨帧持续水平高度模式，并用更低持续面识别 per-frame dominant plane 的桌面/高架误标。
- `materialize_ag_st_tum_support_identity_factors.py` 与 `validate_ag_st_tum_support_identity_factors.py`：
  用通过 identity 的 sequence height 重物化 TUM support/boundary pseudo-label，并验证 UNKNOWN、gravity
  alignment、camera-height binding 与旧/新 support-positive correction。
- `run_ag_st_analytic_support_boundary_canary.py`：解析 floor+dominant-table exact renderer；同时检验
  support identity、table false-positive 和 2px level-change boundary，全部为 deterministic CPU mechanics。
- `run_ag_st_icl_mesh_support_identity.py`：用 ICL-NUIM 官方 living-room OBJ 的 `room_floor` 与 global poses
  检验最低持续高度；只接受 upward-facing exact mesh surfaces，过低相机或稀疏视角保持 UNKNOWN。
- `materialize_ag_st_sequence_identity_labels.py` 与 `validate_ag_st_sequence_identity_labels.py`：把
  sequence identity 推广到 16-parent multi-Teacher TRAIN 标签；parent 至少 2/3 帧 camera-height plausible，
  每帧仍守 `0.45–2.20 m`，否则 factor denominator 为零。
- `train_ag_st_masked_student.py`：冻结 DepthART-S 或 MobileNetV3 encoder，只训练小型 dense factor head；
  可直接拼接多个互不重叠的 Stage0A/label batch，每个 orientation 留 1 selection + 1 canary parent；
  也可在已有独立 confirmation 时把全部 consumed parents 纳入 fit。支持 multifactor、depth/support-only、
  metric-precision + calibrated-support 与 boundary-only 目标；train-only scalar temperature/bias 可折叠回
  support head。也支持 DepthART 四层 decoder pyramid、dilated pyramid head 与显式 base-depth guidance。
  A/B/C tier weights 保留，UNKNOWN 权重恒为零，不调用 reducer 或 task outcome。
- `test_train_ag_st_masked_student.py`：12 个 focused tests，覆盖 16/32-parent split 重放、非对称
  orientation roster、零残差与 identity-gate 初始化、multi-scale/base-depth head、objective
  UNKNOWN/NaN 隔离、tier 权重和 scalar calibration。
- `train_ag_st_bonn_anchored_student.py`：把三批 40-parent ARKit factor labels 与冻结 Bonn FIT 的
  registered source depth 合并；Bonn 只提供 A-tier depth，其他 factor 全 UNKNOWN。使用 5x Bonn 重放形成
  domain-balanced optimizer visits，并训练初始回退 DepthART base 的 `identity_sigmoid` correction gate。
- `test_train_ag_st_bonn_anchored_student.py`：2 个 focused tests，锁定 8/8 cohort disjointness、排除旧
  fixed-8，并验证 Bonn adapter 只开放 source depth、其余 factor 分母恒为零。
- `train_ag_st_no_regret_selector.py`：冻结 base 与 correction expert，只训练/评价 base-vs-correction
  selective router，并报告 perfect signed-advantage oracle 的安全 coverage/headroom。threshold admission
  要求每个 calibration parent 的 MAE 与 `>0.10 m` error 都 no-regret，且至少一半 parent 有非零
  correction coverage；没有 admissible threshold 时确定性回退 base。
- `test_train_ag_st_no_regret_selector.py`：9 个 focused tests，覆盖 deterministic split、可扩展 TUM
  calibration parent 数、oracle headroom、
  fallback，以及“macro 改善但单 parent 受伤”必须拒绝的回归测试。
- `run_ag_factorwise_no_regret_oracle_parent_gate_canary.py`：重放冻结 prior/expert/selector，并显式加入
  perfect signed-advantage oracle；逐 parent 同时约束 MAE 与 `>0.10 m` error，至少一半 parent 必须有
  非零 coverage，R21 boundary 只作 SHA/结果重放。
- `train_ag_st_frame_advantage_lcb_router.py`：冻结 pixel selector/correction expert，以 neural quantile
  ensemble 和跨 parent kNN lower bound 形成只会 veto 的 frame gate；可把 checkpoint fallback 与显式
  Development pixel candidate threshold 分开记账，也可限制为 TUM-only 并纳入已消费 evaluation parent。
- `evaluate_ag_st_frame_advantage_lcb_router_tum.py`：对冻结 frame gate 执行 parent-disjoint TUM 评估，
  检查 fit/calibration firewall，并报告 neural/kNN 分数、逐 frame 真值 advantage 与严格 parent gate；
  consumed 诊断不能重新包装成 fresh evidence。
- 对应 8 个 focused tests 覆盖 gate、fallback、pinball asymmetry、veto-only、kNN parent exclusion 与
  selector threshold provenance；当前 TUM14 结果为 runtime-observability fail-stop，不授权继续用同一
  observable 重训。
- `run_ag_runtime_correction_gain_observability_canary.py`：在 14 个 consumed TUM parent 上以
  leave-one-parent-out 检验 flip equivariance、temporal reprojection 与合取 observable；三候选均为全
  fallback，按停止条件关闭当前 correction expert/router，未打开 fresh3。
- `run_ag_angular_boundary_body_swept_task_canary.py`：固定 ICL source-exact depth/support/obstacle，仅替换
  R20/R21 boundary probability；R21 虽提高 task-reference agreement，但 naive conjunction 会产生
  unsupported CLEAR，不能晋级。
- `run_ag_angular_boundary_fail_closed_task_canary.py`：把 boundary absence 改为 UNKNOWN，只允许通过
  component-edge localization sigma 增加不确定性；危险放行归零，但 R20/R21 task state 完全相同，故
  boundary-to-task mapping 停止。对应 focused tests 锁定 one-sided evidence 与 UNKNOWN 语义。
- `run_ag_positive_obstacle_support_task_effect_audit.py`：固定 source-exact depth/boundary，prediction-first
  执行 learned support、learned obstacle 与合取替换臂。结果显示 support 可越权打开 reference-UNKNOWN，
  obstacle 则安全但丢失 `23/33` known cells 且不产生 OCCUPIED；下一步只允许 obstacle 三态校准，support
  保持 veto-only。对应 focused tests 锁定单 factor 替换和 completion 保留。
- `run_ag_obstacle_evidence_tristate_calibration_canary.py`：冻结 R21 obstacle logit，在六个 checkpoint-held
  ARKit/TUM consumed parent 上完成 RGB+K prediction 后才打开 source-valid obstacle truth；嵌套前的首轮
  leave-one-parent-out 双阈值校准为 `0/6` 可评 fold，因此 current scalar score fail-stop，禁止 reducer seam。
- `run_ag_obstacle_selective_interaction_head_canary.py`：只组合 frozen obstacle/support/boundary/depth 与
  image-row observable；inner calibration parent 始终由排除自身的模型预测，outer parent 也只评一次。
  嵌套结果仍为 `0/6` 可评 fold，关闭同一 RGB factor observable 家族上的后续 selector。
- `run_ag_depth_pose_analytic_obstacle_canary.py`：只从 RGB+K 预测 DepthART metric depth，并把
  `camera_to_world` 作为 runtime-equivalent VIO/IMU pose-gravity；跨帧恢复最低水平高度后用冻结 factor
  geometry 解析 obstacle。`3/6` parent 几何完整但 `0/6` fold 获得安全双阈值，当前 obstacle task route
  fail-stop；对应 tests 锁定 persistent mode、world-z pose 变换与最小帧数。
- `run_ba_clear_qplane_o0a_representation_headroom.py`：两阶段检验 body-query-conditioned inverse-depth
  ray-plane residual。Phase A 只从 source-support inverse-depth residual 拟合并哈希 1080 个三参数向量；
  Phase B 才比较 A0–A5 与 shuffled/wrong-gravity/wrong-K/globalized 负控，且不保存 corrected dense depth。
  120 帧结果为 FAIL：A4 parent-macro MAE `0.17284 m`，差于 A1/A2/A3，不授权 O0-B 或训练。
- `test_run_ba_clear_qplane_o0a_representation_headroom.py`：7 个 focused tests，覆盖冻结 authority、
  三参数恢复、support/evaluation 零交集、临时深度不污染 base、UNKNOWN 记账和非对称 query margin。
- `replay_ba_clear_qplane_o0a_query_decomposition.py`：只读校验原 result/candidate SHA 后复算冻结 Phase B，
  补齐 9 个 `band@horizon` query 的独立指标；不 refit candidate、不改变参数、gate 或终态。
- `evaluate_ag_st_student_checkpoint.py`：在 parent-disjoint cohort 上零样本评估冻结 checkpoint；默认
  fresh 模式，也可显式签署 `consumed_development_comparison`，防止把已看过的 cohort 再包装成新证据。
- `test_evaluate_ag_st_student_checkpoint.py`：6 个 focused tests，覆盖 objective-specific core factors、
  all-consumed fit parent firewall、macro improvement、diagnostic split 与 consumed-mode fresh-claim 禁用。
- `evaluate_ag_st_student_bonn_depth.py`：默认固定 Bonn RGB-D Dynamic 8 sequence × 3 帧，也可读取
  SHA-frozen cohort manifest；以 used-set 形成唯一 RGB-depth 对。模型只读 RGB+K，推理后才打开
  registered source depth，比较 initialized DepthART 与 frozen student 的 parent-macro MAE/`>0.10 m`
  error。输出仅为 cross-dataset depth Development，不评价 support/boundary/task，也不作许可结论。
- `test_evaluate_ag_st_student_bonn_depth.py`：6 个 focused tests，覆盖固定 cohort、唯一 pairing、缺失
  depth member、uint16 `/5000`、parent-macro 与非单调 index fail-closed。
- `export_assistive_geometry_onnx.py`：把未来选定 checkpoint 导出为 portrait/landscape 静态 ONNX，
  保留五个 raw GeometryState tensor 与 host camera prompts；gravity/UNKNOWN 后处理不塞入图内。
- `evaluate_teacher_complementarity.py`：在未来另行授权的 truth-bound cohort 上比较 metric 与 temporal
  geometry 教师的单体、oracle、独占正确 parent、分歧错误浓度和时序优势；任一 kill gate 失败即停止 C1。
- `temporal_geometry_ablation.py`：为未来 phase D 提供同一 8-frame GeometryState 下的因果
  GRU/TCN/diagonal-SSM 候选，统一 future-clearance/TTC/compute-gate 输出和 50k 参数上限；不决定最终三态。
- `run_hypothesis_canary_lite.py`：只用 deterministic synthetic CPU geometry 审查 censored
  survival、profile-conditioned clearance、widest-path bottleneck 与 one-sided conformal
  uncertainty 的数学不变量和反例；不读取任何数据 role outcome、模型或 checkpoint。

## 输出

大体积输出只允许写入 `artifacts.local/datasets/`、`artifacts.local/experiments/`、
`artifacts.local/evidence/hftf/` 或 `artifacts.local/evidence/assistive-geometry/`。
roster 选择只依据冻结 metadata/hash，不读取模型输出或 task outcome。当前合同和结果真源位于
`docs/research/assistive-geometry/`。

## 安全边界

本模块的 B1-A0 及 A1–A4 已永久关闭；teacher 只有未激活的历史 C0 complementarity mechanics，
历史 C0 teacher 路线当前不读取 teacher output，也不授权 C1、QNN/HTP、默认 App、产品或 safety。
独立 AG-ST 已在 `WILD_LAB` 中完成 MapAnything Stage 0A、factor-label 物化和冻结 DepthART encoder
masked-head 训练。两条独立的 train-parent 到 fresh-parent 零样本链均复现 depth/support 学习信号；
obstacle 一条改善、一条退化，只是 diagnostic；boundary 在 multifactor、boundary-only 和两条 fresh
zero-shot 评价中均未通过。后续 combined-32 depth/support-only checkpoint 又在 8 个全新 parent 上取得
depth MAE `85.8%` 和 support BCE `47.5%` 的 parent-macro 相对下降；累计 44 个互异 ARKitScenes parent
被消费后，combined-40 precision checkpoint 在最终 CONFIRMATION 8 上取得 depth MAE `85.4%`、support
BCE `83.2%` 的相对下降。累计 52 个互异 ARKitScenes parent 已消费，这仍不是跨数据源泛化。
后续 multi-scale head 在已消费 confirmation 上结果混合且不晋级；两个 ARKit-trained depth residual head
在 Bonn registered depth 上又分别把 DepthART MAE `0.2533 m` 恶化到 `1.0146 / 1.1755 m`，均为
`0/8` parent 改善。因此当前 depth transfer 明确不支持，下一算法必须保留域外 identity fallback 或加入
source-diverse metric anchors，不能继续用同域 head-capacity scaling 代替跨数据源监督。
后续 mixed-domain identity-gated student 已把新 Bonn EVAL MAE 收回到 `0.2713 m`，大幅消除上述
catastrophic collapse；但仍差于冻结 DepthART baseline `0.2517 m`，且只有 `1/8` parent 改善，故不晋级。
当前 factor-wise Development 先比较 perfect signed-advantage oracle 与现有 selector，回答 correction 是否
存在安全覆盖；只有 oracle 有 headroom 而 selector 失败时，才训练 one-sided advantage-LCB router。selector
准入已从 domain/macro no-regret 收紧到逐 parent 双指标 no-regret 与非集中 coverage，不在已看 EVAL 调 threshold。
support/boundary 仍是 conservative pseudo-label、
sigma 是 proxy，不产生完整 truth、正式 F1、产品或 safety authority。
时序模块同样只有未激活 mechanics；没有新 temporal cohort、训练、任务收益或设备性能 authority。
移动导出受历史 M0 质量先于性能协议约束；现有 DepthART D1 cohort 不得复用为 Assistive Geometry
选模证据。新 R2 已完成 F0 reducer mechanics、F1-P schema/loss/selection/Kill Gate、正式 source-native
supervision frontdoor 和 deterministic `FactorTensorAdapter`；adapter synthetic evidence 仍为
8/8 cases、10/10 gates、8/8 双进程 replay 与 7/7 sigma mutation PASS。其后 direct SuperTeacher
real seam 又在 12 帧上完成全部 tensor/adapter/reducer receipt。该结果证明 reference factor mechanics，
不证明 learned student、移动推理或真实任务收益。
`UNKNOWN` 不得当作负例；synthetic shape 与 benchmark geometry 不得冒充任务质量。

## 停止条件

合同违规、checkpoint/shape 不匹配、非 finite 输出、camera prompt drift 或 ONNX checker
失败均立即 fail closed。当前 roster、source integrity、truth reader 与 registration 已关闭，
且 B1 target/loss/confidence、dual-orientation overlay、4,800-frame target cache 与模型/loss
implementation lock 与 A0 execution lock 已关闭，三个正式 seed 均完成。合成 evaluator dry-run
与真实 Development Selection 评价均已执行；A0 虽通过前门，但 clearance MAE、false-block 和
geometry transition agreement 均为 `0/3` seed 通过，终态为
`B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES`。旧 A1 条件 successor 未激活，A1–A4、teacher、
移动和时序执行继续禁止。只读 failure anatomy 已完成且不可晋级；Selection 已消费且不得复用，
Calibration 与 Confirmation 保持封存。R2 F0 已签署
`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS`；历史 F1-P frontdoor 和
adapter blocker 后续均已关闭。当前冻结终态是
`AG_R2_SUPERTEACHER_TO_AG_FINAL_V2_SEAM_PASS / QPLANE_O0A_REPRESENTATION_HEADROOM_FAIL_CLOSE_NO_TRAINING / CONFIRMATION_OUTCOMES_UNOPENED`。
历史 ETH3D implementation-lock successor 已关闭，不提供当前执行权限。Q-Plane O0-A 是最后一次
materially different scale/plane representation 重开并失败；当前无 active Assistive Geometry successor。
不得枚举/解压七个 opaque archive、运行 Confirmation、结果后调参/重跑 Q-Plane、创建 O0-B 或训练 learned
Q-Plane head，也不得把任何 consumed result 写成 HTP、默认 App、产品或 safety 结论。

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/assistive_geometry -p "test_*.py"
```
