# USTRF route-target evidence closure

状态：R1 `DATA_BLOCKED / STOP_SOURCE_SEARCH` / elastic evidence R1 active / JRDB cross-sequence support/bias R0 `CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION / VALID` / JRDB sensor support/bias R0 `SENSOR_SUPPORT_AND_BIAS_PROFILE_AVAILABLE_WITH_ABSTENTION / VALID` / JRDB native multisensor P1B `NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT / VALID` / P2 R0 history `FAIL_CLOSED_LABEL_JOIN / VALID` / P2 R1 `ANNOTATION_DERIVED_PERSON_GEOMETRY_AVAILABLE_WITH_ABSTENTION / VALID` / candidate replay R2 `CANDIDATE_REPLAY_COMPLETE / VALID` / R2-L1 metric profiles `METRIC_PROFILES_COMPLETE / VALID` / route-invalid + reset-scoped lifecycle `MECHANISM_DIAGNOSTIC_COMPLETE / VALID / overall gate false` / eligible-attribution ordered isolated opening `ORACLE_MECHANISM_REPAIR_DIAGNOSTIC_COMPLETE / VALID` / truth-blind causal per-track token R0 `HOLD_FOR_POLICY_GATE / VALID` / causal token policy/risk R1 `POLICY_COVERAGE_REJECT / VALID` / policy failure attribution R1 `POLICY_FAILURE_ATTRIBUTION_CLOSED / VALID` / current-input policy feasibility bound R0 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID` / causal route-intrusion signal R0 `SIGNAL_REJECT / VALID` / route-conditioned scale-growth R0 `FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED / VALID` / canonical observation authority R0 `SOURCE_AUTHORITY_ABSENT / VALID` / L2+L3 prereg frozen but not authorized

## JRDB person 3D trajectory sensor support and bias cross-sequence replication R0

在读取任何新支持结果前，只按 timestamp / ZIP central-directory metadata 一次性冻结 3 个新 train sequence × 120 帧；input packet/eligibility 全部物化后再整体 hash-bind。support 与 validator hash-check 并直接复用原 R0 PCD、oriented-box、`>=3` 点门、四类 ledger 与 quantile kernel。结果 pooled object/pair support 为 `83.08% / 80.81%`，residual median/P95 为 `0.168/0.446m`；3D-only residual 方向复现，但远距仅 1/3 sequence 可评，终态 `CROSS_SEQUENCE_PROFILE_AVAILABLE_WITH_PARTIAL_REPLICATION / VALID`。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CROSS_SEQUENCE_REPLICATION_R0_RESULT_2026-07-25.md)。

```powershell
$env:PYTHONPATH="artifacts.local/work/python-deps/rosbags-cpu-20260720;scripts/research/ustrf_route_target_evidence_closure"
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json --phase materialize-inputs
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json --phase freeze-inputs
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json --phase support
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.json
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/test_jrdb_person_3d_trajectory_sensor_support_and_bias_cross_sequence_replication_r0.py -v
```

## JRDB person 3D trajectory sensor support and bias canary R0

真实解码 Meyer Green 120 帧的 240 份 `binary_compressed` PCD，分别保留 upper/lower 支持后，以冻结 oriented-box 合同审计父 R1 的 annotation-derived 轨迹。结果为 1,105/1,350 object-frame、1,044/1,336 pair sensor-supported；局部 0 点、1–2 点和 2D-only 不升级整段失败。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_JRDB_PERSON_3D_TRAJECTORY_SENSOR_SUPPORT_AND_BIAS_CANARY_R0_RESULT_2026-07-25.md)。

```powershell
$env:PYTHONPATH="scripts/research/ustrf_route_target_evidence_closure"
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0.json
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0.py --repo . --config configs/ustrf_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0.json
.\.python311\python.exe scripts/research/ustrf_route_target_evidence_closure/test_jrdb_person_3d_trajectory_sensor_support_and_bias_canary_r0.py -v
```

## Elastic evidence standard / JRDB person geometry R1

R1 不覆盖 R0 receipt，而是修正 claim dependency：结构完整性、逐 claim availability 与 authority ceiling 分轴报告；普通缺失只在最小依赖单元 abstain，且 `expected = eligible + abstained + invalid`。同一 R0 packet 上，1,350 个 3D geometry 与 1,336 个 adjacent annotation-derived motion pair 全部可计算；29 个 3D-only 只降低 cross-modal coverage。详见[弹性标准](../../../docs/research/ustrf-sc/USTRF_ELASTIC_EVIDENCE_AND_DEGRADATION_STANDARD_R1.md)与 [R1 结果](../../../docs/research/ustrf-sc/USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R1_RESULT_2026-07-25.md)。

```powershell
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/validate_ustrf_elastic_evidence_and_degradation_standard_r1.py --repo . --config configs/ustrf_elastic_evidence_and_degradation_standard_r1.json
$env:PYTHONPATH="artifacts.local/work/python-deps/rosbags-cpu-20260720;scripts/research/ustrf_route_target_evidence_closure"
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1.py --repo . --config configs/ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1.json
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1.py --repo . --config configs/ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r1.json
```

## JRDB single-sequence native multisensor person geometry canary R0

Meyer Green 前 120 帧的 120 stitched RGB、240 PCD、2D/3D labels、bag RGB/LiDAR header、动态 pose、IMU 与静态 TF 已物化为 immutable packet。第二进程从 raw payload + bag 精确重建 packet/receipt；clock、PCD、frame chain 与 interpolation 门通过，但 29/1,350 个 3D object-frame 无唯一同帧 2D `label_id`，故 `FAIL_CLOSED_LABEL_JOIN / VALID`，motion 未计算。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_JRDB_SINGLE_SEQUENCE_NATIVE_MULTISENSOR_PERSON_GEOMETRY_CANARY_R0_RESULT_2026-07-25.md)。

```powershell
$env:PYTHONPATH="artifacts.local/work/python-deps/rosbags-cpu-20260720;scripts/research/ustrf_route_target_evidence_closure"
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.py --repo . --config configs/ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.json --phase materialize
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/run_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.py --repo . --config configs/ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.json --phase audit
E:\codex-tools\projects\blindassist\toolchain\python311\python.exe scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.py --repo . --config configs/ustrf_jrdb_single_sequence_native_multisensor_person_geometry_canary_r0.json
```

## JRDB single-rosbag native pose / IMU / time authority canary R0

27 条 train bag 中最小的 Meyer Green member 通过 704 MiB 单成员门获取并 CRC/SHA 绑定。原生动态 `odom -> base_link` TF、`imu/data` 与上下 Velodyne 均覆盖外部前 120 帧 timestamp，第二进程完整重解码为 `NATIVE_POSE_IMU_TIME_AUTHORITY_PRESENT / VALID`。这只完成 P1B 并允许另立 P2；不计算 person-relative motion，不开放 route/event/safety。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_JRDB_SINGLE_ROSBAG_NATIVE_POSE_IMU_TIME_AUTHORITY_CANARY_R0_RESULT_2026-07-25.md)。

```powershell
python scripts/research/ustrf_route_target_evidence_closure/acquire_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.py --config configs/ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.json --output-bag artifacts.local/datasets/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/meyer-green-2019-03-16_0.bag --output-receipt artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/acquisition.json
$env:PYTHONPATH="artifacts.local/work/python-deps/rosbags-cpu-20260720;scripts/research/ustrf_route_target_evidence_closure"
python scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.py --repo . --config configs/ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.json --bag artifacts.local/datasets/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/meyer-green-2019-03-16_0.bag --acquisition artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/acquisition.json --output artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/receipt.json
python scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.py --repo . --config configs/ustrf_jrdb_single_rosbag_native_pose_imu_time_authority_canary_r0.json --bag artifacts.local/datasets/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/meyer-green-2019-03-16_0.bag --acquisition artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/acquisition.json --receipt artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/receipt.json --output artifacts.local/evidence/jrdb-single-rosbag-native-pose-imu-time-authority-canary-r0/validation.json
```

## JRDB native pose / 3D person motion authority audit R0

该 P1 只读取 40 GB rosbag、22.3 GB images、11 GB pointcloud 与 labels 的 ZIP central directory，并有界解压两个 label JSON；不下载完整 archive。选定 train sequence 的前 120 帧在 RGB、双 Velodyne、timestamp 与 2D/3D track 目录中完整，静态 robot-camera-LiDAR transform 与同名 rosbag 均存在；但 dynamic pose 和 IMU 仍缺直接 topic/message/header-time 覆盖核验，故 P2 关闭。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_JRDB_NATIVE_POSE_AND_3D_PERSON_MOTION_AUTHORITY_AUDIT_R0_RESULT_2026-07-25.md)。

```powershell
python scripts/research/ustrf_route_target_evidence_closure/audit_jrdb_native_pose_and_3d_person_motion_authority_r0.py --repo . --config configs/ustrf_jrdb_native_pose_and_3d_person_motion_authority_audit_r0.json --output artifacts.local/evidence/jrdb-native-pose-and-3d-person-motion-authority-audit-r0/receipt.json
python scripts/research/ustrf_route_target_evidence_closure/validate_jrdb_native_pose_and_3d_person_motion_authority_r0.py --repo . --config configs/ustrf_jrdb_native_pose_and_3d_person_motion_authority_audit_r0.json --receipt artifacts.local/evidence/jrdb-native-pose-and-3d-person-motion-authority-audit-r0/receipt.json --output artifacts.local/evidence/jrdb-native-pose-and-3d-person-motion-authority-audit-r0/validation.json
python scripts/research/ustrf_route_target_evidence_closure/test_jrdb_native_pose_and_3d_person_motion_authority_r0.py
```

## Canonical observation authority / repairability audit R0

A 进程不复用会先解码 candidate/lifecycle 的旧 scale producer，而是从 source bundle、source frame ledger、canonical observation transport 和逐帧 RGB 独立重建 41 条 frame-ledger。结果覆盖 `41/41` sequence、`62,229/62,229` frame 与 263,680 person box：source geometry、RGB、capture time、frame membership 为 authoritative，bbox frame 为 verifiable transform；canonical transform 全部 unknown，authoritative severe truncation 全部 absent。

B 新进程先复验 inventory SHA，再只读取 aggregate denominator projection；event/cell/negative identifier 与 truth/oracle/outcome/signal/candidate 解码均为 0。由于 severe truncation 全局 absent，availability 上界为 `0/11` independent event、`0/33` mechanical cell、`0/836` negative interval，第三进程复算为 `SOURCE_AUTHORITY_ABSENT / VALID`。G1、signal、Android 与更高权限保持关闭；详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_CANONICAL_OBSERVATION_AUTHORITY_AND_REPAIRABILITY_AUDIT_R0_RESULT_2026-07-25.md)。

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_canonical_observation_authority_inventory_r0.py --repo . --config configs/ustrf_canonical_observation_authority_inventory_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_canonical_observation_denominator_availability_r0.py --repo . --config configs/ustrf_canonical_observation_denominator_availability_r0.json --inventory artifacts.local/evidence/ustrf-canonical-observation-authority-repairability-r0/authority-inventory-r0.json --inventory-sha256 <frozen-inventory-sha256>
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_canonical_observation_authority_repairability_r0.py --repo . --config configs/ustrf_canonical_observation_denominator_availability_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_canonical_observation_authority_repairability_r0.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure audit_jrdb_rgb_time_frame_transform_access_canary_r0.py --repo . --config configs/ustrf_jrdb_rgb_time_frame_transform_access_canary_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_jrdb_rgb_time_frame_transform_access_canary_r0.py --repo . --config configs/ustrf_jrdb_rgb_time_frame_transform_access_canary_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure materialize_jrdb_single_frame_rgb_time_transform_canary_r1.py --repo . --config configs/ustrf_jrdb_single_frame_rgb_time_transform_canary_r1.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_jrdb_single_frame_rgb_time_transform_canary_r1.py --repo . --config configs/ustrf_jrdb_single_frame_rgb_time_transform_canary_r1.json
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts/run_research_tool.py ustrf-route-target-evidence-closure audit_jrdb_rgb_continuity_egomotion_availability_r0.py --repo . --config configs/ustrf_jrdb_rgb_continuity_egomotion_availability_r0.json
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_jrdb_rgb_continuity_egomotion_availability_r0.py --repo . --config configs/ustrf_jrdb_rgb_continuity_egomotion_availability_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure audit_ustrf_observability_program_authority_terminal_r0.py --repo . --config configs/ustrf_observability_program_authority_terminal_r0.json
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_ustrf_observability_program_authority_terminal_r0.py --repo . --config configs/ustrf_observability_program_authority_terminal_r0.json
```

## Route-conditioned scale growth separability R0

`configs/ustrf_route_conditioned_scale_growth_separability_r0.json` 冻结 normalized bbox-area `S_t=0.5*log(w_norm*h_norm)`、past-only 600ms / 至少 5 观测 / 最大 150ms gap、真实 timestamp 与 Theil–Sen slope；唯一可扫描变量为全部实际 slope breakpoint。父 evaluator 没有 alertable deadline，本轮在任何 signal outcome 前独立冻结 5000ms event-window delay 门，并明确不冒充父门。

producer-preflight 复核全部父 SHA 和 123 条候选投影，折叠为 41 序列 / 62,229 帧后发现：逐帧 source-size 与 rotation receipt 均未绑定，263,680 个 observed-track 也没有 severe-truncation authority。因此在 signal score、truth/event/oracle/negative/candidate decode 均为 0 时合法终止为 `FAIL_CLOSED_INPUT_OR_CONTRACT_BLOCKED`；inventory、frontier 与 candidate 均未生成。独立 audit 和 validator 在不同进程复算为 `VALID`。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_CONDITIONED_SCALE_GROWTH_SEPARABILITY_R0_RESULT_2026-07-25.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure test_route_conditioned_scale_growth_separability_r0.py -v
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_route_conditioned_scale_growth_separability_r0.py --repo . --config configs\ustrf_route_conditioned_scale_growth_separability_r0.json
```

## Candidate-independent causal route-intrusion signal R0

本轮加入 track/relation/timing family 不含的新测量变量，而不是继续扩 policy：对同一 track/reset 的 5 帧 causal history，计算 bbox footpoint 相对 route UV 的径向接近、横向收敛及 normalized bbox-height expansion，固定 `2-of-3` 符号门。producer 先证明 123 条候选投影 bbox-exact，再折叠为 41 序列 / 62,229 帧；冻结 inventory 前 truth/event/oracle/负暴露解码均为 0。

结果为 `SIGNAL_REJECT / VALID`：1,903 个激活只覆盖 `7/11 = 21/33`，低于旧 timing family 的乐观 `8/11 = 24/33`；负暴露有 `43` 个，即 `8.6759/min`，远高于 `<=2 / <=0.50/min`。该信号直接淘汰，不调窗口/组合/阈值，不生成 policy 或连接 opener。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_CAUSAL_ROUTE_INTRUSION_SIGNAL_R0_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_causal_route_intrusion_signal_r0.py --repo . --config configs/ustrf_causal_route_intrusion_signal_r0.json
.\.venv-export312\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure test_causal_route_intrusion_signal_r0.py -v
```

## Current-input policy feasibility bound R0

`configs/ustrf_current_input_policy_feasibility_bound_r0.json` 冻结 current-input monotone lease family：同一共享 active-relation 持续时长、至少连续 2 帧、one-token-per-track/reset、fail-closed、no-renewal。求解先把 36 个 candidate cell 去重为 12 个候选无关事件，再穷尽全部正整数 duration breakpoint；为形成 coverage 上界只乐观忽略 nominal TTL，不跨越 relation/route/track/reset 失效，也不输出任何 threshold、TTL、activation map、witness 或候选 policy。

结果为 `CURRENT_INPUT_POLICY_FAMILY_NOT_FEASIBLE / VALID`：完整 frontier 的最大 coverage 仅 `8/11 = 24/33`；冻结负暴露 `4.956min` 在 `0.50/min` 点率门下最多容许 2 个负 token，而风险约束下 coverage 上界仅 `2/11 = 6/33`。可信风险证据仍不足且没有被冒充为不可行原因。不能继续调资格时长/TTL/renewal，也不能接 opener；若继续，须另行预注册新的候选无关因果判别信号。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_CURRENT_INPUT_POLICY_FEASIBILITY_BOUND_R0_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_current_input_policy_feasibility_bound_r0.py --repo . --config configs\ustrf_current_input_policy_feasibility_bound_r0.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\validate_current_input_policy_feasibility_bound_r0.py --repo . --config configs\ustrf_current_input_policy_feasibility_bound_r0.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_current_input_policy_feasibility_bound_r0.py -v
```

## Candidate-independent policy failure attribution R1

`configs/ustrf_candidate_independent_policy_failure_attribution_r1.json` 只读绑定并复算冻结 policy gate 的 inventory、risk、terminal 与 oracle。归因按 timestamp 半开有效期进行；96 次 miss oracle qualification opportunity 各自唯一分类，同时让 24 个 cell 保留 mixed 原因集合。

结果为 `POLICY_FAILURE_ATTRIBUTION_CLOSED / VALID`：opportunity 为资格不足 `39`、TTL 后 oracle `39`、relation gap 提前失效 `12`、route unknown 提前失效 `6`、track `0`；24 个 miss cell 中 6 个为混合原因。34 个负暴露 token 全部按 source/sequence/invalidation reason 归因：TTL `16`、relation gap `9`、track unobserved `8`、route unknown `1`。该结果不修改 policy，也不授权 successor policy 或 opener。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_CANDIDATE_INDEPENDENT_POLICY_FAILURE_ATTRIBUTION_R1_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_candidate_independent_policy_failure_attribution_r1.py --repo . --config configs\ustrf_candidate_independent_policy_failure_attribution_r1.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\validate_candidate_independent_policy_failure_attribution_r1.py --repo . --config configs\ustrf_candidate_independent_policy_failure_attribution_r1.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_candidate_independent_policy_failure_attribution_r1.py -v
```

## Candidate-independent causal token policy/risk gate R1

`configs/ustrf_candidate_independent_causal_token_policy_risk_gate_r1.json` 在任何 R1 输出前冻结 `2 frames + 500ms` active-relation 资格、500ms token TTL、reset/route unknown/track unobserved/relation gap/TTL 的 fail-closed 失效，以及同 track/reset 再资格化只记账不重发。producer 只重放父 R0 的 candidate-independent runtime facts；41 条 policy ledger / 62,229 帧全部落盘并冻结 inventory 后，第二进程才读取 oracle 与半开负暴露。

结果为 `POLICY_COVERAGE_REJECT / VALID`：supported oracle cell 在 token 有效期内仅 `9/33`，3 个无 active relation cell 继续关闭；完整序列 `1,448` token、`1,445` extra，负暴露 `34/4.956min=6.86/min`，95% Poisson UCB `9.13/min`，并且两个 LILocBench source 不满足每 source 3 sequence 的 cluster floor。不能接 isolated opener，也不能用扩负样本回救 coverage reject。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_CANDIDATE_INDEPENDENT_CAUSAL_TOKEN_POLICY_RISK_GATE_R1_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_causal_token_policy_risk_gate_r1.py --repo . --config configs\ustrf_candidate_independent_causal_token_policy_risk_gate_r1.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_causal_token_policy_risk_gate_r1.py --repo . --config configs\ustrf_candidate_independent_causal_token_policy_risk_gate_r1.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\validate_causal_token_policy_risk_gate_r1.py --repo . --config configs\ustrf_candidate_independent_causal_token_policy_risk_gate_r1.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_causal_token_policy_risk_gate_r1.py -v
```

## Truth-blind causal per-track attribution-token producer audit R0

`configs/ustrf_truth_blind_causal_per_track_attribution_token_producer_audit_r0.json` 只允许 detector/T0 track、per-track route relation、route validity 与 reset。producer 进程先验证 C1/C2/C3 的 runtime 输入投影在 123 条 trace 上逐帧一致，再折叠并冻结 41 条候选无关 full-sequence ledger / 62,229 帧；truth、event window、oracle token 在 inventory 冻结前的解码数均为 0。第二个进程先复验所有 ledger SHA，之后才联结既有 oracle token 与负暴露 mask。

结果为 `HOLD_FOR_POLICY_GATE / VALID`：33 个 oracle-supported candidate-event cell 达到 33/33 coverage，3 个无 active relation cell 继续关闭，unknown-route、cross-reset 和 duplicate token 都为 0。但 5,126 枚 producer token 中 5,113 枚为 extra；4.956 分钟负暴露内有 153 枚（30.87/min），另有 6,328 次重复激活被记录并抑制。该结果不允许连接 isolated opener；下一边界必须先冻结并通过 candidate-independent token policy/risk gate。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_TRUTH_BLIND_CAUSAL_PER_TRACK_ATTRIBUTION_TOKEN_PRODUCER_AUDIT_R0_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_causal_per_track_attribution_token_audit_r0.py --repo . --config configs\ustrf_truth_blind_causal_per_track_attribution_token_producer_audit_r0.json --phase producer
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_causal_per_track_attribution_token_audit_r0.py --repo . --config configs\ustrf_truth_blind_causal_per_track_attribution_token_producer_audit_r0.json --phase audit
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\validate_causal_per_track_attribution_token_audit_r0.py --repo . --config configs\ustrf_truth_blind_causal_per_track_attribution_token_producer_audit_r0.json
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_causal_per_track_attribution_token_audit_r0.py -v
```

## Eligible target attribution → isolated one-shot opening R1

`configs/ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json` 绑定上一轮 failure-attribution config/terminal/validation 与 event-scope blind inventory。它先从父 full blind trace 重建每个 event 的 reset-scope 前缀，再联结冻结 proxy/model truth，生成并冻结 36 个 eligible-attribution token ledger。one-shot opener 明确消费 truth-derived event token/window scope，但拒绝 raw truth、observed box、baseline key、delivery 与 guard event 输入。

结果为 `ORACLE_MECHANISM_REPAIR_DIAGNOSTIC_COMPLETE / VALID`：父 formed-delivery 的 6 个单元仍 token-qualified，27 个 pre-open/quarantine 单元反事实恢复，C1/C2/C3 均为 `11/12`；唯一从未形成 active relation 的 event 在三候选上继续 fail closed。36 次 background namespace 强制变异均不改变目标 opening，27 个恢复单元都在 qualification 前或当帧有 background activity；opening-before-qualification、one-shot cardinality violation、duplicate key 与 accounting gap 均为 0。该阶段是 truth-assisted oracle upper bound，不是可部署修复；causal candidate-blind token producer、候选比较、selection、clearance 修复及所有更高权限均为 0。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_ELIGIBLE_TARGET_ATTRIBUTION_ORDERED_ISOLATED_OPENING_R1_RESULT_2026-07-24.md)。

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_eligible_target_attribution_ordered_isolated_opening.py --config configs\ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json --repo .
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\validate_eligible_target_attribution_ordered_isolated_opening.py --config configs\ustrf_eligible_target_attribution_ordered_isolated_opening_r1.json --repo .
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_eligible_target_attribution_ordered_isolated_opening.py -v
```

## Route-invalid + reset-scoped lifecycle diagnostic R1

`configs/ustrf_route_target_route_invalid_reset_lifecycle_diagnostic_r1.json` 只读取 A2 的 123 条权威 trace，在冻结候选之外加入单一 lifecycle guard：route invalid 同帧终止 active，discontinuity reset 终止旧 scope，episode key 绑定 source/sequence/reset/local key/activation ordinal。guard 不读取 truth；123 条 guarded trace 全部构造后，才用既有 12-event clearance 分母检查同 scope 的 known-route relation closure。

结果为 `MECHANISM_DIAGNOSTIC_COMPLETE / VALID`，但 overall gate 为 false。三个候选的 unknown/stale active 帧从 `12,621 / 7,165 / 12,759` 降为 0，跨 reset key 为 0；clearance 仍为 `0/12 / 1/12 / 0/12`。`route_invalid` 与 `reset_scope_end` 只表示 fail-closed terminalization，永不计 truth clearance。没有候选重跑、比较、selection、consume timestamp 修复或更高权限。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_INVALID_RESET_LIFECYCLE_DIAGNOSTIC_R1_RESULT_2026-07-24.md)。

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_route_invalid_reset_lifecycle_diagnostic.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_route_invalid_reset_lifecycle_diagnostic.py --config configs/ustrf_route_target_route_invalid_reset_lifecycle_diagnostic_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_route_invalid_reset_lifecycle_diagnostic.py --config configs/ustrf_route_target_route_invalid_reset_lifecycle_diagnostic_r1.json --terminal artifacts.local/evidence/ustrf-route-invalid-reset-lifecycle-diagnostic-r1/terminal-receipt-r1.json --output artifacts.local/evidence/ustrf-route-invalid-reset-lifecycle-diagnostic-r1/validation-receipt-r1.json --repo .
```

## R2-L1 trace-only metric profiles

`configs/ustrf_route_target_r2_l1_metric_profile_r1.json` 只读取 A2 terminal 中的 `123` 条权威 trace，并绑定 A3 completion、A4 4 GiB memory validation、eligibility protocol/mask/receipt 与三份 post-output truth。入口逐条复核 trace/authoritative-receipt SHA、四元 frame identity、每候选 `41` ledger / `62,229` 帧 / `15` reset，再做 truth join；不运行候选、不创建新 trace。

评分器按 delivery track 独立归因，critical miss 使用精确 critical interval 上的 active relation，closure key 附加 reset segment，clearance 使用 capture timestamp + `1500ms` horizon。现有 trace 的 consume timestamp 为 `0/62,229`，因此三个 evidence-age profile 均严格为 `not_evaluable`。结果为 `METRIC_PROFILES_COMPLETE / VALID`：critical miss 都是 `0/8` 但 bound 不足；三个 profile 都存在 unknown/stale active-alert 硬 veto 且 clearance 点估计失败；repeat 仅 underpowered。没有候选比较、排名、selection 或更高权限。详见[日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_R2_L1_METRIC_PROFILE_R1_RESULT_2026-07-24.md)。

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_metric_profiles_r2_l1.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_metric_profiles_r2_l1_from_traces.py --config configs/ustrf_route_target_r2_l1_metric_profile_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_metric_profiles_r2_l1.py --config configs/ustrf_route_target_r2_l1_metric_profile_r1.json --repo .
```

## L1 candidate replay R2

`configs/ustrf_route_target_l1_candidate_replay_r2*.json` 把旧 exploratory R1 失败收据、R3 `41/41` input completion、两个唯一 canonical root、冻结 C1–C3 和新 retry namespace 分开绑定。replay-only runner 只生成 candidate trace，不读取 truth、不生成 metric profile，也不携带 comparison/selection/shadow/H2/生产权限。

初始 R2 与 A1 分别保留 Windows 长路径失败收据；A1 已完成的 10 条 first-valid trace 在 A2 中按父哈希引用继承，没有候选重跑。用户明确将本次 replay 的可用内存门从 6 GiB 修订为 4 GiB 后，A2 使用短哈希 authority path 完成其余 113 条。最终为 C1/C2/C3 各 `41/41`、总 `123/123` trace、`186,687` candidate-frame、`45` reset；独立 validator 重放确定性状态输出，A3 再以严格 schema 绑定 A2 terminal/validation。由于 A2 的启动时 4 GiB 观测未持久化为逐 ledger 回执，A4 在 123 条独立确定性复演前逐条真实采样可用内存并 fail closed，最小观测 `7,592,321,024` bytes，未创建新权威 trace。结果见 [日期化报告](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_CANDIDATE_REPLAY_R2_RESULT_2026-07-24.md)。

运行与验证使用项目 Python 环境：

```powershell
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure run_candidate_replay_r2_continuation_a2.py --config configs\ustrf_route_target_l1_candidate_replay_r2_continuation_a2.json
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_candidate_replay_r2_continuation_a2.py --config configs\ustrf_route_target_l1_candidate_replay_r2_continuation_a2.json
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_candidate_replay_r2_a3.py --config configs\ustrf_route_target_l1_candidate_replay_r2_finalization_a3.json
E:\codex-tools\projects\blindassist\toolchain\venv-corridor-causal-py311\Scripts\python.exe scripts\run_research_tool.py ustrf-route-target-evidence-closure validate_candidate_replay_r2_memory_guard_a4.py --config configs\ustrf_route_target_l1_candidate_replay_r2_memory_guard_a4.json
```

## R2-L1E materialization recovery R3

`R2-L1E-RECOVERY-B1` 保留父 R2/A1 failure receipts，只修复执行运输与资源调度。canary 使用 `adb push /data/local/tmp` 后由 `run-as com.linnan.blindassist` 复制到 `targetContext.filesDir`，逐一验证 manifest 中全部图像哈希且不加载 TFLite。one-shard runner 继续使用同一目标私有目录，raw 通过 `adb exec-out run-as ... cat` 直接流入文件，不捕获到主机内存；compact successor 验证后删除 raw 和设备 staging。

B1 首分片阶段保留冻结的 6 GiB 可用物理内存门。canary 与 materialization 都要求连续 6 次 readiness 采样，并在输入加载后、instrumentation 前复查。2026-07-24 真机结果为 canary `1,455/1,455` RGB，通过后首条 CrowdBot ledger `1,455/1,455` 帧闭合；当时跨阶段累计 `3/41` ledger、`6,049/62,229` 帧。后续 continuation 经用户明确授权才将门修订为 4 GiB；数据、覆盖与权限标准未改变。详见 [R3 日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1E_MATERIALIZATION_RECOVERY_R3_RESULT_2026-07-24.md)。

剩余 38 条使用 continuation A1–A3 严格串行完成：父编排器只串行启动 child，每个 child 在双 canonical root 上重算下一条缺失 ledger、获取独占锁、复用 B1 单分片 materializer，验证恰好新增一个 successor 后立即退出。无效/半写 pair、重复权威根、额外 ledger、并发 child、重试耗尽或覆盖不精确都会 fail closed。父编排器只能在 `41/41`、`62,229/62,229`、`15/15` reset 同时成立后写出 input-complete receipt；它不导入或执行 C1–C3。

A1 在原 6 GiB 门下完成 9 条至 `12/41`，随后由一次真实内存失败和两次 Windows 长控制路径失败闭合。用户明确授权将门修订为 4 GiB 后，A2 用短控制路径完成 1 条至 `13/41`，但在 successor 后写长 host receipt 时失败；A3 使用 Windows extended-path 原子写，完成其余 28 条且 0 失败。最终独立重算为 `41/41`、`62,229/62,229`、`15/15 reset`，C1–C3、trace/profile 仍为 0。

验证入口：

```powershell
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_l1e_materialization_recovery_r3.py -v
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_l1e_materialization_recovery_r3_remaining.py -v
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\test_l1e_materialization_recovery_r3_continuation_a3.py -v
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_l1e_materialization_recovery_r3_canary.py --config configs\ustrf_route_target_l1e_materialization_recovery_r3_canary.json --repo .
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_l1e_materialization_recovery_r3_one_shard.py --config configs\ustrf_route_target_l1e_materialization_recovery_r3_one_shard.json --repo .
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_l1e_materialization_recovery_r3_remaining.py --config configs\ustrf_route_target_l1e_materialization_recovery_r3_continuation_a1.json --repo .
.\.venv-export312\Scripts\python.exe scripts\research\ustrf_route_target_evidence_closure\run_l1e_materialization_recovery_r3_continuation_a3.py --config configs\ustrf_route_target_l1e_materialization_recovery_r3_continuation_a3.json --repo .
```

## R2-L1X-L2P recovery and preregistration

`configs/ustrf_route_target_r2_l1x_l2p_prereg_r1.json` 在任何新 C1–C3 输出前绑定旧 R1 failure receipts，并冻结 L2 fresh-selection 与 non-executable L3 lockbox。`run_r2_l1x_l2p.py` 只在独立 namespace 恢复逐 ledger canonical raw；`validate_r2_l1x_l2p.py` 复建 41 ledger / 62,229 frame / 15 reset、权限和唯一终态。`validate_l2_l3_prereg_r1.py` 独立校验 L2 required metrics/门/primary/tie-break/source/veto/role/selection 语义，以及 L3 的 `executable=false`、`candidate_id=null` 和 lockbox/statistics floors。

原 R2 在三次远端清理白名单失败后保留 `FAIL_CLOSED_EXECUTION_ABORTED`。outcome-unseen A1 只修远端路径白名单，但两次 instrumentation 仍无法从 app external-files 识别 shell materialized manifest，第三次又触发该阶段不可降低的 6 GiB 内存门；A1 尝试耗尽并成为最终合法终态。该历史阶段终态只有 2/41 ledger、4,594/62,229 frame canonical input，C1–C3、trace/profile、机制成绩审计和 selection 均为 0；其缺口后来由独立 R3 materialization closure 闭合。详见 [R2-L1X-L2P 日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_R2_L1X_L2P_RESULT_2026-07-24.md)。

验证入口：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_l2_l3_prereg_r1.py --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_l2_l3_prereg_r1.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_r2_l1x_l2p.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_r2_l1x_l2p_transport_amendment_a1.py
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_r2_l1x_l2p.py --config artifacts.local/evidence/ustrf-route-target-r2-l1x-l2p-a1/frozen-merged-prereg-a1.json --repo .
```

## R2-L1E receipt-aware exploratory profiles

`configs/ustrf_route_target_l1_exploratory_profile_r1.json` 精确绑定父 R2-L1 protocol/mask/denominator/validation、冻结 C1–C3 实现、41 条 masked sequence ledger、62,229 帧和 15 个 discontinuity reset。独立 runner 只允许逐 ledger Android Canvas canonical raw 分片、host compact successor、冻结 T0 replay-local association 和因果 route input；truth 只能在候选输出后 join。终态 schema 只允许 `EXPLORATORY_PROFILES_COMPLETE`、`FAIL_CLOSED_INPUT_BLOCKED` 或 `FAIL_CLOSED_EXECUTION_ABORTED`，且所有 selection、Android shadow、H2、人体和生产权限固定关闭。

运行：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure run_metric_eligibility_exploratory_profiles_r2_l1.py --config configs/ustrf_route_target_l1_exploratory_profile_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_exploratory_profiles_r2_l1.py --config configs/ustrf_route_target_l1_exploratory_profile_r1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure test_exploratory_profiles_r2_l1.py
```

当前机器收据为 `FAIL_CLOSED_EXECUTION_ABORTED`：冻结的 6 GiB 系统可用内存门在初始尝试和两次有界重试中均触发，首个 CrowdBot device attempt 未创建。validator 重建为 2/41 ledger、4,594/62,229 帧 canonical raw 已验证，39 ledger、57,635 帧缺失；候选、trace 和 profile 均为 0。结果见 [R2-L1E 日期化结果](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_R1_RESULT_2026-07-24.md)。

## R2-L1 metric eligibility materialization

`configs/ustrf_route_target_metric_eligibility_r2_l1.json` 将当前 6,369 个 LILocBench/CrowdBot 事件或提案逐项物化为 8 指标 eligibility mask，并把非事件粒度的负暴露和 preoutput frame support 放入独立 ledger。输入只允许读取 11 个哈希冻结、candidate-blind 的 truth/route/review 文件；禁止目录扫描、候选模块执行和候选输出读取。

运行：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure materialize_metric_eligibility_r2_l1.py --config configs/ustrf_route_target_metric_eligibility_r2_l1.json --repo .
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_metric_eligibility_r2_l1.py --config configs/ustrf_route_target_metric_eligibility_r2_l1.json --repo .
python scripts/research/ustrf_route_target_evidence_closure/test_metric_eligibility_r2_l1.py -v
```

输出位于忽略的 `artifacts.local/evidence/ustrf-route-target-metric-eligibility-r2-l1/`：`eligibility-mask-r2-l1.json`、`denominator-receipt-r2-l1.json` 和 `validation-receipt-r2-l1.json`。mask 同时包含 62,188 个相邻 frame-pair 的 eligibility/exclusion audit 与 62,229 行显式 preoutput frame ledger。validator 会完整重建前两者并检查规范 JSON 精确一致，同时硬拒绝 0/0 pass、pre-clear 进入 clearance、truth pool 冒充 repeat 分母、pair universe 缺口、负暴露重叠和任何候选输出访问。

当前物化结论是：`critical_miss`、`clearance`、`unknown_or_stale_alert` 为 L1 探索资格；`repeat`、`evidence_age` 为候选观测完整后才成立的条件资格；`event_recall`、`regeneration`、`false_alerts_per_minute` 仍为 L0。该结论只授权另开独立任务生成单次探索 profile，不授权选择候选或进入 Android/H2/生产。

下一独立任务使用 [R2-L1E 单次探索 profile 通宵目标](../../../docs/research/ustrf-sc/USTRF_ROUTE_TARGET_L1_EXPLORATORY_PROFILE_OVERNIGHT_GOAL_2026-07-24.md)：先检查全量 canonical input，再让 C1–C3 各对 41 条 masked sequence ledger 单次 replay，并在冻结观测断点重置状态；canonical raw 逐 ledger 分片验证和清理，输出只有分指标探索 profile 与机器收据，不产生 winner、排名或晋级。

## 稳定 Interface

运行 `python scripts/research/ustrf_route_target_evidence_closure/validate_prereg.py --config configs/ustrf_route_target_evidence_closure_r1.json --repo .`。validator 重算父 evidence 哈希，并冻结五态逐人路线角色、三条 oracle 臂、最多三个累积结构候选和逐来源 holdout 门；任一漂移均 fail closed。

运行 `python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_evidence_maturity_v2.py --config configs/ustrf_route_target_evidence_maturity_v2.json --repo .` 校验当前证据成熟度协议。V2 不改写 R1：它只允许按 event recall、critical、repeat、clearance、false-alert exposure、evidence age 和 unknown-route veto 分别冻结 eligibility/分母，并按 L0–L4 提升权限。空分母必须是 `not_evaluable`，低样本只能是 `evaluable_underpowered`；L1 不选胜者，L2 不直接进 Android，L3 才能申请 production-isolated shadow admission。

`prepare_route_role_review_bundle.py` 在 detector/candidate 输出隐藏状态下，逐帧重算 4,594 张 RGB 哈希并联结 source timestamp、因果 route receipt、既有 target/negative person seed truth。seed box 只提供既有审查事实；每帧仍要求 full-frame all-person discovery，不能把未发现的共现者当作 absent。

`annotate_seen_persons_closed_vocab.py` 是第二条独立 person annotation proposal pass：固定闭词表模型 SHA、960 输入、`.01` proposal floor，只生成 truth-review proposal，不是 detector 候选，也不获得 benchmark 或晋级 credit。它与既有 prompted YOLOE-11s-seg pass 融合；非 seed 的单模型 proposal 必须隔离，不能直接进入真值。另保留既有负窗 YOLO11x-960 proposal 作为负窗附加审计证据。

`fuse_seen_person_proposals.py` 先以既有冻结 seed truth 为优先，以固定 Ultralytics ByteTrack 默认参数分别形成 annotation-only 身份提议，再按预注册 IoU 联结两条 proposal pass，并只生成 `proposal_track_id`。ByteTrack 在未观察到的帧号间隙强制重置，且片段编号写入 ID，不能跨不连续窗口串联。含单模型节点、多人关联歧义或 identity 冲突的 tracklet 必须进入第三模型 adjudication；在此之前禁止命名为稳定 `person_id`。

`prepare_third_model_adjudication_bundle.py` 只抽取争议 tracklet 涉及的去重帧；`annotate_third_model_disagreements.py` 使用冻结 SHA 的 YOLO11x-960 作为第三条闭词表 person proposal pass，仍看不到评分标签、App detector 或候选告警。`resolve_third_model_adjudication.py` 对单模型节点和一对多身份歧义执行 fail-closed 裁决；未被第三模型重合确认或仍跨 tracklet 冲突的整段必须 quarantine。

`build_route_role_model_proxy_truth.py` 只把 registered RGB-D 用作离线注释支持，route 只读 causal prediction；route/depth/脚点无效均输出 unknown。`run_seen_oracle_attribution.py` 运行 T0 与三条单接缝 oracle。`candidates.py` 是冻结的 C1–C3 因果状态机实现；它不读取 proxy truth 或 RGB-D，也不修改 detector/tracker。

`inspect_remote_zip_inventory.py`、`extract_remote_zip_entry.py` 与 `stream_remote_zip_entry.py` 只通过官方 Range 请求核对 ZIP 目录/条目并做 CRC+SHA 收口；`stream_remote_zip_entry.py` 可显式启用有限并发 Range，把压缩分片暂存到独立缓存盘，按原始字节顺序解压并复用同一 ZIP CRC、输出 SHA 与帧哈希合同，成功后删除分片。长时间运行的旧物化器也可由 evidence root 中不可变、版本化的 `transport-acceleration-r*.json` 在下一条 stream 子进程边界启用同一模式，优先使用最高受支持版本；旁路配置必须声明 candidate-blind，并将自身路径和 SHA 写入下载收据。`qualify_crowdbot_route_capacity.py` 只用发布 tracks 与同步 pose 做来源容量代理，不运行候选。`materialize_crowdbot_rgbd_sequence.py` 无损导出 forward RGB 与精确同时间戳 aligned depth；`rgb8` 原样保存，来源原生 `bgr8` 只允许通过通道反转规范化成 RGB PNG，不允许颜色增强或坏行修复。`materialize_crowdbot_holdout_sources.py` 支持同一来源的多 part raw inventory，每次只保留一个临时 bag，bundle 验证通过后删除 raw 并留下可重取收据；Windows 若在子进程退出后短暂保留 raw bag 文件句柄，只允许有界退避重试同一已验证文件的删除，不跳过 bundle/hash 校验。全量完成后，`audit_materialized_holdout.py` 才会逐文件复算 RGB/depth 哈希并执行跨来源 exact SHA 与 dHash 近重复审计；该审计本身不授予来源准入或 H2 权限。

Holdout candidate route 不得复用容量筛查的未来真实轨迹。`materialize_crowdbot_rgbd_sequence.py` 同时保存 candidate-blind TF frame inventory，`backfill_crowdbot_tf_inventory.py` 只用于给已验证的早期 bundle 补齐同一 raw bag 的 TF 绑定。16/16 后，`build_crowdbot_causal_route_ledger.py` 继承 R3 的 past-pose-prefix-only 合同：仅用当前/过去 Qolo pose，经静态 `tf_qolo→camera` 外参与相机内参生成 causal route UV；future pose 只允许生成 annotation route truth，并以所有可投影 pose 样本形成完整 UV polyline，不能只保留终点。

`build_crowdbot_projected_track_role_proposal.py` 把发布 LiDAR track 投影进 RGB，仅生成 candidate-blind identity/metric-role proposal；它不能据 track 缺失宣称画面无人。`approaching_route` 必须由 actual-future track 在 1.6s 内实际进入 route 支持，`receding` 必须有 prior intersection；missing/间断不会形成 clear。该 proposal 还必须与两条独立 visual person pass 做全帧共识与歧义隔离，才能冻结为最终 holdout truth。

`annotate_crowdbot_holdout_person_proposals.py` 只按预注册的 YOLOv8n/Yolo11x 模型哈希和 `.01` proposal floor 生成两条全帧 person proposals；它们看不到 App detector/event 或 C1–C3。首组 truth/window 因“无关未知人使整窗失效”和“继承相机不可见 LiDAR onset”以 `0/2` 失败，相关来源只保留协议诊断权限。替换协议由 `configs/ustrf_route_target_evidence_closure_r1_replacement_holdout.json` 哈希冻结：`fuse_crowdbot_holdout_person_role_truth.py` 只从视觉确认的 metric-person 连续角色生成正事件；raw LiDAR event 只作容量代理。负帧要求 causal route known 且所有路线相关人物已解决，路线外未知人不抹掉整帧，路线内或可能路线相关的未知人仍使帧不可评。`freeze_crowdbot_holdout_truth_windows.py` 只有在隔离后仍逐来源满足正/critical 事件、同序列等长负窗和 10 分钟负 exposure 时才签发 2/2 selection authority。

`prepare_crowdbot_holdout_detector_device_bundle.py` 只有在上述 2/2 truth/window 收据已冻结后才可整理完整 RGB manifest；随后复用 hash-bound Android `ImagePreprocessor` Canvas + 正式 App TFLite CPU-4-thread exporter 产生 canonical raw tensors。`run_crowdbot_holdout_app_detector.py` 只解码该设备 raw stream，并绑定权重、labels、`.35/.45`、manifest/receipt 与完整 RGB 哈希；禁止用 host PIL reconstruction 冒充 App detector。`run_crowdbot_holdout_candidates.py` 固定使用 T0 association，让 C1–C3 各自对每条完整 sequence 一次运行；truth 只在状态更新后用于归因。候选告警若重合 unresolved person，则该来源以 `unknown_person_active_alert_count > 0` 硬失败。`configs/ustrf_route_target_evidence_closure_r1_scoring_amendment.json` 在候选运行前修正 false-alert numerator：完整序列内所有 route-known、无真事件归因且非 unresolved-person 的 delivery 都计入，不能只计匹配窗口却除以全量负暴露。报告保留逐序列 trace hash、delivery/closure 收据、逐来源全部硬门与 worst-source tie-break，任何来源失败都不会打开 Android shadow 或 H2。

Replacement 的 23 条完整序列最终仍只形成 2 个可接受事件，说明纯 LiDAR/pose 容量代理不能预测 camera-visible metric identity continuity 与 terminal clear。后续来源必须先过 `configs/ustrf_route_target_evidence_closure_r1_camera_source_prescreen.json`：每来源只解码两条 candidate-blind canary，事件 canary 按 positive/critical/active 容量选取，负窗 canary按 `negative_route_seconds / compressed_GiB` 选取；两条均永久排除未来 lockbox。canary 门按正式门乘 `2 / metadata_sequence_count` 冻结，只具有 reject-only 权限；失败立即停止其余下载，通过也只允许物化非 canary，不能直接准入来源或运行候选。`inspect_remote_zip_inventory.py` 在读取 body 之前硬检查 HTTP 206、Content-Range 与 Content-Length，服务器忽略 Range 时不得误拉整包。

## 输出

首组诊断 evidence 保留在 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/`；替换来源写入 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1-replacement/` 与 `artifacts.local/datasets/ustrf-route-target-evidence-closure-r1-replacement/`。`0410 mds + 1203 shared-control` 已完成 23/23，但 truth/window admission 为 0/2，只保留诊断和回归权限。0327 reject-only canary 位于 `artifacts.local/camera-source-prescreen-r1/`，物理存储在获用户授权的 D 盘；两条共 4,422 RGB，最终为 0 positive、0 critical、0 matched negative、0.0764min negative exposure，已拒绝并停止剩余 11 条。NavWareSet 只下载 181.6MB Stage A 即因 robot/GRS 原始时间区间不重叠而拒绝，没有启动约 5.97GB Stage B。Bi3 完整 41.7GB 包也未下载。候选输出始终关闭。

最终有界来源审计还拒绝了 REveL：已查看的 `dynamic` 只能保留 development/diagnostic 权限，排除后全部未查看 footage 的发布总时长上界约 7.903 分钟，低于每来源 10 分钟负暴露门；匿名小包又不同时包含相机、稳定身份、metric sensor/person pose 与 terminal clear。JRDB、Oxford-IHM、KTP/IAS-Lab、FLOBOT、THÖR-MAGNI、SCAND 也均未同时通过五项来源门。最终收据为 `artifacts.local/evidence/ustrf-route-target-evidence-closure-r1/source-search-final-bounded-decision-r1.json`，决定为 `DATA_BLOCKED_STOP_SOURCE_SEARCH`、可用来源 `0/2`、本轮新增下载 `0 bytes`。不继续扩大来源搜索，不运行候选。

## 安全边界

当前 15+15 seen 窗口只做故障归因，不能选择候选或调标量。detector、`.35`、NMS、tracker 均冻结；深度、TTC、route-risk flip、Android shadow、训练和生产权限关闭。模型生成的 route-role truth 是 hash-bound benchmark evidence，不是真实用户安全事实。

## 停止条件

R1 来源搜索已经停止。V2 每轮最多审计两个新来源 family、每来源两个 canary，默认自动下载上限 2 GiB；连续两个 family 不合格即 `STOP_DATA_COLLECTION_AT_CURRENT_LEVEL`，保留已有局部指标证据。真实性/unknown 硬门失败才 `STOP_MECHANISM`。轮内不得改语义、分母、阈值或 tie-break；轮间改变必须升协议版本并让已查看数据失去 selection/confirmation/shadow lockbox 权限。
