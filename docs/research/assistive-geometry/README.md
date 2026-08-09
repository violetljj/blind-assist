# BlindAssist Assistive Geometry

状态：`current / RESEARCH_MAINLINE / A0_TRAIN_EXECUTION_LOCK_PASS / FORMAL_A0_THREE_SEED_TRAINING_AUTHORIZED_NOT_STARTED / DEVELOPMENT_AND_CONFIRMATION_SEALED`

本路线把 DepthART-S 从研究终点降为可替换的轻量 encoder/initialization 候选，核心问题改为：

> 能否学习直接面向身体通行空间的 Ground、Clearance、Confidence、UNKNOWN 与
> Body-swept Occupancy，同时保持移动端可部署性？

当前真源：

- [路线决策](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.md)
- [机器合同](BLINDASSIST_ASSISTIVE_GEOMETRY_PROGRAM_ROUTE_DECISION_2026-08-09.json)
- [B0 task contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.md)
- [B0 machine contract](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TASK_CONTRACT_2026-08-09.json)
- [B0 input/data preflight result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT_RESULT_2026-08-09.md)
- [B0 preflight machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_INPUT_DATA_PREFLIGHT_RESULT_2026-08-09.json)
- [B0 runtime geometry receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_RUNTIME_GEOMETRY_RECEIPT_2026-08-09.md)
- [B0 runtime geometry machine receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_RUNTIME_GEOMETRY_RECEIPT_2026-08-09.json)
- [B0 data capability and roster lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK_RESULT_2026-08-09.md)
- [B0 data capability and roster machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_CAPABILITY_AND_ROSTER_LOCK_RESULT_2026-08-09.json)
- [B0 data-use authorization receipt](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_DATA_USE_AUTHORIZATION_RECEIPT_2026-08-09.md)
- [B0 upsampling TRAIN materialization protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_ARKIT_UPSAMPLING_TRAIN_MATERIALIZATION_PROTOCOL_2026-08-09.json)
- [B0 truth-reader validation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_VALIDATION_PROTOCOL_2026-08-09.json)
- [B0 truth-reader and registration lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK_RESULT_2026-08-09.md)
- [B0 truth-reader machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B0_TRUTH_READER_AND_REGISTRATION_LOCK_RESULT_2026-08-09.json)
- [B1 training protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.md)
- [B1 machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09.json)
- [B1 protocol-lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_CONFIDENCE_THRESHOLD_AND_TRAINING_PROTOCOL_LOCK_RESULT_2026-08-09.md)
- [B1 protocol-lock machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_CONFIDENCE_THRESHOLD_AND_TRAINING_PROTOCOL_LOCK_RESULT_2026-08-09.json)
- [B1 orientation geometry preflight](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_INPUT_ORIENTATION_GEOMETRY_PREFLIGHT_RESULT_2026-08-09.md)
- [B1 orientation geometry machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_INPUT_ORIENTATION_GEOMETRY_PREFLIGHT_RESULT_2026-08-09.json)
- [B1 current dual-orientation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09_ATTEMPT_02.md)
- [B1 current dual-orientation machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_TRAINING_PROTOCOL_2026-08-09_ATTEMPT_02.json)
- [B1 dual-orientation lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_PROTOCOL_LOCK_RESULT_2026-08-09.md)
- [B1 dual-orientation machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_PROTOCOL_LOCK_RESULT_2026-08-09.json)
- [B1 implementation execution protocol Attempt 02](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_EXECUTION_LOCK_PROTOCOL_2026-08-09_ATTEMPT_02.json)
- [B1 dual-orientation implementation lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_IMPLEMENTATION_LOCK_RESULT_2026-08-09.md)
- [B1 dual-orientation implementation machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_DUAL_ORIENTATION_IMPLEMENTATION_LOCK_RESULT_2026-08-09.json)
- [B1 A0 TRAIN execution lock protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_TRAIN_EXECUTION_LOCK_PROTOCOL_2026-08-09.json)
- [B1 A0 current execution protocol Attempt 02](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_TRAIN_EXECUTION_LOCK_PROTOCOL_2026-08-09_ATTEMPT_02.json)
- [B1 A0 TRAIN execution lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_TRAIN_EXECUTION_LOCK_RESULT_2026-08-09.md)
- [B1 A0 TRAIN execution machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_TRAIN_EXECUTION_LOCK_RESULT_2026-08-09.json)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEPTH_ONLY_THREE_SEED_FORMAL_TRAIN_EXECUTION`

ARKitScenes `16/8/8` visit/video-disjoint roster 与 9,600-frame integrity 已冻结；B0 reader 又以
6 个 TRAIN 视频、157 个 AppleDepth/FARO exact-timestamp 对照和主 TRAIN 的 480 个固定 stride
帧关闭 source scale、registration、pose、ground、三通道 body-swept clearance 与 UNKNOWN
fail-closed。B1 又冻结了 target/loss/confidence、训练角色与停止条件，并把原 8 个 DEVELOPMENT
identity 在 outcome 前拆成 4 calibration / 4 selection。implementation preflight 随后发现
TRAIN 有 `2,724` portrait 与 `2,076` landscape 帧，Attempt 1 单一 portrait 协议已被取代；
Attempt 2 冻结 dual-shape full-FOV、orientation bucket 和重新平衡的 Development split。当前已
逐 SHA/语义验证 4,800 个 TRAIN target，并关闭 dual-shape shared decoder/head/loss 与真实
checkpoint 的训练 Autograd smoke；带缺失 Autograd-key 警告的部署 operator smoke 保留为 HOLD，
训练改走包内显式 custom Function。A0 TRAIN loader、orientation carry、BF16 optimizer step 与
全状态 checkpoint roundtrip 现已关闭，正式 seed `17/29/43` 的 A0 depth-only TRAIN execution
得到授权但尚未启动；执行期间仍不读取 DEVELOPMENT/CONFIRMATION outcome，不运行 A1–A4 或
双教师，也不改默认 App。
