# BlindAssist Assistive Geometry

状态：`current / RESEARCH_MAINLINE / A0_SEED_17_COMPLETE / SEED_29_GUARDED_EXECUTION_RUNNING / SEED_43_NOT_STARTED / DEVELOPMENT_EVALUATION_IMPLEMENTED_NOT_ACTIVATED / DEVELOPMENT_AND_CONFIRMATION_SEALED`

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
- [B1 A0 formal TRAIN execution protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FORMAL_TRAIN_EXECUTION_PROTOCOL_2026-08-09.json)
- [B1 A0 host performance preflight result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_TRAIN_PERFORMANCE_PREFLIGHT_RESULT_2026-08-09.md)
- [B1 A0 host performance machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_TRAIN_PERFORMANCE_PREFLIGHT_RESULT_2026-08-09.json)
- [B1 A0 evaluation synthetic dry-run protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PROTOCOL_2026-08-09.md)
- [B1 A0 evaluation synthetic dry-run machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PROTOCOL_2026-08-09.json)
- [B1 A0 evaluation synthetic dry-run result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_RESULT_2026-08-09.md)
- [B1 A0 evaluation synthetic dry-run machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_RESULT_2026-08-09.json)
- [B1 A0 Development evaluation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.md)
- [B1 A0 Development evaluation machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.json)
- [C0 heterogeneous-teacher complementarity protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_PROTOCOL_2026-08-09.md)
- [C0 heterogeneous-teacher complementarity machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_PROTOCOL_2026-08-09.json)
- [D0 temporal ablation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_D0_TEMPORAL_ABLATION_PROTOCOL_2026-08-09.md)
- [D0 temporal ablation machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_D0_TEMPORAL_ABLATION_PROTOCOL_2026-08-09.json)
- [并行 WILD_LAB 数学假设 canary](BLINDASSIST_ASSISTIVE_GEOMETRY_HYPOTHESIS_CANARY_LITE_R0_2026-08-09.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_SEED_29_GUARDED_FORMAL_TRAIN_EXECUTION`

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
全状态 checkpoint roundtrip 现已关闭。正式 runner 又以同一真实路径比较 `workers=0/1/4`，
选出 `workers=1` 的 `0.5453 step/s`，外推每 seed `3.06h`、诊断上界 `4h`；三档 CPU 输入
摘要一致，但 CUDA 权重不签署 bit-exact。seed 17 已完成 20 epochs / 6000 steps，四个留存点、
最终 carry、模型状态和 TRAIN-only 防火墙均闭合；seed 29 正在 guarded TRAIN-only 执行，43 尚未
启动。与此同时，纯合成 evaluator dry-run 已覆盖 12 个 tiny
checkpoint、三 seed 无选择聚合、九格指标、UNKNOWN、全局零分母、缺 horizon、coverage 塌缩、
协议漂移和失败相邻日志并通过。它不读取 Development/Confirmation outcome，也不授权真实评价；
正式 Development evaluator v2 现已在 outcome 前冻结：补齐 ground recovery、clearance coverage、
valid→UNKNOWN 和 geometry transition，并将 truth/pred clearance validity 分离。物化器会在读取首个
Development frame 前强制验证三 seed 全部完成；执行期间仍不得运行 A1–A4、双教师或改默认 App。
异质教师只冻结到 C0 complementarity kill gate mechanics：教师 identity、评估 cohort 和输出仍未
授权，未通过 oracle 增益、独占正确 parent、分歧错误浓度和时序优势四类门前不得启动 C1 蒸馏。
时序 D0 也只冻结因果 GRU/TCN/diagonal-SSM 的统一 GeometryState 接口、参数/设备预算和未来
clearance/TTC/compute-gate 输出；单帧候选与新时序 cohort 未就绪，不授权训练或读取 outcome。

## 并行 WILD_LAB 数学 canary（不改变 successor）

纯合成 CPU canary 已审查四个 forward hypothesis：censored robust-contact survival、
profile-conditioned swept configuration clearance、maximum-bottleneck corridor loss 和
cluster-level one-sided conformal/CRC uncertainty。数学不变量与反例均通过，终态为
`MATH_MECHANICS_SUPPORTED_PAPER_NOVELTY_AND_LEARNABILITY_NOT_ESTABLISHED`。其中 H1/H2
优先，H3 只能作为已有 max-min/topology 文献下的 task-specific loss，H4 受 calibration
parent 数量阻塞：8% finite-sample risk 至少需 12 个独立 parents，当前 4 个不可能。

该 canary 不读取模型、checkpoint 或任何数据 role outcome，不修改冻结 A0–A4，不授权 B2、
真实 Development、部署、默认 App 或 safety；唯一 successor 保持 seed 29 guarded A0 execution。
