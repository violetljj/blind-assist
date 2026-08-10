# BlindAssist Assistive Geometry

状态：`current / B1_A0_PERMANENT_NEGATIVE_TERMINAL / R2_F0_SYNTHETIC_REDUCER_PASS / F1_P_PROTOCOL_FROZEN / FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS / SUPERVISION_FRONTDOOR_UNSATISFIED / F1_EXECUTION_NOT_AUTHORIZED / ALL_CALIBRATION_AND_CONFIRMATION_SEALED / DEFAULT_APP_UNCHANGED`

本路线把 DepthART-S 从研究终点降为可替换的轻量 encoder/initialization 候选，核心问题改为：

> 可替换的轻量视觉 encoder 能否先学习 metric-ish depth、support surface 与 obstacle boundary
> 连续因子，再由确定性 body-swept reducer 构造 Clearance、Occupancy 与 UNKNOWN？

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
- [B1 A0 seed 29 formal TRAIN retry Attempt 02](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FORMAL_TRAIN_EXECUTION_PROTOCOL_2026-08-09_ATTEMPT_02.md)
- [B1 A0 seed 29 formal TRAIN retry machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FORMAL_TRAIN_EXECUTION_PROTOCOL_2026-08-09_ATTEMPT_02.json)
- [B1 A0 host performance preflight result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_TRAIN_PERFORMANCE_PREFLIGHT_RESULT_2026-08-09.md)
- [B1 A0 host performance machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_TRAIN_PERFORMANCE_PREFLIGHT_RESULT_2026-08-09.json)
- [B1 A0 evaluation synthetic dry-run protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PROTOCOL_2026-08-09.md)
- [B1 A0 evaluation synthetic dry-run machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PROTOCOL_2026-08-09.json)
- [B1 A0 evaluation synthetic dry-run result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_RESULT_2026-08-09.md)
- [B1 A0 evaluation synthetic dry-run machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_RESULT_2026-08-09.json)
- [B1 A0 Development evaluation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.md)
- [B1 A0 Development evaluation machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_PROTOCOL_2026-08-09.json)
- [B1 A0 Development evaluation result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_RESULT_2026-08-09.md)
- [B1 A0 Development evaluation machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_DEVELOPMENT_EVALUATION_RESULT_2026-08-09.json)
- [B1 A0 permanent closure](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_PROGRAM_CLOSURE_2026-08-09.md)
- [B1 A0 permanent closure machine record](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_PROGRAM_CLOSURE_2026-08-09.json)
- [B1 A0 failure-anatomy protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_PROTOCOL_2026-08-09.md)
- [B1 A0 failure-anatomy machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_PROTOCOL_2026-08-09.json)
- [B1 A0 failure-anatomy result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.md)
- [B1 A0 failure-anatomy machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_FAILURE_ANATOMY_RESULT_2026-08-09.json)
- [Geometry R2 factorized hypothesis protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.md)
- [Geometry R2 factorized hypothesis machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_FACTORIZED_GEOMETRY_HYPOTHESIS_PROTOCOL_2026-08-09.json)
- [Geometry R2 F0 synthetic reducer protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PROTOCOL_2026-08-09.md)
- [Geometry R2 F0 synthetic reducer machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PROTOCOL_2026-08-09.json)
- [Geometry R2 F0 synthetic reducer result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_RESULT_2026-08-10.md)
- [Geometry R2 F0 synthetic reducer machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_RESULT_2026-08-10.json)
- [Geometry R2 F1 factor schema](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTOR_SCHEMA_2026-08-10.json)
- [Geometry R2 F1-P protocol lock](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK_2026-08-10.md)
- [Geometry R2 F1-P machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_TRAIN_ONLY_FACTOR_LEARNABILITY_PROTOCOL_LOCK_2026-08-10.json)
- [Geometry R2 F1-P lock result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_PROTOCOL_LOCK_RESULT_2026-08-10.md)
- [Geometry R2 F1-P machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_PROTOCOL_LOCK_RESULT_2026-08-10.json)
- [Geometry R2 F1 FactorTensorAdapter gap audit](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_GAP_AUDIT_2026-08-10.md)
- [Geometry R2 F1 FactorTensorAdapter machine audit](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_GAP_AUDIT_2026-08-10.json)
- [Geometry R2 F1 FactorTensorAdapter protocol lock](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_2026-08-10.md)
- [Geometry R2 F1 FactorTensorAdapter protocol machine contract](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_2026-08-10.json)
- [Geometry R2 F1 FactorTensorAdapter protocol result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_RESULT_2026-08-10.md)
- [Geometry R2 F1 FactorTensorAdapter protocol machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_PROTOCOL_LOCK_RESULT_2026-08-10.json)
- [Geometry R2 F1 FactorTensorAdapter implementation canary lock](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_IMPLEMENTATION_CANARY_LOCK_2026-08-10.json)
- [Geometry R2 F1 FactorTensorAdapter synthetic canary result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_RESULT_2026-08-10.md)
- [Geometry R2 F1 FactorTensorAdapter synthetic canary machine result](BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_RESULT_2026-08-10.json)
- [AG-ST R0 source-anchored selective labelability protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SOURCE_ANCHORED_SELECTIVE_LABELABILITY_PROTOCOL_LOCK_2026-08-10.json)
- [AG-ST R0 source / Teacher / ancestry / license audit](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SOURCE_TEACHER_ANCESTRY_LICENSE_AUDIT_2026-08-10.md)
- [AG-ST R0 machine audit](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SOURCE_TEACHER_ANCESTRY_LICENSE_AUDIT_2026-08-10.json)
- [AG-ST R0 SuperTeacher factor-label factory result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SUPERTEACHER_FACTOR_LABEL_FACTORY_WILD_LAB_RESULT_2026-08-10.json)
- [AG-ST R0 multi-Teacher factor-label factory result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_MULTITEACHER_FACTOR_LABEL_FACTORY_WILD_LAB_RESULT_2026-08-10.json)
- [AG-ST R1 TUM cross-source multi-Teacher result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R1_TUM_CROSS_SOURCE_MULTITEACHER_RESULT_2026-08-10.json)
- [AG-ST R2 fresh-TUM third-Teacher + gravity-factor result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R2_TUM_THIRD_TEACHER_AND_GRAVITY_FACTORS_RESULT_2026-08-10.json)
- [AG-ST R4 ICL pixel-exact boundary result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R4_ICL_PIXEL_BOUNDARY_RESULT_2026-08-11.json)
- [AG-ST R5 fresh exact-depth boundary result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R5_FRESH_EXACT_DEPTH_BOUNDARY_RESULT_2026-08-11.json)
- [AG-ST R6 source-native boundary corpus result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R6_SOURCE_NATIVE_BOUNDARY_CORPUS_RESULT_2026-08-11.json)
- [AG-ST R7 source-boundary learnability result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R7_SOURCE_BOUNDARY_LEARNABILITY_RESULT_2026-08-11.json)
- [AG-ST R8 soft-boundary Bonn canary result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R8_SOFT_BOUNDARY_BONN_CANARY_RESULT_2026-08-11.json)
- [AG-ST R9 continuous boundary factors result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R9_CONTINUOUS_BOUNDARY_FACTORS_RESULT_2026-08-11.json)
- [AG-ST R10 unified factor labels result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R10_UNIFIED_FACTOR_LABELS_RESULT_2026-08-11.json)
- [AG-ST R11 unified factor student result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R11_UNIFIED_FACTOR_STUDENT_RESULT_2026-08-11.json)
- [AG-ST R12 external Bonn unified-student result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R12_EXTERNAL_BONN_UNIFIED_STUDENT_RESULT_2026-08-11.json)
- [AG-ST R0 frozen-DepthART masked-student result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_MASKED_STUDENT_DEPTHART_WILD_LAB_RESULT_2026-08-10.json)
- [AG-ST R0 fresh-parent zero-shot replication](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_FRESH_PARENT_ZERO_SHOT_RESULT_2026-08-10.json)
- [AG-ST R0 combined-32 depth/support result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_COMBINED32_DEPTH_SUPPORT_RESULT_2026-08-10.json)
- [AG-ST R0 combined-40 precision confirmation](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_COMBINED40_PRECISION_CONFIRMATION_RESULT_2026-08-10.json)
- [C0 heterogeneous-teacher complementarity protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_PROTOCOL_2026-08-09.md)
- [C0 heterogeneous-teacher complementarity machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_C0_TEACHER_COMPLEMENTARITY_PROTOCOL_2026-08-09.json)
- [D0 temporal ablation protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_D0_TEMPORAL_ABLATION_PROTOCOL_2026-08-09.md)
- [D0 temporal ablation machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_D0_TEMPORAL_ABLATION_PROTOCOL_2026-08-09.json)
- [M0 task-preserving mobile deployment protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_M0_TASK_PRESERVING_MOBILE_DEPLOYMENT_PROTOCOL_2026-08-09.md)
- [M0 task-preserving mobile deployment machine protocol](BLINDASSIST_ASSISTIVE_GEOMETRY_M0_TASK_PRESERVING_MOBILE_DEPLOYMENT_PROTOCOL_2026-08-09.json)
- [并行 WILD_LAB 数学假设 canary](BLINDASSIST_ASSISTIVE_GEOMETRY_HYPOTHESIS_CANARY_LITE_R0_2026-08-09.md)
- [AG-QSF 并行路线 current](../assistive-geometry-qsf/README.md)
- [AG-CBF 并行路线 current](../assistive-geometry-cbf/README.md)
- [算法研究入口](../ALGORITHM_RESEARCH_CURRENT.md)
- [DepthART 部署支线](../hftf/README.md)

## 唯一 successor

`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F1_SUPERVISION_SOURCE_AND_LABEL_CONTRACT_LOCK`

该 successor 当前 execution authority=false，只允许另行冻结 pre-outcome supervision source/label
contract；不得物化标签、读取真实 task outcome、定义模型、训练或分配 R2 Development，也不得启动
teacher / temporal / mobile。adapter PASS 只移除了 deterministic tensor-to-frame seam blocker。

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
摘要一致，但 CUDA 权重不签署 bit-exact。seed 17/29/43 均已完成 20 epochs / 6000 steps，四个
留存点、最终 carry、模型状态和 TRAIN-only 防火墙均闭合；seed 29 Attempt 01 在 2097 steps 收到 CUDA
OOM 并保留失败收据，Attempt 02 从共同初始化完整重跑，没有恢复或挑选中间状态。与此同时，纯合成 evaluator dry-run 已覆盖 12 个 tiny
checkpoint、三 seed 无选择聚合、九格指标、UNKNOWN、全局零分母、缺 horizon、coverage 塌缩、
协议漂移和失败相邻日志并通过。它不读取 Development/Confirmation outcome，也不授权真实评价；
正式 Development evaluator v2 在 outcome 前冻结了 ground recovery、clearance coverage、
valid→UNKNOWN 和 geometry transition，并将 truth/pred clearance validity 分离。三 seed 完整后只物化
四个固定 Selection parent / 1,200 帧并完成 3,600 个 seed-frame 观察。A0 前门通过，但 clearance
MAE `0.3152 m`、false-block `0.7501`、geometry transition agreement `0.7728` 均为 `0/3` seed
通过，终态为 `B1_A0_DEVELOPMENT_EVALUATION_FAIL_TASK_GATES`。因此冻结的 A1 条件 successor 未激活，
B1-A0 与 A1–A4 ladder 已永久关闭。只读 post-mortem 又确认 841/852/870 个 false-block 全部与
predicted-clearance 阈值穿越内部一致，clearance signed bias 为 `-0.216/-0.226/-0.256 m`，跨 seed
false-block mask Jaccard 为 `0.924–0.936`；transition failure 的 81.9%–82.3% 是持续 truth-clear /
predicted-occupied，而非 flip。全部 truth-clear 支持集中于 parent `464241`，故不能外推全场景，也
不能因果指定 depth scale 或 ground/support 为单一罪因。该诊断不具有晋级资格。

新 R2 的 F0 已以零模型、零真实数据、零训练完成 synthetic mechanics kill gate：23 个冻结 case 中
22 个解析真值逐项精确匹配，1 个 learned final-state shortcut 负控被拒绝；10/10 gate PASS，
4 条 uncertainty degradation ladder 无 `CLEAR→OCCUPIED`，12 个反 A0 counterexample 的 unsupported
occupancy 为 0。学习图仍禁止输出 final clearance/occupancy，只能在未来提供 metric-ish depth、support
和 boundary/evidence 连续因子及 uncertainty；版本化 deterministic reducer 是最终三态唯一 producer。
最小顺序仍为 F0 synthetic mechanics → F1 TRAIN-only factor learnability → F2 全新至少 8-parent
Development，数字 task 门不得继承 B1 或使用 anatomy/F0 outcome 事后选取。F0 PASS 只允许冻结下一份
F1 协议，不构成 factor learnability 或真实任务收益证据。

F1-P 现已冻结 14 个 factor prediction 字段、13 个独立 loss、`8/2/2` parent-disjoint
FIT/CHECKPOINT_SELECTION/TRAIN_CANARY 最低角色、factor-only checkpoint Pareto 规则与 8 项 Kill Gate；
aggregate loss 和 reducer/task metric 均不能选 checkpoint 或拯救 factor failure。AG-DCA 只读能力矩阵
同时证明当前监督前门未闭合：metric-depth 有 `4,767/16` 支持，但 support 仅 `320/11`，continuous
boundary truth 与 complete R2 factor schema truth 均为 `0/0`。因此 F1-P 终态是协议已冻结、执行未授权；
本轮没有 label materializer、factor model、trainer、optimizer step 或 checkpoint。
其后静态接口审计确认 F1 tensor 不能直接喂给 F0 reducer：dense `depth_log_sigma_hw` 没有变成
scalar `scale_sigma_m` 的冻结规则，support 缺 `normal_sigma_rad/height_sigma_m`，dense boundary/evidence
没有变成 ordered metric obstacle list 的规则，camera receipt 也没有逐字段 frame binding。当前已完成其
CANARY_LITE protocol lock：`14/14` prediction-field coverage、17 个 operation、8 个 synthetic cases 与
10 个 canary gate 均冻结，通用/专项 validator VALID，13/13 mutation tests PASS。随后零参数实现、
runner 与 10-test focused suite 按 SHA 入库后首次执行：8/8 case、A01–A10 10/10、8/8 双进程 replay
与 7/7 sigma-only mutation 全部 PASS，形成
`R2_F1_FACTORTENSOR_ADAPTER_SYNTHETIC_CANARY_PASS`。这只建立 synthetic seam mechanics，不建立
real factor learnability、headroom 或 task utility。
异质教师只冻结到 C0 complementarity kill gate mechanics：教师 identity、评估 cohort 和输出仍未
授权，未通过 oracle 增益、独占正确 parent、分歧错误浓度和时序优势四类门前不得启动 C1 蒸馏。
时序 D0 也只冻结因果 GRU/TCN/diagonal-SSM 的统一 GeometryState 接口、参数/设备预算和未来
clearance/TTC/compute-gate 输出；单帧候选与新时序 cohort 未就绪，不授权训练或读取 outcome。
移动 M0 只冻结选模后的双 shape ONNX、单 fixed-mixed HTP 候选、新 MOBILE_DEVELOPMENT roster 与
“质量先于性能”门；当前无选定模型、转换、HTP partition 或任务保持证据。

## AG-ST R0 selective labelability handoff（不改变 successor）

AG-ST R0 已冻结为独立并行的 `WILD_LAB / DISCOVERY` 诊断问题：不是让超级 Teacher 直接造
truth，而是用 source anchor 与隐藏的 registered geometry，研究一个跨未见 parent 的
`ACCEPT / UNKNOWN` 准入函数。Stage 0A 只允许未来比较 MapAnything、source-only baseline、
confidence-only 与可解释 gate；只有 0A 可评价后，Stage 0B 才能在**新 pre-outcome 协议和未消费
canary** 下加入 pose-conditioned DA3，测量第二个 Teacher 在同风险下增加多少 coverage。
UniDepthV2、Metric3Dv2 与 semantic mask 都不进入初始 R0。

协议之外，现已按用户要求直接完成一次可逆的 TRAIN-only Stage 0A 实验，结果见
[WILD_LAB result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_STAGE0A_WILD_LAB_RESULT_2026-08-10.json)。
factor-only adapter 从 B0 raw source 恢复 RGB/K/pose/partial depth，不读取 B1 clearance/occupancy；
MapAnything Apache checkpoint 在 16 个 TRAIN parent × 3 帧、`1,009,190` 个确定性隐藏参考像素上完成推理。
64 px 大孔洞下，source-only nearest baseline 的 MAE/`>0.10 m` 比例为 `0.04933 m / 9.41%`；
逐 view source-anchor 校准后的 Teacher 为 `0.03351 m / 6.78%`，完整覆盖下分别改善 `32.1% / 28.0%`。
保留约 `50.1%` confidence coverage 时为 `0.03021 m / 5.12%`。这支持继续做 depth label factory，
也证明 metric anchor 不是可选修饰：未校准 Teacher 的 MAE/`>0.10 m` 比例反而是
`0.05699 m / 15.93%`。

随后不等待完整真值，已直接完成
[SuperTeacher factor-label factory](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_SUPERTEACHER_FACTOR_LABEL_FACTORY_WILD_LAB_RESULT_2026-08-10.json)：
把 source-first depth、Teacher confidence、observed-anchor residual 和 multi-view reprojection residual
合成为 A/B/C/UNKNOWN 分级监督，并为 48 帧物化约 103 MB 可训练 NPZ。metric depth、dense-normal
diagnostic、conservative support 与 obstacle/boundary evidence 的有效覆盖率分别为 `96.45%`、`94.66%`、
`63.57%` 与 `71.60%`；support plane 在 `36/48` 帧成立。physical-boundary seed 只占 evidence-valid
像素的 `0.12%`，2 px 训练带占 `0.48%`，避免把连续斜面大面积误标成边界。

关键突破是独立 multi-view gate。在约 50% coverage 下，仅 confidence 的 MAE/`>0.10 m` 为
`0.03021 m / 5.12%`，加入 anchor 与 multi-view 后为 `0.01607 m / 0.85%`，且 16 个 parent 全部仍可评；
对应相对下降 `46.8% / 83.4%`。anchor-only 收紧到 10% 会饿死部分 parent，而 combined gate 不会。

主线随后回到标签工厂，而不是继续要求某个 student 或 correction module 击败 DepthART。
[multi-Teacher factor-label factory](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_MULTITEACHER_FACTOR_LABEL_FACTORY_WILD_LAB_RESULT_2026-08-10.json)
在同一 16 parent/48 帧上加入独立的 Depth Anything V2；它不替换 MapAnything depth，只把逐帧 source-anchor
后的跨 Teacher 相对分歧加入 quality/uncertainty/UNKNOWN。冻结 `C=0.30` 阈值保留隐藏参考像素的
`90.40%`，接受区 MAE `0.02399 m`，拒绝区 `0.12311 m`，相差 `5.13x`；在 50% coverage 下，
primary 的 MAE 从 `0.01607` 降到 `0.01575 m`，`>0.10 m` 从 `0.85%` 降到 `0.68%`。第二 Teacher
自身全覆盖 MAE 较差（`0.06710 m`），但这不构成失败：它的角色是补充独立分歧证据，而不是取代主 Teacher。
新工厂物化 48 个 NPZ，metric/support/boundary-evidence 覆盖为 `97.72% / 64.24% / 72.36%`；无法形成
足够一致证据的 teacher-only 像素继续为 UNKNOWN。下一步继续扩 source/Teacher 覆盖，不回到 backbone 竞赛。

这一不确定性信号随后在
[TUM cross-source R1](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R1_TUM_CROSS_SOURCE_MULTITEACHER_RESULT_2026-08-10.json)
接受了真正的跨数据源检验。4 个 TUM sequence/12 帧只作 FIT 诊断，冻结的 R0 `C=0.30` 阈值不作回调，
再一次性评到 3 个 parent-disjoint sequence/9 帧。held-out 接受覆盖为 `67.31%`，接受区 MAE
`0.06776 m`，UNKNOWN/拒绝区 `0.28470 m`，相差 `4.20x`；3/3 parent 都保持接受区低风险。
因此 multi-Teacher disagreement 不再只是 ARKitScenes 内部现象。21 个 source-first depth NPZ 已物化：
source-native 覆盖 `66.35%`，Teacher 新增 `9.01%`，总 metric-depth 覆盖 `75.36%`，其余 `24.64%`
明确 UNKNOWN。TUM gravity basis 尚未验证，所以 support/boundary 在这批标签中全部 UNKNOWN。

随后 [fresh-TUM R2](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R2_TUM_THIRD_TEACHER_AND_GRAVITY_FACTORS_RESULT_2026-08-10.json)
换到另 7 个此前未引用的 TUM parent，并在 FIT 选择后才打开 3 个 held-out parent。冻结两教师配方在
held-out 上保留 `89.14%` 接受覆盖，接受/拒绝 MAE 为 `0.02934 / 0.28228 m`，3/3 parent 均保持
接受区低风险。DepthART 单体 MAE `0.09998 m` 优于 DA2 的 `0.13267 m`，但其 union/consensus witness
都没有在 FIT 中形成 no-regret 增益，因此第三 Teacher 不晋级，也没有为了三教师结论回调阈值。

同批 source accelerometer + mocap 又把 Freiburg1/2 的世界 `+Z` gravity basis 验证出来：15 帧的
world-specific-force 角误差 median/P95 为 `4.50° / 9.92°`，最优轴映射相对 runner-up 有 `14.03°`
余量。由此 15/15 帧成功物化 gravity-relative dominant support-plane pseudo-label；eligible 像素上的
normal/support/evidence 覆盖为 `84.14% / 69.25% / 84.13%`。Freiburg3/Xtion 的 6 帧无 accelerometer，
继续全 UNKNOWN。这里的 dominant horizontal support 可能是桌面，不是 source-native walkable-ground
truth；下一步必须在 gravity-native 或 synthetic-exact source 上区分地面与抬高水平面，而不是训练 student。

这一缺口现已由
[R3 support identity](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R3_SUPPORT_IDENTITY_RESULT_2026-08-10.json)
直接定位并修正。TUM 的 5 个 gravity-evaluable parent 中，旧 per-frame dominant-plane 在 3 个 parent、
9/9 帧稳定选到比最低持续水平面高 `0.56–0.80 m` 的桌面/高架面；只有 1 个 parent 的 3 帧确实贴近
最低面，另 1 个保持 ambiguous。采用跨帧 world-height persistence 后，9 个错误帧的 support-positive
像素由 `110,587` 降到 `5,959`，减少 `94.61%`；21/21 NPZ invariant PASS，9 个重力缺失或身份不明帧
继续全 UNKNOWN。

机制又通过了两层独立检查。解析 floor+table exact canary 中，旧规则 3/3 帧选到 `0.75 m` 桌面，
新规则 support precision `1.0`、最低 floor recall `0.8623`，table false-positive 从至少 `0.9507`
降为 `0`，boundary 的 2px precision/recall 为 `1.0/0.9871`。随后只下载 `7.4 MB` 的 ICL-NUIM
官方 living-room exact OBJ 与 global pose，而非重下整段 RGB-D；算法把 source-native `room_floor`
高度 `0.1331 m` 恢复为 `0.1199 m`，误差 `1.32 cm`，9/9 可评视角持续命中，并保留 4 个高架模式。

因此相同 sequence identity 已应用到 16-parent/48-frame multi-Teacher TRAIN labels：11 个 parent、31 帧
通过 parent-level `2/3` camera-height consistency，17 帧 fail-closed 为 UNKNOWN；48/48 NPZ invariant
PASS。identity-valid 像素上的 support-valid/support-positive 为 `86.46%/29.57%`。这打开的是
WILD_LAB masked depth/support 学习入口，不是正式 F1 或 safety；boundary 只有解析 exact mechanics，
外部 pixel-exact validation 未完成，仍不得进入训练主目标。

这项 boundary 缺口随后由
[R4 ICL pixel-exact boundary](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R4_ICL_PIXEL_BOUNDARY_RESULT_2026-08-11.json)
直接检验，而不是继续用解析场景自证。固定沿用 support-identity 实验在任何像素输出前选定的 12 个 ICL
视角，用官方 exact depth、pose 与 OBJ surface identity 构造选择性像素真值。5 个视角满足冻结可评条件，
共 `725` 个 exact target 像素；当前 geometric seed 的 2 px recall 为 `0.9160`，但 precision 只有
`0.1805`，且可评视角少于门槛 `6`，所以冻结 canary 明确 FAIL，boundary 仍不得作为 dense 训练目标。

失败分支没有回调阈值，而是只物化 exact-mesh target 与 geometric seed 在 2 px 内一致的正锚点：5 个
NPZ 共保留 `664/725`（`91.59%`）positive，negative 定义为空，其余像素全部 UNKNOWN。这样边界支线已有
可审计的 source-exact positive supervision，但还没有完整正负监督；下一步应在新的外部 scene/source 上
冻结更完整的 exact geometric boundary target，或增加独立几何 Teacher consensus，不能拿这 12 个视角
继续调参后声称通过。

这个下一步已在未消费的 ICL trajectory 1 上直接完成，结果见
[R5 fresh exact-depth boundary](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R5_FRESH_EXACT_DEPTH_BOUNDARY_RESULT_2026-08-11.json)。
在读取任何 trajectory-1 像素结果前，固定从 965 poses 均匀取 12 个视角，并把 local physical boundary
冻结为：相邻 source-exact depth 反投影点的 3D gap 至少 `0.06 m`；有有效邻域且无 gap 才是 negative，
无有效邻域保持 UNKNOWN。该定义不再依赖不完整的 OBJ material identity，也没有用 Teacher 输出当 truth。

12/12 帧可评，共 `20,625` 个 exact target 像素；当前 geometric seed 的 2 px macro
precision/recall 为 `0.8257/0.9455`，4/4 冻结 gate PASS。12 个 dense NPZ 已物化 source-exact
positive/negative/UNKNOWN、tier 与 provenance。这打开的是外部 synthetic source-exact boundary 监督和
几何规则证据；ARKit/TUM 中由 Teacher 填充的 boundary 仍需 source/Teacher agreement 与 uncertainty
gate，不能借此直接变成训练真值。

这一 Teacher-filled 缺口随后没有靠降低门槛“修过”，而由
[R6 source-native boundary corpus](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R6_SOURCE_NATIVE_BOUNDARY_CORPUS_RESULT_2026-08-11.json)
正面裁决。48 帧 ARKit TRAIN hidden-reference 上，source 自身 point-to-plane seed 对冻结 source factor
boundary 的 overall precision/recall 为 `0.8185/0.5836`；MapAnything、DA2 与 pixelwise-quality 双 Teacher
共识只有 `0.0341/0.3015`、`0.1048/0.2059` 与 `0.1891/0.1433`，parent-macro 共识仅
`0.0594/0.0642`。把 RGB Sobel edge 再加入共识也只有 macro `0.0444/0.0339`。因此当前
Teacher-filled boundary 和 SAM-style refinement 支线均不授权，缺失区继续 UNKNOWN；Teacher 仍只扩 depth。

boundary 主线改为只消费 source-native 或 synthetic-exact depth。现已统一物化 ARKitScenes 16 parent/48 帧、
TUM RGB-D 7 parent/21 帧、ICL exact 1 parent/12 帧，共 3 source、24 parent、81 帧；22/24 parent
有正边界，positive 数分别为 `1,838 / 25,914 / 20,625`。81 个 compact NPZ 全部绑定精确 RGB 文件或
tar member、预处理方式及 RGB/label SHA，5/5 corpus gate 与 5/5 binding gate PASS。这打开的是
source-balanced boundary-only masked-student canary，不改变正式 F1 authority，也不把 Teacher 分歧当负例。

该 canary 已由
[R7 source-boundary learnability](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R7_SOURCE_BOUNDARY_LEARNABILITY_RESULT_2026-08-11.json)
直接执行。冻结 ImageNet MobileNetV3-Small，只训练 `155,129` 参数多尺度 decoder；固定
ARKit `12/2/2`、TUM `5/1/1` parent split，单一 ICL exact parent 只进入 FIT。40 epochs、4,320
source-balanced steps 后才打开 2 个 ARKit + 1 个 TUM canary parent。TUM AP 从常数先验 `0.00718`
升到 `0.03265`，2 px F1 `0.3715`；ARKit AP 从 `0.000115` 升到 `0.00310`，但绝对 AP 与 F1
`0.0414` 仍低。macro F1 `0.2065`，5 项 gate 中 4 项通过，唯一失败是两个 source 都须 AP 绝对
增加至少 `0.02`，故终态仍为 FAIL。

这不是退回 Teacher boundary，也不是降低 gate。结果定位出 representation/denominator 问题：ARKit
canary 453,519 个 valid 像素仅 52 个 positive，单像素硬边界极度稀疏；TUM 同一 decoder 已形成学习信号。
下一候选只能在未消费外部 source 上检验 boundary-distance/soft-band supervision，Bonn 8 parent 可作为
fresh cross-source canary；已消费的 R7 canary 不再用于调门或宣称新候选通过。

[R8 soft-boundary Bonn canary](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R8_SOFT_BOUNDARY_BONN_CANARY_RESULT_2026-08-11.json)
已把 source boundary core 转为 3 px 连续距离热图，只使用 FIT/SELECTION 训练并保持 R7 canary 标签读取数为 0。
冻结后才打开 boundary-branch fresh 的 Bonn 8 parent/24 帧 source depth。Bonn AP 从常数先验 `0.04421`
升到 `0.06409`，4 px precision/recall/F1 为 `0.6450/0.3117/0.4203`；但 AP 绝对增量 `0.019879`
比预冻结 `0.020000` 门槛少 `0.000121`，因此不事后降门，正式终态仍为 FAIL。该结果保留的是连续软边界
表示与跨传感器学习信号，下一步直接物化 `boundary_distance + soft probability + validity/provenance`，不在 Bonn 上调参。

该数据产物已由
[R9 continuous boundary factors](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R9_CONTINUOUS_BOUNDARY_FACTORS_RESULT_2026-08-11.json)
完成：3 source、24 parent、81 帧均物化独立 NPZ，直接提供 `boundary_distance_px_hw`、软概率、uncertainty、
validity/UNKNOWN、tier/provenance 和逐文件 SHA。ARKit 的 1,838 个 core 扩展为 11,219 个 ≤3 px 连续带像素，
TUM/ICL 分别为 147,743/55,061；无效像素的 distance 为 NaN 且显式 UNKNOWN，未被改成负例。6/6 物化 gate
PASS，因此连续 boundary factor 已可与 metric depth/support 一起进入按 factor mask 训练。

[R10 unified factor labels](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R10_UNIFIED_FACTOR_LABELS_RESULT_2026-08-11.json)
随后完成 48/48 frame identity merge：R5 的全部非边界数组逐元素保留，旧 boundary 三字段由 R9 连续边界替换，
每帧统一为 50 个字段并显式区分 depth/normal/support/obstacle/boundary 五套 validity mask。16 parent/48 帧、
104,140,450 bytes，6/6 merge gate PASS；覆盖率分别为 `96.45%/94.66%/63.57%/71.60%/94.03%`。
因此当前已有可直接进入统一 masked-factor 训练的五因子 SuperTeacher 包，不要求完整真值，最终 task state 仍由
deterministic reducer 决定。

[R11 unified factor student](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R11_UNIFIED_FACTOR_STUDENT_RESULT_2026-08-11.json)
已修正旧 trainer 将 boundary/obstacle 共用 validity 的接口错误，并用冻结 DepthART-S 执行共享 head、boundary-only、
factor-split 和 factor-split continuous 四个固定开发实验。最终联合候选在 canary 上将 depth MAE
`1.988→0.610 m`、support BCE `0.718→0.246`、obstacle BCE `0.816→0.527`、boundary soft-BCE
`0.0829→0.0679`；support F1 达 `0.889`，boundary hard/distance F1 达 `0.146/0.141`。但 depth
`>0.10 m` error rate 仍为 `91.78%`，全图 boundary distance MAE 也恶化，因此只冻结为 partial candidate，
停止内部 canary 调参，下一步只做无拟合外部 source evaluation。

该冻结 checkpoint 的
[R12 external Bonn evaluation](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R12_EXTERNAL_BONN_UNIFIED_STUDENT_RESULT_2026-08-11.json)
已在 8 parent/24 帧上完成。DepthART prior 的 parent-macro MAE `0.252 m` 被学生残差恶化为 `1.411 m`；
boundary probability AP 仅 `0.04624`（prevalence `0.04421`），4 px F1 约 `0.000045`，均不支持跨源使用。
这拒绝的是 ARKit-only unified checkpoint，不是 SuperTeacher 标签：R8 source-balanced boundary specialist 在同一 Bonn
协议上已有 F1 `0.4203`。因此下一执行转为 depth identity/no-regret gate 与 source-balanced factor specialist，
不再调 R11 内部 canary，也不允许无条件覆盖 DepthART prior。

因此没有先寻找“完全真值”或等待第二 Teacher，而是直接按 factor validity 与 tier weight 完成了
[冻结 DepthART masked student](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_MASKED_STUDENT_DEPTHART_WILD_LAB_RESULT_2026-08-10.json)。
模型只训练 `11,109` 参数 factor head，固定 `12/2/2` parent split、80 epochs、2,880 steps，总耗时
`33.2 s`。两个 canary parent 的 macro depth MAE 从 `1.988 m` 降到 `0.486 m`，support BCE 从
`0.718` 降到 `0.260`、F1 从 `0.512` 升到 `0.775`，obstacle-evidence BCE 从 `0.816` 降到 `0.456`。
但 selection support BCE 由 `0.760` 恶化为 `0.845`，boundary BCE/距离在 canary 也恶化，且最终
depth `>0.10 m` error rate 仍为 `90.52%`。所以这是 depth 与部分 factor 的跨 parent 学习信号，
不是可用模型；boundary 必须作为独立稀疏因子问题继续处理。

[fresh-parent zero-shot replication](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_FRESH_PARENT_ZERO_SHOT_RESULT_2026-08-10.json)
随后把一次内部 canary 推进成两条独立的 `train parents -> fresh parents` 链，评估端均不拟合、不选阈值，
且 checkpoint 训练/selection/canary parent 与 fresh evaluation parent 的交集严格为空。第一条链把原
16-parent checkpoint 直接评到另一批 16 parent：depth MAE `1.7628 -> 0.5074 m`，support BCE
`0.7574 -> 0.4438`、F1 `0.2675 -> 0.4379`；obstacle BCE 也由 `0.9377 -> 0.7637`。第二条链在
这 16 个 parent 上重新训练 checkpoint，再评到第三批 4 个未见 parent：depth MAE
`1.7496 -> 0.4041 m`，support BCE `0.4138 -> 0.2781`、F1 `0 -> 0.4622`。

因此 depth/support 的跨 parent 学习信号已经完成两链复现，不再只是一次 split 偶然；但 obstacle 在
第二条链由 `0.7132` 恶化到 `0.7753`，只能保留为 diagnostic。boundary 在原多任务、boundary-only
负控和两次 fresh zero-shot 中都未通过，当前 target/representation 不支持 transferable claim。
两次 zero-shot 的 depth `>0.10 m` error rate 仍分别为 `87.14% / 77.01%`，所以结果仍不是任务可用模型。

这一步现已按
[combined-32 depth/support result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_COMBINED32_DEPTH_SUPPORT_RESULT_2026-08-10.json)
直接执行：两批 16-parent 标签不复制地合并为 32 parent/96 帧，固定 `28/2/2` split，80 epochs，
只让 depth/support 进入梯度，obstacle/boundary heads 保持训练先验。随后一次性评到另行预留且与
checkpoint 全角色零重叠的 8 个 DEVELOPMENT parent/24 帧；没有 fresh fitting 或 threshold selection。
depth MAE `2.0418 -> 0.2907 m`，相对下降 `85.8%`，8/8 parent 改善；support BCE
`0.7084 -> 0.3722`，相对下降 `47.5%`，7 个可评 parent 中 6 个改善，F1 `0 -> 0.6554`。
这把 depth/support 路线从两次小模型复现推进成 combined-data scaling signal。

随后把这 8 个已消费 DEVELOPMENT parent 并入 fit，形成
[combined-40 precision confirmation](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_COMBINED40_PRECISION_CONFIRMATION_RESULT_2026-08-10.json)：
40 parent/120 帧全部用于训练，depth 加入 absolute-meter Huber 与 `0.10 m` soft margin，support 改用
unweighted BCE + soft Dice，并只在训练 logits 上拟合 temperature/bias 后折叠回 head。checkpoint 冻结后，
才一次性打开与此前 44 parent 零交集的 CONFIRMATION 8（`180/210/240`，24/24 adapter PASS）。
confirmation depth MAE `2.2745 -> 0.3321 m`，下降 `85.4%`，8/8 parent 改善；support BCE
`0.5791 -> 0.0972`，下降 `83.2%`，5/5 可评 parent 改善，F1 `0 -> 0.6436`。

这确认 depth/support supervision scaling 已成立，但 frozen 11k head 的精度瓶颈仍直接可见：confirmation
depth `>0.10 m` error rate 仍为 `67.68%`，并非任务可用精度；support 也只有 5/8 parent 可评。
obstacle/boundary 因未训练而不作 rescue claim。下一轮应测试更强 multi-scale factor decoder 或 bounded
DepthART adapter；当前 B0 的 DEVELOPMENT/CONFIRMATION 已消费，新的泛化结论必须另留外部数据源。
第二 Teacher 保留为 coverage/独立性增益实验，而不是训练前门。

该容量假设现已由
[multi-scale + Bonn cross-dataset development result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_MULTISCALE_AND_BONN_CROSS_DATASET_DEVELOPMENT_RESULT_2026-08-10.json)
直接裁决。`136,517` 参数、DepthART 四层 decoder pyramid、dilation `1/2/4` 且显式读取 base-depth
guidance 的 head 完成 40-parent/120-frame 全量拟合。在已消费 confirmation8 上，它只把 depth
`>0.10 m` error 从 `67.68%` 小幅降到 `65.89%`、support BCE 从 `0.0972` 降到 `0.0914`；同时
depth MAE 从 `0.3321` 恶化到 `0.3868 m`，support F1 从 `0.6436` 降到 `0.6105`，故不晋级。

更关键的是，两个 frozen head 又在 Bonn RGB-D Dynamic 的固定 8 sequence × 3 帧上接受 registered
source-native depth 检验。原始 DepthART baseline 的 parent-macro MAE 为 `0.2533 m`；11k 小头与
multi-scale 头分别恶化到 `1.0146 / 1.1755 m`，且都是 `0/8` parent 改善，`>0.10 m` error 均接近
`99.9%`。这否定了当前 ARKitScenes residual 的跨数据源 depth transfer，也说明继续堆 decoder 不是
答案：它学到的是域特定 metric correction，覆盖了在 Bonn 上更好的 DepthART prior。下一轮只值得做
source-diverse A-tier depth anchor + identity-preserving/OOD-gated residual，或冻结 depth 只学 support；
Bonn 本轮没有 support label，因此没有 support、task 或 safety claim。

该 mixed-domain 实验现已按
[冻结 cohort](BLINDASSIST_AG_ST_BONN_MIXED_DOMAIN_COHORT_R0_2026-08-10.json) 与
[identity-gated result](BLINDASSIST_ASSISTIVE_GEOMETRY_AG_ST_R0_BONN_ANCHORED_IDENTITY_GATED_RESULT_2026-08-10.json)
执行。排除前一轮 Bonn fixed-8 后，以固定 SHA 从其余 sequence 锁定 8 FIT / 8 EVAL / 2 reserve；
FIT 的 24 帧 registered depth 作为 A-tier depth-only anchor，support/boundary/obstacle 全部 UNKNOWN。
学生使用 shared DepthART feature、base-depth guidance 和初始仅 5% 开放的 identity gate，并把 Bonn
frame 重放 5 倍形成 120:120 domain-balanced optimizer visits。10 epochs / 2,400 steps 后，FIT 内 Bonn
MAE `0.2871 -> 0.1610 m`，ARKit MAE `1.8933 -> 0.3120 m`，两域都通过 FIT 门。

checkpoint 冻结后一次性打开 8 个 disjoint Bonn EVAL parent。灾难性 collapse 确实被消除：相较此前
ARKit-only head 的 `1.0146/1.1755 m`，新 student 为 `0.2713 m`；但原始 DepthART baseline 是
`0.2517 m`，student 仍恶化 `7.8%`，`>0.10 m` error 由 `73.10%` 升到 `77.65%`，只有 `1/8`
parent 改善。因此当前 gate 不晋级，也不在已消费 EVAL 上回调参数。关键算法诊断是：当前 gate 学的是
“需要多大 correction”，不是“correction 是否比 base 更可靠”。下一轮应拆成两阶段：先冻结 correction
expert，再训练 no-regret selector 预测 `error(corrected) < error(base)`；没有正改善证据时直接回退 base。
这需要至少第三个 metric RGB-D sensor domain，而不是继续在 Bonn EVAL 上调 gate。

这些文件是分级 pseudo-label，不是完整 truth；uncertainty 字段仍是 proxy，dense normal 仍是派生诊断。
它们足以启动 WILD_LAB masked training，但不把当前正式 `SUPERVISION_FRONTDOOR_UNSATISFIED`、F1、
跨数据源泛化或 safety 改成 PASS。当前 WILD_LAB 角色合计已消费 52 个互异 ARKitScenes parent；它们
不能再次被称为 fresh evaluation，下一轮确认必须使用仍未消费的 parent/source。

## 并行 WILD_LAB 数学 canary handoff（不改变 successor）

纯合成 CPU canary 已审查四个 forward hypothesis：censored robust-contact survival、
profile-conditioned swept configuration clearance、maximum-bottleneck corridor loss 和
cluster-level one-sided conformal/CRC uncertainty。数学不变量与反例均通过，终态为
`MATH_MECHANICS_SUPPORTED_PAPER_NOVELTY_AND_LEARNABILITY_NOT_ESTABLISHED`。其中 H1/H2
优先，H3 只能作为已有 max-min/topology 文献下的 task-specific loss，H4 受 calibration
parent 数量阻塞：8% finite-sample risk 至少需 12 个独立 parents，当前 4 个不可能。

该 canary 不读取模型、checkpoint 或任何数据 role outcome，不修改冻结 A0–A4，不授权 B2、
部署、默认 App 或 safety；B1 A0 的负终态不向该并行路线转移 Selection outcome 或晋级权限。
H1/H2 的后续实现与 TRAIN-only canary authority 已移交独立
[AG-QSF current](../assistive-geometry-qsf/README.md)；两条路线只读共享冻结资源，run state、
checkpoint、target cache、outcome、scheduler 与 artifact root 保持隔离。
