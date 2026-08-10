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
合成为 A/B/C/UNKNOWN 分级监督，并为 48 帧物化约 105 MB 可训练 NPZ。metric depth、dense-normal
diagnostic、conservative support 与 obstacle/boundary evidence 的有效覆盖率分别为 `96.45%`、`94.66%`、
`25.05%` 与 `71.60%`；support plane 在 `36/48` 帧成立。

关键突破是独立 multi-view gate。在约 50% coverage 下，仅 confidence 的 MAE/`>0.10 m` 为
`0.03021 m / 5.12%`，加入 anchor 与 multi-view 后为 `0.01607 m / 0.85%`，且 16 个 parent 全部仍可评；
对应相对下降 `46.8% / 83.4%`。anchor-only 收紧到 10% 会饿死部分 parent，而 combined gate 不会。
因此现在不需要先寻找“完全真值”或等待第二 Teacher：下一步可以直接按 factor validity 与 tier weight
训练 masked student。第二 Teacher 保留为后续 coverage/独立性增益实验，而不是首轮训练前门。

这些文件是分级 pseudo-label，不是完整 truth；uncertainty 字段仍是 proxy，dense normal 仍是派生诊断。
它们足以启动 WILD_LAB masked training，但不把当前正式 `SUPERVISION_FRONTDOOR_UNSATISFIED`、F1、
跨源泛化或 safety 改成 PASS。当前 16 个 TRAIN parent 已消费，正式泛化评价必须使用 fresh parent/source。

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
