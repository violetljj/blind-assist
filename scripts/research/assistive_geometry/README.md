# Assistive Geometry research scripts

状态：`B1_A0_PERMANENT_NEGATIVE_TERMINAL / R2_F0_SYNTHETIC_REDUCER_PASS / F1_P_PROTOCOL_FROZEN / SUPERVISION_FRONTDOOR_UNSATISFIED / F1_EXECUTION_NOT_AUTHORIZED / CALIBRATION_AND_CONFIRMATION_SEALED`

本目录包含 BlindAssist Assistive Geometry B0 的冻结合同、shape/export、metadata roster、
可恢复媒体物化与 label-blind integrity 工具：

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

大体积输出只允许写入 `artifacts.local/datasets/`、`artifacts.local/evidence/hftf/` 或
`artifacts.local/evidence/assistive-geometry/`。
roster 选择只依据冻结 metadata/hash，不读取模型输出或 task outcome。当前合同和结果真源位于
`docs/research/assistive-geometry/`。

## 安全边界

本模块的 B1-A0 及 A1–A4 已永久关闭；teacher 只有未激活的历史 C0 complementarity mechanics，
当前不读取 teacher output，也不授权 C1、QNN/HTP、默认 App、产品或 safety。
时序模块同样只有未激活 mechanics；没有新 temporal cohort、训练、任务收益或设备性能 authority。
移动导出受历史 M0 质量先于性能协议约束；现有 DepthART D1 cohort 不得复用为 Assistive Geometry
选模证据。新 R2 已完成 F0 reducer mechanics，并冻结 F1-P schema/loss/selection/Kill Gate；当前
continuous boundary 与 complete factor-schema truth 均为 0，F1 supervision frontdoor 不满足；此外
F1 tensors 与 F0 reducer 之间的 deterministic `FactorTensorAdapter` protocol 已冻结，但实现与 synthetic
canary 尚未运行。没有 factor model、label materializer、trainer、optimizer、checkpoint、真实任务收益
或 F1 execution authority。
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
`BLINDASSIST_ASSISTIVE_GEOMETRY_R2_F0_SYNTHETIC_FACTOR_GEOMETRY_CANARY_PASS`；F1-P 又签署
`R2_F1_PROTOCOL_FROZEN_EXECUTION_NOT_AUTHORIZED_SUPERVISION_FRONTDOOR_UNSATISFIED`。后续接口审计签署
`R2_F1_EXECUTION_BLOCKED_FACTORTENSOR_ADAPTER_ABSENT`；其后 protocol lock 已将缺口收缩为
`R2_F1_ADAPTER_STATIC_CONTRACT_COMPLETE_CANARY_NOT_RUN`。当前唯一 successor 只允许实现冻结的
outside-graph adapter 并运行 8-case/A01–A10 synthetic canary；标签物化、模型定义、训练、real outcome
与 F1 authority 仍为 false，监督源/label contract 仍是后续独立必要门。

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/assistive_geometry -p "test_*.py"
```
