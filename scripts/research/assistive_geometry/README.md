# Assistive Geometry research scripts

状态：`A0_FORMAL_RUNNER_AND_HOST_PERFORMANCE_PILOT_PASS / WORKERS_1_SELECTED / SEED_17_GUARDED_EXECUTION_NOT_STARTED / DEVELOPMENT_AND_CONFIRMATION_SEALED`

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

## 输出

大体积输出只允许写入 `artifacts.local/datasets/` 或 `artifacts.local/evidence/hftf/`。
roster 选择只依据冻结 metadata/hash，不读取模型输出或 task outcome。当前合同和结果真源位于
`docs/research/assistive-geometry/`。

## 安全边界

本模块不训练 student、不运行 teacher matrix，也不授权 QNN/HTP、默认 App、产品或 safety。
`UNKNOWN` 不得当作负例；synthetic shape 与 benchmark geometry 不得冒充任务质量。

## 停止条件

合同违规、checkpoint/shape 不匹配、非 finite 输出、camera prompt drift 或 ONNX checker
失败均立即 fail closed。当前 roster、source integrity、truth reader 与 registration 已关闭，
且 B1 target/loss/confidence、dual-orientation overlay、4,800-frame target cache 与模型/loss
implementation lock 与 A0 execution lock 已关闭；当前只授权按冻结合同执行 A0 三 seed 的
TRAIN-only 正式训练。训练中仍不得打开 Development/Confirmation outcome，不得运行 A1–A4、
teacher、部署或默认 App 路径。

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/assistive_geometry -p "test_*.py"
```
