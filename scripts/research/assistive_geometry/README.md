# Assistive Geometry research scripts

状态：`B0_TRUTH_READER_AND_REGISTRATION_LOCK_PASS / NO_TRAINING_AUTHORITY`

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
但 B1 target/loss/confidence threshold 和 training protocol 尚未冻结；不得从本目录启动训练
或打开独立 outcome。

验证：

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest discover `
  -s scripts/research/assistive_geometry -p "test_*.py"
```
