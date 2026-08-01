# hftf

状态：`development / candidate-side-lane / Stage-B-D0-admitted / fresh-R3-authorized`

## 研究问题与版本

本 Module 服务 `HFTF_CANDIDATE_LANE_R0`：检验历史 RGB 能否预测面向行人身体包络的
短时未来可通行/碰撞风险场，而不是继续给 YOLO 增加后处理规则。当前只执行
`HFTF_H0_SOURCE_FEASIBILITY_R0`，允许的 claim 是来源与教师接口可行性，不是模型效果、
创新性、用户效果或安全性。

当前章程与终态见 `docs/research/hftf/README.md`。通用 H0 的 partial terminal 仍保留；
source-specific H0.1/H0.2 已准入下一阶段的 geometry proxy canary。

## 稳定 Interface

从仓库根目录运行：

```powershell
$runId = 'h0-source-feasibility-r0-REPLACE_WITH_NEW_RUN_ID'
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_source_feasibility.py `
  --replay-root artifacts.local/evidence/datasets/sanpo-synthetic-replay-25frames-20260720 `
  --output "artifacts.local/evidence/hftf/$runId/source_feasibility.json"
```

输入必须包含 hash-bound RGB、panoptic mask、metric depth、相机内参、pose CSV、
`dataset_spec.json`、`manifest.replay.jsonl` 和既有 source-integrity QA。相对路径必须
保持在 replay root 内；报告路径必须位于 `artifacts.local/` 且已存在时拒绝覆盖。
静态 projection 资格由脚本独立复算全部文件 hash、完整 PNG decode/dimensions、depth
header/shape 与 finite-positive samples；QA 还必须以 schema、`ok`、frame count 和逐
depth path 与 manifest 一致。输入中的 `SANPO-Synthetic`/official split 字段只作为
内部一致性声明；本 H0 不把本地 manifest 自报内容当作来源身份的密码学认证。
重复 canonical asset path 或完整 RGB/mask/depth hash triplet 会 fail closed；QA
布尔字段必须是精确 JSON boolean，字符串 `"false"` 不视为 false declaration；
frame count、fraction 与相机内参拒绝 bool 或字符串伪数值。

multi-height/future 的**结构准备度**不能由普通 CSV 列名或非空占位字段获得。通用
H0 可检查下列精确合同：

- `hftf_body_frame_contract`：精确 schema、frame/axis/unit/direction、有限且归一化
  SE(3)、ground reference 和 provenance；
- `hftf_pose_binding`：hash-bound JSONL，把每个 manifest row 一一映射到唯一 raw pose
  row，并核对 session/sequence/frame/time、admitted tracking state、有限 position 与
  归一化 quaternion。

即使上述结构检查全部通过，本工具仍把 multi-height/future 判为 `NOT_EVALUABLE`。
真实准入还必须由 source-specific verifier 分别复算标定 receipt 与原始
pose-frame/time mapping；hash-bound sidecar 不能给自己签发权威。

### SANPO source-specific H0.1/H0.2

H0.1 discovery：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/verify_sanpo_pose_geometry_authority.py `
  --evaluation-mode discovery `
  --replay-root <single-session-replay-root> `
  --official-repo artifacts.local/downloads/sanpo_dataset_official_repo `
  --output artifacts.local/evidence/hftf/<run-id>/authority.json
```

H0.2 replication 对 H0.1 已冻结的
`p_world = R_xyzw @ p_opencv_camera + translation_m` 做跨 session 检验：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/verify_sanpo_pose_geometry_authority.py `
  --evaluation-mode frozen_canonical_replication `
  --replay-root <independent-single-session-replay-root> `
  --official-repo artifacts.local/downloads/sanpo_dataset_official_repo `
  --output artifacts.local/evidence/hftf/<run-id>/authority.json
```

verifier 固定 official repository commit/common.py hash，在线复核 GCS object
generation/size/MD5/CRC32C，再验证本地 MD5、official pose-row/frame-index 规则、48 个
pose/basis hypothesis、metric-depth reprojection 和 semantic-ground local plane。
`frame_num / session fps` 只表示 nominal relative time。

三个或更多独立 frozen-replication reports 用以下命令聚合：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/aggregate_sanpo_proxy_replication.py `
  --report <session-a-authority.json> `
  --report <session-b-authority.json> `
  --report <session-c-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort.json
```

聚合器拒绝重复 source session，并保持 physical calibration、student/effect、主线和
产品层为 `NOT_EVALUABLE`。

### H1 geometry teacher canary

H1 必须使用已提交的 frozen protocol 和其中四个精确 authority/report hashes。R0
360° evidence version 已执行并 burned；后续正式运行使用 R1 forward-sector protocol
与四个全新 sessions：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_geometry_teacher_canary.py `
  --protocol docs/research/hftf/HFTF_H1_FORWARD_SECTOR_GEOMETRY_TEACHER_CANARY_PROTOCOL_R1_2026-08-01.json `
  --session <fresh-replay-a> <fresh-authority-a.json> `
  --session <fresh-replay-b> <fresh-authority-b.json> `
  --session <fresh-replay-c> <fresh-authority-c.json> `
  --session <fresh-replay-d> <fresh-authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_canary.json
```

runner 会重算 protocol、authority、manifest、dataset spec、pose 与每个实际消费
depth/mask 文件 hash；theta edges 同时约束 cell probes 与 obstacle binning，partial
sector 外 points 不 wrap；R0/R1 future field 保持 anchor-centric，所有版本的 UNKNOWN
都留在冻结 denominator。
输出终点只可能是 H1 的 `NOT_EVALUABLE`、multi-height/future stop 或
`GEOMETRY_PROXY_MECHANISM_SUPPORTED`，后者也不会自动授权 H2。

R2 protocol 额外绑定 source-preparation contract。runner 对每个 anchor 选择冻结
lookback/tolerance 下的严格历史 frame，仅用 history-to-anchor pose 计算
ground-tangent velocity，并为 `.4/.8 s` 分别生成 horizon-specific rolling origin、
probes 与 obstacle bins。future pose 只作为 observation，不定义 origin；predicted 与
observed ground-origin error 仅作 diagnostic，不进入 gate。

### Stage B swept-envelope label mechanics D0

R2 只关闭 angular-cell point-support proxy；它没有实现原始 Stage B 所需的人体横向
包络、候选轨迹 swept collision 与足部 ground continuity。D0 因而只在已烧毁的 R2
sources 上检查这套标签 mechanics，不是 fresh evidence，也不评价未来轴或 student：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_swept_envelope_label_mechanics.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --r2-protocol docs/research/hftf/HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.json `
  --session <burned-r2-replay-a> <authority-a.json> `
  --session <burned-r2-replay-b> <authority-b.json> `
  --session <burned-r2-replay-c> <authority-c.json> `
  --session <burned-r2-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/mechanics.json
```

实现使用冻结的 synthetic effective half-width、9 个 swept-prism probes 和 5-section
ground support；只有 known 且数值 risk 为零的 cell 才编码为 SAFE，缺失 ground
support 保持 UNKNOWN。输出准入 fresh R3 也只代表 mechanics 可执行且非退化，不代表
风险真值、Stage B 增益、H2 或主线替换。

D1 在同一 burned cohort 上比较 candidate、baseline 与 disjoint dense reference：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/pilot_swept_envelope_reference_metrics.py `
  --pilot docs/research/hftf/HFTF_STAGE_B_REFERENCE_METRIC_PILOT_D1_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --r2-protocol docs/research/hftf/HFTF_H1_CAUSAL_ADVECTED_ORIGIN_GEOMETRY_TEACHER_PROTOCOL_R2_2026-08-01.json `
  --session <burned-r2-replay-a> <authority-a.json> `
  --session <burned-r2-replay-b> <authority-b.json> `
  --session <burned-r2-replay-c> <authority-c.json> `
  --session <burned-r2-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/reference_metrics.json
```

candidate/reference pixel grids 不相交；四个 reference count thresholds 必须全部报告。
D1 只设计 R3 gate，不选择 fresh outcome。

## 输出

只写入显式的 `artifacts.local/evidence/hftf/<run-id>/source_feasibility.json`。报告分别
裁决静态 metric projection、多高度身体包络教师、短时未来教师和独立 student-effect
评价，不把上一级可用性自动传递给下一级。本通用 H0 只可能准入静态 projection；
multi-height/future 需要后续 source-specific admission，student-effect 必须由 H2/H3
的独立 hash-bound parent-event ledger validator 裁决。

## 安全边界

这是 host-only `DEVELOPMENT_STANDARD` 审计。不训练模型，不读取 fresh/blind，不修改
Android、提醒或默认 App。合成深度/位姿派生结果只能叫 geometry-derived proxy；
没有独立人类事件真值时不得称为风险真值。

## 停止条件

报告产生下列一个终态即停止：

- `HFTF_H0_SOURCE_INTEGRITY_NOT_EVALUABLE`
- `HFTF_H0_SOURCE_FEASIBILITY_PARTIAL`
- `HFTF_H0_1_SOURCE_AUTHORITY_NOT_EVALUABLE`
- `HFTF_H0_1_POSE_MAPPING_ONLY`
- `HFTF_H0_1_SANPO_PROXY_FRAME_ADMITTED`
- `HFTF_H0_2_CANONICAL_PROXY_NOT_REPLICATED`
- `HFTF_H0_2_SANPO_CANONICAL_PROXY_REPLICATED`
- `HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_NOT_EVALUABLE`
- `HFTF_H0_2_INDEPENDENT_SESSION_REPLICATION_ADMITTED`

任何 blocker 只关闭相应 evidence instance。修复来源合同必须生成新输出路径，不覆盖
旧报告；不得靠默认行号、跨 session 时间差、自报事件数量或 session 改名补出
pose-frame binding、future span 或 effect eligibility。

## 假设与规则质疑

方向、距离、高度、人体包络、dynamic 与 uncertainty 都是历史 USTRF 的继承
primitive；不构成新颖性。唯一待证表示增量是 action-agnostic、history-only RGB 对
显式 short-future layered cells 的预测。falsifier 是：多高度或未来轴相对
single-height/current-field 没有独立增量，或 student 在相同事件账本与算力约束下不能
优于 incumbent。

## 失败资产复用

失败报告可作为数据来源缺口、pose/body-frame 合同、teacher leakage 与 evaluation
readiness 的 regression fixture；不能重包装为 HFTF 模型负结果、创新性结论或
unseen Confirmation。
