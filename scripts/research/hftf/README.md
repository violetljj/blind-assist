# hftf

状态：`development / candidate-side-lane /
R3.1-reference-opportunity-not-evaluable / split-source-successor-only`

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

formal R3 必须使用已绑定四 source hashes 的 protocol：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_b_reference_comparison.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_PROTOCOL_R3_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-preparation docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_REFERENCE_COMPARISON_SOURCE_PREPARATION_R3_2026-08-01.json `
  --session <fresh-r3-replay-a> <authority-a.json> `
  --session <fresh-r3-replay-b> <authority-b.json> `
  --session <fresh-r3-replay-c> <authority-c.json> `
  --session <fresh-r3-replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/stage_b_r3.json
```

runner 先裁决 source/reference/known readiness，再裁决 obstacle 增益，最后单列 ground
opportunity 与 agreement。full terminal 也只允许冻结下一 Stage C protocol，不直接授权
future execution 或 student。

R3.1 单 source qualification 只能运行 reference arm：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/qualify_stage_b_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan artifacts.local/evidence/hftf/r3-1-inventory-plan-20260801/inventory_plan.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --replay-root <candidate-replay> `
  --authority <candidate-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

runner 固定 D0 mechanics hash，复核实际消费的 depth/mask 与 authority bindings，并拒绝
16 个 burned sessions。报告不包含 candidate、baseline、confusion 或 arm delta。

40-session bounded inventory plan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_r3_1_inventory_candidates.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

planner 验证 official split generation/hash，只读 description 与 RGB/mask/depth 对象清单，
记录 burned/ineligible 跳过原因及前 40 个 eligible sessions 的确定性 frame indices。

完成全部 source reports 后，使用 cohort aggregator 验证冻结顺序与 reference-only
firewall：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/aggregate_r3_1_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_REFERENCE_ONLY_OPPORTUNITY_QUALIFICATION_R3_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R3_1_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan artifacts.local/evidence/hftf/r3-1-inventory-plan-20260801/inventory_plan.json `
  --report <rank-001-qualification.json> `
  --report <...in exact contiguous inventory order...> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort_result.json
```

若先得到 4 个 qualified source，报告数必须精确停在第 4 个 qualified rank；若不足 4
个，则必须提供全部 40 个报告才能得到 budget-exhausted terminal。R3.1 实际终态为
`R3_1_REFERENCE_OPPORTUNITY_COHORT_NOT_EVALUABLE`，不得在同一队列继续扫描或降门。

### Stage B split-source R4

R4 obstacle source 先按 56-session burn ledger 生成最多 12 个的新 inventory plan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_r4_obstacle_inventory_candidates.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

每个 source 的 qualification 只可计算 obstacle dense reference：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/qualify_r4_obstacle_reference_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --replay-root <candidate-replay> `
  --authority <candidate-authority.json> `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

terrain component 完全由冻结解析 profiles 生成，不读取 SANPO outcome：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_r4_analytic_terrain.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/terrain_result.json
```

terrain pass 只代表 controlled mechanics component 通过，不能独自签发 joint R4
terminal 或 Stage C 权限。

前四个 qualification 通过后，先锁定 source hashes，再运行 obstacle arm 和 joint
aggregation：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_r4_obstacle_opportunity_cohort.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --report <contiguous-rank-qualification.json> `
  --output artifacts.local/evidence/hftf/<run-id>/cohort_lock.json

E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_r4_obstacle_reference_comparison.py `
  --protocol docs/research/hftf/HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_R4_SOURCE_POOL_BURN_LEDGER_2026-08-01.json `
  --inventory-plan <inventory-plan.json> `
  --cohort-lock <cohort-lock.json> `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --session <replay-a> <authority-a.json> `
  --session <replay-b> <authority-b.json> `
  --session <replay-c> <authority-c.json> `
  --session <replay-d> <authority-d.json> `
  --output artifacts.local/evidence/hftf/<run-id>/obstacle_result.json
```

`aggregate_r4_split_source_result.py` 是唯一可签发 joint R4 terminal 的工具；单个
component 不得提前开放 Stage C。

Stage C C0 在任何 EgoWalk RGB/depth media 内容打开前，先用 exact dataset revision
和四个 metadata hashes 复算 239 条 trajectory 的健康门与冻结 cohort：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_c0_egowalk_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json `
  --metadata-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/inventory.json
```

planner 只读 parquet/meta 与远端 LFS size/hash，不下载或打开 RGB/depth，也不读取
annotation、teacher label 或 student output。必须精确复现冻结的两个不同日期 source，
否则在 media acquisition 前 fail closed。

cohort lock 后只下载其绑定的两组 media，再以官方 `gray16le mm -> m / zero ->
UNKNOWN` 规则完整解码，并运行冻结的 32-frame transport/surface-support audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_c0_egowalk_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_2026-08-01.json `
  --inventory <locked-inventory.json> `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport_audit.json
```

audit 会完整解码全部 RGB/depth 帧并核对 LFS/local SHA、帧数、5 Hz rate 与 PTS；
32-frame canary 只检查正有限 depth 和 bottom-half/common support，不读取 semantic
class、annotation 或 hazard/safe truth。

C0 的 container nominal-rate 门失败后，C0.1 只允许在 hash-bound consumed replay 上
用 parquet frame/timestamp + meta fps 修复 timebase authority：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_c0_1_egowalk_timebase_repair.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/timebase_repair.json
```

runner 要求 predecessor 的唯一 failures 精确为 RGB/depth nominal-rate mismatch；
若存在任何其他 C0 failure，禁止用 C0.1 越过。

Stage C D0 在 consumed calibration sources 上运行冻结的 depth-only ground-plane /
horizontal-support reader、七个 structural canaries 与第二遍 determinism：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_d0_semantic_independent_label_readiness.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SEMANTIC_INDEPENDENT_LABEL_READINESS_D0_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/label_readiness.json
```

formal runner 不读取 semantic class、annotation 或 RGB outcome。`<1.2 m` 与缺失
support 永远 UNKNOWN；即使 D0 full pass，也只允许冻结 fresh-source label/student
canary protocol。

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
