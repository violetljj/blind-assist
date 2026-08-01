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

### Stage C SANPO body/head temporal-student F0

F0 的 source planner 只读取 official split、description、intrinsics、pose object
receipt 与 RGB/mask/depth object inventory；不下载媒体，不计算 geometry/student
outcome：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_f0_sanpo_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/inventory_plan.json
```

planner 必须排除 effective 60-session burn union，按完整 ID 字典序固定 12 个 source，
并按 rank 固定 `6 train / 3 dev / 3 heldout`。任何 geometry outcome 打开后不得重新
规划或替换。

F0.1 在任何 media/geometry/student outcome 前把 heldout 加强为 official test
split。它复用 F0 metadata plan 的前九个 train/dev candidates，只对 official test
文件顺序做 heldout metadata scan：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/plan_stage_c_f0_1_sanpo_cross_split_inventory.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --f0-plan artifacts.local/evidence/hftf/<f0-run-id>/inventory_plan.json `
  --output artifacts.local/evidence/hftf/<f0-1-run-id>/inventory_plan.json
```

输出必须固定 `6 train / 3 dev / 3 official-test heldout`，且所有 outcome firewall
保持 false，才授权 exact media acquisition。

在下载前用 source-lock validator 固化 exact sessions、split、物理 timeline 与
description/pose GCS receipts：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_f0_1_sanpo_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --burn-ledger docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_SOURCE_POOL_BURN_LEDGER_F0_2026-08-01.json `
  --f0-plan artifacts.local/evidence/hftf/<f0-run-id>/inventory_plan.json `
  --cross-split-plan artifacts.local/evidence/hftf/<f0-1-run-id>/inventory_plan.json `
  --output artifacts.local/evidence/hftf/<lock-run-id>/source_lock.json
```

`aggregate_r4_split_source_result.py` 是唯一可签发 joint R4 terminal 的工具；单个
component 不得提前开放 Stage C。

F0.1 exact media 获取后，先用
`audit_stage_c_f0_1_sanpo_acquisition.py` 对 12 个包的 300 组 RGB/mask/depth、
split、物理索引、GCS/local hash 与 pose 文件做统一审计；再逐 source 运行
`verify_sanpo_pose_geometry_authority.py --evaluation-mode
frozen_canonical_replication`，最后由
`aggregate_stage_c_f0_1_sanpo_authority.py` 封口 exact authority cohort。

teacher opportunity 必须使用 outcome 前冻结的
`HFTF_STAGE_C_SANPO_TEACHER_OPPORTUNITY_EXECUTION_CONTRACT_F0_1_2026-08-01.json`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_f0_1_teacher_opportunity.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_SANPO_TEACHER_OPPORTUNITY_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --f0-protocol docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --f0-1-protocol docs/research/hftf/HFTF_STAGE_C_SANPO_CROSS_SPLIT_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_1_2026-08-01.json `
  --mechanics-protocol docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-lock <source_lock.json> `
  --acquisition-audit <acquisition_audit.json> `
  --authority-cohort <authority_cohort.json> `
  --datasets-root artifacts.local/evidence/datasets `
  --authority-root artifacts.local/evidence/hftf/<authority-run-id> `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

该工具只输出 source/role/horizon/height 汇总，不物化 cell corpus。只有终态
`F0_1_SANPO_TEACHER_OPPORTUNITY_READY_FOR_CORPUS` 才可物化 train candidate
corpus 与 dev reference targets；official-test heldout targets 必须继续封闭到
checkpoint 冻结后的 ordered evaluation。

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

Stage C D1 在同一 consumed cohort 上检验 history-origin-causal 的 future observation
label increment：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_d1_causal_future_label_mechanics.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_CAUSAL_FUTURE_LABEL_MECHANICS_D1_2026-08-01.json `
  --media-root artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/future_label_mechanics.json
```

runner 同时计算 current-observation-only baseline 与 current+future candidate；future
pose 只能重投影 observation，不能决定 causal origin/grid orientation。D1 不训练
student。

Stage C E0 source lock 必须在读取六条 fresh RGB/depth 前复算：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_fresh_student_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --inventory artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-lock-r1-20260801/inventory.json `
  --pretrained-weight artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 只读已消费的 metadata inventory、parent hashes 与通用预训练权重，不打开
fresh RGB/depth。只有 `E0_FRESH_SOURCE_LOCK_VALIDATED` 才授权获取机器合同中精确
绑定的六条媒体；仍不授权 teacher corpus 或 student training。

正式 source lock 通过后，只允许按协议 exact allow-list 获取 E0 媒体：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/acquire_stage_c_e0_fresh_media.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-fresh-source-lock-20260801/source_lock.json `
  --output-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --manifest artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801/acquisition_manifest.json
```

首次打开 RGB/depth 后六条 source 全部 burned。acquisition 只验证 exact bytes 并授权
transport decode audit；它不读取 geometry label outcome，也不授权 teacher corpus 或
student training。

E0 fresh transport audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_fresh_media_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --acquisition-manifest artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801/acquisition_manifest.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport.json
```

audit 完整 decode pose/RGB/depth、核对 PTS 与 parquet 物理 timebase，但不计算
geometry labels。只有 `E0_FRESH_MEDIA_TRANSPORT_SUPPORTED` 才授权 teacher mechanics
和 role-opportunity audit。

E0 teacher mechanics + role opportunity audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_teacher_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FRESH_FOOT_GROUND_STUDENT_CANARY_E0_2026-08-01.json `
  --transport artifacts.local/evidence/hftf/stage-c-e0-fresh-transport-20260801/transport.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-fresh-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

audit 复用 hash-bound D0/D1 mechanics，输出 source/role aggregates 而不持久化完整
teacher corpus。只有 `E0_FRESH_TEACHER_AND_ROLE_OPPORTUNITY_SUPPORTED` 才授权后续
corpus generation；任何 dev/heldout opportunity failure 都不得换样。

E0.1 fresh evaluation source lock：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_1_fresh_evaluation_sources.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --pretrained-weight artifacts.local/models/hftf/torch/hub/checkpoints/mobilenet_v3_small-047dcff4.pth `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 先复核 E0 负终态与八条 consumed exclusion，再从原 inventory 重算新的
dev/heldout。它不打开新媒体；只有
`E0_1_FRESH_EVALUATION_SOURCE_LOCK_VALIDATED` 才授权 exact acquisition。

E0.1 exact fresh evaluation media acquisition：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/acquire_stage_c_e0_1_fresh_evaluation_media.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-1-source-lock-20260801/source_lock.json `
  --output-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --manifest artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801/acquisition_manifest.json
```

acquisition allow-list 只含新 dev/heldout 与公共 metadata。首次打开后两条永久 burned；
仍不计算 label 或 student。

E0.1 fresh transport audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_1_fresh_transport.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --acquisition-manifest artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801/acquisition_manifest.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/transport.json
```

transport 只完整 decode 与核对 timebase；通过后只授权 `.4 s` teacher opportunity，
不重开 `.8 s`。

E0.1 `.4 s` fresh teacher/opportunity audit：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/audit_stage_c_e0_1_teacher_opportunity.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_FOOT_GROUND_STUDENT_CANARY_E0_1_2026-08-01.json `
  --transport artifacts.local/evidence/hftf/stage-c-e0-1-fresh-transport-20260801/transport.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-1-fresh-evaluation-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/teacher_opportunity.json
```

runner 只解码 anchor 与 `anchor+2` teacher depth，报告中明确
`zero_point_eight_second_output_computed=false`。新 dev/heldout mechanics 与 opportunity
全过后才授权 corpus/training。

E0.2 fixed multi-source batch lock：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/lock_stage_c_e0_2_fixed_batch.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json `
  --output artifacts.local/evidence/hftf/<run-id>/source_lock.json
```

validator 同时排除 consumed trajectory 与 recording date，复算唯一固定的 3 dev +
3 heldout；只有 `E0_2_FIXED_BATCH_SOURCE_LOCK_VALIDATED` 才允许获取该 batch。

E0.2 fixed-batch ordered qualification：

```powershell
E:\codex-tools\bin\blindassist-python.cmd `
  scripts/research/hftf/run_stage_c_e0_2_fixed_batch_qualification.py `
  --protocol docs/research/hftf/HFTF_STAGE_C_MULTI_SOURCE_EVALUATION_QUALIFICATION_E0_2_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-e0-2-source-lock-20260801/source_lock.json `
  --media-root artifacts.local/evidence/hftf/stage-c-e0-2-fixed-batch-media-20260801 `
  --output artifacts.local/evidence/hftf/<run-id>/qualification.json
```

runner 内部仍严格按 acquisition → transport → `.4 s` teacher → role opportunity
顺序执行；前门失败不运行后门。它不计算 `.8 s` 或 student。固定 batch 无 successor
expansion。

F0.1 SANPO official-test heldout one-shot 必须按下列顺序、canonical 路径逐步执行；
任一步失败即停止，尤其 consumption ledger、truth-join receipt 或 terminal-validation
receipt 出现后不得重跑对应阶段：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/materialize_stage_c_f0_1_heldout_package.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --f0 docs/research/hftf/HFTF_STAGE_C_SANPO_BODY_HEAD_TEMPORAL_STUDENT_CANARY_F0_2026-08-01.json `
  --mechanics docs/research/hftf/HFTF_STAGE_B_SWEPT_ENVELOPE_LABEL_MECHANICS_CANARY_D0_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --authority-cohort artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-authority-cohort-20260801/authority_cohort.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --training-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-student-training-validation-20260801/validation.json `
  --datasets-root artifacts.local/evidence/datasets `
  --authority-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-authority-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_f0_1_heldout_package.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --datasets-root artifacts.local/evidence/datasets `
  --package-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/predict_stage_c_f0_1_heldout.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --inference-inputs artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/inference_inputs.jsonl `
  --source-lock artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-source-lock-20260801/source_lock.json `
  --opportunity artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-teacher-opportunity-20260801/teacher_opportunity.json `
  --datasets-root artifacts.local/evidence/datasets `
  --checkpoints-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-student-training-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/evaluate_stage_c_f0_1_heldout.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --truth artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/heldout_truth.jsonl `
  --prediction-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801 `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-effect-result-20260801

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_f0_1_heldout_result.py `
  --contract docs/research/hftf/HFTF_STAGE_C_SANPO_HELDOUT_EXECUTION_CONTRACT_F0_1_2026-08-01.json `
  --package-validation artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-validation-20260801/validation.json `
  --truth artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-package-20260801/heldout_truth.jsonl `
  --prediction-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-predictions-20260801 `
  --result artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-effect-result-20260801/result.json `
  --output-root artifacts.local/evidence/hftf/stage-c-f0-1-sanpo-heldout-terminal-validation-20260801
```

T0 short-path transport 只允许在合同固定的 outcome-open Development source 上执行。
合同及实现必须先提交推送并确认远端一致；不得用该 CLI 打开 fresh/reserved source：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_t0_sanpo_short_path_transport.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json `
  --transport-root artifacts.local/evidence/hftf/t0-short-path-transport-20260801 `
  --session-id 12b65d2c76d7ad0c17d7ac791089b8cae0bb059c9b02a6f23129044192bc93bb `
  --official-split train --start-frame 0 --target-fps 10 --frame-count 25 `
  --report-output artifacts.local/evidence/hftf/stage-c-t0-short-path-acquisition-20260801/acquisition.json

E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/validate_stage_c_t0_sanpo_short_path_equivalence.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_T0_CONSUMED_DEVELOPMENT_SHORT_PATH_TRANSPORT_CONTRACT_2026-08-01.json `
  --candidate-root artifacts.local/evidence/hftf/t0-short-path-transport-20260801/r/50bce40f5469ad75 `
  --output artifacts.local/evidence/hftf/stage-c-t0-short-path-equivalence-20260801/equivalence.json
```

acquirer 在首个网络请求前验证 exact contract/source/root/config、自身 hash、G0
outcome-open role 与 canonical consumed package。validator 完全离线，逐帧验证 remote
object identity、本地 SHA/MD5、metadata、transport receipt 以及 final/`.tmp` 路径
`<240`。candidate manifest/spec hash 是 post-open transport receipt，不允许在合同中
预填。失败不重跑、不补 partial、不换源。

D2 metadata qualification 只读取 generation/SHA 绑定的 official-train split、
candidate `description.json` 与对象 receipts/listings；不读取 RGB/mask/depth bytes，
也不读取 `camera_poses.csv` 内容。合同、planner 与 planner test 必须先提交推送，且
CLI 在首个网络请求前验证三者 tracked、clean、hash-bound，并确认
`HEAD == origin/master`：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/plan_stage_c_d2_official_train_metadata.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_OFFICIAL_TRAIN_METADATA_QUALIFICATION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --output artifacts.local/evidence/hftf/stage-c-d2-official-train-metadata-qualification-20260802/qualification.json
```

planner 固定排除 78 个 burned/consumed/closed/reserved parents，按 official-train
`session_id` 升序选择前 6 个 metadata-eligible 新 parents。candidate 级请求经三次
内部 retry 后仍 404 或 metadata 不合法时写入 ineligible ledger 并继续；完整 split
不足 6 个即 `STOP_NO_ELIGIBLE_NEW_DEVELOPMENT_COHORT`。扫描只执行一次，不追加或
替换 sources。合同、planner 与 planner test 必须来自同一 clean remote HEAD，
`--retries` 必须精确为 3；CLI 在首个网络请求前写入不可覆盖的 durable attempt
marker，失败或中断后也不允许重扫。成功只允许冻结下一份 media/mechanics 合同，
不直接授权媒体、pose 内容、teacher、student 或 D2 mechanics。

2026-08-02 的唯一 metadata scan 已以
`D2_OFFICIAL_TRAIN_METADATA_COHORT_QUALIFIED` 锁定 6 条升序 official-train
parents。durable qualification SHA-256 为
`63a217c3e658bbe4fee9e351c5c9abf68379ec2ccb89a6c3449f1581e385ee47`；
独立审计重算 13 项 bindings、900 个媒体对象 receipts 与 18 个 modality receipt
hashes 后 `CLEAR`。这些 source 只在 metadata 层被打开并锁定；媒体与 pose 内容仍未
读取。不得重扫、追加或替换，下一步必须先冻结另一个 hash-bound one-shot
media/mechanics contract。

D2 mechanics 实现还必须绑定 D2.1 definition clarification。exact G0 不允许预先
过滤全局 theta/distance domain 外点；全部 admitted obstacle points 都对每个 cell
产生 signed proxy，nonmember 以正 closed-box SDF 参与 second-smallest。ground-aligned
rotation 使用 history/current forward 在 current ground tangent plane 上的最短有符号
角，并以 Rodrigues 绕 current up 延拓；predicted right 固定为
`cross(predicted_forward,current_up)`。每个 anchor 只读自身 history/current inputs，
其 0.4/0.8 s records 必须在处理后续 anchor 前 durable 写入。D2.1 JSON SHA-256 为
`51ed1c0bc2a98481b4991f237d44979cf0c455624031c2c0ee41715ec0d6a8f0`。

D2 六源媒体获取合同只物化 metadata scan 已锁定的 6 个 official-train Development
parents。正式 CLI 必须从 tracked、clean、pushed 的 exact contract/acquirer/test
及 SANPO network transport dependency 启动，并在首网前再次确认
`HEAD == origin/master`、固定 `--retries 3`、canonical root 不存在且 durable
attempt 可独占创建、完成 `flush + fsync`。source-blind 路径预检命令为：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_d2_six_source_media.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_SIX_SOURCE_MEDIA_ACQUISITION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3 `
  --preflight-only `
  --preflight-output artifacts.local/evidence/hftf/stage-c-d2-six-source-media-path-preflight-20260802/preflight.json
```

已封存的 preflight 覆盖 1510 个 final/staging/downloader `.tmp` 内容路径，最大长度
173；它不联网、不读取媒体，也不创建 acquisition root。正式一次性命令为：

```powershell
E:\codex-tools\bin\blindassist-python.cmd scripts/research/hftf/acquire_stage_c_d2_six_source_media.py `
  --execution-contract docs/research/hftf/HFTF_STAGE_C_D2_SIX_SOURCE_MEDIA_ACQUISITION_EXECUTION_CONTRACT_2026-08-02.json `
  --retries 3
```

acquirer 逐项绑定 frozen generation/size/MD5；完整 pose CSV 校验后只把 13 个 selected
rows 写成独立 hash-bound pose slices，供后续 future-blind preprocessor 按 anchor
最小读取。RGB/mask/depth 不在获取阶段解码。任何 source 失败都只产生
`D2_MEDIA_ACQUISITION_NOT_EVALUABLE_NO_RETRY_NO_SOURCE_REPLACEMENT`，不得重跑、换源、
追加或 partial fill；成功也只允许另冻 mechanics execution contract，不直接授权
preprocessor、future truth、effect 或 student。

唯一一次正式获取已达到
`D2_SIX_SOURCE_SHORT_PATH_MEDIA_COHORT_ACQUIRED`：254/254 下载请求均在 attempt 1
成功，6/6 source 原子发布。独立离线复算闭合 378 个 files、234 个媒体对象、
6 个 pose CSV 与 78 个 pose slices；媒体只做 hash/size/MD5，未解码，future truth
未打开。per-frame acquisition index SHA-256 为
`60e63e2df8b2813519e90a287b841dbcfa2b2c9a9b0765b1f10ebcf7c9c8b2a8`。
下一步只能先冻结并推送 mechanics execution contract，再运行 future-blind
preprocessor；不得直接打开 future truth。

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
