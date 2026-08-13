# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D1_TASK_QUALITY_FAIL_TERMINAL / D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_TERMINAL / D3R1_D3R2_EXECUTION_INVALID_TERMINALS_PRESERVED / D3R3_EXACT64_CENSUS_PASS_9597_OF_9600_PAIRED / D3R4_D3R5_DIRECT_VETO_NEGATIVES_PRESERVED / D3R6_BUDGETED_UNKNOWN_DEFERRAL_FRESH_CONFIRMATION_PASS / DEVELOPMENT_CANDIDATE_ONLY / USER_PAUSED / R2_CANDIDATE_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

本页只维护 DepthART 当前摘要、权限和唯一 successor。完整旧历史在
[2026-08-07 快照](archive/README_FULL_HISTORY_2026-08-07.md)，不得据此恢复权限。

## 当前主张

DepthART-S 是 [Assistive Geometry](../assistive-geometry/README.md) 的 encoder/initialization、
depth baseline 与部署研究载体，不是算法终点。部署或单设备性能不能替代 task-quality
admission，也不等于正式 App 能力。

## 当前结论

- strict G4-D、D1、D2 负终态与 D3R1/D3R2 invalid evidence versions 保持不可改写；
  R2 candidate 未授权。
- D3R3 fresh transport recovery 完成 exact 64/64 GET 与独立 census：`5,580,879,686` bytes，
  exact 9,600 stems 中 `9,597` 个 depth+confidence paired；3 个缺失 stem 固定为
  `SOURCE_UNAVAILABLE_UNKNOWN`，未替邻帧、未改分母。
- 原 all-horizon 双向门在 32 个身份上为 `0/32`，扩到另 21 个同分布身份仍不能增加 far-CLEAR
  parent diversity。D3R4 pooled hard veto 与 D3R5 parent-relative direct veto 都显著降低 false-clear，
  但分别把 fresh false-block 推高至 `46.78%` 与 `27.92%`，两项负结果保留且不得晋级。
- D3R6 保留 D3R5 的 parent×band×horizon 相对秩风险分数，但动作改为每个 parent 最多
  `54/2700 = 2%` 个 baseline-CLEAR cell 转 `UNKNOWN`，永不输出 CLEAR/OCCUPIED。预算只在 TRAIN
  冻结；第二组 8 个此前未读模型输出的 parent 上，false-clear `31.73% -> 29.92%`，false-block
  `18.92% -> 18.92%`，coverage 恰好下降 `2.00%`，fresh confirmation PASS。
- D3R6 只锁为 Development candidate。按用户要求，本轮完成后路线暂停；不自动进入 R2、设备、
  Android 默认、产品或安全。

## 当前证据入口

- [D3R2 execution stop](DEPTHART_TASK_PRESERVING_D3R2_PHASE_B_SOURCE_COVERAGE_EXECUTION_STOP_2026-08-13.md)
- [D3R3 recovery protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-13.md) · [machine protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-13.json)
- [D3R3 source-scope receipt](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_SCOPE_RECEIPT_2026-08-13.json)
- [D3R3 fresh-HEAD activation](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_HEAD_ACTIVATION_2026-08-13.json)
- [D3R3 census protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_PROTOCOL_2026-08-13.json) · [activation](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_ACTIVATION_2026-08-13.json)
- [Missing-source UNKNOWN experiment](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_MISSING_SOURCE_UNKNOWN_EXPERIMENT_2026-08-13.json)
- [D3R4 selective router](../../../scripts/research/hftf/deployment/depthart/run_depthart_d3r4_selective_router_canary.py) · [D3R5 parent-relative veto](../../../scripts/research/hftf/deployment/depthart/analyze_depthart_d3r5_parent_relative_veto.py)
- [D3R6 budgeted deferral](../../../scripts/research/hftf/deployment/depthart/analyze_depthart_d3r6_budgeted_deferral.py) · [fresh confirmation](../../../scripts/research/hftf/deployment/depthart/confirm_depthart_d3r6_budgeted_deferral.py)
- D3R6 fresh result：`artifacts.local/evidence/hftf/depthart-d3r6-fresh-confirmation-20260813-r0/result.json`，
  `14,408 bytes / SHA-256 B089A050C729C9D9513CE0728B9F075615C3D4021CDBCF4F144765FA99855EDC`。
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [HFTF Module index](../../../scripts/research/hftf/INDEX.md)

## 唯一 successor

无。状态为 `NONE / USER_PAUSED_AFTER_D3R6_FRESH_CONFIRMATION_PASS`。

未来只有用户明确恢复后，才可为 D3R6 建立新的 held-out Confirmation 或任务交互实验；当前不执行
训练、调参、R2、设备或产品 successor。

## 禁止与权限边界

- 不复用 D3R2 44 bodies，不改写 D3R4/D3R5 negative，也不把 missing source 当 negative；
- 不在已打开的两组 fresh parent 上重训、改 2% budget、改 checkpoint 或改 PASS 门；
- 不读取 sealed R2 outcome，不把 Development/fresh-parent confirmation 写成正式 R2、默认 App、
  产品或安全证明。

## Claim ceiling

当前证明：D3R6 的 parent-relative risk ranking 加固定 per-parent UNKNOWN budget，在 TRAIN-frozen
规则下通过一组 8-parent fresh confirmation，并把 coverage/false-block 代价结构性封顶。它是可信的
Development candidate 和负结果驱动的新机制，不证明 sealed R2、跨数据域、部署、默认 App 或用户安全。
