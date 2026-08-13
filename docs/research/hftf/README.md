# DepthART 算法路线

状态：`current / DEVELOPMENT_STANDARD / INNOVATION_NOT_EVALUABLE / R1_RESEARCH_MAINLINE / STRICT_G4D_NEGATIVE_TERMINAL / D1_TASK_QUALITY_FAIL_TERMINAL / D2_DEVELOPMENT_FROZEN_HEAD_QUALITY_FAIL_TERMINAL / D3R1_PHASE_B_EXECUTION_INVALID_INCOMPLETE_NO_SCIENTIFIC_TERMINAL / D3R2_PHASE_B_COVERAGE_CENSUS_EXECUTION_INVALID_INCOMPLETE / D3R3_FRESH_HEAD_64_OF_64_PASS_ZERO_HEADER_DRIFT / D3R3_EXACT64_CENSUS_ACTIVATED / R2_CANDIDATE_NOT_AUTHORIZED / DEFAULT_APP_UNCHANGED`

本页只维护 DepthART 当前摘要、权限和唯一 successor。完整旧历史在
[2026-08-07 快照](archive/README_FULL_HISTORY_2026-08-07.md)，不得据此恢复权限。

## 当前主张

DepthART-S 是 [Assistive Geometry](../assistive-geometry/README.md) 的 encoder/initialization、
depth baseline 与部署研究载体，不是算法终点。部署或单设备性能不能替代 task-quality
admission，也不等于正式 App 能力。

## 当前结论

- strict G4-D、D1、D2 负终态与 D3R1/D3R2 invalid evidence versions 保持不可改写；
  R2 candidate 未授权。
- D3R2 exact-64 census 在第 45 个资产因正文长度不匹配 terminal；44 个 partial bodies 不可复用。
- D3R3 是独立 transport-recovery version，不继承 D3R2 bodies、checkpoint 或 execution authority。
  fresh HEAD 64/64 PASS，声明总正文 `5,580,879,686` bytes，Content-Length/ETag/Last-Modified
  相对冻结 snapshot 零漂移，redirect/body read 为 0。
- 当前 fresh-root exact-64 census 已激活。只有 HTTP 200、headers 匹配且 premature EOF 时可
  删除 partial、从 byte 0 完整重试，最多 3 次；禁止 Range、member payload、pixel、truth/selection。
- 即使 coverage 完成，也不自动授权模型、训练、Development、R2、Android 默认、产品或安全。

## 当前证据入口

- [D3R2 execution stop](DEPTHART_TASK_PRESERVING_D3R2_PHASE_B_SOURCE_COVERAGE_EXECUTION_STOP_2026-08-13.md)
- [D3R3 recovery protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-13.md) · [machine protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_RECOVERY_PROTOCOL_2026-08-13.json)
- [D3R3 source-scope receipt](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_SCOPE_RECEIPT_2026-08-13.json)
- [D3R3 fresh-HEAD activation](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_HEAD_ACTIVATION_2026-08-13.json)
- [D3R3 census protocol](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_PROTOCOL_2026-08-13.json) · [activation](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_SOURCE_COVERAGE_CENSUS_ACTIVATION_2026-08-13.json)
- [Missing-source UNKNOWN experiment](DEPTHART_TASK_PRESERVING_D3R3_PHASE_B_MISSING_SOURCE_UNKNOWN_EXPERIMENT_2026-08-13.json)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [HFTF Module index](../../../scripts/research/hftf/INDEX.md)

## 唯一 successor

`RUN_D3R3_EXACT64_SOURCE_MEMBER_COVERAGE_CENSUS`。

使用 fresh D3R3 root 对 64 个资产全量重新 GET；只有 matching-header premature EOF 可从
byte 0 重试。完整 64 项与独立 validation PASS 后，才进入 missing-source policy registration。

## 禁止与权限边界

- 不复用 D3R2 44 bodies，不改变 exact32/9600 stems/64 URLs；
- 不使用 Range/redirect，不读取 member payload、pixel 或 truth，不运行 selection；
- 不把 partial coverage、synthetic、HTP/Android 或性能写成 accuracy、默认 App、产品或安全证明。

## Claim ceiling

当前只证明 D3R3 transport-recovery 前门与 fresh HEAD；不证明完整 coverage、候选质量、
scientific admission、部署晋级或用户安全。
