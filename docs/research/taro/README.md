# BlindAssist TARO

状态：`current / PARALLEL_WILD_LAB / R10_NOT_EVALUABLE_DUAL_CLASS_COVERAGE / R10_NO_PROMOTION / R11_WEAK_DISTAL_ABSTENTION_DEVELOPMENT_ONLY / R11_SOURCE_DOWNLOAD_144_OF_144_INTEGRITY_PASS_ONE_SHOT_CONSUMED / R11_INVENTORY_48_PARENT_1043_FRAME_PASS / R11_PHASE_A_ONE_SHOT_CONSUMED_PRODUCER_PASS / R11_PHASE_A_INDEPENDENT_VALIDATION_PASS / R11_PHASE_A_PIPELINE_HOLD_RELEASED / R11_SOURCE_ONLY_TOP24_ONE_SHOT_CONSUMED_PASS / R11_TOP24_INDEPENDENT_VALIDATION_PASS / R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_PASS_NON_EXECUTING / R11_FORMAL_ZIP_MEMBER_PAYLOAD_READS_ZERO / R11_SCIENTIFIC_NOT_RUN / DEFAULT_APP_UNCHANGED`

本页只维护 TARO 当前状态、权限和唯一 successor。较早完整 R0–R11 叙事保存在
[14d8ad7e 历史快照](archive/README_FULL_HISTORY_2026-08-13.md)，不能从中恢复旧权限。

## 当前主张

TARO 是独立并行 `WILD_LAB`：在声明的米制锚和冻结 factor/reducer 下，以低维 residual
gauge posterior、可观测子空间和受限相机微基线，让 body/path-specific clearance query
先于完整场景达到局部可识别。`UNKNOWN`、缺字段和不可观测方向永不转成 negative。

TARO 与 [Assistive Geometry](../assistive-geometry/README.md) 并列，不从 DepthART、Android、
HTP 或默认 App 自动继承权限。

## 当前结论

- R10 以 dual-class coverage `NOT_EVALUABLE` 收口；强正占用信号保留，但不得改 selector、
  threshold、denominator 或 gate 回救。
- R11 exact-48 source-first Phase A 与独立验证已 PASS；highres/FARO/truth/model 为 0。
- 冻结 R9 source-only 48→24 selection one-shot 与独立 validator 均 PASS；selected 24
  identities 已不可变封存。
- selected-only FARO Phase-B implementation lock 已完成：冻结 24 parents、674 frames、6,066
  queries、完整 dual-class gates、678-file atomic terminal 和 no-FARO-replay validator；19 个
  聚焦测试 PASS，但正式 FARO 尚未读取。
- 当前只授权另立 one-shot execution lock；科学状态仍为 `NOT_RUN`，默认 App 与产品权限不变。

## 当前证据入口

- [R10 terminal](TARO_O1R_R10_FRESH_CLEAR_ENRICHED_CONFIRMATION_RESULT_2026-08-12.md)
- [R11 Phase-A status](TARO_O1R_R11_FRESH_48_PARENT_SOURCE_ONLY_PHASE_A_EXECUTION_STATUS_2026-08-13.md)
- [R11 Phase-A independent validation](TARO_O1R_R11_PHASE_A_INDEPENDENT_VALIDATION_RESULT_2026-08-13.json)
- [Top-24 result](TARO_O1R_R11_FRESH_48_TO_24_SOURCE_ONLY_SELECTION_RESULT_2026-08-13.json)
- [Selected-top24 FARO Phase-B implementation lock](TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_IMPLEMENTATION_LOCK_2026-08-13.md)
- [算法路线总表](../ALGORITHM_RESEARCH_CURRENT.md) · [TARO Module](../../../scripts/research/taro/README.md)

## 唯一 successor

`TARO_O1R_R11_SELECTED_TOP24_FARO_PHASE_B_ONE_SHOT_EXECUTION_LOCK`。

只能绑定 sealed selected 24 与已冻结 implementation，正式读取 selected-only FARO；不得修改
selection、selector、candidate、threshold 或 gates，也不得读取 24 个 unselected parents 的 FARO。

## 当前允许

- 准备并验证上述 exact one-shot execution lock；
- 重放 hash-bound synthetic tests、manifest/hash validator 和只读证据复核；
- outcome-blind 审计公开来源许可、字段、接口和数据能力。

## 当前禁止

- 覆盖、resume、删除或重跑已消费的 R10/R11 producer、Phase A 或 top-24 one-shot；
- 修改 selected 24，读取 unselected FARO/highres，或用 truth/knownness 重做 selector；
- execution lock 前读取正式 FARO，或越过阶段锁训练、Android/QNN/HTP、默认 App、产品或安全结论。

## Claim ceiling

当前只证明 source-first Phase A、source-only top-24 selection 与 FARO Phase-B implementation
seam；不证明 FARO outcome、fresh dual-class confirmation、移动端可行性、产品有效性或用户安全。
