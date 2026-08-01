# HFTF Stage C G0-D1 timeline amendment

日期：2026-08-01

状态：`FROZEN_BEFORE_D1_DEVELOPMENT_CORPUS_OR_STUDENT_OUTCOME`

原 D1 scientific design 把 target timeline 简写为
`ALL_25_CURRENT_10FPS_FRAMES`。执行前核对冻结 G0 source plan 发现，9 个
Development 来源中 7 个 target FPS 为 10，另 2 个 target FPS 为 5：

- `1405e451...`
- `1480ad06...`

因此执行语义修订为：

`ALL_25_CURRENT_FRAMES_AT_EACH_SOURCE_PLAN_FROZEN_TARGET_FPS`

每源仍只使用 source plan 已冻结的 25 个 current frames；frame step 必须等于
`round(source_fps / target_fps)`，首帧为 0。此修订只消除时间线文字歧义，不改变
source identity/role、student input、模型、target 几何、loss、checkpoint selection、
success gates 或 fresh firewall。修订完全基于已存在的 metadata，在任何 D1 corpus
或 student outcome 打开前冻结。
