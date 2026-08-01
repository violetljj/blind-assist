# HFTF Stage C source feasibility C0.1

日期：2026-08-01

状态：`FROZEN_BEFORE_C0_1_SCHEMA_REPAIR_REPLAY`

## 1. 为什么允许 successor

C0 已不可变地关闭为 `C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE`。失败不是文件、
frame count、ordinal PTS 或 depth support，而是协议错误地要求 video container
nominal rate 等于真实采样率 5 Hz。

两个 trajectory 的 container 都声明 100 Hz；dataset `meta/info.json` 为 5 Hz，
parquet timestamp 每行约 200 ms。官方 API 也以 ordinal frame index 结合 container
base-rate/time-base 定位 PTS，而不是用 container 播放时长替代 parquet capture time。

C0.1 是 schema/timebase 修复，不是同一 outcome 上降效果门。

## 2. 冻结 replay

只允许复用：

- `2024_08_15__19_45_11`
- `2024_07_11__12_33_57`
- inventory SHA
  `98a99f07089e3497d533c29b788f236c4752ef730cf47b49a084b6d8a177f90a`
- C0 audit SHA
  `3dafbef91d09f13f63826d6f004be28da9d9af1ad8a680a5df83f26ad7887057`

不准替换或新增 media。C0 的 file SHA、完整 decode、frame count、ordinal PTS、
32-frame depth support、UNKNOWN firewall 和全部权限边界保持不变。

## 3. 唯一修复

container nominal rate 必须记录，但不定义物理 timeline。物理时间权威固定为：

1. parquet `frame=0..n-1`；
2. parquet timestamp 严格递增，中位 `195–205 ms`、每步 `150–250 ms`；
3. `meta/info.json fps=5`；
4. RGB/depth decoded frame count 等于 parquet rows；
5. RGB/depth 各自 PTS 严格递增且 constant-step，冻结 32 个 ordinal index 全部可解码。

这允许 container 在内部用 100 Hz timebase 编码 frame ordinal，但不允许用它把
129 秒 source 错解释成 6.5 秒。

## 4. 终态与权限

顺序终态：

1. `C0_1_FROZEN_REPLAY_BINDING_NOT_EVALUABLE`
2. `C0_1_FRAME_INDEX_TIMEBASE_REPAIR_NOT_EVALUABLE`
3. `C0_1_NATURAL_SURFACE_OBSERVABILITY_NOT_EVALUABLE`
4. `C0_1_STAGE_C_SOURCE_TRANSPORT_FEASIBILITY_SUPPORTED`

成功仍只授权冻结 Stage C label-and-student canary protocol；不授权 label execution、
student training/effect、主线、Android/App 或安全/产品 claim。

机器可读真源：
[HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.json](HFTF_STAGE_C_SOURCE_FEASIBILITY_C0_1_2026-08-01.json)
