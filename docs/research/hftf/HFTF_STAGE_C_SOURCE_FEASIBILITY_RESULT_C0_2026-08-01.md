# HFTF Stage C source feasibility result C0

日期：2026-08-01

终态：`C0_EGOWALK_MEDIA_TRANSPORT_NOT_EVALUABLE`

## 1. 结论

C0 的 metadata selection、文件绑定、完整解码、帧序对齐和 natural depth support
均通过，但冻结的“容器 reported rate 必须为 5 Hz”假设失败。两个选中 trajectory 的
RGB 与 depth container 都声明 `100 Hz`，而 dataset `info.json` 与 parquet timestamp
间隔给出 `5 Hz / 约 200 ms`。

因此原 C0 按顺序门关闭为 media transport `NOT_EVALUABLE`。不能把其余通过项越过该门
包装成 C0 success，也不能覆盖报告、换 trajectory 或静默删除 rate gate。

## 2. 报告绑定

- inventory lock R1：
  `artifacts.local/evidence/hftf/stage-c-c0-egowalk-inventory-lock-r1-20260801/inventory.json`
- inventory SHA-256：
  `98a99f07089e3497d533c29b788f236c4752ef730cf47b49a084b6d8a177f90a`
- transport audit：
  `artifacts.local/evidence/hftf/stage-c-c0-egowalk-transport-audit-20260801/transport_audit.json`
- audit SHA-256：
  `3dafbef91d09f13f63826d6f004be28da9d9af1ad8a680a5df83f26ad7887057`

正式实现先于 media acquisition 推送为 `e6d2c7b`。

## 3. 通过项与 blocker

| trajectory | pose/RGB/depth rows | local/LFS SHA | RGB/depth rate | depth support |
| --- | ---: | --- | --- | --- |
| `2024_08_15__19_45_11` | `647/647/647` | all pass | `100/100 Hz` | `32/32`, adjacent `31/31` |
| `2024_07_11__12_33_57` | `664/664/664` | all pass | `100/100 Hz` | `32/32`, adjacent `31/31` |

两条 source 的 RGB/depth 都是单 stream、frame count 精确等于 pose rows，PTS 严格递增
且 constant-step；全部 32 个冻结 index 均可解码。RGB 为 `960x600 yuv444p`，
depth 为 `960x600 gray16le`。所有 sampled depth 的正有限比例和 bottom-half support
均过门，31 个相邻 pair 也全部达到共同支持 `.25`。

唯一失败是 container nominal rate `100 != 5`。该 rate 还会把 647 帧解释为约
`6.46 s`，与 parquet 约 `129.2 s` 的 5 Hz timeline 不一致。

## 4. successor 边界

这是 source schema/timebase assumption 的失败，不是 depth/pose/RGB 对齐或自然深度
支持的负结果。官方 EgoWalk API 以 ordinal `frame_idx` 和 container base-rate 计算
seek PTS；真实采样时间另由 parquet timestamp 与 `meta/info.json` 表示。允许的最小
successor 是 C0.1：

- 复用同一 consumed media、同一 frozen cohort 和全部原门；
- container nominal rate 只记录、不作为物理 timeline；
- 物理 timeline authority 固定为 parquet timestamp，要求约 200 ms、有效 5 Hz；
- 仍要求 RGB/depth frame count、ordinal index、PTS 单调与完整解码全部一致；
- 不修改 depth support 门，不计算 teacher label/student output。

C0.1 即使通过，也只允许冻结 Stage C label-and-student canary protocol。
