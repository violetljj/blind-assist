# P0-S1 Crossview Entrance Identity

状态：`FROZEN_DEVELOPMENT_SLICE / SAME_SEQUENCE_STRONG_ONLY / CROSS_SEQUENCE_SUPPORT_ONLY`

## Question

对于两个或多个已通过 proposal、building、wall 与 map-anchor mechanics 的 entrance candidates，能否保守地建立：

> 它们是同一个物理入口，而不只是同一建筑、同一 facade 或同一段墙附近的不同门？

P0-S1 不更换 detector，不修改 Grounding DINO revision、prompt、threshold 或 NMS，不扩数据源，也不修改
冻结的 P0-S0 materializer。它是 materializer 之后的独立身份门；现有机器 receipt 与 nominal verdict 保持原样。

## Evidence levels

1. `BUILDING_ASSOCIATED`：ray 与一个 admitted building footprint 相交；不证明 facade、wall 或 entrance identity。
2. `WALL_ASSOCIATED`：candidate 的 ray-wall hit、range、map anchor 与 ambiguity gate 有效；不证明 same entrance。
3. `CROSS_SEQUENCE_SUPPORT_ONLY`：跨 sequence candidates 可记录相容或冲突线索，但 V1 永远不能单独建立
   `ENTRANCE_IDENTITY_ESTABLISHED` 或支撑 `SILVER_A_PRIMARY`。
4. `ENTRANCE_IDENTITY_ESTABLISHED`：仅当同 sequence pair 同时通过全部冻结 identity gates。

因此必须写死：

> `same building wall != same physical entrance`

## Strong same-sequence rule

同 sequence pair 必须同时满足：

- 不同 source image，capture gap `<=30 s`；
- camera baseline `3–30 m`；
- same admitted building 与 exact same OSM entrance anchor；
- 两个独立 ray-wall positions 相距 `<=1 m`；
- ray angle difference `10–120°`；
- bbox aspect-ratio ratio `<=1.5`；
- distance/focal-normalized physical-height proxy ratio `<=1.75`；
- crop RGB histogram intersection + grayscale dHash similarity `>=0.65`。

任一失败即不建立 identity。跨 sequence appearance 不一致可记录为 conflict diagnostic，但不能用相似度把跨 sequence
升级为 strong identity。

## Authority and leakage boundary

- P0-S1 只读取冻结 proposal receipt、Mapillary image/pose metadata、候选 crop 与 map/geometry outputs。
- 不读取人工 visual audit disposition、Silver label、evaluator truth 或目标答案坐标作为 identity 输入。
- 人工目测只用于结果后审计，不能写回 candidate truth。
- P0-S1 建立的 identity 只解除 multiview identity gate；其余 P0/Silver gates 仍由冻结 materializer 独立满足。

## Verdicts

- `P0_S1_SAME_SEQUENCE_IDENTITY_ESTABLISHED`：至少一个 nominal primary record 有一个通过全部强门的同 sequence pair；
- `P0_S1_IDENTITY_RULE_TOO_WEAK`：mechanics 可运行，但没有 record 获得强 identity；
- `P0_S1_SCHEMA_INADEQUACY`：必要 image、pose、timestamp、bbox、geometry 或 provenance 字段缺失。

当前 20-image canary 已被 S0 结果与人工审计消费。P0-S1 首次 replay 只能是 Development mechanics evidence，
不得称 fresh confirmation。只有 P0-S1 通过后，才允许在保持视觉配置不变的前提下原样重跑 S0-R1。
