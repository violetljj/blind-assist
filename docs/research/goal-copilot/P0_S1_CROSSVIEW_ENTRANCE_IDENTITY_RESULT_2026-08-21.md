# P0-S1 Crossview Entrance Identity result

状态：`COMPLETE / P0_S1_IDENTITY_RULE_TOO_WEAK / 0_STRONG_IDENTITY / NO_S0_RERUN / NO_SCIENTIFIC_VERDICT`

协议与冻结 rule：[`P0_S1_CROSSVIEW_ENTRANCE_IDENTITY.md`](P0_S1_CROSSVIEW_ENTRANCE_IDENTITY.md) /
[`p0_s1_crossview_identity_config.json`](p0_s1_crossview_identity_config.json)

## Verdict

已消费的 20-image P0-S0 canary 通过独立 P0-S1 identity gate replay，终态为：

> `P0_S1_IDENTITY_RULE_TOO_WEAK`

这表示 identity mechanics 可运行并且成功阻止原 nominal `SILVER_A_PRIMARY`，但当前 evidence 中没有一对
candidate 满足保守的 same-sequence strong identity rule。原 materializer 的 `1` 个 nominal primary 保持原 receipt；
P0-S1 接受的 primary 为 `0`，不原样重跑 S0-R1。

## Pair accounting

唯一 nominal record 有 4 个 `WALL_ASSOCIATED` candidates，共 6 个 pair：

- 3 个 same-sequence pairs：crop appearance similarity `0.963908–0.984823`，shape/scale/local-wall position 均相容；
  但 camera baseline 只有 `0.004270–0.011859 m`，ray angle 只有 `0.079955–0.160994°`。它们是近乎同一相机位姿的
  重复观察，不是满足 `3–30 m / 10–120°` 的 multiview parallax，因此全部为
  `SAME_SEQUENCE_IDENTITY_NOT_ESTABLISHED`。
- 3 个 cross-sequence pairs：camera baseline `15.805787–15.810028 m`、ray angle
  `18.935955–19.096949°`、wall-position delta `0.145955–0.203642 m`，说明 wall-level projection 看似一致；
  但 capture gap 约 `154,487,675–154,487,680 s`，appearance similarity 仅 `0.379476–0.405762`，bbox aspect 与
  physical-height proxy 也不相容。全部为 `CROSS_SEQUENCE_APPEARANCE_INCONSISTENT_SUPPORT`，且 V1 规则无论如何
  都禁止跨 sequence 单独建立 strong identity。

这个分解验证了 P0-S1 的核心边界：

> `same wall association != same physical entrance identity`

## What was and was not established

已建立：

- Grounding DINO、prompt、threshold、NMS 与图像集均未改变；
- frozen P0-S0 materializer 未修改；
- same-sequence、cross-sequence、wall association 与 entrance identity 已机器可执行地分离；
- manual visual audit disposition 未进入 P0-S1 输入，forbidden-input audit 为 0；
- synthetic/fail-closed tests 与真实 report replay 均确定性通过。

未建立：

- 当前 canary 中没有同 sequence 且具有真实 `3–30 m` parallax 的 map-anchored pair；
- 没有 `ENTRANCE_IDENTITY_ESTABLISHED`；
- 没有科学上可接受的 `SILVER_A_PRIMARY`；
- 不证明 same-sequence rule 已在 fresh real pair 上成功，也不授权 baseline、模型比较、Android、导航或安全主张。

## Evidence and next boundary

Ignored evidence:
`artifacts.local/evidence/p0-s0/2026-08-21-grounding-dino-tiny-s0-r1-anchor-aware/p0-s1-crossview-identity-result.json`.

- config SHA-256: `b26c9ebefb7512ba4a0abb09cc8435e081bab212b3b3c569675a72ff5f46b369`;
- input bundle SHA-256: `085c06b78278192e3a2f8aa4c88dd73c5b9067965bf2fddfb8739d0fc8b22665`;
- parent materialization report SHA-256: `9efb129d51934410f84ef2f765fcea1d5e1c05b14dd9627ce191b75ad773e20d`;
- implementation SHA-256: `eef2c218292de784da1af88ea941f5363c7b2d056a0e62835fb4c10b176ce722`;
- P0-S1 report SHA-256: `53670f1175485d4d966d71f245aef17e638083229f35110b209e546284c8df65`.

当前 blocker 已进一步压缩为：在同一 Mapillary source 内取得或筛出 target-anchor-facing、同 sequence、不同相机位置且
每个 view 都独立通过 map/geometry 的 candidate pair。不得通过降低 3 m baseline、10° ray angle、appearance gate，
或把跨 sequence evidence 升级来救当前 consumed record。
