# RCLE periodic self-motion counterfactual R2：P2 response-blind quality calibration R0

日期：2026-07-29
阶段：`DEVELOPMENT / P2_RESPONSE_BLIND_QUALITY_CALIBRATION`

## 结论

三轴终态为：

```text
NO_GLOBAL_QUALITY_STRENGTH / VALID / HOLD_P2
```

冻结 blur grid 没有全局可行值。最小候选 `sigma=0.75 px` 的总体
Laplacian-variance ratio 仅为 `0.132784`，8 个 block×motion subgroup 为
`0.128384–0.136307`，全部低于冻结下界 `0.35`；继续增大 sigma 只会进一步下降，
因此不能锁定 blur strength，也不得扩网格回救。

Low-texture 单轴存在可行值：`alpha=0.15` 是满足总体、全部 8 个 subgroup 和解析
fixture edge-spread 门的最大 alpha。其总体 multiscale-gradient-density ratio 为
`0.413205`，subgroup 范围为 `0.386632–0.463782`；CAL 与 fixture edge-spread ratio
均约为 `1.0`。`alpha=0.30` 的总体 ratio 已为 `0.558009`，且 subgroup 最大值为
`0.603944`，故不能选择。

`alpha=0.15` 只记录为单轴可行结果；由于 blur 不可行，本 R0 没有形成可供 P3 使用的
两项全局 strength pair。

## 冻结执行

- CAL panel：4 block × 4 CAL seed × static/periodic × 16 帧；
- 每帧固定 `1 clean + 6 blur + 5 low-texture`，共 `6144/6144` 个
  response-blind image evaluation；
- 先对每个 sequence 的 16 个逐帧 ratio 取 average-rank median，再对每个
  block×motion 的 4 个 CAL seed 取 median，最后对 32 个 sequence 取 overall
  median；
- 未读取或运行 RCLE estimator/output，未按 trigger/response 调 strength，未换
  seed、扩候选、做分 block strength 或运行 P3/P4；
- 未修改 R3、`0.01/s`、三-pair、sequence16、CoTracker、Android 或实时集成。

## 最小证据

- [global strength lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_R0_STRENGTH_LOCK_2026-07-29.json)，
  SHA-256 `d04c06d544c6780e8b86e4eb32b3c181ffe3940626e1f4127fa9cbf497dd41ea`；
- [independent validation receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_R0_INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json)，
  SHA-256 `37d2f09d0e8764aee904ebc7998d2b629e1cba9dde024e739fbe4d1ea667fd06`，
  `errors=[] / validated=true`；
- response-blind metric ledger：
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p2_quality_calibration_r0/response_blind_metric_ledger.jsonl`，
  `6144` 行，SHA-256
  `0356a85d0901426a5ed7142d348863501f6d1d9f14fd1c76520c15ee2f9d0c3a`。

独立 validator 不导入 producer、quality-intervention implementation 或 RCLE
algorithm；它重算全部 ledger ratio、16-frame/4-seed/overall 层级、fixture
identity、门、选择方向、hash、read allowlist 与 firewall。专项实现与 mutation
tests 为 `23/23 PASS`。

## 后继边界

本 R0 按冻结 no-fit rule 关闭，不得扩展 blur grid、换 CAL seed、降低门或改成
per-block strength。P3 仍未授权；若未来挑战 blur grid/renderer 的 proxy
适配性，只能另立经审查的新协议版本，不能回写或重跑本 R0。
