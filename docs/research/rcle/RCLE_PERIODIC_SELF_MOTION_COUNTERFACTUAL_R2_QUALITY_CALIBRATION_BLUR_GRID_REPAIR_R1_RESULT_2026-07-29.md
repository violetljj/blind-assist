# RCLE periodic self-motion counterfactual R2：P2 blur-grid repair R1

日期：2026-07-29
阶段：`DEVELOPMENT / P2_RESPONSE_BLIND_QUALITY_CALIBRATION`

## 结论

三轴终态为：

```text
QUALITY_CALIBRATION_PASS / VALID / P3_NOT_AUTHORIZED
```

一次性冻结的小 sigma grid 找到两项全局强度：

```text
blur_sigma_px = 0.475
low_texture_alpha = 0.15
```

`sigma=0.475 px` 是满足总体及全部 8 个 block×motion subgroup 门的最小候选。
其总体 Laplacian-variance ratio 为 `0.525336`，subgroup 范围为
`0.513451–0.533071`；总体 local-RMS-contrast ratio 为 `0.931832`，
subgroup 最小值为 `0.919136`。更小的 `sigma=0.45 px` 总体 ratio 为
`0.604904`，subgroup 范围为 `0.594027–0.611581`，高于冻结上界 `0.55`，
因此不能选择。

Low-texture 未重跑或重调；`alpha=0.15` 由 hash-bound R0 lock、ledger 与独立
receipt 继承。R1 只补足 blur 单轴，并形成一个全局 pair lock。

## 冻结执行

- 候选 sigma：`[0.35, 0.40, 0.425, 0.45, 0.475, 0.50, 0.55, 0.60, 0.65]`；
- 复用原 CAL panel：4 block × 4 CAL seed × static/periodic × 16 帧；
- 每帧固定 `1 clean + 9 blur`，共 `5120/5120` 行 response-blind ledger；
- 未读取或运行 RCLE estimator/output，未读取 trigger/response；
- 未换 seed、扩候选、重调 low-texture、做分 block strength 或自动开启第二次修复；
- 未运行 P3 预检、480+16 正式序列、sequence16、CoTracker、Android 或实时集成；
- 未修改 R3、阈值或三-pair。

## 最小证据

- [R1 contract](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_CONTRACT_2026-07-29.json)，
  SHA-256 `ac2c9fa9b499d60492d01d542e7401bca4058cd5c786870804a1ebfff3a845ca`；
- [global strength lock](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_GLOBAL_STRENGTH_LOCK_2026-07-29.json)，
  SHA-256 `aa24b908f71a45bb9cd6c3ffbc5ab947938c6e29726667402487ec1dd01d28dc`；
- [independent validation receipt](RCLE_PERIODIC_SELF_MOTION_COUNTERFACTUAL_R2_QUALITY_CALIBRATION_BLUR_GRID_REPAIR_R1_INDEPENDENT_VALIDATION_RECEIPT_2026-07-29.json)，
  SHA-256 `c88a631dff3393cc6c05070f761977be7543a4dab8b04cb1692b9fd34eb15b01`，
  `errors=[] / validated=true`；
- response-blind blur metric ledger：
  `artifacts.local/evidence/rcle_periodic_self_motion_counterfactual_r2/p2_quality_calibration_blur_grid_repair_r1/response_blind_blur_metric_ledger.jsonl`，
  `5120` 行，SHA-256
  `00353899cb7462829a8cb8b4dfe1e4da325f101985afb9d85ce0f430bda865d2`。

独立 validator 未导入 R1 producer、quality-intervention implementation 或 RCLE
algorithm；它复算全部逐帧 ratio、16-frame/4-seed/overall 层级、8 个 subgroup、
最小 sigma 选择、R0 alpha 继承、manifest、hash、read allowlist 与 firewall。

## 权限边界

本结果只关闭 P2 quality calibration，不等于 RCLE 性能通过。P3 仍未授权；
不得据此运行 8 条 P3 性能预检、480+16 正式序列、RGB/Android/实时集成，也不得
回写 R0 或继续细化 sigma grid。
