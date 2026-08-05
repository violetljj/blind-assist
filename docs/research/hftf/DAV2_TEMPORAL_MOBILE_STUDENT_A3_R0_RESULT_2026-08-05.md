# DA V2 Temporal Mobile Student A3 R0 结果

日期：2026-08-05

终态：`A3_TEMPORAL_MOBILE_STUDENT_ENGINEERING_NONINFERIORITY_NOT_SUPPORTED`

## 结论

固定的 1,271,281 参数 MobileNetV3-Small + FPN student 完成五轮 teacher-only temporal
distillation，但在 P1 上只通过 3/11 工程门。它把 false-clear 压到 `0.095%`，代价是几乎
持续判为 occupied：clearance MAE `1.118 m`、collision agreement `38.38%`、几何状态
`0/120` 与 canonical 相同。A3 不进入 Android 转换或性能 profile。

## 训练

- train/validation temporal pairs：`1937 / 417`；
- 参数量：`1,271,281`；
- validation combined loss：`0.5094 -> 0.4946 -> 0.4849 -> 0.4710 -> 0.4606`；
- 固定规则选择 epoch 5；
- checkpoint SHA-256：
  `E8C9EFA36E56933DC86528F21FCDFB879F1C271F0B3A80D66A54AE7AF2AFDB4F`；
- training 与 P1 cache 物化均记录 `truth_inputs_opened=false`。

首次执行上下文在完整 receipt 前中断，只留下 checkpoint、没有五轮 history 或
`training_result.json`。该 root 已写入 `INVALID_INTERRUPTED_RUN.md`，残余 checkpoint
`640748C7...4788303` 永不用于 P1。R1 使用完全相同合同在新 root 完成，不改模型或超参。

## P1

| 指标 | canonical | A3 | 结果 |
|---|---:|---:|---|
| raw metric AbsRel median | 29.43% | 33.94% | 失败 |
| scale-aligned AbsRel median | 8.33% | 16.17% | 失败 |
| ground recovery | 100% | 98.32% | 失败 |
| clearance MAE | 0.380 m | 1.118 m | 失败 |
| collision agreement | 75.27% | 38.38% | 失败 |
| false-clear / all known | 24.25% | 0.095% | 通过但由 over-occupied 塌缩获得 |
| temporal clearance-delta MAE | 0.113 m | 0.207 m | 失败 |
| status exact vs canonical | 100% self | 98.33% | 通过 |
| geometry state exact vs canonical | 100% self | 0% | 失败 |
| transition change agreement | 100% self | 49.14% | 失败 |

absolute consumed task 门同样只通过 paired-valid 与 false-clear 两项。低 false-clear 不能覆盖
false-block、clearance 和时序失败。

## 机制与门审计

training-only same-pixel log-depth-delta supervision没有建立 clearance dynamics；相机运动下，
相同像素不对应同一 3D 表面，这是 A3 时序监督的机制弱点。不得在已打开 P1 上调整 temporal
weight、pair gap 或 decoder 重跑 A3。

P1 R0 还暴露出一个前瞻性设计问题：`geometry_state_exact_agreement` 与
`transition_change_agreement` 都以 canonical 为真，而 canonical 自身 false-clear 为 24.25%。
这会把 truth-improving state changes 也计为失败。A1–A3 的 R0 终态保持不变；后续候选必须先
冻结 truth-referenced R1 gate，以 sensor occupancy/status/transition 判断有害变化，
canonical 差异只保留诊断，不得回溯重标 A2。

## 证据

- A3 protocol SHA-256：`17B29A14D20A0C85E9B0288A218AC77A761C6551F69B15F1AF6DF87F647ADD47`；
- training result SHA-256：`2372D987A4F1794BBF9E7CC7FAC1232E7A7D9D1772AF5233D8585AD1F7723F1F`；
- P1 cache SHA-256：`27EA1EC7DAFECFBE1FCF1CEDB72B36B689BE4C12B75EF0474916DBEE610DB93A`；
- P1 cache manifest SHA-256：`B9FC78840A0DF640F94B62B8EC9A5F2E708889DF3D01176D75C608A162A4C60F`；
- machine result SHA-256：`82DF9CAED328F0645DE6F95F8DB633A87A6323702BEE8AFFB6090FEAEC807307`。
