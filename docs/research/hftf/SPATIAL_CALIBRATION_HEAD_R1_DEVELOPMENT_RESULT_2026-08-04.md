# Spatial Calibration Head R1 development result

## 结论

Spatial Calibration Head R1 在 fresh、visit-disjoint ARKitScenes 开发评价中失败，纯 RGB
尺度校准扩展在此停止；不打开 sealed 米制真值，不进入手机 shadow mode。按预冻结规则，
下一路线切换为多区 ToF E 臂。

机器可读摘要见
[`SPATIAL_CALIBRATION_HEAD_R1_DEVELOPMENT_RESULT_2026-08-04.json`](SPATIAL_CALIBRATION_HEAD_R1_DEVELOPMENT_RESULT_2026-08-04.json)，
原始完整结果位于 ignored
`artifacts.local/evidence/hftf/spatial-calibration-head-r1-development-run-20260804/result.json`
（SHA-256 `A2EE8A5709A012296F7924AEAA85EA6CB1893C88BB6DE7279CB72676813DA326`）。

## 数据与防火墙

- 24 个互不相交 `visit_id`：16 train、4 validation、4 sealed。
- 开发缓存 3,000 帧；sealed 只打开 600 帧 RGB 做身份审计。
- 3,600 帧 SHA/pHash 审计产生 0 条跨 parent 候选边。
- sealed depth、confidence、intrinsics、trajectory 与米制结果均未打开。
- 已消费 TUM 未进入该 cohort。

## 固定 validation

| 臂 | coverage | MAE (m) | agreement | false-clear | temporal MAE (m) | ECE |
|---|---:|---:|---:|---:|---:|---:|
| Raw DA V2 | 78.65% | 0.460 | 77.53% | 18.14% | 0.408 | — |
| 常数校准 | 78.65% | 0.396 | 81.61% | 11.89% | 0.387 | — |
| global CLS 770 | 75.27% | 0.439 | 79.26% | 13.98% | 0.432 | — |
| spatial 9,423 | 78.55% | 0.573 | 73.85% | 19.45% | 0.469 | 0.601 |

空间头的六项任务/置信门全部失败，并且 MAE 与 false-clear 都劣于常数校准。
四个 CV fold 的联合优胜数为 `0/4`，低于 `3/4` 晋级门。该负结论不能通过调
threshold、loss、feature layer、seed 或重新划分 session 救援。

## 决策

- `STOP`：继续扩展纯 RGB 空间尺度校准。
- `NOT_AUTHORIZED`：sealed 米制评价和手机 HTP shadow。
- `ACTIVATE`：同摄像头、同 session 的多区 ToF E 臂；硬件选择沿用既有
  `VL53L8CX_DEFAULT_CANARY_VL53L5CX_AVAILABILITY_FALLBACK` 权威。
