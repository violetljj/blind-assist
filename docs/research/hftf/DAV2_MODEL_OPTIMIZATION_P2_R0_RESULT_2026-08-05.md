# DA V2 模型侧优化 P2 R0 结果

日期：2026-08-05

终态：`A1_RESOLUTION_ONLY_NOT_SUPPORTED / A2_DISTILLED_STATIC_QUALITY_SIGNAL_TEMPORAL_AND_STATE_GATE_FAIL`

## 结论

P1 准确率/false-clear 门建立后，P2 严格按冻结顺序执行了两个 `392` arms：

- A1 只降分辨率，失败 7/11 工程非劣化门；
- A2 用独立 training RGB 做固定 teacher-only 蒸馏，修复了 metric depth、collision agreement
  和 false-clear，但仍失败 temporal delta 与两项几何状态门。

两个 arm 都不进入 Android 转换或性能 profile。Host CUDA cache latency 只证明候选可执行，
不能用来覆盖质量失败。A2 的正信号应保留为下一代轻量 student 的训练机制依据，但不能将其
重标为通过。

## A1：392 resolution-only control

| 指标 | canonical 518 | A1 392 | 结果 |
|---|---:|---:|---|
| raw metric AbsRel median | 29.43% | 52.68% | 明显恶化 |
| scale-aligned AbsRel median | 8.33% | 8.70% | 通过相对结构门 |
| clearance MAE | 0.380 m | 0.698 m | 失败 |
| collision agreement | 75.27% | 65.12% | 失败 |
| false-clear | 24.25% | 34.76% | 失败 |
| temporal delta MAE | 0.113 m | 0.131 m | 超过非劣化增量门 |
| geometry state exact | 100% self | 29.17% | 85/120 帧改变 |
| transition change agreement | 100% self | 60.34% | 失败 |

cache SHA-256：`3B588771EAB851710AC013A4A154BB9412153035738E2369230C9F2FFA8921FF`。
A1 说明较低 token 数仍保留相对结构，但 metric scale 与下游几何不能靠直接缩图维持。

## A2：392 teacher-only distilled student

A2 使用 2,400/600 张 ARKitScenes RGB，按 video/parent 分离。518 FP16 teacher cache 在
训练前物化，未打开 sensor depth、affine target 或 P1 truth。固定三轮 validation total loss
为 `0.2904 -> 0.2258 -> 0.2002`，按合同选择第 3 轮；checkpoint SHA-256：
`8464C67D33010EF0F3225B0CDF9ACFFB2C6581150C3B2F42F45C73522A6CC0E9`。

| 指标 | canonical 518 | A2 distilled 392 | 冻结门 |
|---|---:|---:|---|
| raw metric AbsRel median | 29.43% | 10.53% | 通过，绝对改善 |
| scale-aligned AbsRel median | 8.33% | 7.73% | 通过 |
| ground recovery on sensor-valid | 100% | 99.16% | 通过 |
| clearance MAE | 0.380 m | 0.343 m | 非劣化通过，绝对 `0.25 m` 门仍失败 |
| collision agreement | 75.27% | 83.90% | 非劣化通过，绝对 90% 门仍失败 |
| false-clear / all known | 24.25% | 2.86% | 非劣化与绝对 5% 门均通过 |
| false-clear / truth occupied | 62.46% | 8.24% | 强正信号，诊断分母 |
| temporal delta MAE | 0.113 m | 0.221 m | 失败 |
| status exact vs canonical | 100% self | 99.17% | 通过 |
| geometry state exact vs canonical | 100% self | 0.83% | 119/120 帧改变，失败 |
| transition change agreement | 100% self | 50.00% | 失败 |

A2 通过 8/11 工程门，绝对 task 门通过 2/5。虽然其 false-clear 改善很大，但冻结协议是
AND gate，不能用静态准确率补偿 temporal/state 失败。`MODEL_VARIANT_ENGINEERING_NONINFERIORITY_FAIL`
保持不变。

## 机制判断

A1 与 A2 的对比支持三个机制结论：

1. token 降低本身主要破坏 metric scale，不一定破坏 scale-aligned 相对结构；
2. teacher-only metric/log-scale distillation 可以强力恢复尺度和 false-clear；
3. 单帧蒸馏会重构 occupancy state，并未保住连续帧 clearance dynamics，下一学生不能只优化
   per-frame depth loss。

这不授权在 P1 cohort 上调 A2 loss、epoch、seed 或 threshold。若继续 A3，必须在打开任何
A3 P1 输出前另行冻结：轻量 backbone/head、训练-only 连续帧 teacher-delta supervision、一个
seed 和一个 checkpoint selection rule。A3 仍需一次性通过同一个 P1 gate。

## 多速率架构边界

`student 8–12 Hz + teacher 1–3 Hz + YOLO/segmentation 15–30 Hz` 仍是合理目标，但本轮只
建立了 teacher distillation 的离线正信号，没有建立 student 实时或多速率系统效果。
teacher 可做周期校正、disagreement 与 confidence supervision；若没有独立 ToF/量距绑定，
同源 DA V2 teacher 不能被称为独立 metric scale anchor。stale、disagreement 或无独立尺度时
仍必须 `UNKNOWN`。

## 证据

- A1 machine result SHA-256：`42CDCE895CF9A5417EEA0193B066F2B7715DDD3D64CD6E8FCA209F3857523F4B`；
- A2 teacher manifest SHA-256：`9C0D58D6FC8342E21EABB2BA1EAADE340A483601C3E7888B8FB6C5B637D24CA9`；
- A2 training result SHA-256：`3C962F9174F596CEE4B74CEA34CD0C98BD50CBB47D5D892ACFC9BB1432ED0E89`；
- A2 P1 cache manifest SHA-256：`824A5B51AEB572606574642DC76CD85DCEE0A7B3F35A81C13FDC700D91D876E7`；
- A2 machine result SHA-256：`3ADA97149130317BE20598E521268952D3BAFEDF33E966B71FCC9B18B2A8416B`。
