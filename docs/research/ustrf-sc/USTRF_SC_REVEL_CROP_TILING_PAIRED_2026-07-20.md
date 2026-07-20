# REveL YOLO11n 8/32 帧 crop/tiling 配对实验（2026-07-20）

状态：8 帧 canary 已完成并按预注册停止；32 帧未运行

证据权限：`bounded-public-rgb-tiling-screening-only`

生产授权：否

## 结论

固定的 `full-frame + 4 corner crops` 候选确实恢复了部分二维 small person box，但新增误报过多，未通过 8 帧 canary，因此没有进入 32 帧阶段。

这次停止不是 GPU 或系统不稳定：两臂均在前台守护下正常完成，无相关 System event。失败来自检测质量权衡，而不是运行故障。

## 冻结合同

- 基线：YOLO11n、全帧、`imgsz=256`、FP32、`batch=1`。
- 候选：每帧先跑全图，再跑四个固定 60% 角落 crop；共 5 views。窗口为 `[0,0,.6,.6]`、`[.4,0,1,.6]`、`[0,.4,.6,1]`、`[.4,.4,1,1]`。
- 融合：映射回原图归一化坐标后，score 降序做 class-agnostic NMS，IoU 固定 `.5`。
- 安全：GPU memory fraction `.15`、源帧间隔 `250ms`、view 间隔 `50ms`、温度硬停 `72°C`、8 帧超时 `180s`。
- 8 帧来自 512 r2 已冻结逐帧收据的 8 个不同 small-miss segment；32 帧合同另含 20 miss、8 hit 与 4 个空控制帧，但只允许在 8 帧 canary 通过后运行。
- 8 帧进入门：至少恢复 2 个 small miss、不得丢失任何基线已匹配 GT、F1 不低于 `.5`、FP 不超过 `6`，且两份守护收据均有效。

机器可读协议与选帧合同位于：

- `artifacts.local/evidence/ustrf-sc/revel-yolo11n-crop-tiling-pair-20260720-r1/protocol.json`，SHA-256 `157efdc130ad064eb436ecab4c1ac39e65a04a49d6d8188da22ec3f506da5364`；
- 同目录 `selection-8.json`，SHA-256 `9a9e68ce6f0c82ce5dc404bf8d130c11696629715c1267ace812a1820001b999`。

## 8 帧配对结果

| 指标 | full-frame | full + 4 crops | 变化 |
| --- | ---: | ---: | ---: |
| TP | 6 | 10 | +4 |
| FP | 4 | 14 | +10 |
| FN | 8 | 4 | -4 |
| precision | .6000 | .4167 | -.1833 |
| recall | .4286 | .7143 | +.2857 |
| F1 | .5000 | .5263 | +.0263 |
| small matched | 0/8 | 4/8 | +4 |
| medium matched | 3/3 | 3/3 | 0 |
| large matched | 3/3 | 3/3 | 0 |

逐 GT 配对为 small recovered `4`、small regressed `0`、all regressed `0`，恢复分布于 4 个 failure segment。成对报告 SHA-256 为 `f38cf353f13199ba9fd4f9167083beee26285ecdc7b10c3e392aa8a441ad01f7`，决定是 `stop_after_8_frame_canary`。

候选的 14 个 fixed-score FP 全部由 crop view 的融合结果贡献；恢复 4 个 small miss 的同时，没有任何 full-frame 基线已匹配 GT 回退。这说明放大确实能暴露小目标证据，但当前无条件四角 tiling 也把局部伪人形/重复候选放大，不能只看 recall 或 F1 略升便扩大样本。

## 系统收据

| 项目 | full-frame | full + 4 crops |
| --- | ---: | ---: |
| inference views | 8 | 40 |
| 最高温度 | 47°C | 50°C |
| 最高整卡显存 | 1,276 MB | 1,508 MB |
| 最高功耗 | 20.2 W | 18.08 W |
| 相关 System event | 0 | 0 |
| stop reason | 无 | 无 |

因此 GPU 恢复与守护路径保持有效；本次否决只针对这份冻结 tiling 检测候选。

## 决定与边界

- 32 帧阶段按预注册停止规则跳过；不得事后把 FP 上限从 6 放宽，也不得把 8 帧 failure-enriched canary 当作总体效果估计。
- 不运行 128/512/全量，不改默认 YOLO、不接 App、不改 risk/feedback、不开设备或生产门。
- 下一次若仍研究 detector，应把“降低 crop-view FP”作为新的独立变量，例如先冻结跨 view 一致性或小目标专用候选准入，再重新从独立 canary 开始；不能在本结果上扫描 overlap/NMS/score 来回救。
- 更高价值主线仍是 route-conditioned、object-agnostic risk field 与真实 assistive event truth。这里的 `small` 只是归一化 2D box area `<.02`，不是用户距离、physical TTC 或风险事件标签。
