# Dense Metric Depth Propagation R0 Development result

日期：2026-08-03

终态：`DENSE_PROPAGATION_CONSUMED_DEVELOPMENT_NOT_SUPPORTED`

## 结论

冻结的方案二没有通过。它在每个已完成 Metric3D 关键帧上拟合 DA→Metric3D 的稠密
仿射关系，再用双向 RAFT 一致区域传播 Metric3D 相对仿射 DA 的残差；遮挡/新区域由
当前帧 DA 填充。候选路径没有读取任何非关键帧 Metric3D 深度，因果违规为 0。

光流本身不是主要瓶颈：一致区域平均覆盖 `87.60%`，P05 为 `75.58%`。但 30 帧因
锚点仿射残差门失败而整帧 UNKNOWN，另有 8 帧处于启动期，最终只有 `81/120 =
67.5%` 帧能与 sensor proxy 配对。仅 clearance MAE 过门，任务门为 `1/5`。

| 指标 | 冻结结果 | 门 | 通过 |
| --- | ---: | ---: | :---: |
| paired-valid | 67.50% | >=90% | 否 |
| clearance MAE | 0.17694 m | <=0.25 m | 是 |
| 包络一致率 | 89.81% | >=90% | 否 |
| false-clear | 7.02% | <=5% | 否 |
| temporal delta MAE | 0.15114 m | <=0.15 m | 否 |

独立 CUDA worker 假设下，DA 加双向 RAFT 的稳态均值/P95 为 `112.55/131.59 ms`；
其中光流均值/P95 为 `55.81/78.31 ms`。这不含异步 Metric3D worker 的 service time，
也不是手机端共驻延迟。已知输出 anchor source-age P95 为 `0.60408 s`。

## 解释边界

结果否定的是当前固定 RAFT-small、双向 `1.5 px` 一致门、50% coverage、固定稠密
affine 与 period-5/TTL-1s 的窄实现。不能在该 consumed outcome 上搜索 flow、阈值、
fit、period 或 TTL 救援，也不能据此宣称所有稠密传播无效。

该回放只用四个已消费 TUM RGB-D 窗口；sensor depth 是 clearance proxy，不是人工通行
真值。没有 final-camera、手机共驻、温度、安全或 ToF 替代权限。

## 可复现性

- 协议 SHA-256：`6F93050C2FBF8C5769CA363EC23E34D69356B318331475FA0AD75A90B299C2E1`
- cache manifest SHA-256：`49C3F9426D0D658B61741569BB5AD1DB6839DBECF76227531E628FE662D9D8B4`
- evaluator SHA-256：`9C676D3363A6299791B209DA8760F415C6B97744793A766828F5D8087A2A0A4F`
- ignored result SHA-256：`2B5937E4B8242F6220590745D041DE69F5F907C206FDA51AD9F249D20F4EC51E`
- ignored trace SHA-256：`80DE838625ACAA0EB3E7D2C4BE69A75EE7FC5DC644133AAC085859C0D6BC653E`

机器摘要见
[DENSE_METRIC_DEPTH_PROPAGATION_R0_DEVELOPMENT_RESULT_2026-08-03.json](DENSE_METRIC_DEPTH_PROPAGATION_R0_DEVELOPMENT_RESULT_2026-08-03.json)。

