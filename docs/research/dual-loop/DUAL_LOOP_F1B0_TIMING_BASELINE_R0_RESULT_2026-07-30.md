# F-1B0 双源时序基线 R0 结果

状态：`COMPLETE / READY / TIMING_PROTOCOL_VALID`

执行者：`viojjet`

## 结论

既有 QNN 与 Sparse LK 凭据只有聚合延迟或不完整的完成时间，不能回答结果何时真正
发布、可用和被消费。按连续推进授权补做一次 baseline-only 真机测量后，绑定设备
`SM-S9280 / SM8650` 上的生产 QNN 语义轨迹与隔离 Sparse LK 几何轨迹均形成完整因果
时间账本：

```text
semantic results = 24
semantic backend = qualcomm_qnn_htp
geometry results = 24
clock domain = ANDROID_ELAPSED_REALTIME_NANOS
camera timestamp source = REALTIME
camera clock mapping = VERIFIED_ANDROID_REALTIME
future-use violations = 0
alerts invoked = false
risk outputs serialized = false
```

独立确定性校验因此给出：

```text
TIMING_STATUS = READY
TIMING_PROTOCOL_STATUS = VALID
```

这只产生 F-1B 科学比较资格，不说明几何具有事件增量，也不授权生产双环接线。

## 观测延迟

单位均为毫秒，`available age` 从可辩护的源观测时间到结果真正可用，`work` 只计算
本次算法工作区间：

| 路径 | available age P50 | available age P95 | work P50 | work P95 |
| --- | ---: | ---: | ---: | ---: |
| production QNN semantic | 86.017 | 107.773 | 9.125 | 9.999 |
| isolated Sparse LK geometry | 9.397 | 11.309 | 4.474 | 5.402 |

几何轨迹 `abstain count = 0`。这些数值是一次开发时序基线，不是功耗、热稳定、组合路径
吞吐或跨设备结论。

## 因果与隔离边界

- 语义路径使用目标 App 的 `RuntimeObjectDetectorFactory`，实际后端为
  `qualcomm_qnn_htp`；未改变生产路由。
- CameraX 报告 `SENSOR_INFO_TIMESTAMP_SOURCE=REALTIME`，因此 capture 与
  elapsed-realtime 时间可在同一域内校验。
- 每条语义记录满足
  `receivedAt <= queuedAt <= startedAt <= completedAt <= publishedAt <= availableAt <= consumedAt`。
- 每条几何记录满足前后观测、排队、开始、完成、发布、可用、消费的单调顺序。
- 几何只在 hash-bound benchmark RGB 上隔离运行；未与 YOLO 融合，未访问事件标签、
  风险或提醒效果。
- receipt 明确记录 `effect_outputs_accessed=false`、`alerts_invoked=false`、
  `risk_outputs_serialized=false`。

## 可复算凭据

```text
timing_receipt_sha256:
7c2b0d7caab29161cf256fec63cfa64b3f0eca285d9312401124b26d9d4ce5e7

validation_sha256:
6b0ebe06c8ba4c64027128c5a48b1f167d4cfd5662f72b9cf2401788ac75bd6d
```

正式本地凭据位于：

`artifacts.local/evidence/dual-loop/f1b0-timing-baseline-r0/`

真机选择性 instrumentation 结果为 `OK (1 test)`；确定性时序校验结果为
`READY / VALID`。

## 下一阶段

F-1B 必须在读取 decision 输出前，用 development session 冻结 Sparse LK 身份、质量门、
区域规则、TTL、连续支持、同一提醒状态/冷却、唯一主要终点与最小有意义差异。F-1B0
的隔离几何时间只允许把正向首次提醒结论写成
`EARLY_INFORMATION_OPPORTUNITY / DEVELOPMENT_SCREEN`，直至 F-1C 在真实组合路径复现。
