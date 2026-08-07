# DEPTHART_ADMISSION_R0 结果

日期：2026-08-07

终态：`DEPTHART_S_METRIC_INDOOR_R0_QUALITY_NOT_ADMITTED`

## 结论

官方 `DepthART-S metric indoor` 在 120 帧 TUM consumed Development 回归上，深度误差、
clearance 和 false-clear 都显著优于冻结 DA2 baseline；但 false-block 从 `0.48%` 升至
`3.10%`，超过冻结的 `+1 percentage point` 容差。因此 Gate 1 失败，不能进入 ONNX
数值一致性或 Snapdragon QNN/HTP 可行性预算，也不能替换、删除或降级 DA2。

这不是“模型完全失败”：它是一个有明确任务收益、但当前风险代价不合格的候选，后续
只能在新协议授权下研究 false-block/unknown 边界；本轮不允许在这 120 帧上调阈值、重做
scale、选 checkpoint 或搜索后处理。

## 固定身份与执行证据

- 候选：`depthart-s-metric-indoor-448-official-fp32-r0`。
- 官方源码提交：`0384521b3bcb4c64adf03eeb5d55ebdb1cbdd84c`。
- checkpoint SHA-256：`597631AC7AEAB8346F4DB013C3C65EF3203DF373E21C7265D7A147093C667E65`。
- 物化缓存 SHA-256：`4C3FFE1C4B931137E5DF313F4A91B24F87A3A90349C63A077965C0FE5BB1BA37`。
- 物化形状/类型：`120×480×640`, `float32`；固定 TUM 内参，未读取 sensor truth 做模型输入。
- 当前 Python 运行时实际参数计数：`8,072,022`；这条运行事实优先于论文中的约数，不把“约 6M”外推为本项目资产计数。
- host 仅记录为工程诊断：RTX 5060、reference Selective Scan fallback；加载约 `15.8 s`，
  逐帧中位 `125.6 ms`、P95 `314.4 ms`。这不是 Snapdragon 或移动端性能证据。

完整 receipt、逐帧进度、cache 和机械 R2 projection 位于
`artifacts.local/evidence/hftf/depthart-admission-r0/`。

## 门结果

| 门 | 结果 | 关键证据 |
| --- | --- | --- |
| Gate 0 资产/有限输出 | PASS | 官方 source/checkpoint/hash/size 均一致；输出全有限 |
| Gate 1 任务质量 | FAIL | clearance、false-clear、collision、ground、truth status 通过；false-block 失败 |
| Gate 2 时序质量 | PASS | clearance delta `0.1279 m`、depth delta `0.1088 m`、scale drift `0.01427` 均在容差内 |
| Gate 3 PyTorch→ONNX | NOT_EVALUATED | Gate 1 未通过，不消耗导出预算 |
| Gate 4 Snapdragon QNN/HTP | NOT_EVALUATED | 未启动 Android/设备工作 |

### 任务指标（候选 / DA2 baseline）

- metric AbsRel median：`5.31% / 29.43%`（诊断，不作为单独晋级理由）。
- scale-aligned AbsRel median：`4.23% / 8.33%`（诊断）。
- clearance MAE：`0.1582 m / 0.3804 m`。
- false-clear：`6.76% / 24.25%`。
- false-block：`3.10% / 0.48%`，Gate 1 veto。
- temporal clearance delta MAE：`0.1279 m / 0.1131 m`，在 `+0.015 m` 容差内。

## 权威边界与下一步

DA2 P1/P2 的既有终点保持不变；FRESH-TF 及 successors 仍为用户暂停。DepthART-S 当前
是 `candidate-core / not admitted`，不是主 backbone。若要继续，只能另立并冻结一个
false-block 机制协议，优先检查 metric 输出与 ground/occupancy envelope 的关系；不能
在本轮 consumed cohort 上做 outcome-driven rescue。最终 camera/Android 结论仍需要新的
session/parent-disjoint、独立 RGB-D 或量距真值。
