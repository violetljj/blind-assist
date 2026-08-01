# BlindAssist 信息上限三臂审计 D0 结果

状态：`COMPLETE / VALID / MIXED_DETECTOR_AND_REPRESENTATION_GAPS /
CONSUMED_DEVELOPMENT_CURRENT_CHAIN_ONLY / DEFAULT_APP_UNCHANGED`

协议：
[INFORMATION_CEILING_THREE_ARM_D0_PROTOCOL_2026-08-01.md](INFORMATION_CEILING_THREE_ARM_D0_PROTOCOL_2026-08-01.md)

## 结论

在本 cohort 和当前决策链上，继续只在 YOLO 输出后补规则已经没有充分依据：

- 当前 YOLO 在两个正 parent events 上均未交付提醒，关键事件漏报 1 个；
- 完整真值风险框把正事件命中恢复到 `2/2`、关键漏报降到 0，证明当前 detector
  的类别/数据/输出确实缺少决策链需要的信息；
- 但真值框同时产生 53 个误提醒帧、1 个负事件误报，且两个 passed event 均未清除，
  说明“框存在 -> 当前风险规则”本身不能稳定表达路径占用与事件结束；
- 真值 mask 经当前 adapter/source policy 后为 `2/2` 命中、0 关键漏报、0 误提醒帧、
  `2/2` passed 清除，事件计数上同时支配当前 YOLO 与真值框；代价是两个正事件的
  首次响应分别比真值框晚 5 帧和 2 帧。

所以该结果不是“目标检测完全无潜力”，也不是对“纯 bbox 几何上限”的独立证明。
它支持的工程选择是：**冻结当前 YOLO 为 baseline，停止继续为同一失败模式增加
post-YOLO 规则；若只推进一个主模型候选，下一候选优先做轻量风险/可通行性分割。**

## 冻结执行

- 设备：SM-S9280，serial `R5CX10M8Y8X`；
- device run：`20260801-104938`，benchmark payload：`20260801-105134`；
- cohort：90 帧、3 个 `risk_event_id`，其中 2 个正事件、1 个负事件；
- manifest SHA-256：
  `3d7168ac975aed57ac6b437ecfa0e668c13dc5d509c7b6584353383eed19d217`；
- dataset spec SHA-256：
  `6815d8b613eca34d840255e66f02bd751196cb55e1b3b74f8ff0f659babf07bb`；
- YOLO asset SHA-256：
  `00edb41a528b0a7e709c4af8ce3e685491492c4539274804e5cfc17a1a867cd2`；
- `riskConfig=current`、`AlertProfile.STANDARD`、100 ms 因果时钟；
- decision kernel：`blindassist_shared_decision_kernel_v1`；
- 每臂每帧 3 次 app run，序列边界 reset；
- Gradle/instrumentation：1/1 test，`BUILD SUCCESSFUL`；
- benchmark JSON SHA-256：
  `1bd2d3cc156af1706428f642f393e43efa30d7c3cc37af96f77354edb595f1d7`。

三臂实际输入互斥：

1. `A_CURRENT_YOLO`：当前 YOLO detections only；
2. `B_ORACLE_RISK_BOX`：318 个 source-region/mask-derived 风险类别框，
   `OBJECT_DETECTOR` source，禁止混入 YOLO；
3. `C_ORACLE_RISK_MASK`：source-native mask 经当前
   `TraversabilitySegmentationAnalyzer`，`SEGMENTATION` source，禁止混入 YOLO。

## 事件结果

| Arm | 正事件命中 | 关键漏报 | 负事件误提醒 | 误提醒帧 / 9 s | passed 清除 | center 首次响应 | stairs 首次响应 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `A_CURRENT_YOLO` | 0/2 | 1 | 0 | 0 | 2/2 | MISS | MISS |
| `B_ORACLE_RISK_BOX` | 2/2 | 0 | 1 | 53 | 0/2 | 0 帧 | 1 帧 |
| `C_ORACLE_RISK_MASK` | 2/2 | 0 | 0 | 0 | 2/2 | 5 帧 | 3 帧 |

误提醒率同时保留分母：cohort 暴露仅 9 秒，因此 B 的 53 个误提醒帧换算为
`353.33/min`，该 rate 只用于本 cohort 描述，不外推为现场频率。

成对事件计数：

- `B vs A`：多命中 2 个正事件、少 1 个关键漏报，但多 1 个负事件误报并少清除
  2 个 passed event，属于不可支配 trade-off；
- `C vs B`：命中和关键漏报持平，少 1 个负事件误报、多清除 2 个 passed event，
  达到 `MATERIAL_EVENT_GAIN`；
- `C vs A`：多命中 2 个正事件、少 1 个关键漏报，其余 parent-event 计数不恶化，
  达到 `MATERIAL_EVENT_GAIN`。

按 outcome 前冻结规则，字面终态为
`MIXED_DETECTOR_AND_REPRESENTATION_GAPS`。

## 独立 validator

稳定入口 `scripts/validate_information_ceiling_three_arm_d0.py` 不读取 benchmark
aggregate 作为结果，逐项完成：

- manifest、dataset spec、90 个 RGB/mask 内容 hash 与 3 个 event identity；
- 三臂完全相同的 90-frame membership、risk config、alert profile、时钟与 kernel；
- A 的 YOLO-only source/class 范围；
- B 每帧 `risk_inputs` 与 manifest `source_regions` 的精确 label/class/box 对应；
- C 从原始 mask 独立执行 256×256 nearest resize、taxonomy、4-connected component、
  corridor/boundary gate、排序与 `take(1)`，再与设备账本比较；
- 逐帧 scoring truth、event hit、关键漏报、误提醒、首次有效响应和 passed 清除复算；
- 三组 pairwise deltas 与终态。

结果：`PASS`，errors 0。最终本地 evidence：

`artifacts.local/evidence/information-ceiling-three-arm-d0/20260801-105134/`

## 归因上限

三个限制阻止把结果写成“语义分割已经被证明优于目标检测”：

1. 只有 3 个 parent events、2 个正事件；event/phase truth 来自既有 AI-review 派生，
   mask 又含 24 帧人工标注与 66 帧机器标注；
2. Arm B 是从同一 mask ancestry 派生的风险框，不是独立 detector-native instance GT；
3. Arm C 在进入 `RiskAnalyzer` 前已经执行 corridor/连通域/边界筛选，最后最多只转发
   一个框；B/C 还触发不同 source-specific temporal/event policy。

因此 C 的优势可能来自像素形状、走廊筛选、候选压缩和 source policy 的组合。
它与 bbox 表征限制一致，但不隔离 bbox 几何；C 的 truth-oracle 成功也不证明一个
learned segmentation model 能复现该结果。

## 路线决定与停止条件

- 当前 YOLO 保留为冻结 baseline 和工程回退，不再为本 cohort 的漏报继续加类别后规则、
  taxonomy patch、阈值或事件特例；
- 不依据本 pilot 立即删除检测链或切换默认 App；
- 若只推进一个学习模型候选，下一 Development 问题应是轻量风险/可通行性分割，
  用新的 parent-event/source cohort 检验 learned mask 是否能保留 C 的事件收益与清除，
  同时满足设备 P95、内存和热预算；
- 本 90-frame outcome 已 consumed，只可做回归与机制诊断，不再用于阈值、mask adapter、
  taxonomy 或 event rule 选择；
- 若论文要声称“bbox 几何上限”，另立后继 matched-control：B/C 使用共同中性 source
  policy，并让 bbox rasterization 与 dense mask 计算同一组空间风险特征；本轮不补做。
