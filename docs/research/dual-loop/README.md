# BlindAssist 神经—几何双环研究主线

状态：`current / PHASE_MINUS1_ADMISSION / PROPOSAL_ONLY / EXECUTION_NOT_AUTHORIZED`

最后核验：2026-07-30（Asia/Hong_Kong）

## 当前决定

BlindAssist 当前论文系统路线切换为“先准入、后实现”的神经—几何双环候选：

```text
DATA_STATUS: NOT_RUN
TIMING_STATUS: NOT_RUN
SCIENCE_STATUS: NOT_RUN
RUNTIME_STATUS: NOT_RUN
ROUTE_CONTRACT_STATUS: DESIGN_FROZEN
DATA_PROTOCOL_STATUS: NOT_RUN
TIMING_PROTOCOL_STATUS: NOT_RUN
SCIENCE_PROTOCOL_STATUS: NOT_RUN
RUNTIME_PROTOCOL_STATUS: NOT_RUN
EXECUTION_AUTHORITY: NONE
CLAIM_CEILING: THESIS_ROUTE_PROPOSAL_ONLY
```

顺序固定为：

```text
F-1A 数据能否评价
  ↓
F-1B0 当前语义与几何结果何时真实可用
  ↓
F-1B 几何证据是否产生事件级增量
  ↓
F-1C 指定手机是否承载得住
  ↓
才决定是否实现正式双环
```

详细输入、状态、判定和停止门以
[双环阶段−1准入合同 R0](BLINDASSIST_DUAL_LOOP_PHASE_MINUS1_ADMISSION_CONTRACT_R0_2026-07-30.md)
为准。合同已经进入项目主线，但**合同存在不等于任何实验已获授权**。

## 研究候选

- 语义证据：现有 YOLO11n 与现有生产检测/风险接口；目标设备若已准入，可使用其真实
  QNN 路由。
- 几何证据：阶段−1默认只评估已有 Sparse LK 候选输出能否形成区域级几何/接近候选
  证据；它是可替换证据源，不是已经成立的接近判断或独立告警系统。
- 汇合位置：同一事件/区域、真实时间戳、质量、时效和失效原因进入既有统一决策接缝；
  两个环不得分别提醒。
- 论文候选贡献：不是“双环”这个框图本身，而是双环相对 YOLO-only 是否产生可重复的
  首次有效提醒提前、风险判别改善或风险连续性改善。

当前不实现正式融合器、自适应调度、深度、分割、ARCore、新风险场、新状态机或第二套
反馈系统。

## 与 RCLE、USTRF 和 Project Guideline 的关系

- [RCLE](../rcle/README.md) 已由用户于 2026-07-30 暂停。既有科学终态、协议终态、
  one-shot 消费状态和未消费的 `480+16` 全部保留；双环不重跑或救援 RCLE。
- Sparse LK 是阶段−1的默认轻量候选。RCLE 若未来恢复，只能通过新的、独立的准入决定
  成为可选证据源，不能自动替换 Sparse LK。
- 已关闭的
  [USTRF route-conditioned program](../ustrf-sc/USTRF_ROUTE_CONDITIONED_PROGRAM_CLOSURE_R1_2026-07-25.md)
  不因“双环”名称重启；不恢复旧 dense risk field、route、lifecycle 或旧数据门。
- [Project Guideline 适配审计](../../PROJECT_GUIDELINE_COMPONENT_ADAPTATION_AUDIT_2026-07-30.md)
  只提供失效语义、时间戳、最小证据账本和可重算原则；仍是
  `REFERENCE_ONLY / NO_IMPLEMENTATION_AUTHORITY`。

## 当前证据边界

已有仓库证据只说明候选具有继续做准入检查的基础：

- 正式 CameraX 把 ImageAnalysis 配置为 `640×480 / KEEP_ONLY_LATEST`，Preview 请求
  `24 FPS`；该请求不等于真实 analysis 或结果频率。
- SM-S9280 / SM8650 的现有 QNN HTP 路由已经晋升，完整检测已有同机延迟与十分钟
  持续观测；能耗优势没有证据，SM8550 也没有被该结论覆盖。
- `LatestOnlySidecar` 与 `RgbaLumaSidecar` 已实现单槽替换、拥有的 luma 副本和过期
  结果拒绝，但不包含视觉算法、风险或提醒语义。
- 旧 CPU-era Sparse LK 回放与真机 shadow 结果相互提示：并发形态可能可承载，但真实
  CameraX 组合路径曾超过旧 `70 ms` 门，且没有 matched live YOLO-only 因果对照。
- 当前没有足以证明双环事件增量的独立事件级结果。

因此当前只允许把这些内容写成 `OBSERVED / CANDIDATE / HOLD`，不得写成双环已经有效。

## 当前权限

| 能力 | authority |
| --- | --- |
| 编写和维护阶段−1准入合同 | `AUTHORIZED` |
| F-1A 数据审计 | `NOT_AUTHORIZED` |
| F-1B0 双源时间基线补测 | `NOT_AUTHORIZED` |
| F-1B 离线几何增量评价 | `NOT_AUTHORIZED` |
| F-1C 手机双环 A/B | `NOT_AUTHORIZED` |
| 一次有限修复 | `NOT_AUTHORIZED / CONDITIONAL_ONLY` |
| 正式融合器或生产 CameraX 接线 | `NOT_AUTHORIZED` |
| 自适应调度、深度、分割、ARCore | `NOT_AUTHORIZED` |
| 默认模型、提醒、反馈或产品行为变更 | `NOT_AUTHORIZED` |
| 真人、独立助行、安全、产品或跨设备结论 | `NOT_AUTHORIZED` |

## 下一步

下一项可单独授权的动作只有 `F-1A_DATA_AUDIT_ONLY`。在用户明确授权前，不读取候选
输出、不补标、不采集、不运行设备实验，也不修改 Android 正式链。
