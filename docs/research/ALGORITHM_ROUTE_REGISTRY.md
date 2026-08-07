# 算法路线总表

状态：`current`。本表是跨研究域的短入口；详细协议、结果和历史过程仍由各路线唯一真源维护。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作 / 唯一 successor | 禁止动作 | 是否影响默认 App |
|---|---|---|---|---|---|---|
| YOLO + 语义分割双环 | 论文级风险/可通行性机制研究，探索分割是否提供稳定增量 | `THESIS_DEVELOPMENT / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE`；YOLO 保持 incumbent | [dual-loop README](dual-loop/README.md) | 先解决 event-eval `HOLD`；仅能新建有独立标签依据的 RISKSEG successor | 不重算 consumed 终态、不把训练/benchmark 结果写成生产或安全结论 | 否 |
| DA2 Metric teacher/baseline | 提供冻结的 metric-depth teacher、baseline、回归参考和 fallback | `FROZEN_BASELINE / DEVELOPMENT_REFERENCE`；不参与当前新候选晋级 | [DA2 P1/P2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 只作为统一 ledger 下的 reference；若重开必须新版本、明确问题和独立数据 | 不把 DA2 的 teacher/HTP 性能诊断写成 accuracy、生产或安全授权；不因新候选失败而删除/降级 DA2 | 否 |
| DepthART-S candidate backbone | 在相同 clearance/false-clear 约束下替代或挑战 DA2 的深度 backbone | `R0_QUALITY_NOT_ADMITTED / R1_RESEARCH_MAINLINE / A3_BLOCKED_SELECTIVESCAN`；仍属 HFTF Development | [DepthART R1 result](hftf/DEPTHART_ADMISSION_R1_RESULT_2026-08-07.md) | 先完成图级 numerical parity 与 SelectiveScan lowering feasibility；科学 admission 仍需新 session/parent-disjoint holdout | 不在 R0 120 帧上调 admission 阈值；不提前做 quantization/partition/真机 latency；不接入默认 App | 否 |
| SANPO 训练/评测 | 为论文双环提供事件、分割和设备候选证据 | `THESIS_DEVELOPMENT`；生产晋级未激活 | [SANPO current](../SANPO_CURRENT_STATUS.md) | 只在明确任务下选择 `THESIS_DEVELOPMENT` 或 `PRODUCTION_PROMOTION` lane | 不用生产门阻塞论文开发；不以单次银标/设备结果替换默认模型 | 否 |
| RCLE-RF | 旋转补偿局部扩张风险场机制 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE README](rcle/README.md) | 仅在用户明确授权后，以新 scoped contract 恢复一个 successor | 不自动恢复旧 one-shot、旧 fresh split 或旧“下一步” | 否 |
| USTRF-SC route-conditioned | 历史路线条件化、looming 与事件级证据链 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF-SC README](ustrf-sc/README.md) | 若重开，必须是全新信号假设 + 独立证据；否则无 successor | 不沿用旧窗口调阈值、route、quantile 或架构收敛 | 否 |
| 候选事件自动挖掘 | 发现候选事件和失败模式，不产生事件真值 | `DISCOVERY_ONLY` | [candidate-event-mining README](candidate-event-mining/README.md) | 将候选交给已授权的事件数据合同；没有合同就停在 candidate pool | 不把 candidate、模型复核或公开视频直接当 truth、训练集或 holdout | 否 |

## 状态与权限规则

- `current` 入口只维护摘要、权限和唯一 successor；日期化结果属于 `snapshot`，完整流水属于 `archive`。
- `closed`、`paused`、`diagnostic` 路线没有隐含的下一步；后继必须在当前真源中显式命名并绑定新的问题、数据或实现差异。
- 任何研究结果都不自动修改正式 App、默认模型、生产权限或安全结论。
