# 项目研究路线总表

状态：`current`。本表是跨研究域的短入口；当前算法研究焦点是 DepthART。项目整体同时包含可独立并行、也可与算法主线耦合的数据、链路、延迟、性能和部署研究。详细协议、结果和历史过程仍由各路线唯一真源维护。

## 当前研究结构

- 算法研究焦点：DepthART-S。
- 并行研究 workstream：数据集/数据治理、通信链路、端到端延迟、性能优化、部署可行性。
- 关系：workstream 可以独立形成问题和结论，也可以在明确接口、数据角色和证据边界后与
  DepthART 或其他路线耦合；不默认属于 DepthART 的附属实验。
- 权限：任何 workstream 都只产生自己范围内的证据，不自动获得默认 App、生产或安全权限。

| 路线 | 主张 | 当前状态 | 唯一真源 | 下一动作 / 唯一 successor | 禁止动作 | 是否影响默认 App |
|---|---|---|---|---|---|---|
| DepthART 算法路线 | 当前算法研究主要围绕 DepthART-S | `ALGORITHM_FOCUS / DEVELOPMENT_STANDARD / DEFAULT_APP_UNCHANGED` | [HFTF README](hftf/README.md) | 新 parent-disjoint admission；另有明确接口时可接收数据/平台研究输入 | 不把数据、链路或平台结果直接写成算法 admission | 否 |
| YOLO + 语义分割双环 | 论文级风险/可通行性机制研究，探索分割是否提供稳定增量 | `THESIS_DEVELOPMENT_SECONDARY / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE`；YOLO 保持 incumbent | [dual-loop README](dual-loop/README.md) | 仅在 event-eval 数据 repair successor 满足后继续；不与 DepthART workstream 混写 | 不重算 consumed 终态、不把训练/benchmark 结果写成生产或安全结论 | 否 |
| DA2 Metric teacher/baseline | 提供冻结的 metric-depth teacher、baseline、回归参考和 fallback | `FROZEN_BASELINE / DEVELOPMENT_REFERENCE`；不参与当前新候选晋级 | [DA2 P1/P2 closure](hftf/DAV2_P1_P2_EXECUTION_CLOSURE_2026-08-05.md) | 只作为统一 ledger 下的 reference；若重开必须新版本、明确问题和独立数据 | 不把 DA2 的 teacher/HTP 性能诊断写成 accuracy、生产或安全授权；不因新候选失败而删除/降级 DA2 | 否 |
| DepthART-S candidate backbone | 在相同 clearance/false-clear 约束下替代或挑战 DA2 的深度 backbone | `R0_QUALITY_NOT_ADMITTED / R1_RESEARCH_MAINLINE / A3_BLOCKED_SELECTIVESCAN`；仍属 HFTF Development | [DepthART R1 result](hftf/DEPTHART_ADMISSION_R1_RESULT_2026-08-07.md) | 先完成图级 numerical parity 与 SelectiveScan lowering feasibility；科学 admission 仍需新 session/parent-disjoint holdout | 不在 R0 120 帧上调 admission 阈值；不提前做 quantization/partition/真机 latency；不接入默认 App | 否 |
| SANPO 训练/评测 | 为论文双环提供事件、分割和设备候选证据 | `THESIS_DEVELOPMENT`；生产晋级未激活 | [SANPO current](../SANPO_CURRENT_STATUS.md) | 只在明确任务下选择 `THESIS_DEVELOPMENT` 或 `PRODUCTION_PROMOTION` lane | 不用生产门阻塞论文开发；不以单次银标/设备结果替换默认模型 | 否 |
| RCLE-RF | 旋转补偿局部扩张风险场机制 | `PAUSED / NO_ACTIVE_EXECUTION` | [RCLE README](rcle/README.md) | 仅在用户明确授权后，以新 scoped contract 恢复一个 successor | 不自动恢复旧 one-shot、旧 fresh split 或旧“下一步” | 否 |
| USTRF-SC route-conditioned | 历史路线条件化、looming 与事件级证据链 | `CLOSED / HISTORICAL_DIAGNOSTIC` | [USTRF-SC README](ustrf-sc/README.md) | 若重开，必须是全新信号假设 + 独立证据；否则无 successor | 不沿用旧窗口调阈值、route、quantile 或架构收敛 | 否 |
| 候选事件自动挖掘 | 发现候选事件和失败模式，不产生事件真值 | `DISCOVERY_ONLY` | [candidate-event-mining README](candidate-event-mining/README.md) | 将候选交给已授权的事件数据合同；没有合同就停在 candidate pool | 不把 candidate、模型复核或公开视频直接当 truth、训练集或 holdout | 否 |
| 数据集与数据治理 | 建立 parent/session 隔离、truth、coverage、质量和可复用数据角色 | `DEVELOPMENT_DATA_WORKSTREAM / PARALLEL_OR_COUPLED` | 各路线数据合同与 [HFTF README](hftf/README.md) | 可独立修订数据合同，也可为 DepthART admission 提供合规输入 | 不用候选输出反推 truth、不把 candidate pool 当 event truth、不重用被选择污染的数据作 confirmation | 否 |
| 通信链路与延迟 | 分解外设/网络/拷贝/排队/推理/反馈链路，定位真实端到端瓶颈 | `DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED` | 对应 timing/latency result 与 [HFTF README](hftf/README.md) | 可独立做链路优化，也可在接口冻结后接入 DepthART 端到端测量 | 不把吞吐、RTT 或 accelerator occupancy 写成准确率、安全或 App 全链路 authority | 否 |
| 性能与部署可行性 | 优化 CPU/GPU/HTP、内存、热、稳定性、模型导出和 runtime 可行性 | `DEVELOPMENT_PLATFORM_WORKSTREAM / PARALLEL_OR_COUPLED` | 对应 deployment/performance result 与 [HFTF README](hftf/README.md) | 可独立做平台工程研究，也可在 numerical parity 后为 DepthART 提供部署证据 | 不在科学 admission 前做无边界 quantization/partition 搜索，不把平台 benchmark 伪装成模型质量 | 否 |

## 状态与权限规则

- `current` 入口只维护摘要、权限和唯一 successor；日期化结果属于 `snapshot`，完整流水属于 `archive`。
- 新路线先登记短 current 入口，再开始实验；旧路线退出当前执行职责后必须转为 `closed`、`paused` 或 `diagnostic`，详细材料归档到同域 `archive/`。
- `closed`、`paused`、`diagnostic` 路线没有隐含的下一步；后继必须在当前真源中显式命名并绑定新的问题、数据或实现差异。
- 任何研究结果都不自动修改正式 App、默认模型、生产权限或安全结论。
