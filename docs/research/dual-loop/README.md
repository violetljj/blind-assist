# BlindAssist YOLO + 语义分割双环研究主线

状态：`current / THESIS_DEVELOPMENT / RISKSEG_R0_NEGATIVE_NOT_PROMOTABLE / DEFAULT_APP_UNCHANGED`

本页只维护当前摘要、权限和唯一 successor。完整历史已保留在 [archive/README_FULL_HISTORY_2026-08-07.md](archive/README_FULL_HISTORY_2026-08-07.md)，日期化协议与结果继续作为 snapshot。

## 当前主张

验证论文级的 YOLO + 语义分割双环是否能在事件级风险/可通行性问题上提供可复现增量；不主张产品安全、独立行走有效性或默认模型替换。

## 当前状态

- YOLO 是正式 App 的 incumbent；默认模型不变。
- RISKSEG R0 的开发训练/技术链路已完成，但事件质量/稳定性与晋级门未形成可推广正证据，当前仅保留 `DEVELOPMENT` 诊断与失败学习。
- event-eval 数据门仍是当前科学阻塞点；未冻结 truth 前，不启动新的正式效果、量化或 Android 晋级链。

## 稳定入口

- [RISKSEG R0 task/data contract](RISKSEG_R0_TASK_DATA_AND_EXECUTION_CONTRACT_2026-08-01.md)
- [RISKSEG data role ledger](RISKSEG_R0_DATA_ROLE_LEDGER_2026-08-01.json)
- [RISKSEG event-eval gate result](RISKSEG_R0_EVENT_EVAL_DATA_GATE_RESULT_2026-08-01.md)
- [项目算法路线总表](../ALGORITHM_ROUTE_REGISTRY.md)

## 唯一 successor

`RISKSEG_EVENT_EVAL_DATA_REPAIR_SUCCESSOR`：只有补齐独立事件标签、达到当前数据合同门槛并重新登记版本后，才能进入新的 Development/Confirmation 选择；未满足条件时没有其他并行 successor。

## 禁止与权限边界

禁止重算或改写 consumed 终态、为旧结果调阈值回救、把 host/device benchmark 写成科学效果、把研究结果接入默认 App，或把正向论文证据写成安全认证。任何生产晋级必须另行显式激活 `PRODUCTION_PROMOTION`。
