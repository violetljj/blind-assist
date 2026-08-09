# Assistive Geometry B1 A0 评估合成 dry-run 协议

状态：`FROZEN / SYNTHETIC_ONLY / OUTCOME_SEALED`

机器合同：[JSON](BLINDASSIST_ASSISTIVE_GEOMETRY_B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PROTOCOL_2026-08-09.json)

## 目的

在任何 Development 或 Confirmation outcome 打开前，用纯合成 fixture 验证 A0 训练后
评估器。它只签署评估器与 checkpoint 包装机制，不签署模型质量。

## 最小硬门

- seed 必须严格为 `17 / 29 / 43`，不允许缺失、重复、额外 seed 或 best-seed 选择；
- 每 seed 必须有 epoch `5 / 10 / 15 / 20` 四个 checkpoint，对应累计 optimizer step
  `1499 / 2999 / 4499 / 6000`；
- 外部 bytes/SHA 与内部协议 SHA、初始化 SHA、seed、epoch、step、model-state SHA、
  optimizer/scheduler/scaler/sampler/RNG/history 必须一致；
- 九格必须保持 `left/center/right × 1.0/1.5/2.0 m`，`UNKNOWN` 不得进入负类；
- 全局 truth-known、truth-clear 或 truth-occupied 关键分母为零时 fail-closed；细分 strata
  的条件分母为零只报告 `undefined`，不把已定义的 pooled 指标一起判废；
- pooled coverage 每个 seed 都必须通过。任务阈值按每项至少 `2/3` seed 通过汇总，
  同时保留三个 seed 的全部值、mean、sample std、median、min/max，不选择某个 seed。

## 指标与分层

主指标为 known coverage、false-clear/all-known、false-clear/truth-occupied、
false-block/truth-clear、band-level clearance MAE 和同 parent/session/band 连续帧 temporal
clearance-delta MAE。九格、parent macro/worst、orientation、near-field、indoor/outdoor 与
low-light/blur 作为诊断分层，不额外制造阻塞门。

## 失败收据

checkpoint 不完整、协议漂移、seed 集合错误、schema/九格错误、全局零分母、coverage
塌缩、任务门失败与 best-seed 企图都有独立机器终态。每个失败场景必须同时保留相邻
`failure.json` 与短 `failure.log`。

## 防火墙与边界

dry-run 只允许 `SYNTHETIC_EVALUATOR_FIXTURE`。不得读取 Development/Confirmation
文件、运行模型推理、导入 teacher、选择 seed 或改默认 App。通过只表示训练结束后无需再
临时补写评估器；真实 Development 执行仍须等三 seed 训练完整并另行激活。
