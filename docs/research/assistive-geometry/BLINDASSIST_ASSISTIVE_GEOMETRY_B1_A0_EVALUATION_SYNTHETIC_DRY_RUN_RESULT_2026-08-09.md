# Assistive Geometry B1 A0 评估合成 dry-run 结果

终态：`B1_A0_EVALUATION_SYNTHETIC_DRY_RUN_PASS`

合成 dry-run 已通过。它没有读取 Development 或 Confirmation outcome，也没有运行模型
推理；这里只签署 evaluator mechanics 与 checkpoint package integrity。

## 已验证

- seed 固定顺序 `17 / 29 / 43`，`selected_seed = null`；
- 共 `12` 个 tiny checkpoint，epoch `5 / 10 / 15 / 20` 的精确累计步数为
  `1499 / 2999 / 4499 / 6000`；
- 每个 checkpoint 的 bytes、SHA、协议/初始化 SHA、seed、epoch、step、model-state SHA、
  optimizer、scheduler、scaler、sampler、RNG 与 history 均通过；
- 每 seed 都跑通 `3 band × 3 horizon` 九格、pooled、parent、orientation、near-field、
  indoor/outdoor 和 low-light/blur 报告；
- `UNKNOWN` truth 每 seed 各有一格，均从负类分母排除；
- 通过 fixture 的 clearance MAE 为 `0.02 / 0.04 / 0.06 m`，temporal delta MAE 为
  `0.04 / 0.08 / 0.12 m`；这里只验证算术，不是模型质量结果。

## 失败终态

`checkpoint_incomplete`、`protocol_drift`、`missing_horizon`、`zero_denominator`、
`coverage_collapse` 与 `best_seed_forbidden` 六个反例均命中预期终态，并各自生成相邻
`failure.json` 和短 `failure.log`。

机器总收据：
`artifacts.local/evidence/hftf/assistive-geometry-b1-a0-evaluation-dry-run-20260809-r1/dry_run_result.json`
（SHA-256 `FAFC57127B07486672B5E66218E26BCA9DF53EA7396B3ACCCD51B9055AE47407`）。

## 下一步与边界

评估实现已就绪。真实 Development evaluation 仍须等 seed `17/29/43` 训练包全部完整，
再单独激活；不得挑 best seed，也不得打开 Confirmation outcome。本结果不授权模型质量、
部署、默认 App、产品或 safety 主张。
