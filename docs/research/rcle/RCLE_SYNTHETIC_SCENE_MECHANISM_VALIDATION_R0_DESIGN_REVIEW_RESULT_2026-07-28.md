# RCLE Synthetic Scene Mechanism Validation R0 design review

独立评审终态：`REVISE`

评审确认：scene family 内完整交叉 motion 能解除旧 external cohort 的
source-role overconstraint；scene-level split、truth/algorithm 概念防火墙和 synthetic
claim ceiling 方向正确。

执行前必须修订：固定 `101` 帧/`100` pair 与 trigger timestamp；为 coverage、
retention、delay 和 truth/algorithm error 定义固定分母、未定义值及逐 sequence AND；
增加旋转接近、纯 yaw 和 truth agreement 直接门；修复会越出房间的 receding 轨迹；
冻结坐标/pose/visibility/truth fit；使 algorithm firewall 可执行；把 sealed 激活改为
hash-bound exclusive one-shot claim；补环境、磁盘、运行时和双运行确定性 preflight。

上述项目已进入 `dataset_spec.json` 的 design revision R1。首次评审不授权 candidate
generation 或 sealed execution；revision R1 需第二次独立 review。

## Revision R1 implementation disposition

development 执行前的后续复核确认 signed projection、pose/truth 方向和 algorithm
staging 设计成立；validator、valid-mask digest、staging payload hash 和 Windows
CRLF 下冻结协议身份的实现缺陷随后均已修复。完整测试、双生成确定性、development
结果和最终独立复核见同日结果文档。

本轮评审与修复仅打开 development 执行，不授权 sealed。sealed 仍为
`CLOSED / NOT_AUTHORIZED`。

## Final review

最终独立复核结论：

- renderer/truth/interface：`PASS`
- development evidence package：`PASS`
- sealed：`CLOSED / NOT_AUTHORIZED`

最终复核确认 25/25 tests、四类 truth-only firewall mutation、staged payload
tamper rejection、双生成确定性、完整 dataset validator 和保存 ledger/summary
hash 均闭合。validator 明确只报告 `algorithm_ledger_parse_checked=true` 与
`scientific_outcome_recomputed=false`，不冒充独立科学结论复算。
