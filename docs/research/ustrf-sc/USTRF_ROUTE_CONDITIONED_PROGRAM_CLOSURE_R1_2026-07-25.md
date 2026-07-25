# USTRF route-conditioned program 收口 R1（2026-07-25）

状态：`VALID / ROUTE_CONDITIONED_PROGRAM_CLOSED`

最大权限：`DOCUMENTATION_CLOSURE_ONLY`

## 决策

[bbox-route 归因 R1](USTRF_BBOX_ROUTE_ATTRIBUTION_R1_RESULT_2026-07-25.md) 已触发唯一停止终态 `STOP_ROUTE_CONDITIONED_USTRF_DOWNGRADE_TO_DETECTOR_BASELINE`。因此，USTRF 当前 route-conditioned program 到此结束，不再把任何旧 `active`、`conditional`、`blocked`、`next stage` 或持续研究授权解释为待执行队列。

以下方向全部关闭：

- 当前 dense / object-agnostic risk-field 表达；
- 当前 bbox-route 表达；
- causal lifecycle 后继；
- 120 episode / 60 matched pair 与正式 U0；
- architecture convergence、student/producer 收敛和同一路线的算法变体。

现有 YOLO 与 bbox 能力保留为普通 detector baseline。保留不等于产品授权，也不触发删除、重构、默认模型替换、阈值变更或 App / Kotlin / Android 行为改变。

## 冻结与重启规则

本轮 15 对正负窗口及其既有结果永久保留为 discovery / falsification evidence。不得继续使用它们调整 route 宽度或形状、bbox 权重、quantile、窗口汇总、来源规则、detector 或报警阈值，也不得以换名变体继续回救。

未来若重新研究算法，必须另立全新、版本化、独立预注册的研究目标，并同时具备：

1. 与当前 dense、bbox-route、timing/token 或 lifecycle 延伸不同的新信号假设；
2. 不从本轮 15 对窗口选择、调参或形成接受门的独立证据；
3. 明确的新数据角色、对照、停止门和权限上限。

缺少任一项时，不恢复 USTRF route-conditioned program。

## 产品边界与下一步

- 正式 BlindAssist App、默认 YOLO 模型、CameraX、风险规则与反馈行为均不改变。
- 既有研究代码、结果和收据不删除、不重构，继续作为历史证据与普通 detector baseline 参考。
- 本收口没有 lifecycle、120 episode、U0、人体、安全、shadow 或生产权限。
- 收口完成后，工作面切换到其他产品工作；USTRF 当前路线没有自动后继。

## 事实来源

- [USTRF bbox-route 归因 R1 结果](USTRF_BBOX_ROUTE_ATTRIBUTION_R1_RESULT_2026-07-25.md)
