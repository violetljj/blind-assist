# HFTF Stage C D2 因果 signed-clearance transport 结果

## 结论

唯一一次 D2 mechanics 执行终态为
`D2_NOT_EVALUABLE_OPPORTUNITY_INADEQUATE_NO_SOURCE_REPLACEMENT`。

这不是 transport 有效或无效的结论。冻结的 24 个
`parent × height × horizon` opportunity strata 只有 8 个通过，16 个失败，因此 effect
gates 按合同没有获得判定权限。该六源 cohort 已消费，不得换源、补样本、重跑或同
cohort 调参。

## Future-blind 与一次性 truth join

execution contract 与 exact implementation/test bytes 由 commit
`ed56242178538cb2c83ee465615cf9073e78caad` 提交推送并确认
`HEAD == origin/master` 后执行。

preprocessor 只启动一次并自然退出。它在首个 history/current pose 或 current
depth/mask read 前写入 durable attempt，然后按冻结顺序完成 6 parents × 7 anchors =
42 个 points/prediction 对、84 个 horizon records。completion 离线复核 exact
order/count/hashes，全部 prediction 均记录
`future_depth_mask_or_pose_read=false`。

evaluator 只启动一次。在打开任何 future pose/depth/mask 前，它先排他写入并 `fsync`
truth-join receipt；随后生成 exact 84 个 synthetic geometry-proxy truth records 并自然
退出。两个进程 stderr 均为 0 bytes；没有 failure artifact、重跑、换源、追加或
partial fill。

关键 evidence SHA-256：

- contract：`2afb530400b157990474523f4157630f9bf1bc225f15e32bfe9a0ffd4f034c56`
- preprocessor attempt：`5203515259ac66fb63529efe24073d2f5304c484531364cb553ba73a0136ece0`
- preprocessor completion：`da01d2abe5ba3f07e87f2f68d0862abbddd7a119cc67e76e00c91e231a158ca3`
- truth-join receipt：`b6186923b1fdc051ae9af6984d973a07475c14c3e2ae1bba642d00661a15ef99`
- effect result：`a6c34d28876c46b09b3507ab46468530c04ea9b409d5fdd3e0d0701b91356276`

## Opportunity gate

每个 stratum 必须同时满足 common-known coverage `>=0.10`、known risk `>=5`、
known safe `>=20`。独立离线重算与正式 result 完全一致：

- 24 个 strata 中 8 个通过、16 个失败；
- 16 个失败 strata 的 known risk 全部少于 5；
- 其中 3 个同时低于 0.10 coverage，3 个同时少于 20 个 known safe；
- 全部 strata 的最低 coverage 为 `0.027777777777777776`，最低 known risk 为 0，
  最低 known safe 为 7；
- 6 个锁定 parents 按顺序分别失败 `1 / 4 / 4 / 4 / 0 / 3` 个 strata；
- UNKNOWN→SAFE violation 为 0。

独立最终只读审计再次闭合 42 个 prediction receipts、84 个 truth keys/future offsets、
24 个 strata 与完整 hash chain，得到 0 mismatch、`CLEAR`；审计未解码媒体、未重跑
执行器，也未创建新 artifact。

最主要的可判定性缺口是 parent-stratified risk support 不足，而不是 effect gate
本身失败。宏观 MAE、各 height/horizon 非劣、5/6 parent 改善及 risk-sign F1 增量均
不应从本结果中选择性读取或声明。

## 科学与权限边界

本结果只证明：当前冻结六源 D2 cohort 不足以裁决 history-only constant-velocity
transport 相对 current-field persistence 的效果。它既不支持也不否定 transport
假设；geometry teacher 仍只是 synthetic proxy，不是人类事件或 safety truth。

本协议内 source replacement、追加、重跑和同 cohort retuning 均关闭。RGB student
protocol、training/execution、reserved official-test、研究主线、默认 App、Android、
生产与 safety 权限全部关闭。若未来继续，只能建立新的 protocol/data-role 边界，并在
任何新 mechanics outcome 前重新冻结独立 opportunity-adequate cohort 规则；不得把本
cohort 的失败 strata 用作定向选源或阈值调参依据。
