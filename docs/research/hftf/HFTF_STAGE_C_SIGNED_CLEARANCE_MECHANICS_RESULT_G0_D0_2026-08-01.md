# HFTF-G0-D0 support-equivalent clearance mechanics 结果

终态：
`G0_SIGNED_CLEARANCE_SOURCE_AND_MECHANICS_TERMINAL_VALIDATED`

机制终态：
`G0_SIGNED_CLEARANCE_MECHANICS_SUPPORTED_FOR_FRESH_LEARNABILITY_CANARY`

## 结论

G0 可以进入独立的 current-RGB learnability canary，但目前只证明 frozen
support-equivalent clearance proxy 在已 consumed synthetic sources 上定义正确且
非退化，不证明 RGB student 已经会学。

source planner 在未打开新媒体或 outcome 的条件下固定了三组 parent sessions：

- 9 个 outcome-open development reuse：原 6 train + 3 model selection；
- 3 个固定 one-shot fresh evaluation；
- 3 个仅 metadata 预留的 official-test future heldout。

后三个 heldout 在整个 G0 中仍不得获取或打开。

## D0 证据

runner 使用 12 个已 consumed F0.1 parent sessions 的全部 300 个 current frames。
每个 `source × body/head` 单独过门，共 24 个独立 source-height 单元：

- positive known 最小值 `5`，negative known 最小值 `148`；
- clipped target 的 1 mm bins 最小值 `55`；
- `|target| <= 0.2 m` 近边界 known cells 最小值 `10`；
- risk 在 `-0.5 m` 的最大饱和比例为 `0`；
- safe 在 `+1.0 m` 的最大饱和比例为 `0.888889`，但每个单元都有非饱和 safe；
- proxy `< 0` 与原 `support_count >= 2` 的不一致为 `0`；
- unknown 非 null target、unknown→safe 违规均为 `0`。

独立 validator 未导入 mechanics runner，并从冻结协议、source plan 与结果文件重新计算
角色隔离、firewall、全部 source-height gates 和终态。validation SHA-256 为
`4659e1fbb7938a637c157c6ceaad1186bc2b9ec919951fca6cb252b61acacd62`。

## 解释边界

D0 支持的是：

> reference-lattice clearance proxy 在 consumed synthetic geometry 上既与旧 binary
> support rule 精确等价，又保留足够连续、近边界和非饱和监督。

它没有回答：

- current RGB 能否学习该 proxy；
- fresh source 上是否优于 direct binary risk；
- history、future 或 causal transport 是否有效；
- human collision、真实助盲安全、主线或 App 是否应改变。

下一步只允许先冻结 D1 current learnability 合同。模型、loss、activation、checkpoint、
threshold、gates 和 fresh execution 顺序必须在任何 fresh media/outcome 打开前固定。
