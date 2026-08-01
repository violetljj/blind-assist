# HFTF Stage C D3-Q0.1 screening 预算耗尽结果

## 结论

唯一一次 Q0.1 screening 终态为
`D3_REFERENCE_SUPPORT_OPPORTUNITY_COHORT_NOT_EVALUABLE_BUDGET_EXHAUSTED_NO_EXPANSION`。

原 40-slot 预算中，slot 1 作为旧 Q0 schema-invalid 执行被永久 burned；Q0.1 从原
slot 2 开始，严格按原顺序各打开一次 slots 2–40。最终只有原 slots
`3 / 14 / 20 / 29 / 37` 通过资格门，未达到预先冻结的 first-six / 6-source formal
cohort 要求。因此 D2 effect 不可评价，future-blind preprocessor 和 sealed-effect
evaluator 均未获授权、未运行。

这不是 transport 有效或无效的结论，也不是 HFTF 假设的正负结论。它只证明：在当前
冻结的 roster、顺序、40-slot 预算与资格门下，没有形成 formal effect evaluation
所需的六来源 cohort。

## 一次性执行与闭合计数

execution contract 由 commit
`ef248690e60a77ba5ab4f98443fefaa64fbc1b50` 精确提交推送；执行前已确认
`HEAD == origin/master`，runner、aggregator、preprocessor 与 evaluator 的 formal
`verify_git=True` 全部通过。

第一次 runner 调用只写入 global screening attempt 与 outcome-free slot-1
carry-forward burn receipt，没有打开媒体、pose、support 或 truth。随后 39 次
new-slot 调用严格对应原 slots 2–40，没有重跑、重开、替换、重排、扩容或同 cohort
调门。闭合计数为：

- 1 个 carry-forward burned slot；
- 5 个 qualified selectors；
- 32 个合法 not-qualified selectors；
- 2 个 execution failures；
- 合计 40 个 consumed slots。

原 slots 2 和 28 因 `D2 current ground sample is inadequate` 封存为
`D3_QUALIFICATION_SLOT_NOT_EVALUABLE_CONSUME_SLOT_CONTINUE_FROZEN_ORDER`；failure
按冻结合同消耗槽位，不触发替换。slot 40 闭合后，aggregator 只运行一次，并在读取
首个 selector/failure receipt 前 durable 写入 attempt；它没有打开 sealed payload。

## Durable evidence

- contract：
  `268f1491835fb8b4d365a24064eac94edc5046633fa7861b7fbd1588ded7225a`
- screening attempt：
  `eb3c035f850a3c527e4f6079aec2ee356db62c7b506088af81cb09edd4d7551a`
- slot-1 carry-forward：
  `19ef0745d254009677b8100b1c0e39270ae0ab85b3b32b000fea5ce43f6e4503`
- aggregate attempt：
  `409aee3dd51e3aee483a888e1a21a1024c8b8fa64717b88e5ec4da1b69d5242c`
- budget-exhausted terminal：
  `e992a8117184b2f97dbfd4ac81805cc665a003fbf6f85167fec1d213d2b9e89b`

五个 qualified selector SHA-256 分别为：

- slot 3：
  `17b0ed07a3951fd4e7abfd4a3adea39c4f682876c17b0401b78ea7fa3d553c09`
- slot 14：
  `809c02b8430e6330065a10b75b91bbaa2c168c17e2abd1a3aee7807e9b0032f1`
- slot 20：
  `5b35b637e36a52f7ff3902563e42146f653af9ac6a07817dc7976ec51d0bb06c`
- slot 29：
  `c1f1931c7c6b4b2e3d169e7a2b0da56424f0ca1a6a1d55bdd1132fba1a1c7dee`
- slot 37：
  `d49a359d1993357993766fccef1066bc5de64912b5a5247731729e366f98c99d`

`selection.json`、`screening_invalid.json` 与 `formal/` 均不存在。slot-1 carry receipt
明确记录 `invalid_selector_read=false`、`outcome_fields_imported=false`；旧 Q0
selector/outcome 没有进入 Q0.1 cohort。

## 独立终审

独立科学终审使用冻结 scanner 只读重建完整 state，和
`budget_exhausted.json` 逐字段完全相等；确认 40/39/1/5/2 计数、qualified/failure
slot 集合、first-six 未达与权限边界，结论 `CLEAR`、0 blocker。

独立工程终审复核 40 个连续 slot receipt、attempt/hash 绑定、日志与负终点闭集，
确认 aggregate attempt-first、没有 selection/invalid/formal artifact、没有遗留执行
进程或第二次执行，结论 `CLEAR`、0 blocker。

## 科学与权限边界

不得从本结果读取或估计任何 D2 effect metric/gate，也不得表述为 HFTF/transport
“支持”或“不支持”。geometry teacher 仍是 synthetic proxy，不是人类事件或 safety
truth。

本 Q0/Q0.1 cohort 已全部消费。screening 重跑、slot 重开、替换/重排来源、扩预算、
同 cohort 调门、future-blind preprocessing、sealed effect、RGB student、
reserved official-test、研究主线、默认 App、Android、生产与 safety 权限全部关闭。

若继续研究，只能先建立新的独立 protocol/data-role 边界。允许对已消费 receipts 做
描述性、只读的资格瓶颈归因，用于提出新假设；不得据此回填本 cohort、修改已冻结门或
把诊断性结果冒充 fresh validation。
