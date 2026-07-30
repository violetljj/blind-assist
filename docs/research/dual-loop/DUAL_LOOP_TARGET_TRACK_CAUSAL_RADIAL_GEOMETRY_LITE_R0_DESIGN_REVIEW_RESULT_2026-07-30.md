# target/track-conditioned causal radial geometry LITE R0 设计评审

状态：`DESIGN_REVIEW_PASS / IMPLEMENTATION_NOT_RUN / REPLAY_NOT_AUTHORIZED`

阶段：`DEVELOPMENT / F1_INTERFACE`

执行者：`violjjet`

## 结论

[LITE R0 设计锁](DUAL_LOOP_TARGET_TRACK_CAUSAL_RADIAL_GEOMETRY_LITE_R0_DESIGN_LOCK_2026-07-30.json)
通过两轮独立只读评审。第一轮先给出 `HOLD`：既有 512-frame/770-box 稀疏账本不能形成
连续因果 replay 或自然事件分母。修订后，完整 REveL Dynamic capture 已冻结为
候选可见的 RGB/ROI allowlist 和候选不可见的 Vicon/event truth：

```text
source frames = 8,580
unique-ROI replay opportunities = 13,014
truth rows = 17,160
raw truth-only events = 1,660
primary parent events (>=5 frames) = 469
target x anchor-region x truth-state cells = 18/18
minimum primary events in one cell = 9
```

最终评审为 `DESIGN_REVIEW_PASS`。该 PASS 只授权实现两条冻结 arm、evaluator、
pre-truth deterministic fixtures，并在任何全量 producer replay 前生成和独立评审
implementation identity lock。它不授权全量 replay、truth join、科学终点消费、
Confirmation、Android、融合、提醒、产品或安全工作。

## 冻结设计

- 候选只能读取当前与紧邻过去帧、同一 opaque `target_id + track_epoch` 和 source-GT
  ROI；lookahead 为 `0`。
- `BBOX_LOG_AREA_GROWTH` 使用
  `0.5 * d(log box area)/dt`；`ROI_SPARSE_RADIAL_FLOW` 使用同一 ROI 内 LK track
  相对各帧 ROI 中心的 median `d(log radius)/dt`。
- 两臂共享输入、frame pair、timestamp、region、reset、TTL 和固定事件分母；两臂
  均不做 global-motion compensation。
- 最小输出固定为 `target_id + region + positive-is-approach signed rate + quality +
  100 ms TTL + abstention`，并包含 frame/time、track epoch 和实现/参数 hash。
- REveL Vicon 只允许在 producer ledger 写完并固定 SHA-256 后由独立 evaluator join；
  green/yellow oracle identity 不进入 producer。
- parent natural event 是同一 target、同一 truth state、连续帧且 gap 不超过
  `100 ms` 的最大 run。region 变化不拆 parent event，首个 truth-eligible frame 的
  region 只作 anchor stratum。
- capture 是唯一独立组；target、event、frame 是嵌套单位。R0 只报告描述性
  Development 计数，不把 frame/event 当跨 capture 独立样本。

## 评价和停止

每个 arm 的主分母固定为全部 `469` 个 primary parent events。event coverage 分母是
该 parent event 的全部 truth-eligible rows，包含首帧和 epoch-reset abstention；
非弃权有限输出为分子。少于 `3` 行或 coverage `<0.50` 的事件不可评价，但仍以错误
事件留在固定分母，不能从共同成功子集删除。

flow 只有同时满足以下条件才可写成
`FLOW_CANDIDATE_READY_FOR_FUTURE_CONFIRMATION_DESIGN`：

- 正确 event 比 bbox 至少多 `2`；
- wrong-signed event 不增加；
- 可评价 event 最多损失 `23`；
- 两个 target 均有正增益，且至少两个 anchor region 有正增益；
- 总体正确率 `>=0.60`、可评价率 `>=0.80`、wrong-signed 率 `<=0.20`；
- approaching/quasi-static/receding 每个 truth state 正确率均 `>=0.50`。

`wrong-signed` 只表示 approaching↔receding 反号；quasi-static 与方向状态的互错计为
incorrect，但不伪装成反号。任一未来输入、truth 泄漏、旧 F-1B decision 访问、
arm 分母不一致、abstention 删除或 outcome 后改门均使本 evidence version
`INVALID`。fixture 失败最多允许一次纯实现修复，且必须发生在第一次 truth join 前；
第二次失败关闭 R0。

## 身份与验证

| 对象 | SHA-256 |
| --- | --- |
| design lock | `ada0bfb22d84b0600354e438555c6a603456a302ffba6d0d3d418fe4e2cd6a2a` |
| input-freeze manifest | `ee7073c311832c1866addd6440c917fdcd45fb46a41c7f9c8072413a60f9b642` |

验证结果：

- research protocol validator：`VALID / errors=[] / warnings=[]`；
- input-freeze validator：`VALID / errors=[]`；
- manifest preparation/firewall tests：`5/5 PASS`；
- replay、truth、event 行数、key、SHA-256、image-root scope 与 18-cell coverage
  均由独立评审复算一致。

全程未运行候选、未 join 新候选 truth、未读取旧 F-1B decision 输出。

## 唯一后继

实现两个冻结 arm、evaluator 和全部 pre-truth fixtures；生成 hash-bound
implementation identity lock 并做独立评审。只有 implementation review 再次
`PASS` 后，才可另行激活一次全量 producer replay。设计 PASS 本身不是算法 PASS。
