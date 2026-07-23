# USTRF route-target 证据成熟度分层标准 V2

状态：`current / governance active / L0 only / R2 candidates unrun`

## 结论

R1 的 `DATA_BLOCKED / STOP_SOURCE_SEARCH` 保持不变，但“来源未通过完整 holdout 准入”不再等于“整批数据毫无用途”。V2 改用 **指标级可评估性 + 分层授权**：

- recall、critical、repeat、clearance、false-alert exposure、evidence age 各用自己的有效分母；
- 缺 `terminal clear` 只限制 clearance，不再抹掉已闭合的 onset/alertable 证据；
- 样本不足输出 `evaluable_underpowered / PARTIAL_METRIC_EVIDENCE`，不输出通过；
- 只有全部必需指标在新鲜数据上达到确认级分母并通过冻结性能门，才能逐级进入候选选择、离线确认和隔离 Android shadow；
- 来源搜索每轮有明确预算和停止条件，不再以“继续找数据”作为无限下一步。

当前最高授权仍为 `L0_ENGINEERING_DIAGNOSTIC`。本标准只改变证据如何分层使用，不运行 C1–C3，不开放 Android shadow、H2、人体或生产权限。

机器合同：`configs/ustrf_route_target_evidence_maturity_v2.json`。验证入口：

```powershell
python scripts/run_research_tool.py ustrf-route-target-evidence-closure validate_evidence_maturity_v2.py `
  --config configs/ustrf_route_target_evidence_maturity_v2.json `
  --repo .
```

## 不可动态放松的真实性底线

以下条件跨所有成熟度永久保持：

1. 真值、指标 eligibility mask、窗口与分母必须在候选/App 输出暴露前冻结并绑定哈希。
2. 人物身份在被评分区间内必须唯一且连续；歧义、失配、缺测一律为 `unknown`。
3. 候选路线只能读取当前及过去输入；未来轨迹只允许服务离线注释。
4. missing、occluded、stale、unknown 永远不能产生 clear。
5. 候选输出不能改变真值、隔离决定或评分分母。
6. 每条完整序列只连续运行一次，不能按评分窗口重置。
7. 必须逐来源报告；pooled 结果不能覆盖单来源失败。
8. 对 unknown route 或 unresolved person 的 active alert 是晋级 veto，但不会删除其他指标已经形成的局部证据。
9. Android 结论仍要求 canonical Canvas/raw tensor 与正式 TFLite/决策内核哈希证据。
10. R1 的失败、已打开数据和候选未运行事实永久保留，不能追溯改写为 R2 成功。

## 指标级可评估性

| 指标 | 进入分母所需证据 | 不再连带要求 |
| --- | --- | --- |
| event recall | 同一人物从 onset 到 alertable deadline 身份连续；causal route 已知 | terminal clear、匹配负窗 |
| critical miss | critical interval 已观察；同人身份和 causal route 在该区间闭合 | terminal clear、负暴露 |
| repeat within observation | 首次 delivery 后，active episode 连续观察到 observation end | terminal clear；删失后只称 observed repeat |
| clearance | 同一人物真实 terminal clear 已观察，且有 post-clear follow-up | 正事件数量、负暴露 |
| regeneration after clear | clear 后仍有足够观察，可确认新 onset | 未闭合的右删失事件 |
| false alerts/min | causal route known 且所有 route-relevant person 已解决的连续负暴露 | 正事件必须闭环、matched negative |
| evidence age | capture/consume 时间戳已绑定 | 人物事件闭环 |
| unknown/stale active alert | 完整 replay 带 route validity 状态 | terminal clear；任一命中仍是晋级 veto |

`matched negative` 改为配对因果比较的独立证据，不再是完整序列 false-alert rate 的前置条件。

每个指标必须输出：

`support_status`、`result_status`、numerator、denominator、value、CI、`bound_sufficient`、`gate_result` 与排除原因计数。空分母必须是 `not_evaluable + null`；禁止把 `0/0` 写成零失败或通过。

false alerts/min 的 numerator 是完整序列中所有无法归因到 eligible active truth event + 正确人物的 delivery，包括正事件期间错人、`adjacent_safe/receding/cleared`、匹配窗外 delivery；unknown/unattributable delivery 另行触发 promotion veto，不能被静默丢弃。

## 删失规则

- `right_censored_administrative`：候选盲的序列结束或固定观察结束；可在声明独立删失合理时进入 KM/RMST 描述。
- `right_censored_identity_loss`：身份丢失属于信息性删失；晋级时保持 unresolved，不能用乐观生存分析处理。
- `competing_route_change`：路线任务改变后原 clearance 问题不再同义，默认是独立 competing event，不算 clearance 成功；只有 truth event type 在候选输出前明确预注册处理方式，才能使用，否则保守记为 promotion unresolved。
- `not_evaluable_pre_clear`：在 clear 发生前真值已失效。
- 所有 censored event 都不得被写成成功、失败或 `0ms clearance`。

clearance assessment horizon 固定为 `1500ms`：truth clear 后候选在 horizon 内 clear 才成功；完整观察 `1500ms` 仍未 clear 是失败；只观察到不足 `1500ms` 才是 right-censored；truth clear 前结束为 `not_evaluable_pre_clear`。confirmation 必须同时得到 clearance-rate LCB 与 P95 或 survival quantile；任一不可估时 `bound_sufficient=false`。

repeat 若观察到一次即可失败；右删失且观察到 0 次只能 `estimate_only`，只有完整 active episode 才能通过 repeat 门。regeneration 的 post-clear horizon 固定为 `2000ms`：观察到一次即失败，0 次只有完整 follow-up 才能通过。selection 的 clearance censor fraction 上限为 `.20`，confirmation/shadow 为 `.10`。超限只阻止对应指标晋级，其他指标仍独立保留。

clearance censor fraction 只从 **已观察到 truth terminal clear** 的事件开始计时：numerator 是 clear 后发生的 administrative、identity-loss 与未预注册处理的 competing route change，denominator 仅为已观察 truth clear 的事件。truth clear 前结束的事件不进入 KM、clearance rate 或该 censor fraction，而是单独报告 `terminal_clear_observability = observed truth clear / recall-eligible events`。

## L0–L4 成熟度与授权

| 层级 | 证据要求 | 能做什么 | 不能做什么 |
| --- | --- | --- | --- |
| L0 Engineering/Diagnostic | 有哈希绑定的真实数据，或明确标注的 proxy；限制已记录 | 管线验证、故障归因、回归 | 运行候选、效果结论、选胜者 |
| L1 Exploratory Metric Profile | 每项独立判定；非零分母可报 underpowered，达到相应 `5 events / 5min` screening floor 才形成该指标的 L1 profile | 点估计+CI、候选 profile、因真实性/unknown veto 淘汰候选 | 选胜者、Android、泛化或安全结论 |
| L2 Candidate Selection | 至少 2 个独立 session family；总计 20 recall、5 critical、15 clearance、15 complete repeat、15 complete regeneration、20min 负暴露；全 replay 带时间戳/route validity；相对主张另需 10 matched pair | 所有 required metric 的 point/worst-source 门通过且 veto=0 后，一次性选 1 个 provisional candidate 进入全新 confirmation lockbox | 把 `estimate_only` 当确认、直接进入 Android 或声称安全有效 |
| L3 Offline Confirmation | 全新 lockbox；6 session、至少 2 provenance family、60 个完整正负 matched pair、5 strata、6-fold LOSO；每项 bound 与 cluster/worst-session 门闭合 | 授予 offline-confirmed research candidate，并允许进入 Android admission | 拆 pair 跨 fold、直接运行 shadow、人体或生产 |
| L4 Android Shadow | L3 通过；Canvas/raw tensor、内核、时钟/route-age、隔离路由和所需米制几何门闭合；至少 10 runs，并分别达到 60 recall、59 critical、60 complete repeat、60 clearance、60 complete regeneration interval、120min 负暴露 | 仅形成 production-isolated research shadow evidence | 用某一指标样本补另一个指标、人体效果与生产授权 |

L1 不要求所有指标同时可评；L2 起才要求全部必需指标相对该层达到 `evaluable_powered`。这个状态只说明达到了所声明层级的分母门，绝不自动代表 `bound_sufficient=true` 或性能门通过。因此局部数据可以被利用，但不能越权晋级。

L2 的 `pass` 专指：全部 required metric 达到 L2 分母、point gate 与可判定的 worst-source gate 通过，且 promotion veto 为 0。它允许 `bound_sufficient=false`，此时结果状态必须是 `estimate_only`，selection decision 只能是 `PROVISIONAL_SELECTION_FOR_FRESH_CONFIRMATION_ONLY`。L3 的 `requires_l2_pass` 指这个 provisional selection 加全新 confirmation lockbox；L3 才要求每项 `bound_sufficient=true`。

来源样本不足不等于算法失败：普通 rate 只有达到预注册 per-family floor 才能判 source pass/fail；不足时为 `insufficient_support`。critical miss、unknown-route/unresolved-person alert 和 truth invariant violation 等硬 veto 则不论样本量逐来源生效。pooled 结果不能覆盖一个已可评来源的失败或硬 veto，但一个 underpowered 小来源也不会凭空把整个算法判失败；它只会使独立 family 数不足，从而停止晋级。

## 性能门与统计边界

性能门不随低成熟度降低：

- event recall `>= .90`
- observed critical miss `= 0`
- false alerts `<= .50/min`
- clearance rate `>= .90`，P95 `<= 1500ms`
- repeat / regeneration `= 0`
- evidence age P95 `<= 200ms`
- unknown/stale route active alert `= 0`

低样本即使点估计达标，也只能是 `estimate_only`。单侧 95% 边界至少满足：

- 零失败时，recall 下界达到 `.90` 需要至少 `29` 个事件；
- 零 critical miss 要证明 miss rate 上界不超过 `.05`，需要至少 `59` 个 critical event；
- 零 false alert 要证明 Poisson 上界不超过 `.50/min`，需要至少约 `5.9915min` 负暴露。

未达到边界时写 `bound_sufficient=false`，不能用“观察到 0 次失败”替代“风险已被充分约束”。L2 的一次性研究选择与 L3/L4 的更强确认权限必须明确区分。

L3 只对 recall、critical、FA 和 clearance 这四个 inferential-rate metric 要求 confidence bound 充分；其中 critical miss 若要把零观察失败解释为 miss rate 上界不超过 `.05`，至少需要 `59` 个可评 critical event。repeat/regeneration/unknown 是完整分母上的 hard veto，evidence age 是 point + worst-session 工程门，这四项的 `bound` 为 `not_applicable`，不能伪造 CI。`29/59/5.9915` 的 event-level exact bound 只是必要条件：L3 把 provenance family 固定为分层并检查 worst-family sentinel，在每个 family 内以 session 为单位、至少每 family 3 个 session，使用固定 seed `20260723` 做 `10,000` 次 bootstrap，并通过 LOSO worst-session sentinel；不足 5 个 family 时禁止把 family 当作随机重采层，任何 cluster CI 未定义或退化时仍是 `bound_sufficient=false`。Poisson 上界只作为 count-rate working model，不是无条件真实分布证明。

## 标准怎样调整

允许的“动态调整”只发生在轮与轮之间：

1. 先结束当前轮并保留原结论；
2. 发布新 protocol version，写明由缺失模式或方法学问题触发的原因；
3. 在目标候选输出不可见时冻结新语义、分母、阈值、primary metric、tie-break 和来源预算；
4. 已查看的数据自动降为 seen/development，不得继续充当新 selection/confirmation/shadow lockbox；
5. 新版本使用新鲜数据验证。

只影响下载、解包或物化且 outcome-unseen 的修复，可以走 hash-bound amendment。任何语义、分母、性能门或候选选择规则变化都必须升协议版本，不能在同一轮回救结果。

## 数据搜索预算与停止状态

每轮默认最多：

- `2` 个新 source family；
- 每来源 `2` 个 canary；
- `2 GiB` 自动下载；
- 连续 `2` 个来源 family 不合格后停止。

超预算必须在候选输出不可见时另行预注册；禁止候选特定来源搜索。未达到分母时输出 `STOP_DATA_COLLECTION_AT_CURRENT_LEVEL`，而不是继续无限搜索，也不是用全局 `DATA_BLOCKED` 删除已存在的局部指标证据。真实性或安全硬门失败才输出 `STOP_MECHANISM`。

## 现有数据的重新定位

- LILocBench 15+15 seen 集：当前 L0；允许重新冻结指标 eligibility mask 后进入最高 L1，不得成为 R2 selection/confirmation lockbox。
- CrowdBot 首组与 replacement 23 条：当前 L0；已有 2 个完整 clear event 和约 `5.533min` 可评分负暴露可作为局部探索输入。另有 10 个 **R1 以 `censored_without_terminal_clear` 隔离**的 proposal；V2 必须按指标重新定性：recall 只审计 onset→alertable deadline，critical 审计 critical interval，repeat 审计到 observation end；truth clear 从未观察时，clearance 是 `not_evaluable_pre_clear`，不能整批自动计入任何分母。
- 0327、NavWareSet、REveL 与最终外部清单：只保留 source-rejection/prescreen regression 权限。

因此下一独立边界不是继续找数据，而是 **R2-L1 metric eligibility materialization**：在不读取候选输出的前提下，为现有 seen/development 数据冻结逐指标 eligibility mask、删失状态和分母收据；验证后才能决定是否各运行一次 C1–C3 exploratory profile。该工作仍不产生候选胜者或 Android 权限。
