# HFTF Stage C D3-Q0.1 consumed-selector Failure Atlas R0

## 结论

只读 Failure Atlas 将 Q0.1 的资格瓶颈定位为：

`D3_Q0_1_CURRENT_REFERENCE_TRUTH_RISK_OPPORTUNITY_SCARCITY_DOMINANT_HYPOTHESIS_ONLY`

37 个合法 selector 共包含 148 个
`parent × body/head × .4/.8 s` strata，其中 93 个失败。89/93 个失败 strata 包含
`truth_risk_count < 5`，68/93 更是只有 risk-count gate 失败；coverage 和 safe
分别只在 24/148 与 25/148 个 strata 失败，UNKNOWN→SAFE violation 为 0。

统计独立单位仍是 37 个 parent source，148 个 strata 是 parent 内重复记录，不能当作
148 个独立样本。slots 2 和 28 因 execution failure 没有 selector，Atlas 对这两个
来源存在选择性缺失；slot 1 是 carry burn，也不属于 Atlas。

因此，当前最值得检验的新假设不是“把门从 5 降到 3”，而是：

> 当前 reference/qualification 表示下，可用于四 strata parent-level effect 的
> risk opportunity 稀缺，尤其集中在 head × `.8 s`；下一步应把“表示是否对高度与
> 时间干预敏感”从“自然 parent 是否恰好同时包含四类风险”中解耦。

这是 consumed Development 诊断，不是 confirmatory 结果。它不能区分真实场景风险
本来稀有、当前 teacher/reference 漏掉风险，还是 anchor/parent sampling 与问题不匹配。

## 数据边界与复算

诊断只读取已闭合的 37 个 selector receipts，没有打开 sealed payload、媒体或 formal
artifact。法源为：

- Q0.1 contract SHA-256：
  `268f1491835fb8b4d365a24064eac94edc5046633fa7861b7fbd1588ded7225a`
- budget terminal SHA-256：
  `e992a8117184b2f97dbfd4ac81805cc665a003fbf6f85167fec1d213d2b9e89b`
- 37-entry selector manifest SHA-256：
  `6ef266713ba2e5329768351816b4e131357aea81e70eb4d04f2194adf78922ca`

manifest 每行固定为
`slot_index:workspace-relative-forward-slash-path:sha256\n`，UTF-8 无 BOM，共 6209
bytes。population 闭合为 1 carry + 2 execution failures + 37 selectors = 40 slots；
37 selectors 又闭合为 5 qualified + 32 not-qualified。

## Gate Failure Atlas

93 个失败 strata 的 gate 组合为：

- 68 个仅 risk 不足；
- 21 个 coverage + risk + safe 同时不足；
- 3 个 coverage + safe 不足但 risk 已达到 5；
- 1 个仅 safe 不足；
- 0 个 UNKNOWN→SAFE violation。

四个 strata 的通过数与 risk 分布为：

| Stratum | 通过 | Coverage fail | Risk fail | Safe fail | Risk Q1 / median / Q3 | Risk max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| body × `.4 s` | 20/37 | 4 | 14 | 5 | 2 / 6 / 11 | 29 |
| body × `.8 s` | 17/37 | 12 | 19 | 12 | 0 / 4 / 7 | 23 |
| head × `.4 s` | 11/37 | 3 | 26 | 3 | 0 / 2 / 6 | 19 |
| head × `.8 s` | 7/37 | 5 | 30 | 5 | 0 / 2 / 4 | 14 |

head × `.8 s` 是最稀疏 stratum：只有 7/37 通过，30/37 的 risk count 少于 5，
15/37 为零。selector 层面，5 个通过全部四 strata；其余 selector 中 3 个失败 1
stratum、11 个失败 2、4 个失败 3、14 个失败全部 4 strata。

来源层面，31/32 个 not-qualified selector 至少有一个 risk failure；20/32 是纯 risk
failure signature，11/32 是 risk + support，只有 1/32 是纯 support。16 种 exact
source signature 中，最常见的是仅 head × `.4/.8 s` risk 失败（9 个），其次是四
strata 全部仅 risk 失败（6 个）。body × `.8 s` 另有次级 support 缺口：
coverage/safe 各失败 12/37，common-known count 的 Q1 为 19，低于离散最低计数 26。

资格结果没有明显随 slot 顺序改善：原 slots 2–20 的 18 个合法 selectors 中 3 个
qualified，slots 21–40 的 19 个中 2 个 qualified。这里只作描述，不建立趋势结论。

## 近失样本不是救援许可

slots 4 和 7 都只因 head × `.8 s` risk count 为 3、低于冻结的 5 而失败；slot 24
只因 body × `.8 s` 的 known/risk/safe 为 `13/5/8` 而失败。

这些数值在 outcome 后已知，恰恰是不能改门的理由。把 risk 门降到 3 会追认已消费
样本，破坏 Q0.1 的预注册边界；补开第 41 个 slot、回填 slots 4/7、替换 failure slots
或改成 pooled strata 都同样是救援，不获授权。

## 推荐的新后继

优先候选是 `D4_OPPORTUNITY_ECOLOGY_AND_RECRUITABILITY`。它不再问 HFTF effect，
而先问：

- 在预先声明的 fresh target source population 中，all-four opportunity 的
  source-level 发生率是多少；
- pre-truth metadata 能否以前瞻方式识别可招募来源，并把 acquisition 成本控制在
  冻结预算内。

Q0.1 只作为生成假设的 consumed pilot。新研究必须在结果前冻结 target population、
完整 exclusion union、prospective sampling、最大 acquisition budget、由该预算推导的
最低可行机会率 `p_min`、metadata-only rule、fresh holdout，以及 source-level
confidence bound。若 fresh cohort 的单侧 95% 上界仍低于 `p_min`，或 metadata rule
在 fresh holdout 达不到冻结的 recall/cost 门，则停止该招募路线。成功也只授权另建
一批完全独立的 sealed-effect cohort；生态 cohort 本身不能转作 effect。

第二候选才是 `D4_PAIRED_GEOMETRY_INTERVENTION_CHALLENGE`：在 fresh controlled
scene/pair 中构造已知 motion/geometry/occlusion opportunity，用 paired estimand 检验
HFTF transport 相对 persistence 的 signed change 和噪声失效面。它能隔离机制，但
不能替代 real-source recruitability 或 fresh effect。

## 权限与停止边界

本 Atlas 只授权设计后继，不授权执行。下一步必须先冻结 fresh opportunity-ecology
target population、sampling/holdout、budget-derived `p_min`、metadata rule、
source-level estimand、cost gate 与负终点，再提交推送和独立审计。

Q0/Q0.1 重跑、改门、追认 selector、换源、扩预算、preprocessor/effect、RGB student、
研究主线、默认 App/Android、生产与 safety 权限仍全部关闭。
