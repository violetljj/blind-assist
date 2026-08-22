# BlindAssist Public-real Episode Mining V0 result

状态：`8X89_EXECUTED_AND_SEALED / TRUTH_OR_CONTRACT_INSUFFICIENT_PRIMARY / NO_ALGORITHM_SUCCESSOR_AUTHORIZED`

## 结论

当前 successor 已从 physical capture 改为 `PUBLIC_REAL_EPISODE_MINING_V0`。公开真实数据承担视觉分布、几何校准与
goal-driven mechanics；人工采集不是当前 blocker。物理/用户研究只在公开数据不能回答一个已明确的高价值问题，或要
提出真实视障用户效果 claim 时才另行立项。

已实现的 prospective miner 强制在访问 Mapillary metadata、pixel、model output 和 evaluator truth 前冻结 public Goal
Contract 与 OSM/Overture entrance candidate set；按 `sequence_id + GPS + compass + captured_at` 自动形成距离总体下降、
朝向目标的 approach segment。入口候选数机械映射为 `UNIQUE / SET_VALUED / AMBIGUOUS`，不会为凑 benchmark 强造
唯一 truth。

Truth 顺序冻结为：

```text
native GT
> map/trajectory-derived truth
> independent-teacher consensus
> AMBIGUOUS / UNKNOWN
> manual annotation as last resort
```

## 已执行 smoke run

为验证端到端工具链，复用了已经 sealed 的 Last-10m Mapillary Development sequence；这不是 fresh cohort，也不产生新
model call。来源含 22 个真实 base frames、1 个 OSM `entrance=main` pose proxy 和 6 个既有 approach replay。本轮自动
适配出 `6 episodes / 29 observations`，人工采集 `0`、人工标注 `0`、新 provider call `0`，然后运行 Selective Guidance
V0 baseline 与 evaluator。

所有 29 个 observation 都缺少 exact frame-region visibility truth，因此 visibility、proposal Recall@K、referent
correctness 的 eligible denominator 均为 0；27 次 confident guidance 被正确记为 `unknown=27`，而不是 wrong=27。
六个 episode 的 failure attribution 均为 `TRUTH_OR_CONTRACT_INSUFFICIENT`。OSM pose 只提供 range bucket proxy，不能
升级为物理距离、入口可用性、arrival 或用户 completion truth。

## Fresh prospective metadata cohort

随后从 6 个既有 Overture/OSM source slices 中机械排除历史 Mapillary/roster buildings，并在 Mapillary access 前冻结
4 个新的 venue-taxonomy 合法 public goals。首版 planner 因只要求 place/building/entrance crosswalk，会把楼内商户名
当作 building goal；该问题在 pixel/model/truth 均为 0 时通过公开 roster 审计发现，V0 planning cohort 封存不用。
V1 增加 Overture venue taxonomy 门并排除 V0 buildings，冻结 `Choco-Story`、`Ambassade D Andorre`、
`Instituut Voor Dierkunde`、`Aartsbisschoppelijk Archief te Mechelen` 四个 goal，其中后两处分别保留 `UNIQUE` 与
`SET_VALUED` entrance contract。

Mapillary bbox 查询取得近邻 metadata，再自动展开 14 个具有至少 3 个近邻帧的完整 sequence，共检查 5,151 个 image
metadata；矿工形成 `8 episodes / 89 observations`。全程 `manual capture=0 / manual annotation=0 / pixel download=0 /
provider calls=0`。这是 fresh metadata/trajectory cohort，不是 visibility、proposal 或 guidance performance result。

## 冻结 8×89 正式运行

冻结 roster 未替换、未补抽：8 episodes / 89 observations 全部下载 pixel，三路独立于 baseline 的本地
teacher（YOLOE-26n-seg、base functional-door、domain-adapted functional-door）各执行 89/89，并保留原始
输出。teacher agreement 为 `AGREE=5 / PARTIAL=18 / DISAGREE=66`；没有通过事后挑选 teacher 消除分歧。

Truth coverage 与算法指标分开报告：

- native/map-only strong truth：`0/89`；
- map-bearing + teacher 支持的弱可用 `TEACHER_SUPPORTED`：`4/89`；
- `TEACHER_ONLY_WEAK`：`19/89`；
- `UNKNOWN`：`66/89`。

Truth 冻结后才运行原 Grounding DINO Tiny + `gpt-5.6-terra / medium` V0 baseline；provider journal 为
`89 dispatched / 89 completed / 0 in_doubt`，且不读取 private truth。在仅 4 个
`TEACHER_SUPPORTED` observation 上，proposal Recall@10 为 `4/4`，selection accuracy 为 `1/4`；
另外 `3/4` 的 observation attribution 是 `REFERENT_SELECTION`。这个局部信号不足以建立 selective
commitment H1，因为 8/8 episode 的主 attribution 都仍是 `TRUTH_OR_CONTRACT_INSUFFICIENT`。

公开 map range proxy 同时进入 provider，不能作为独立 range truth；evaluation 中对应的 range
self-consistency 数字不产生 range/geometry accuracy claim。没有用户或独立 completion truth，completion rate
同样不可作效果指标。

正式 headline：

> 0/89 observations 获得 native/map-only strong truth；4/89 仅获得 teacher-supported 弱可用 truth。
> 在这 4 个中 selection 为 1/4，另有 19/89 teacher-only weak 与 66/89 UNKNOWN。当前首先失败在
> truth/substrate coverage，而不是已经建立的算法瓶颈。

终态 receipt：
`artifacts.local/evidence/public-real-episode-mining-v0/prospective-8x89-v0/terminal-receipt.json`。

## 运行前 validity gate（已消费）

在任何 pixel/teacher/provider run 前，已冻结以下机械 validity gate：

- observation truth 使用 `NATIVE_GT / MAP_TRAJECTORY_DERIVED / TEACHER_SUPPORTED / TEACHER_ONLY_WEAK / UNKNOWN`；
- 原样保留 `teacher_A/B/C` 独立 raw output、implementation、运行状态、agreement/disagreement 与 provider-family
  independence；
- `functional_authority=ESTABLISHED` 必须包含 native 或 map/trajectory authority source；teacher consensus 单独存在时
  最高只能是 `TEACHER_ONLY_WEAK / NOT_ESTABLISHED`；
- evaluator 只接受 `truth_frozen=true`，并按 tier 单列 total、UNKNOWN、conditioned denominator、accuracy 与 failure
  attribution；
- current-frame 缺失只输出 `NOT_VISIBLE`；`LOST` 只由 episode FSM 的 `VISIBLE -> NOT_VISIBLE_AFTER_VISIBLE` 派生，
  `NEVER_SEEN -> NOT_VISIBLE` 不产生 LOST。

这些 gate 不包含任何性能阈值，并已按冻结顺序一次执行和封存。不得用 post-outcome resampling、人工唯一门、
threshold/model/provider sweep rescue denominator。当前不授权 acquisition、proposal、selective commitment、range、
P1 或其他算法 successor；若继续，必须另行提出能够增加独立 functional truth coverage 的 substrate/truth-source
工作，且不得改写本次 8×89 终态。

ADT 只承担 depth/bearing/visibility/occlusion/temporal calibration；Ego4D 只承担真实第一视角取景、模糊、遮挡和 out-of-
frame 分布；Habitat 只承担显式 endpoint 的闭环 mechanics。三者都不能替代真实视障用户效果实验。

Claim ceiling：`PUBLIC_REAL_DEVELOPMENT_MECHANICS_ONLY_NO_USER_PRODUCT_SAFETY_OR_NAVIGATION_EFFECT_CLAIM`。

实现：[`real_episode_pilot_v0`](../../../scripts/research/goal_copilot_bridge/real_episode_pilot_v0/README.md)。
