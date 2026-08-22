# BlindAssist Public-real Episode Mining V0 result

状态：`IMPLEMENTED / FRESH_PUBLIC_METADATA_COHORT_READY_8_EPISODES_89_OBSERVATIONS / CONSUMED_BASELINE_SMOKE_RUN / MANUAL_CAPTURE_NOT_BLOCKING`

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

## 当前自动 successor

下一步只对这 8 个已冻结 episode 的 89 个 observation 下载像素，并按冻结 provider/teacher 路径生成 private truth；
只有 native GT、地图/轨迹或独立 teacher 一致性足够时才填入 private evaluator truth；其余保留 `SET_VALUED`、
`AMBIGUOUS` 或 `UNKNOWN`。不得用 post-outcome resampling、人工唯一门、threshold/model/provider sweep rescue denominator。

ADT 只承担 depth/bearing/visibility/occlusion/temporal calibration；Ego4D 只承担真实第一视角取景、模糊、遮挡和 out-of-
frame 分布；Habitat 只承担显式 endpoint 的闭环 mechanics。三者都不能替代真实视障用户效果实验。

Claim ceiling：`PUBLIC_REAL_DEVELOPMENT_MECHANICS_ONLY_NO_USER_PRODUCT_SAFETY_OR_NAVIGATION_EFFECT_CLAIM`。

实现：[`real_episode_pilot_v0`](../../../scripts/research/goal_copilot_bridge/real_episode_pilot_v0/README.md)。
