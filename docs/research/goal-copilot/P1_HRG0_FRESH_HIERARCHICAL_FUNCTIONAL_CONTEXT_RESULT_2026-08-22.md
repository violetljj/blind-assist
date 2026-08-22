# P1-HRG0 fresh hierarchical functional-context result

日期：2026-08-22（Asia/Hong_Kong）

状态：`FRESH_SINGLE_VISIBLE_CASE_TARGET_AVAILABLE_AT_RANK1 / VISIBILITY_YIELD_INSUFFICIENT_FOR_GENERALIZATION / IDENTITY_NOT_EVALUATED / DEFAULT_APP_UNCHANGED`

## 冻结算法

HRG0 在 fresh pixel 访问前以 commit `837424c5dac8826362bfba74e517ba1c0278e33e` 冻结，只组合两个已经固定的
proposal interface：

1. `YOLOE-26n-seg` 用合法 Goal Contract 全局映射出的 `building entrance` 生成 semantic support regions；
2. Grounding DINO Tiny 用既有 frozen functional prompt
   `door . doorway . entrance . building entrance . storefront entrance . gate .` 生成 functional boxes；
3. functional box 中心落入 semantic Top-10 region 时优先，随后按原 functional provider rank 排序；
4. 每个 functional box 固定扩展 `1.5x` 形成 entrance + access-context region，最终 bounded pool `K=10`。

没有 identity selection、private truth access、prompt/threshold/model/scale sweep 或 outcome 后重跑。`1.5x` 来源于上一
consumed-development mechanism diagnostic，因此只能在本批 fresh 数据上接受或拒绝，不能继续调。

## Fresh Goal Contract 与 acquisition

先冻结 7 个未查看过像素或 truth 的 Dutch city-hall product goals，全部使用
`NAMED_BUILDING_ENTRANCE -> building entrance`。英文 city-hall 地址在第一个 Nominatim 查询返回零结果，且此时尚未访问
Mapillary metadata、pixel、truth 或 provider；保留原 plan 和 failure boundary 后，只把 7 个同一地点的 geocoder query
机械改为 Dutch `Stadhuis ...`。goal roster、OSM entrance、Mapillary 几何选择和 no-replacement 规则均未改变。

最终 acquisition：

- 7 个 goal；
- 4 个 materialized frame；
- 2 个 `NO_OSM_ENTRANCE_NO_REPLACEMENT`；
- 1 个 `NO_GEOMETRIC_CANDIDATE_NO_REPLACEMENT`；
- pre-provider truth：`1 VISIBLE / 3 NOT_VISIBLE`。

唯一 visible case 是 Haarlem City Hall。低 visibility yield 是本批证据的一部分，不能通过补图、换地点或把 facade
强行标成 entrance 来扩大分母。

## 一次性结果

Primary 为 visible `SET_VALUED` case 上 `IoU >= 0.30` 的 Recall@1/3/5/10：

| endpoint | 结果 |
|---|---:|
| evaluable | 1 |
| Recall@1 | 1/1 |
| Recall@3 | 1/1 |
| Recall@5 | 1/1 |
| Recall@10 | 1/1 |
| Haarlem first correct rank | 1 |
| Haarlem best IoU | 0.8955 |
| terminal | `P1_HRG0_FULL_HIERARCHICAL_TARGET_AVAILABILITY_ON_FRESH_COHORT` |

Haarlem rank-1 candidate 是 Grounding DINO functional rank 1，经 YOLOE semantic rank 2 支持并固定扩展 1.5x 后得到，
与 private legal target box 的 IoU 为 `0.895515`。这建立了一个 fresh、未见 city-hall 样本上的 hierarchical proposal
availability observation。

## 决策边界

本结果不能解释成 stable coverage 或跨场景 confirmation：primary denominator 只有 1。它也不是 PA3 semantic-only
成功，更没有验证同类 distractor 下的 instance identity。因而 Contrastive Identity Verifier、AMRM 恢复和 App 集成
仍不授权。

当前主要缺口从“visible case 上是否可能形成高质量 candidate”转成“如何在不看 truth 的前提下取得足够多
target-visible current observations”。下一合法算法阶段应是 fresh、geometry-frozen multi-view observation-yield cohort，
将 acquisition/control 与 HRG0 proposal 分开计分；只有它建立足够的 visible proposal coverage，才构造独立
contrastive verifier cohort。

## Evidence identity

- formal run manifest SHA-256 `455c3121a88f70970a5f73063c62b286ee82fda27f21474891f20702d921d04e`
- prediction SHA-256 `926456bae02da446072bf5e9093036c168e5ceb33710f71ff8b5f14d3c38b3ba`
- evaluation SHA-256 `79d85fad005e97b3f1f30549c779caf84bfef06c491eaf8681d8f09ef0e58f42`

Claim ceiling：`FRESH_SINGLE_VISIBLE_CASE_HRG0_PROPOSAL_AVAILABILITY_OBSERVATION_ONLY_NO_GENERALIZATION_IDENTITY_PRODUCT_OR_SAFETY_CLAIM`。
