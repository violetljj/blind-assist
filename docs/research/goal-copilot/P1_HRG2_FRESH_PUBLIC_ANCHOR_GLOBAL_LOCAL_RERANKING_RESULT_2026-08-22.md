# P1-HRG2 fresh public-anchor global-local reranking result

日期：2026-08-22（Asia/Hong_Kong）

状态：`HRG0_FULL_2_OF_2_AT_K3 / HRG2_FULL_2_OF_2_AT_K3 / HRG2_NO_PAIRED_IMPROVEMENT / IDENTITY_NOT_AUTHORIZED_INSUFFICIENT_DENOMINATOR / DEFAULT_APP_UNCHANGED`

## 合法 public contract 与 observation yield

7 个真实产品任务先冻结为 `NAMED_BUILDING_ENTRANCE -> building entrance`，然后才解析公开的命名场所 OSM anchor、访问
Mapillary metadata/pixel 并建立 private truth。provider 只看到 public Goal Contract 与 current frame；具体 entrance node、
visibility 和 bbox 没有进入 provider input。一次 geocoder query 更正和一次 Mapillary transport envelope 缩窄都在 pixel/truth 前以
机械 amendment 固定；没有 goal replacement、truth 后补图、重采样或改框。

修复 acquisition 的静默单帧退化后，6/7 个 episode 取得 16 帧，最多每 episode 3 个空间分离视角；Rembrandt House 没有
geometric candidate。pre-provider adjudication 只有 2 帧 `VISIBLE`，分别是 Miffy Museum 和 Escher in Het Paleis，另外
14 帧为 `NOT_VISIBLE`。因此 episode observation yield 为 `2/7`，frame visibility yield 为 `2/16`。本轮不是零样本，允许按
冻结预算执行；但 primary denominator 只有 2，任何结论都必须保留小样本上限。

## 冻结的配对算法

HRG0 保持既有接口：YOLOE semantic support、Grounding DINO functional proposal、`1.5x` context region、
semantic-supported-first 排序与最终 `K=10`。

HRG2 在查看本批 pixel 前以 commit `ced7fd41859fcf44ca6c43d7de00ffe0677596ef` 冻结：处理 HRG0 全部 Top-10
coarse parent；每个 parent 用同一 Grounding DINO、同一 functional prompt 和同一阈值取得 Top-2 local proposal；映射回
full frame 后只按 local provider score 做全局排序，parent/local rank 仅作 tie-break；再做固定 IoU `0.50` class-agnostic NMS，
最终 `K=10`。没有 identity、named-instance matching、private truth access、prompt/threshold/model/pool sweep 或 retry。

## 一次性结果

Primary 是 2 个 visible frame 上 `IoU >= 0.30` 的 bounded candidate Recall@K：

| arm | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| HRG0 semantic-supported functional context | 1/2 | 2/2 | 2/2 | 2/2 |
| HRG2 global local-score reranking | 0/2 | 2/2 | 2/2 | 2/2 |

HRG0 中，Miffy Museum target 位于 rank 2、best IoU `0.3397`；Escher target 位于 rank 1、best IoU `0.6560`。
HRG2 将二者分别排到 rank 3（best IoU `0.4136`）和 rank 2（best IoU `0.6584`）。HRG2 提高了两个 case 的 best
localization IoU，但没有提高任何预注册 Recall@K，并把 Top-1 coverage 从 `1/2` 降到 `0/2`。

终态为：

- HRG0：`P1_HRG0_FULL_HIERARCHICAL_TARGET_AVAILABILITY_ON_FRESH_COHORT`
- HRG2：`P1_HRG2_FULL_GLOBAL_LOCAL_TARGET_AVAILABILITY_ON_FRESH_COHORT`
- paired verdict：`HRG2_DID_NOT_IMPROVE_HRG0_BOUNDED_AVAILABILITY_ON_TWO_VISIBLE_CASES`

## 决策边界

这两帧证明了一个窄的 mechanics 事实：合法 goal semantics 下，正确入口可以进入 frozen bounded pool，而且本批均在 Top-3。
它不证明跨场所 proposal coverage；`2/7` episode observation yield 和仅 2 个 primary case 使正式 Contrastive Identity
Verifier authorization 仍然不足。不能把 `2/2` 当成对七个 goal 的 `100%`，也不能把 14 个 `NOT_VISIBLE` 当成 detector
negative。

此外，本 cohort 的 Goal Contract 是 `SET_VALUED` 的“任一合法公共入口”，并没有一个必须保持的唯一物理实例。因此后继若继续，
应先扩大一个在 truth 前取得足够 visible episode 的 clean public-spatial observation cohort，并区分：

- `SET_VALUED` 入口任务需要的是 candidate-to-legal-goal verification；
- `UNIQUE` 物理实例任务才需要 contrastive instance identity verification。

当前不恢复 AMRM，不将这批已观察 outcome 的 bank 续作 verifier confirmation，不改 default App。

## Evidence identity

- formal run manifest SHA-256 `4d96ff4b46d69ca9597796bb68c7fb8bc3d66c167d63f8b59c92644b7e031119`
- HRG0 prediction SHA-256 `a7260e36d7cb9aab2fc9a79e739922d71020491e40b543d1df853ec1a2f3ec6a`
- HRG0 evaluation SHA-256 `be60bbfc439e2c6844de0a340dd4aa27b2fddf4367b8929e5e6bf12fb5cb2963`
- HRG2 prediction SHA-256 `6ec83e1a0b0776b6d6b6859ce6a5f514937a1a72150831112c0c7d42b48f5c6e`
- HRG2 evaluation SHA-256 `51d0435af64291c495c3d77c448ee88051568a4ccf43e0fe94b103ddd6cbff00`

Claim ceiling：`FRESH_TWO_VISIBLE_FRAME_HRG0_TO_HRG2_PROPOSAL_AVAILABILITY_ONLY_NO_IDENTITY_GENERALIZATION_PRODUCT_OR_SAFETY_CLAIM`。
