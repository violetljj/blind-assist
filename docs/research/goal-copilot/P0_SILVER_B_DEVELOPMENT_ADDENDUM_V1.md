# P0 Silver-B Development Addendum V1

状态：`ACTIVE / PROSPECTIVE_DEVELOPMENT_AUTHORITY / REVIEWED_COHORT_47_GOALS_43_FRAMES / SINGLE_BRAIN_MECHANICS_RUN / NO_SCIENTIFIC_VERDICT`

本 addendum 不修改 `BA-P0-GOAL-GROUNDING-SILVER-V1`、Grounding DINO、P0-S0 materializer 或
P0-S1 identity rule，也不回写任何历史 receipt。它只允许从本版本开始，把已经通过 provenance、license、
unique building crosswalk、unique entrance anchor、ray-wall geometry 与 conflict/lineage gates 的候选，以
`SILVER_B_MAP_GEOMETRY` 权限用于 P0 Development。

## Silver-B 语义与用途

Silver-B 只表示：

> bbox 是与目标 building wall 和 OSM entrance anchor 几何相容的入口候选。

它不表示 exact same physical entrance identity 已建立。父结果若为 `SILVER_A_PRIMARY`，可向下复用为 B 级
Development 输入，但父 class 与 receipt 保持不变；`REJECT_AMBIGUOUS` 不得导出。

每个 Silver-B Development episode 都必须携带 goal-reference truth。当前 B 证据不能建立 exact physical
referent 集，因此第一批统一为 `resolution=AMBIGUOUS / valid_target_instance_ids=[]`，而不是从 weak bbox 中
擅自指定唯一答案。未来只有独立 evidence 真正建立一个或多个合法物理目标时，才可写成 `UNIQUE` 或
`SET_VALUED`；set-valued 时任一合法目标均不得判错。

允许用途：管线开发/调试、map+geometry conditioned candidate yield、弱候选排序原型、failure analysis 与
abstention 设计。禁止把它用于 exact entrance truth、Grounding DINO recall/precision、exact Brain selection
accuracy、E2E grounding accuracy，或宣称等同 Silver-A/human truth。尤其是 detector 参与了候选生成，因此
仅凭这批数据无法为 detector miss 建立独立 denominator；当前只能报告 yield，不能报告 recall。

## 第一批真实导出

对已消费的 20-image anchor-aware canary 做前瞻性 lower-authority 导出，得到：

- `4` 个 frame-level Silver-B Development episodes；
- `4` 个 map+geometry weak-positive candidates；
- data role：`CONSUMED_DEVELOPMENT`；
- parent nominal class 仍为 `SILVER_A_PRIMARY`；
- exact entrance identity 全部为 `NOT_ESTABLISHED`；
- report SHA-256：`5664be1259400cf342f06f08c38d52f4b69e18407c26420ea6ac90bc4654481a`。

Ignored artifact：
`artifacts.local/evidence/p0-s0/2026-08-21-grounding-dino-tiny-s0-r1-anchor-aware/silver-b-development-cohort-v1.json`。

这批 4 episodes 验证了导出 mechanics 和边界，不足以形成 baseline 或性能结论。下一步直接扩大普通
anchor-facing source coverage，积累几十至约 100 个 Silver-B Development episodes；天然满足冻结 P0-S1 的样本
可另保留为 A subset，但不再把 A coverage 当作 P0 Development 的前置 blocker。

## 扩展 cohort 与单 Brain development 结果

后续按本 addendum 的相同权限扩展到 47 个 goal episodes / 43 个 unique frames，并经独立 score-blind
frame review 得到 `UNIQUE=12 / SET_VALUED=4 / AMBIGUOUS=31`。同帧派生 referring expressions 不增加
unique-frame count；resolved manual regions 不从 proposal score 推导。随后用户指定
`gpt-5.6-terra / medium` 运行一次 conditioned candidate-selection mechanics。结果和失败分解见
[`P0 Silver-B Brain Development Result`](P0_SILVER_B_BRAIN_DEVELOPMENT_RESULT_2026-08-21.md)。

这不是对 addendum 的协议修改，也不提升其 authority：claim ceiling 仍为
`SILVER_B_DEVELOPMENT_ONLY_NO_EXACT_BRAIN_OR_END_TO_END_ACCURACY`。
