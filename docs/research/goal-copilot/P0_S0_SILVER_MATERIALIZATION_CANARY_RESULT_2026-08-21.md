# P0-S0 GoalGrounding-Silver Materialization Canary Result

状态：`COMPLETE / P0_S0_SOURCE_OR_LICENSE_BLOCKED / ZERO_EPISODES / NO_MODEL_RUN / NO_SCIENTIFIC_VERDICT`

协议：`BA-P0-GOAL-GROUNDING-SILVER-V1`

上游 identity：`project-terraforma/mapillary-entrances@3d3b85244b1a1ec2ba05a997d56d000936cc554a`

执行模式：`CANARY_LITE / NON_FRESH_SOURCE_SLICE / ZERO_MODEL`

## 问题与 verdict

本轮只问：冻结 Silver 协议能否在真实公开数据上机械物化合法、可审计、无泄漏的 P0 episode？

终态为：

> `P0_S0_SOURCE_OR_LICENSE_BLOCKED`

两个独立阻塞在任何 episode admission 前成立：

1. `MAPILLARY_ACCESS_TOKEN_MISSING`：冻结审计与上游 commit 都要求 Mapillary access token；本机没有配置，
   本轮没有绕过访问控制、下载 RGB 或伪造 image/sequence metadata。
2. `MANDATORY_CANDIDATE_GENERATOR_NOT_AUTHORIZED`：冻结 episode schema 和 `MAP_ANCHORED` 规则需要真实
   image-space bbox、generator provenance、ray-wall geometry 与 candidate-anchor residual；本轮明确禁止 detector/
   teacher 调用。Overture/OSM crosswalk、OSM entrance 或 5 m cluster 本身不能替代 visual candidate。

因此本轮物化 episode `0`、`SILVER_A_PRIMARY=0`、evaluator episode dry-run `0`。这不是模型失败，也不产生
grounding performance claim。

## 真实 source slice

唯一 bbox：Ghent `3.7215,51.0505,3.7295,51.0565`；角色为 `NON_FRESH_CANARY_SOURCE_SLICE`。

| Source | Frozen identity | Records | SHA-256 |
|---|---|---:|---|
| Overture Buildings | `2026-08-19.0` | 1,102 | `537507e3546c7f41b80ec12c4476a209555f6cc8ae5b4cccc600efc046bcfe77` |
| Overture Places | `2026-08-19.0` | 1,043 | `59768fd3f5246a05fe96ee8833c67bb871759dc209d10fbc36e7129697faa270` |
| OSM map API | exact bounded XML payload | 1,078 building ways / 32 entrance nodes | `6d29a55e02b781d23dad97d184b50fb7a201b56087a36f172734cf2a0ac774f4` |

Source normalization preserved per-record Overture source datasets/licenses and OSM attribution/license. Observed
Overture lineages were ODbL-1.0 building sources and CC0-1.0, Apache-2.0, CDLA-Permissive-2.0 place sources; exact
counts live in the source report. Mapillary license/provenance completeness is not claimed because that source was not
accessed.

Metric crosswalk mechanics produced 630 unique place-inside-building candidates and 20 OSM-entrance-to-Overture-building
candidates: 14 used an explicit Overture OpenStreetMap source-record bridge plus OSM way topology, and 6 used the frozen
`<=3 m` boundary / `>=5 m` second-best margin. They remain `CANDIDATE_ONLY`; none is an episode or label.

Ignored local evidence root:

`artifacts.local/evidence/p0-s0/2026-08-21-ghent-source-slice/`

Authoritative semantic hashes:

- `source-report.json`: `bbd1af722488727f9f44e6adf59b27dd09ca6d48b928b658bb5e147166b24436`;
- `terminal.json`: `c0c7b447c1f7d546306ec8b6639559306fc08f84c2cb0c7cc322cb71f201d956`.

The source report was regenerated from the same bytes and matched exactly.

## 实现与验证

Stable implementation:

- `scripts/research/goal_copilot_bridge/p0_s0_materialization/source_slice.py`;
- `scripts/research/goal_copilot_bridge/p0_s0_materialization/materializer.py`.

The materializer limits one canary bundle to at most 20 normalized records, separates source acquisition from admission,
uses metric distance rather than raw longitude/latitude degrees, preserves provenance/license failures separately, requires
every hard gate, and audits provider-visible input for exact OSM entrance, target-building crosswalk and evaluator-label
leakage. `MULTIVIEW_VERIFIED` requires two real frames, 3–30 m camera baseline, same building/anchor, `<=3 m` anchor
residual and 10–120 degree ray angle; a cluster or repeated candidate cannot set it. A record that passes map and geometry
but lacks multiview coverage remains `SILVER_B_MAP_GEOMETRY`; it is not silently promoted or collapsed into rejection.

Verification result:

- S0 materializer/source tests: `12/12 PASS`;
- existing P0 evaluator mechanics: `12/12 PASS`;
- combined focused unit tests: `24/24 PASS`;
- frozen episode and label-quality schemas: `2/2` valid Draft 2020-12 schemas;
- provider-input leaks in the terminal: `0`.

The 12 evaluator cases are mechanics regression only. Because there was no legal real episode, the evaluator did not receive
a fabricated source-derived episode.

## Claim ceiling 与唯一解阻条件

本轮最多证明 source normalization、crosswalk candidate、hard-gate admission、multiview 独立性检查、泄漏审计、
hash 与 deterministic replay mechanics 可运行，并定位两个真实前置阻塞。它不证明 silver 可覆盖、任何
`SILVER_A_PRIMARY` 可产生、任何模型能力、真实用户效果、导航、安全、生产或默认 App 可用性。

新的真实 S0 只在以下两项同时满足后另建 run：

1. 用户提供合规、可用于本研究 slice 的 Mapillary access token；
2. 另行冻结并授权一个 image-space candidate generator（含 model/config/weights/input lineage 与预算），或提供
   与冻结 schema 等价、许可完整的预计算 visual candidate source。

不得恢复或改写本终态；不得为“让数据出来”把地图 anchor、cluster 或合成 mechanics fixture 升格为真值。
