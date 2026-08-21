# P0 GoalGrounding-Silver Protocol V1

状态：`DESIGN_FROZEN / FEASIBLE_WITH_REQUIRED_CONSTRAINTS / ZERO_MANUAL_ANNOTATION / NO_COHORT / NO_MODEL_RUN / NO_BASELINE / NO_SKY / DEFAULT_APP_UNCHANGED`

协议 ID：`BA-P0-GOAL-GROUNDING-SILVER-V1`

## 1. 冻结问题与 verdict

本协议回答一个问题：能否在零人工标注条件下，用真实 Mapillary 街景、独立地图事实、几何关系、
多视角一致性与可选 teacher 共识，构造可审计的 P0 Goal Grounding silver benchmark？

冻结 verdict：`FEASIBLE_WITH_REQUIRED_CONSTRAINTS`。

这一路线可构造 **map-anchored real-world silver cohort**，用于 P0 的候选覆盖、目标建筑识别、
`entrance_of(target_building)` 排序、hard-negative 错锁与 abstention 评价。它不能构造 human gold，不能
测得 silver label 的真实误差率，也不能单独支持真实用户、导航完成、入口可达性或安全主张。

这里的“零人工标注”仅指不新增 BlindAssist 专用人工框选/关系标注；OSM 社区编辑、Overture 上游整理和
原模型训练数据可能包含人工贡献，不能声称整个数据谱系无人参与。

## 2. 当前授权边界

本轮只冻结数据协议、schema、证据层级、admission、许可证边界、科学 claim ceiling 和下一步最小实现
slice。它补充现有 `BA-P0-NAMED-BUILDING-ENTRANCE-GROUNDING-V1` mechanics contract，不覆盖或修改
该并行 WIP。

本轮禁止：

- 下载、采集或物化 Mapillary、Overture、OSM cohort；
- 调用 YOLO、Florence、OCR、GPT、Qwen 或其他模型；
- 训练、调参、运行 baseline、打开或消费 fresh cohort；
- 调用 Sky，修改 Android/default App，或开始 P1/P2/P3；
- 把自动标签称为 gold truth；
- 在本协议之外放宽 admission 或冲突处理规则。

## 3. 数据链与权威边界

```text
Overture Place identity ──┐
                         ├─ target POI -> target building crosswalk
Overture Building ───────┘
          │
OSM entrance node ───────┼─ independent map anchor
          │              │
Mapillary image/pose ────┼─ real RGB observation
          │              │
entrance detector ───────┼─ visual candidate only
          │              │
ray -> building wall ────┼─ derived geometry support only
          │              │
multiview/OCR/VLM ───────┴─ corroboration, never gold
```

### 3.1 Source facts

- Mapillary image pixels and image metadata are source observations. Position, heading and computed metadata carry
  measurement uncertainty; they are not scene truth.
- Overture Place is a POI identity prior. It does not by itself prove which footprint or entrance belongs to the POI.
- Overture Building is a conflated footprint. It may be roofprint/ML-derived and is not automatically the visible facade.
- OSM `entrance=*` is an independent community map assertion. A node topologically attached to the source OSM building
  outline is stronger than a nearest-neighbour match, but neither proves current visibility or image-space bbox.
- `mapillary-entrances` detector boxes, ray hits, snapped coordinates, building assignments, 5 m clusters and Place
  associations are predictions or derived results. None is ground truth.

### 3.2 Frozen upstream identity

The source audit is frozen against `project-terraforma/mapillary-entrances` commit
`3d3b85244b1a1ec2ba05a997d56d000936cc554a` (2026-01-15). A future implementation must pin the commit and
hash every vendored or executed source/model artifact; later upstream changes require a protocol revision or explicit
compatibility audit.

## 4. Episode unit and independence

The scientific unit is a `natural_episode`, not an image, crop, bbox, frame or teacher call. An episode is one goal,
one target POI/building, one bounded physical encounter, one Mapillary sequence-group and its admitted observation window.

Splits and denominators must be disjoint by:

1. `parent_capture_id` and Mapillary `sequence_id`;
2. target building and OSM entrance ancestry;
3. near-duplicate visual cluster;
4. geographic block declared before outcome access;
5. teacher/model/config lineage where independence is claimed.

Multiple frames, crops, entrances or teachers from one parent episode remain correlated observations, not additional
independent episodes.

## 5. Target identity and spatial matching

All distances use a locally appropriate metric CRS. Raw longitude/latitude degree distances are forbidden.

### 5.1 POI -> building crosswalk

The crosswalk is admitted in descending order:

1. `SOURCE_NATIVE`: Overture source provenance or GERS/source bridge directly identifies the corresponding source
   building record, and the POI is inside or explicitly associated with that footprint.
2. `CONTAINED_UNIQUE`: Place point lies inside exactly one eligible building footprint and no conflicting named POI or
   building association exists.
3. `SPATIAL_UNIQUE`: Place point is outside footprints but the nearest eligible building boundary is within `10 m`,
   the second-best boundary is at least `5 m` farther, and name/category/address evidence has no conflict.

Otherwise the target building crosswalk is `AMBIGUOUS` and the episode is rejected from scientific subsets. Distance-only
association never overrides a source-native conflict.

### 5.2 OSM entrance -> building anchor

An OSM entrance anchor is admitted as:

- `TOPOLOGICAL`: the entrance node is a member of the source OSM building way or a valid building/site relation member;
- `SOURCE_CROSSWALKED`: the Overture building has a unique source-record crosswalk to that OSM building and the node is
  on its outline;
- `SPATIAL_CROSSWALKED`: only when no source-native mapping exists, the node-to-boundary distance is at most `3 m`, the
  nearest competing building boundary is at least `5 m` farther, and topology/name/address evidence has no conflict.

`entrance=main` means mapped primary entrance type, not public access, accessibility, opening status or current usability.
`entrance=yes` supports generic entrance existence only. `entrance=service`, `secondary`, `emergency`, `exit`, `no`,
`access=private/no`, or contradictory `routing:entrance` values must retain their exact semantics.

### 5.3 Visual candidate -> map anchor

A candidate can be `MAP_ANCHORED` only when all hold:

- detector candidate has an image-space bbox and full model/config provenance;
- camera location and heading are present, valid and bound to the source image;
- the candidate ray intersects the same admitted building polygon within `60 m`;
- snapped candidate coordinate is within `3 m` of the admitted OSM entrance node;
- no other admissible entrance anchor is equally close within a `2 m` ambiguity margin;
- candidate class/type does not contradict the OSM entrance semantics used by the goal.

Thresholds are protocol constants, not estimates of sensor accuracy. The pre-materialization canary must report sensitivity
at candidate-anchor radii `2/3/5 m`; changing the primary `3 m` rule requires V2.

## 6. Silver evidence hierarchy

Evidence labels are orthogonal flags; they must not be collapsed into a single undocumented confidence score.

| Flag | Required evidence | Does not establish |
|---|---|---|
| `MAP_ANCHORED` | admitted OSM entrance, target-building crosswalk and candidate-anchor match | visibility, bbox correctness, accessibility |
| `GEOMETRY_VERIFIED` | valid pose, ray-wall intersection, same building, range and ambiguity gates | that detector box is a real door |
| `MULTIVIEW_VERIFIED` | at least two source images with distinct camera positions, consistent same-anchor localization | independent gold truth |
| `VLM_SUPPORTED` | semantic VLM supports target/relation with structured evidence and no conflict | map or geometry truth |
| `TEACHER_CONSENSUS` | required independent teacher roles agree under frozen policy | independence when models/data lineage overlap |
| `AMBIGUOUS` | two or more admissible interpretations or unresolved source conflict | a negative label |
| `ABSTAIN` | insufficient, unavailable, stale, invalid or out-of-scope evidence | target absence unless geometry proves out-of-view |

### 6.1 Multiview rule

`MULTIVIEW_VERIFIED` requires at least two admitted images whose camera centers differ by `3–30 m`; each independently
produces a valid ray-wall candidate for the same building and anchor; estimated anchor points are within `3 m`; the
smallest ray intersection angle is `10–120 degrees`; and no view supports another entrance within the ambiguity margin.
Frames created by slicing one panorama do not count as independent views. Same-sequence observations may verify geometry,
but the episode denominator remains one. A second capture sequence is recorded as stronger provenance, not another label.

## 7. Teacher committee (design only)

| Role | Frozen responsibility | Required output | Forbidden authority |
|---|---|---|---|
| Geometry teacher | pose/FOV/ray/building/anchor consistency | metric residuals, building ID, wall segment, uncertainty | door or semantic truth |
| Entrance vision teacher | propose door/entrance bbox or mask | candidate, score, class, model hash | target-building relation truth |
| OCR/signage teacher | transcribe and localize text/logo | text, region, normalized match, uncertainty | entrance geometry truth |
| Semantic VLM teacher | judge facade/target/entrance relation | structured relation, cited regions, confidence, abstain | sole label authority |
| Optional independent VLM | mechanism/provider-diverse semantic check | same structured contract | required pseudo-independence |

Committee independence must be declared from provider, architecture, weights, training-data family, prompt, candidate source
and input transformation. Two endpoints backed by the same model family are correlated votes. A model that generated a
candidate and judged its relation has one provenance lineage, not two independent teachers.

## 8. Circular-evaluation prohibition

1. A teacher used to create a label cannot serve as the independent truth or adjudicator for that label.
2. `mapillary-entrances` output cannot be compared against itself as truth.
3. Candidate generation, relation judgement and final evaluated system output must record complete provenance.
4. Teacher agreement produces silver support only.
5. Any evaluated system sharing a teacher model, weights, embeddings, prompts or cached outputs with cohort construction
   must be reported as `TEACHER_LINEAGE_OVERLAP`; it is ineligible for the primary independent-evaluation subset.
6. Evaluator code may read frozen episode truth; providers and candidate systems may not.
7. Threshold selection, conflict resolution and exclusions are frozen before baseline outcomes.
8. Exact OSM entrance nodes、target-building crosswalk、visible-building truth 与 silver labels 都是 mining/evaluator
   only；系统最多接收协议允许的 coarse POI prior，不能接收答案坐标或 source IDs。

## 9. Label quality and admission

The normative quality schema is `p0_goal_grounding_label_quality_schema.json`.

### `SILVER_A_PRIMARY`

Required: valid licenses/provenance, unique target-building crosswalk, `MAP_ANCHORED`, `GEOMETRY_VERIFIED`,
`MULTIVIEW_VERIFIED`, exact entrance semantics for the goal, no source/teacher conflict, and no evaluated-system lineage
overlap. VLM support is optional and cannot repair a failed map/geometry gate.

### `SILVER_B_MAP_GEOMETRY`

Required: valid licenses/provenance, unique target-building crosswalk, `MAP_ANCHORED`, `GEOMETRY_VERIFIED`, no conflict.
Multiview is unavailable or fails only for coverage, not contradiction. Eligible for development and a separately reported
secondary silver subset, not the primary subset.

### `SILVER_C_TEACHER_SUPPORTED`

Geometry and multiple teacher roles agree, but an independent OSM entrance anchor or multiview proof is absent. Training,
diagnostic and ablation use only; never primary evaluator truth.

### `REJECT_AMBIGUOUS`

Any source conflict, non-unique crosswalk, near-tied anchor, teacher contradiction, unknown license, invalid pose,
unverifiable visibility, or schema/provenance failure. Rejected samples remain in the rejection ledger and are never
manually patched into the cohort.

## 10. Automatic hard negatives

All negatives preserve a positive statement of what is known; missing evidence is never relabeled as absence.

1. `WRONG_BUILDING_ENTRANCE`: candidate is map/geometry anchored to a visible neighbouring building, while the target
   building identity is independently fixed.
2. `WRONG_ENTRANCE_TYPE`: same target building, but an anchor explicitly tagged `service`, `secondary`, `emergency`,
   `exit` or another non-goal type competes with the admitted `main` anchor.
3. `TARGET_OUT_OF_VIEW`: the complete target-building/anchor angular interval plus uncertainty lies outside the frozen
   camera FOV. This is an abstention episode, not a negative door label.
4. `TARGET_BUILDING_VISIBLE_ENTRANCE_OUT_OF_VIEW`: target facade/building intersects the FOV, but the admitted entrance
   anchor plus uncertainty is outside it. Map geometry alone only proves `IN_FOV`, not actual visibility; without an
   independent visibility source this case is development-only or `UNKNOWN`. Occlusion is never inferred from no detection.
5. `MULTIPLE_SIMILAR_DOORS`: two or more valid visual candidates on the same or adjacent facades are within the score
   margin; only a map-anchored candidate may be positive, otherwise `AMBIGUOUS`.
6. `SIGNAGE_SPATIAL_CONFLICT`: signage/OCR supports the target but its region/facade relation conflicts with the mapped
   entrance/building. Conflict forces rejection; semantic confidence cannot override geometry.

If OSM maps multiple distinct `entrance=main` nodes for one target, every independently admitted main anchor is an allowed
positive for a `MAIN` goal. The protocol must not choose one post hoc or relabel the other main entrance as negative.

## 11. Cohort roles and freezing

Future materialization must create immutable `DEV`, `REGRESSION`, and `FRESH` manifests before model outcomes. Geographic,
building, entrance, capture-sequence and near-duplicate ancestry must be disjoint. `FRESH` is sealed and unavailable to
teachers, prompt tuning, threshold selection and baseline development. A failed or partially consumed fresh root is not
replaced or resumed unless its predeclared recovery contract permits it.

Before any formal run, freeze source snapshot/release IDs, API fields, license/attribution snapshot, upstream commit/model
hashes, geographic query, all thresholds, exclusion counts, split ancestry, teacher budgets, retry/in-doubt semantics and
evaluator hash.

## 12. Claim ceiling

The primary admissible claim form is:

> On a zero-manual, real-Mapillary, map-anchored silver subset constructed from frozen OSM/Overture facts, geometry and
> multiview consistency, system X achieved the reported P0 grounding metrics under the declared lineage exclusions.

The cohort may support: candidate coverage; `entrance_of` ranking; wrong-building/wrong-entrance error; abstention on
geometry-proven out-of-view cases; and stratified failure anatomy within admitted coverage.

It may not support: human-gold accuracy; unlabeled-population accuracy; label error/calibration; global representativeness;
accessibility, public access, opening status or passability; closed-loop navigation/completion; user benefit; safety;
production readiness; or superiority on independent human truth.

## 13. Next authorized slice if separately approved

The minimum implementation slice is a **one-area, zero-model, metadata-only mechanics canary**:

1. pin one Overture release, one OSM snapshot and the audited source commit;
2. query only metadata for a bounded non-fresh area;
3. construct POI-building and OSM entrance-building crosswalks;
4. validate schema, CRS, distance/ambiguity rules, license ledger and rejection reasons;
5. emit no images, detector candidates, teacher calls or scientific verdict;
6. stop for review before any RGB download or model execution.

This slice requires new authorization because it materializes source data.
