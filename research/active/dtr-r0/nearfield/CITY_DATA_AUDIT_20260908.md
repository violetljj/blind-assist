# City 500-group data audit and split

This audits the completed 1,500-frame collection before training. No G13
checkpoint, frozen cohort, model definition or training script is changed.
The dataset remains one-map controlled Development, not blind test evidence.

## Findings

All 500 groups contain clear/center/right-clearance triplets. Native and group
QA reports pass. No severe near-black, near-white or majority-unknown flags
occur under the recorded QA thresholds; this does not establish universal
visibility or absence of occlusion. Exact PNG hashes contain no duplicate
clusters, but there are only 20 camera poses and many near-duplicate backgrounds.

| Label axis | BODY | HEAD |
| --- | ---: | ---: |
| Positive within the 3 m query | 270 | 30 |
| DANGER distance state | 18 | 0 |
| WARNING distance state | 338 | 36 |
| OBSERVE distance state | 114 | 14 |
| NO_VISIBLE_SUPPORT distance state | 1,030 | 1,450 |

The two axes use different ranges; WARNING is not a renaming of the 3 m binary
label. Joint binary labels are 1,230 neither, 240 BODY-only, 30 both and zero
HEAD-only. Every HEAD-positive/range-state example comes from the scaffold
asset. The dataset therefore cannot support HEAD DANGER learning/evaluation or
a claim of category-independent HEAD generalization. A constant negative
HEAD classifier would score 98% raw accuracy, so accuracy alone is unsuitable.

Among 500 center frames, 230 have no 3 m positive support; 200 of these still
have more distant support and 30 have neither body's visible support within
the distance query. The latter are scaffold poses 04/05/06 (yaw 72/90/108),
across five distances and both regions. They are review candidates, not
automatically mislabeled: an open frame may leave the queried corridor clear.
Frames 97/847 have only four positive HEAD pixels each, close to the three-pixel
label threshold; preserve and flag them rather than silently discarding them.
All clear and lateral variants have no positive support in this collection.
Primary inspection of frames 316/391/466 shows the scaffold sides with an open
central corridor, consistent with keeping the query-based labels rather than
forcing asset presence to mean collision. Frames 97/847 render the scaffold
normally and retain their four-pixel HEAD flags. This checks five flagged
representatives, not every possible occlusion or geometry edge case.

## Fixed Development partition

The partition is chosen by region identity, not optimized for label counts:

| Role | Region | Groups | Frames | 3 m BODY / HEAD positives |
| --- | --- | ---: | ---: | ---: |
| TRAIN | west_sidewalk | 250 | 750 | 135 / 15 |
| TEST | plaza | 250 | 750 | 135 / 15 |

All three variants, distances, assets and adjacent poses in one region stay
together. Minimum cross-partition camera XY distance is 135.09 m. No group,
camera pose or exact RGB hash crosses the partition. Each region's adjacent
poses are approximately one metre apart; random group splitting would leak
near-repeat backgrounds despite keeping triplets intact.

There is **no City validation partition**: the two small spatial clusters do
not supply three independent regions. Future fitting must predeclare a
separate validation source (for example the existing Willow validation set,
with its domain limitation), or collect a third region; never tune on this
TEST partition. The plaza has already been inspected for acquisition QA and
its aggregate labels audited, so TEST means Development holdout from fitting,
not blind or unseen-world confirmation. Geometry/asset/camera/group metadata
is audit/supervision-side only, not model input.

## Training contract check

`decoupled_model.py` returns two near logits and two 18x32 support maps from
144x256 full-frame RGB. WARNING/DANGER has no model head or loss. It exists
only in the City native-depth evaluator. Approaching remains UNKNOWN.

The old G13 `decoupled_train.py` requires exact G12 sources, fixed old sample
layouts and batch hashes; it must not be pointed at this three-variant dataset.
Old `grounding_verify.py` first isolates box/bar surfaces; City
`worlds_verify.py` uses all visible scene surfaces. This is an explicit
supervision expansion, not identical labels with new textures.

City support masks contain -1/0/1. Existing `whisker_model.support_bce` masks
whole unknown heads, not unknown pixels. A new City adapter must pool positive
support while retaining unknown cells and use a pixel-known loss; it must not
reinterpret -1 as background. Reuse the D network and RGB preprocessing, but
write a separate training manifest/entry and a new experiment directory.

The next viable fit is a separately specified binary visible-support diagnostic,
with fixed train-only sampling and independent validation. Distance-grade and
HEAD-generalization work first needs HEAD-only, HEAD DANGER and additional
high-obstacle shapes; preserve geometry-based labels when adding those examples.
No training is started by this audit, and no claim that these data improve D
has been made.

The subsequent [City data adapter](CITY_DATA_ADAPTER_20260908.md) implements
the independent cache, tri-valued pooling and pixel-known loss, and converts
all 1,500 frames on the worker. It preserves this split and does not add a
distance prediction head or start training.

## Reproduction

`tools/audit_city_collection.py --capture <capture> --output <fresh-output>`
validates source/report identities, triplets, row labels, region isolation,
exact-duplicate isolation and descriptive distributions. Add `--verify-payloads`
on the worker to hash every RGB/depth/support file against accepted records.
The operation is metadata/hash work on CPU, not a CUDA workload.

Primary metadata result:
`artifacts.local/nearfield/city-pcg-20260908/collection500-audit-v1/` contains
`audit.json` and `split.json`. Raw data remains on the worker. The manifest
records input hashes and the exact frame indices and paths in each partition.

The worker additionally rehashes all 4,500 raw RGB/depth/support payloads and
passes in a 3.087 s audit job. Its split exactly matches the primary split.
Worker audit, split, flagged sample images and release receipt are retained at
`artifacts.local/nearfield/city-pcg-20260908/worker-scale500-v1/collection500-audit-v1/`.
Task-owned audit processes/jobs are released; no UE or training was started.
