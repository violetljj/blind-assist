# BODY-QUERY new-scene source pilot

Completed source outcome: [120-frame results](BODY_QUERY_SCENES_RESULTS_20260909.md).

2026-09-09. Authorized source collection, no training or model scoring. This is
separate from the consumed320-frame BODY-QUERY experiment, the pending336-frame
cross-region contract and the earlier256+256+96 diversity candidate.

Collect120 frames: three previously scouted BigCity pedestrian settings, each
with40 frames (eight parent groups). Four supported fixture families each have
one near and one far group, with target distance centers1.0m and2.4m. Every group
contains CLEAR/BODY_ONLY/HEAD_ONLY/BOTH; the two crossbar groups also contain
LOW/ABOVE/LATERAL_OUT/FAR_OUT. Keep the same local fixtures across regions to
enable matched-background transfer diagnostics. They reuse source-v3 fixture
designs from the prior320-frame collection; these are new placements, not new
object assets. This pilot does not measure increased training diversity.

Roles chosen before new capture/model access: dense_candidate_01 TRAIN,
dense_candidate_07 DEV, dense_candidate_02 EVAL. Their source-only scout specs
previously used the word TRAIN; no model outcome selected these poses. They
are reconnaissance-exposed Development sources. Store exact camera/floor/yaw,
source spec/hash and role in the immutable placements manifest. Shared visible
backgrounds and assets remain possible; do not claim strict background isolation,
new cities, natural scenes or protected blind TEST. One region per split and
one pose per region are a small pilot, not many independent scene trials.

Fixed calibration: optical height1.70m, pitch/roll0,640x360,HFOV100 degrees.
Preserve original depth, RGB, source specifications, native floor probes,
capture/runtime/plugin hashes, render-resource health and process-release logs.
Native depth supplies BODY/HEAD support, twelve query raw/capped visible counts,
exact-ray ownership/known masks and local membership/known fractions. UNKNOWN
is never FREE. Counted query evidence must reconstruct native near labels.

Admission requires complete capture, unchanged map/project inputs, no known
missing-render-resource signature, successful native verifier, and all intended
relations within each complete parent group. Report floor-height discrepancies
and actual near/far positive-group coverage; do not count merely intended HEAD
positives. Retain all failed or mismatched groups with reasons. Repairs create
fresh source/output versions before model access; never alter saved results.
Inspect source RGB for obvious obstruction/support issues before calling the
pilot usable. Training and model-based selection are outside this collection.

The source pilot completes when the captured groups, native label coverage,
explicit rejected groups and resource-release receipts are delivered. Expansion
size is decided from source acceptance, not automatically started here.
