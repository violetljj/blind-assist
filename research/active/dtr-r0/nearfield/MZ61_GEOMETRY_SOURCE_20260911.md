# MZ61: acquire geometry-disjoint obstacle configurations

EXPLORE source experiment. The capability question is whether physically
different target dimensions and poses can supply useful native HEAD/BODY/BOTH
supervision beyond MZ55's repeated geometry. MZ60's source audit found all160
held awning native tensors already present among actually fitted examples.
The comparator is the immutable MZ55 source; this is a source-coverage test,
not a new model, sensor calibration or generalization result.

Freeze the existing [design](MZ61_GEOMETRY_SOURCE_DESIGN_20260911.md) and
spec-v2 manifest SHA256
`2a84f622d516ae7503571b3d42e7bf8f3c1ff47e3c57f27989f44ab8572aa2cb`.
Its 4096 cases comprise four families, four relations, two range anchors,
32 geometry recipes, two sites per configuration and two support contexts.
Roles are 2048 TRAIN_CANDIDATE,1024 CALIBRATION,1024 HELDOUT_GEOMETRY, assigned
by joint geometric recipe, never reassigned using captured outcomes.
The 1024 camera-relative target fingerprints exclude names and roles.
Native masks, rather than these metadata fingerprints, decide actual novelty.

Keep all case payloads, original material/mesh hashes, frozen MZ48 runtime,
native plugin, map hash, settled image quality and weak45-degree ToF recipe.
Only host-local map_file may differ with identical map SHA and all other
spec fields exactly equal. Reuse original owner-local warm DDC. Do not save
changes to the project or assets. New dataset/fullframe adapters bind the
manifest and retain geometry/roles solely in evaluator metadata. Native depth
and native labels never become predictor inputs; missing/invalid returns and
UNKNOWN remain explicit. This does not simulate calibrated VL53L8CX reliability.

The first64 cases are part of4096, split into original28-frame dense05 and
36-frame dense06 shards. Because both GPUs are currently free, initially
allocate dense05 to primary and dense06 to worker. Each shard has one owner
and one durable capture attempt, timeout2400s. Main4032 remains unstarted until
root canary admission. All main ownership changes must name unstarted shards
in a hashed allocation; recheck ownership before each launch. Primary runs
no UE collection concurrently with its model GPU job. Worker collection may
continue during a primary model job. Actual process handles decide occupancy.

Canary admission requires all64 source/render/floor/actor-geometry checks,
32 exact paired target dictionaries and native event masks/depths, at least
56 collapsed relation-intent matches, every family/relation represented
correctly, and both intended near and far event coverage in every family.
Root must view all64 RGB/native panels and bind the actual image/native hashes
with per-case observations. Preserve all actual four-query labels, including
extra range positives and intent mismatches. Failure stops the4032 expansion;
do not tune geometry/materials, replace failed cases or increase this budget.

Before expansion, report native nonempty full-frame cell-tensor equality
across distinct canary configurations and against MZ55 TRAIN, collapsing
deliberate support/site replicas. All-zero tensors are separate. At least one
positive configuration per family must differ from all MZ55 TRAIN positive
tensors; otherwise this source does not demonstrate useful new native geometry
for that family and expansion stops. This modest novelty check is additional
to source health, not a model-quality criterion. On completion, report the
full held-to-TRAIN tensor-collision table, per family/role/query coverage and
native pair integrity over4096. If most held positive configurations repeat
TRAIN, explicitly reject a claim of geometry-generalization evidence.

Preserve full raw RGB/native on its capture owner. Return unchanged RGB bytes,
observable packets, metadata and full-image uint8 counts[45,80,4] plus
valid_counts[45,80]. The80 predeclared native audit cases remain owner-local;
worker raw native arrays are not needed in controller training archives.
Verify compact member-byte roundtrip and bind all derived arrays to native
hashes. Do not delete source, evidence or checkpoints to reduce storage.
Report actual raw/package/allocated bytes and capture/verification/derivation/
transfer timings separately. Estimates1.76GB compact and7.58GB raw are planning
figures only. Reserve12GiB worker and6GiB primary for initial source work;
check actual free space before expansion.

Budget is at most4096 captures, zero model fitting and zero model prediction.
Source failure is NOT_EVALUABLE for model improvement. Successful source
admission supplies a new controlled Development dataset, with consumed sites,
synthetic object installation, reused mesh identities and limited family
diversity stated explicitly. A model experiment declares its own role use.

Mechanical adapter/transport repairs may use the identical already captured
source with preserved failure logs and hashes; they may not recapture or alter
scientific inputs. In finally paths verify owned UE/Zen process and port
release, retain diagnostic evidence and release task invocation/temporary
transfer resources. Root owns registration, source admission and delivery.
