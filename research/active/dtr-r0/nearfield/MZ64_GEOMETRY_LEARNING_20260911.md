# MZ64: matched adaptation to richer obstacle geometry

EXPLORE. MZ63 found limited native BODY_NEAR transfer but no held awning
native addition. MZ61 is now a complete 4096-frame controlled source whose
geometry and native label coverage are broader than the prior fixed recipes.
The question is whether training on its designated TRAIN rows improves the
existing readout under weak ToF, beyond further adaptation to the old source.
This tests an actual learning increment, not another fixed-model transfer.

Use two identical copies of the completed MZ62 CONTROL checkpoint, SHA
`b7106449e4246b7bd7933659cbf0dd815691072c03a8b522633b7aee9a456d2e`.
Keep AnchorQuery GLOBAL, frozen RGB encoder and normalization, 11020 trainable
readout parameters, original local/query loss, Adam lr0.001, seed151 and reset
optimizer state in both arms. MZ60 positive-witness pooling and MZ62 profile
assignment remain negative controls in their original scopes; this does not
claim either recipe is rescued or that model capacity is exhausted.

Both arms perform exactly1536 updates. Every update contains four shared MZ48
native rows, four target-source rows, and eight identical OLD_NEG replay rows
with the original query replay and coefficient0.25. Shared replay is the original
600-step MZ59 sequence cycled to1536; profiles remain IDEAL/MERGE_CLOSE/DROP_CLOSE
in step modulo3 order. No ALL_INVALID training is introduced in this source
comparison. No label or observed score is used to select exposure.

CONTROL uses only MZ55 TRAIN1600 as its target source. Each profile presents all
1600 rows once plus448 extra rows. Seed164 shuffles the800 support pairs once;
three disjoint224-pair groups provide the extra rows for the three profiles.
After a separate deterministic shuffle per profile, take batches of4. Across
all profiles1344 rows appear four times and256 appear three times.
GEOMETRY presents each of MZ61 TRAIN2048 exactly once in each profile, also
shuffled into batches of4. Thus both arms have6144 target presentations and
identical6144 MZ48 plus12288 negative presentations. New source size, dimensions,
poses and per-ID exposure differ together; this is a source-adaptation test,
not a decomposition of those factors. The common warm start previously saw
MZ55 TRAIN and has never fitted MZ61; it is not a from-scratch comparison.

Bind MZ61 COMPLETE source-index SHA
`6f9e4be717b7fe23d66af58a9002a667d7acff304ec0c0af58497dfc502c72c7`.
Only its frozen TRAIN_CANDIDATE2048 may enter fitting; CALIBRATION1024 and
HELDOUT_GEOMETRY1024 remain excluded. MZ55 calibration/held rows are also excluded.
Fit each new cutoff vector by the unchanged original zero-added rule on the
same1256 DEV/MZ48 calibration rows. No new calibration cohort, normalization fit,
threshold search, profile selection, or outcome-dependent retry is allowed.
All inspected sources remain consumed Development, including named held groups.

Encode each unique training RGB once into a temporary task-owned float32 FULL
feature cache. All arms/profiles share those exact features; evaluation encodes
only misses in fixed batches of16. Keep labels, local knownness, source family,
geometry, pair IDs and roles on the loss/evaluator side. Inference uses only RGB
features, existing ranges/validity and calibration. UNKNOWN remains explicit;
missing ToF and an RGB candidate are never measured clearance.

Before fitting, verify both initial states exactly, and compare saved raw/support
outputs on16 MZ48 fit rows for3 profiles and16 MZ61 fit rows for4 profiles
(including ALL_INVALID), atol2e-5/rtol1e-6. This is a32-row initialization probe,
not a new baseline cohort replay. Evaluate both learned heads over the original
seven cohorts under the original3 profiles, plus all4096 MZ61 frames under4
profiles. Reuse and preserve every saved MZ62/MZ63 baseline array. ALL_INVALID
zeroes both ranges and validity and is an evaluation-only absence boundary,
not a calibrated VL53L8CX failure probability.

The primary definition is frozen in the task's primary-definition.json. On
DROP MZ61 HELD1024, GEOMETRY must add more native-winning true BODY_NEAR events
beyond OLD_NEG than both the matched CONTROL and frozen MZ62 CONTROL, and
include at least one native held awning BODY_NEAR addition. Across all four
queries combined it must retain TP and introduce no new false-positive bit
relative to either comparator. Also retain total TP and do not increase total
FP in legacy_noncal, MZ48 nonfit and MZ55 held640 versus both comparators.
Report every clause, full per-query/profile gains/losses, false alerts added and
removed, local UNKNOWN, native/wrong winner attribution and support-pair changes.
Equal net FP is not zero newly false bits on the primary held group.

A positive check retains bounded useful adaptation; a gain with additional
false alerts remains an explicit tradeoff with old comparators retained.
No gain distinguishes this source-adaptation recipe from a fully trained ceiling
or absence of useful input information. MZ61 split label novelty is not proof
of independence from every older training source. Native winner attribution
does not establish causal reliance on that surface.

Budget: two1536-step fits, one shared feature build, one learned-head evaluation
and one independent CPU score. No new capture, encoder fine-tuning or automatic
budget extension. Mechanical repairs may reuse identical registered inputs
with failed logs/bytes retained; no altered scientific conditions. Release
processes/archive/mmap handles and delete only the task-owned disposable cache
after scoring and integrity checks, retaining weights, predictions and receipts.
Primary UE must not overlap this primary GPU run; independent worker work may
continue. No hardware, natural-scene, Android or safety promotion follows.
