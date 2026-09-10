# MZ59: matched training coverage on the admitted diverse source

2026-09-11 EXPLORE. MZ58's frozen scale unions add no detections on the 1920
new-family frames. Under DROP the retained old union finds only 7/160
shallow-awning BODY_NEAR events. This negative transfer motivates a training
coverage contrast. MZ55 and MZ58 have been inspected; all evaluation remains
disclosed consumed Development, including the named heldout-site partition.

Question: does replacing half the native-supervised training presentations
with the admitted MZ55 training partition recover new-shape near-body events
without increasing false alerts relative to an equal-budget old-source
continuation? This tests training coverage, not a new architecture or loss.

Start CONTROL and DIVERSE from the exact completed MZ56 run-v2 GLOBAL_ANCHOR
checkpoint. Both use the same 11020-parameter AnchorQuery, encoder,
normalization, geometry, missing-value handling and native local/query loss.
Strictly compare loaded parameters and buffers. Use Adam 0.001, seed 151,
600 steps per arm, and the original IDEAL/MERGE_CLOSE/DROP_CLOSE cycle.
No optimizer-state reuse, new ranking loss, augmentation, sensor capability,
architecture search or checkpoint selection.

CONTROL uses MZ51's original shared 600x8 MZ48 training schedule. DIVERSE
retains its first four slots per step and replaces the last four with
`default_rng(159).choice(MZ55_TRAIN_CANDIDATE, (600,4), replace=True)`.
Both retain the exact original 600x8 OLD_NEG and query replay schedules and
the original 0.25 selected-negative loss coefficient. MZ55 has 1600 eligible
training frames; its 320 calibration-named and 640 heldout-site frames never
enter fitting. Original MZ48 and legacy splits are unchanged. Save schedule
and split identities before fitting; do not use diagnostic outcomes to select
individual source frames, settings or query examples.

Each arm receives one cutoff vector through the unchanged zero-added rule on
the same original 1000 DEV plus 256 MZ48 calibration frames under DROP, against
MZ37. MZ55 supplies no cutoff rows. Two fitted models and two cutoff vectors
total. Preserve the original MZ56 and MZ58 cutoffs as baseline evidence; they
are not silently overwritten. Report each new candidate and its fixed OR with
the retained OLD_NEG union, preserving all original MZ37 positive decisions.

Primary contrast: DIVERSE versus CONTROL on the 480 new-family frames in
MZ55's heldout-site partition under DROP. Require more newly recovered
BODY_NEAR true events over the unchanged OLD_NEG union with native-positive
winning cells, no higher total FP across all 640 MZ55 heldout-site frames,
and no higher original legacy noncalibration FP. Report losses and total TP
as well; passing this check does not excuse them. If it fails, reject the
corresponding no-added-FP training-coverage claim and retain the detailed
tradeoff rather than changing thresholds or extending training.

Report all three profiles and all old cohorts plus all 2560 MZ55 frames,
keeping fit/calibration/nonfit groups explicit. Include query/site/family/
relation/support-context counts, shallow-awning BODY_NEAR denominators,
paired TP/FP changes, native winning-cell support and UNKNOWN. The MZ55
calibration-named partition is descriptive evaluation here. Its previous
inspection prevents a fresh confirmation claim. Native counts and metadata
are training/evaluator authorities only; the forward function receives only
RGB features, observed ranges/validity and static calibration.

Reuse sealed baseline predictions. No baseline cohort is rerun; at most 16
TRAIN examples may check loaded-model arithmetic against saved raw/support
outputs before fitting, at the inherited atol 2e-5 and rtol 1e-6. Parameter,
buffer, ID, decision and cutoff identity checks remain exact where applicable.
Verify zero context for all-missing observed packets. Independently rebuild
cutoffs, candidates, unions and scalar counts from sealed outputs.

Encode required original full RGB once in batches of 16. Share task-owned
float32 training features between arms and reuse them during evaluation;
stream uncached evaluation frames. Previously deleted MZ54/MZ56 cache files
must not be revived or assumed available. Hash source/schedule bindings and
record feature bytes, actual CUDA/device and stage timings. Use primary GPU
only when no primary UE capture or competing heavy task is active.

Budget: two 600-step fits, two declared cutoff vectors, one shared evaluation
pass for the two heads, and one independent saved-output score. No additional
capture or automatic successor fit. Preserve failed evidence and repair only
observed mechanical faults without overwriting completed scientific outputs.
Release model/process/archive handles and delete only this task's temporary
feature cache after scoring; retain original sources and durable results.
Complete scoped interpretation and delivery before stopping this experiment.
No calibrated VL53L8CX, natural-scene, clearance or App-safety claim follows.
