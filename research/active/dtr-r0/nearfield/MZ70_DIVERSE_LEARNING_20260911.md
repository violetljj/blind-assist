# MZ70: learn from diverse topology under missing inputs

EXPLORE. MZ69 found essentially absent all-missing accepted detections on
MZ67 and weak HEAD_NEAR raw separation (HELD NULL AUC0.582977/AP0.337211).
MZ67 has not trained either tested model. Test whether allocating training
exposure to its shapes improves weak-input native recognition, rather than
only adding old-source updates or changing the operating point.

Fit two matched 4,096-step arms from the exact sealed MZ64 GEOMETRY checkpoint
`fb318f03a598d00b89fc011eb5b1a368873f9ac801d1dc0710618ba119f4d4aa`.
Keep the frozen RGB encoder/normalization, 11,020-parameter AnchorQuery,
original balanced local/query loss plus .25 OLD negative replay, Adam .001
and seed151, full-raster OPEN pooling and original1256 DROP calibration rule.
Do not add distillation, projection, a router or an architecture change.

Each step uses MZ48 native4 + geometry4 + OLD negative8. Cycle IDEAL,
MERGE_CLOSE, DROP_CLOSE, ALL_INVALID, giving1,024 steps per profile per arm.
All-invalid means zero ranges and false validity, including OLD replay;
unavailable anchors must be exactly zero. The stress profiles are conservative
simulation boundaries, not validated VL53L8CX noise or failure probabilities.

CONTROL geometry is MZ61 TRAIN2048 repeated twice per profile. DIVERSE is
MZ61 TRAIN2048 plus MZ67 TRAIN2048, each once per profile. For each profile,
seed170 shuffles the concatenated sorted TRAIN IDs with source tags; CONTROL
replaces each tagged MZ67 ID by its sorted-ordinal MZ61 counterpart. All
other geometry positions remain identical. MZ48/OLD/query rows cycle the
original MZ64 1,536-step schedule and are identical across arms. No score,
label difficulty or evaluator outcome selects repetitions. Every DIVERSE
geometry frame sees every profile once; this statement does not imply every
MZ48 or OLD replay ID sees every profile. Source exposure is the sole difference
between the new arms. Comparisons with older shorter fits do not isolate data
allocation from budget, initialization or schedule.

Use complete sealed MZ67 source-index
`23aa6f552e38a0534052d2723b9a447ee91574bacacd503ecc64957c81b6560b`.
TRAIN-only fitting uses native known cells; query and full-raster UNKNOWN
remain separate. MZ67 CAL/HELD and MZ61 CAL/HELD are never fit or calibration
rows. The original DEV1000 + MZ48 CAL256 DROP pool supplies one unchanged
four-query cutoff rule per arm. No threshold sweep or source-specific recut.
Prediction reads RGB features, observed packets and fixed calibration only;
source IDs, roles, geometry, native labels and known masks are loss/evaluator
information. Invalid ToF and unknown regions are not measured clearance.

The primary source-effect comparison is DIVERSE versus CONTROL on MZ67 HELD
ALL_INVALID: per-query TP/FP/FN/TN, positive gains/losses, newly false/removed
bits, and native/known-wrong/UNKNOWN winning-cell attribution. Include raw
ROC AUC and grouped-tie AP on exactly matched known, supported, MZ37-negative
opportunities, with denominators and excluded rows. Ranking chooses no cutoff.
Report native HEAD_NEAR gain and its false-bit cost explicitly; a descriptive
native-gain flag does not authorize replacement or erase losses elsewhere.

Retain MZ67 DROP, MZ61 DROP/ALL_INVALID, the complete four-profile MZ61/MZ67
results, and all three original profiles on legacy/MZ48/MZ55 cohorts. Report
old retention groups, per-query false-bit exchanges and original MZ64 G,
MZ68 NULL and inherited MZ66 negative-control outputs where already sealed.
Keep all source roles/families/ranges/support partitions; geometry IDs and
raw count novelty do not establish unseen natural categories. Every source
here is consumed, curated controlled Development. No blind confirmation claim.

Gain with retained useful behavior supports a scoped challenger; gain with
costs stays an explicit tradeoff, with old comparators retained. Improved
ranking without useful native detections is a partial learning signal, not
a deployed solution. No gain leaves source allocation insufficient at this
tested representation/budget; it does not establish an information ceiling.
Input/weight/role mismatch is NOT_EVALUABLE. Report these outcomes before any
descriptive flag, without importing MZ68's old11-clause gate as this experiment.

Budget: exactly two4,096-step fits (8,192 total), one shared unique-TRAIN
float32 feature build, one evaluation of each final arm, one independent CPU
score, and32 fixed TRAIN frames for initial replay parity. Decode each
evaluation RGB once and share its full features across arms/profiles. Read
sealed old baselines; do not re-infer/refit them. Record actual device, feature,
fit/evaluation time and cache bytes. Primary model work has priority over
primary UE capture. Source-only checks may run concurrently on the worker.

Preserve mechanical failures and exact scientific inputs during any routing
repair. Check same initial weights, complete geometry/profile exposure,
source-role boundaries, missing anchors and scalar saved-output decisions/cuts.
Close image/archive/mmap handles on all exits; remove only the task-owned
temporary feature cache after terminal scoring with evidence retained. This
experiment ends after actual score and scoped delivery; the broader goal
continues from its measured failures or gains. No Android default, hardware,
natural-scene or safety promotion follows from this controlled comparison.
