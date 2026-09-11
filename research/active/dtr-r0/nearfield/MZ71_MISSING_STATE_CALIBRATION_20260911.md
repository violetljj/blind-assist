# MZ71: calibration matched to the observed missing state

EXPLORE, consumed controlled Development. MZ70 diverse training improves
MZ67 HELD missing BODY_NEAR raw AUC to0.999873 but only57/256 positives
pass the original DROP-calibrated rule. Missing HEAD_NEAR AUC rises to0.782557
yet accepted TP remains0. This changes the next question: can a calibration
rule measured in the same missing-input state expose useful learned evidence?
No architecture or training change is part of this experiment.

Freeze both MZ70 CONTROL/DIVERSE checkpoints and their original four-query
cut vectors, the unchanged MZ37/OLD_NEG pipeline, RGB encoder arithmetic,
normalization and original calibration IDs. MZ70 run receipt is
`a3ddf2e8e6cca4742e9970eb7e3e1f9f2e1adf70f3284883882f15869ceaa1e8`.
Preserve all3363 sealed arrays and all former source roles. MZ64, MZ68 and
MZ66 dispositions remain scoped; both MZ70 arms remain comparators.

For each fixed arm, apply the original zero-added calibration algorithm to
DEV1000 plus the exact original MZ48 calibration256 IDs, with every packet
range zero and validity false. Each of the two new vectors is nextafter of
the maximum eligible original-calibration negative raw score per query.
No threshold sweep, trainable update, or MZ55/MZ61/MZ67 calibration row is used.
This guarantees only that empirical calibration construction, not unseen FP
control, confidence calibration, or a deployment guarantee.

The actual predictor route is `missing = ~valid.any((1,2))`, on the observed
packet. Input validation already requires finite measured ranges within4m and
zero invalid slots. A missing row uses its arm's new missing-state vector;
any row with at least one valid return keeps the arm's original cutoff and
candidate/union bytes. Profile name, source ID, role, native labels, camera
pose and evaluator UNKNOWN never select the branch. Source names below are
evaluation partitions only.

The state rule also covers naturally missing rows in IDEAL, MERGE_CLOSE and
DROP_CLOSE. CPU preparation found94/94/152 such rows in the original1256
calibration pool, respectively. Therefore preserving all named non-ALL_INVALID
profiles would be the wrong route. Audit actual validity and every affected
row in all profiles. Retain MZ37 and OLD_NEG positives with the original union;
do not infer clearance from invalid ToF, empty support or UNKNOWN.

Reuse MZ70 raw/support/winner/anchor outputs on MZ61/MZ67 ALL_INVALID and every
original three-profile cohort. Fill only the absent ALL_INVALID outputs on
the seven old cohorts: DEV1000, relation2000, distance1000, rich44, MZ36
admitted380, MZ48 2560 and MZ55 2560, totaling9544 frames. MZ36 scoring retains
all400 attempted rows and80 query-UNKNOWN bits. Newly inferred comparisons
are original MZ37/OLD_NEG and the two fixed MZ70 heads at their original cuts;
do not invent old G/NULL arrays on these new profile/cohort combinations.

Before that pass, use32 fixed TRAIN frames (first16 sorted MZ61 TRAIN and
first16 sorted MZ67 TRAIN) to check the two MZ70 heads and old baseline pipeline
against sealed ALL_INVALID outputs. Strict checkpoint state tensors must
match exactly. Floating replay tolerance is2e-5 absolute/1e-6 relative and
candidate/union decisions must match; winner equality is reported separately
because old-source native analysis continues to use sealed winner arrays.

Primary results are MZ67 HELD ALL_INVALID per-query TP/FP/FN/TN and exchanged
bits for state calibration versus each same-arm original cutoff. Report
native, known-wrong and UNKNOWN winning-cell attribution of gains and losses;
HEAD_NEAR gains and each false-bit cost are separate outcomes. Include MZ61
HELD and old legacy/MZ48/MZ55 retention under ALL_INVALID, all original profiles
with natural missing rows, every source role/family/range/support partition,
and support-pair effects. No aggregate flag erases a near/head loss or new
false bit. Raw ranking is unchanged; do not claim this calibration learned a
better representation. A useful low-error gain supports a scoped component;
gains with costs remain an explicit tradeoff. No useful gain leaves original
cuts retained and identifies the tested negative tail as insufficient. Any
source/weight/calibration/observation mismatch is NOT_EVALUABLE.

Budget: zero fits; two new four-query calibration vectors from the fixed rule;
one9544-frame shared RGB pass plus32 TRAIN parity frames; one independent CPU
score. RGB is decoded once per inferred frame and global256x144_BOX, crop224
and full640x360 views are shared across the original baseline and both heads.
No permanent dense cache, source61/67 whole-cohort re-inference or threshold
search. Record actual device, timings and view counts. Primary inference has
priority over primary UE capture; independent worker source preparation may
continue. Close all image/archive handles on exit and verify process release.

Mechanical failures retain their receipts and exact scientific inputs.
The experiment stops after the fixed score and scoped delivery. The broader
goal continues from its measured effects. This is not validated VL53L8CX
failure physics, a natural-scene guarantee, a trained information ceiling,
an Android-default change, or a safety claim.
