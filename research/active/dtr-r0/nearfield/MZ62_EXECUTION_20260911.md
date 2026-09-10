# MZ62 matched profile coverage: execution and diagnosis

The complete-profile assignment fails its registered retaining comparison
(3/7 clauses pass). On new-family held480 under DROP, native-winning new
BODY_NEAR additions are CONTROL17 versus COVERAGE6. Final OR TP/FP are
480/88 versus461/82 on held640,2732/57 versus2726/50 on legacy noncal,
and537/24 versus522/24 on MZ48 nonfit. The three FP clauses pass; native
gain and all three TP clauses fail. Neither arm dominates OLD_NEG union.

[Protocol](MZ62_PROFILE_COVERAGE_20260911.md) and
[unchanged score report](MZ62_PROFILE_COVERAGE_RESULTS_20260911.md) define
the comparison. Both arms start from the same MZ56 GLOBAL weights, use the
original loss, and receive1200 steps with identical per-frame total
presentations. CONTROL randomly assigns each TRAIN frame's three uses;
COVERAGE puts one use in each IDEAL/MERGE_CLOSE/DROP_CLOSE condition.
Assignment and ordering change together. Each frame/profile receives one
presentation, so this does not establish a trained ceiling.

All1600 MZ55 TRAIN rows are eligible. CONTROL uniquely covers
1140/1160/1122 rows by profile; COVERAGE covers1600 in every profile.
Awning DROP training positives cover74/100 versus100/100, with101 versus100
presentations. MZ55 CALIBRATION and HELDOUT remain absent from fitting and
cutoff estimation. The two cutoffs use the original1256 calibration rows;
no threshold search or retry follows the failed comparison.

## What the saved outputs explain

Awning BODY_NEAR final OR TP changes65→33/100 TRAIN,13→4/20 CAL, and27→10/40
held. Held native winners rise28→31, yet native-winning final OR TP falls
17→6. Seventeen held losses include12 native→native,2 nonnative→native,
2 nonnative→nonnative, and1 native→nonnative. Losing native location is
therefore not the dominant endpoint explanation for this comparison.

Among27 held rows with native winners in both arms, median paired raw
score changes−0.149165. BODY_NEAR cutoff rises2.531259→3.347590. Score
separation and calibration both change; this diagnosis does not isolate
their causal effects or evaluate an alternative cutoff. COVERAGE's30 held
misses comprise25 native and5 known nonnative winners; UNKNOWN winners
are0 in this awning subset. All other local UNKNOWN cells remain explicit.

Exact new native branch TP, requiring OLD_NEG negative as well as a
native winner and positive candidate, is38/12/17 versus12/4/6 over
TRAIN/CAL/held. Native-winning final OR TP is45/12/17 versus18/4/6;
inherited successes must not be called new branch contributions.

CONTROL's1200-step outputs show a useful measured sensitivity change
relative to older600-step runs, but also add false alerts. That cross-run
comparison changes schedule and budget and is descriptive. Preserve it
with the failed COVERAGE recipe; do not silently promote either arm.

## Execution, storage and evidence

The primary RTX5060 Laptop GPU run completed in282.924260s: shared feature
build92.502017s, CONTROL fit34.956864s, COVERAGE fit28.267734s, and
evaluation110.724572s. Both arms share6228 cached training RGB features;
old baseline predictions are reused with zero baseline reinference.
Independent CPU scoring took3.876217s with no RGB/native reads or models.
The audit preserves1696 prior arrays and5286 prior metric rows, checks
458112 scalar decisions and122880 native winners, reconstructs both
cutoffs, and retains80 MZ36 UNKNOWN query bits.

After run exit0, score exit0 and closed handles, the sole-link task cache
was removed with an exclusive hash/identity check. Actual allocated
storage released is5,739,728,896 bytes; all checkpoints, predictions,
source archives/native arrays, feature indexes and failure evidence remain.
The primary GPU was released before MZ61 main source collection began.

Evidence lives under ignored
`artifacts.local/work/mz62-profile-coverage-20260911/`: `execution-v1.json`,
`run-v1/receipt.json`, `score-v1/receipt.json`, `score-v1/awning-drop-audit.json`,
`score-interpretation-v1/completion.json`, and `cache-cleanup-v1/receipt.json`.
Run receipt SHA256:
`c423418b7b54583b19a334d542d5f832780d7f40e23781b0d4ceb0eb52f985c8`.
Score receipt SHA256:
`0ae3f93b82f33f0ddaebb9b98ccd988e2712490d20247da95b30bac9da60122d`.

Retain the complete-profile assignment as NEGATIVE_CONTROL for this
matched recipe. This is consumed controlled Development evidence using
uncalibrated weak ToF profiles. No natural-scene, device or safety claim
and no change to the default app follow.
