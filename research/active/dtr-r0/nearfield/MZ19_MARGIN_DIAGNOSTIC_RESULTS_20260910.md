# MZ19: correct within-frame ranking does not preserve cross-frame separation

2026-09-10, completed read-only consumed Development diagnostic. MZ5/MZ9 remain
unchanged. No fitting, model inference, cutoff selection or protected access.

Compare saved MZ16 HIGH_DETAIL and MZ18 BODY_WITNESS predictions, known-local
samples and frozen oldDEV cutoffs, verifying their receipt hashes. On the
already trained25-frame pole configuration, BODY_FAR results are:

| Quantity | MZ16 clean/stress | MZ18 clean/stress |
|---|---:|---:|
| Median true-witness maximum |6.469 /6.469|1.517 /1.353|
| Median true-witness cutoff margin |+4.691 /+4.691|-0.174 /-0.337|
| Global winners at true contributors |13/25 /15/25|25/25 /25/25|
| Minimum true-minus-known-wrong local gap |-0.552 /-1.320|+0.349 /+0.208|

MZ18's within-image ordering improves, while its real pole witnesses are
outranked by a median3 of38 oldDEV hard-negative frames (5 under stress).
MZ16 true pole witnesses exceed all inspected oldDEV/placement negatives.
For MZ18, every pole witness is outranked by at least one relationDEV negative.
A shared monotonic recalibration cannot reverse those cross-frame orderings.
This does not prove that another training objective will solve the separation.

Hard negatives here mean original task-negative queries with geometric support
and negative MZ5 decisions. Their BODY_FAR raw maxima are:

| Cohort | Frames | MZ16 median/max | MZ18 median/max |
|---|---:|---:|---:|
| oldDEV |38|-0.237 /1.778|0.337 /1.691|
| relationDEV |53|-0.943 /2.443|0.275 /1.978|
| distanceDEV |58|-1.437 /1.681|1.074 /1.776|

Crucially, only19/38,1/53 and1/58 of these frames have stored known-valid local
false candidates. The oldDEV cutoff-driving frame718 and next418/18 have none.
Known-local-only BODY_FAR negative maximum is0.573 on oldDEV, versus the actual
inference maximum1.691. Thus a loss restricted to known local false positions
can omit the actual negative tail. Preserve unknown local labels; the existing
task-negative output supervision is a separate authority and is not a CLEAR
claim. This finding motivated MZ20's cross-frame ranking with original negative
bag maxima, rather than reclassifying unknown local locations.

The25 correlated pole frames are one configuration, not25 independent obstacles.
Placement cohorts and oldDEV are consumed. Scores come from the same saved
arrays used in original results; no new performance claim or candidate exists.

Code: [mz19_margin_diagnostic.py](mz19_margin_diagnostic.py).
Artifacts: `artifacts.local/work/mz19-margin-diagnostic-20260910/` contains
`result.json` and PASS/hash `receipt.json`. Execution is NumPy CPU saved-array
reduction (TASK_NOT_GPU_SUITABLE); process exited. Retain as a diagnostic
component. The paired witness-versus-negative separation question remains open.
