# MZ37: restoration recovers misses with two false-alert costs

2026-09-10 EXPLORE, consumed Development including MZ36. The fixed additive
rule recovers17 true query bits outside calibration and preserves every MZ35
positive and prior addition, but adds2 false bits. The zero-new-FP criterion
fails (`useful_effect=false`). Retain a CHALLENGER with this explicit tradeoff;
do not replace the frozen MZ35/MZ5 baselines or erase their earlier failures.

[Protocol](MZ37_POSITIVE_RESTORATION_PROTOCOL_20260910.md),
[runner](mz37_restore.py), [independent audit](mz37_audit.py).

No neural inference, fitting, new data capture, bank change or negative-selector
calibration occurs. For MZ5-negative and MZ35-negative disagreements with original
geometric support, the candidate selects the existing positive modality only
above one newly calibrated positive-confidence cutoff. This never undoes an
alarm removed from a baseline-positive case. Original support is evaluated
before packet availability, and is not by itself evidence that an alarm is true.

## One oldDEV calibration

Per-query thresholds maximize restored TP at zero new FP on oldDEV1000 only,
with highest-threshold tie breaking and a disable option. The confidence floor
is0.5. Thresholds were saved before noncalibration replay; no outcome changed them.

| Query | Eligible TP | Eligible FP | Threshold | Restored calibration TP |
|---|---:|---:|---:|---:|
| BODY_NEAR |7|0|0.806730032|3|
| BODY_FAR |3|0|0.604706168|1|
| HEAD_NEAR |5|3|0.612323165|2|
| HEAD_FAR |14|6|0.999982297|1|

BODY_NEAR/FAR have no eligible calibration negatives. Their zero calibration FP
therefore provide no empirical rejection test for false restoration in that
subdomain. This is a coverage limitation, not evidence of perfect reliability.

## Paired task effects

All counts below are query bits except exact, which requires all four frame
judgments to match. OldDEV is the calibration cohort and is separated explicitly.

| Cohort | MZ35 exact | MZ37 exact | FP before/after | FN before/after | Added TP/FP |
|---|---:|---:|---:|---:|---:|
| oldDEV1000 calibration |952|957|18/18|38/31|7/0|
| clean200 |182|182|7/7|13/13|0/0|
| stress200 |181|181|7/7|14/14|0/0|
| relationDEV2000 |1934|1945|26/27|43/30|13/1|
| distanceDEV1000 |976|977|0/1|24/22|2/1|
| MZ36 admitted380 |375|377|3/3|2/0|2/0|

Combined relation/distance placements improve exact2910->2922/3000 and FN67->52,
while FP26->28. MZ36 retains400 attempted frames,380 admitted and20 excluded;
all80 excluded query bits remain UNKNOWN. Its two restored cases are precisely
the earlier far-crossbar BODY_FAR and HEAD_FAR misses, with positive confidence
0.995125949 and0.999997020. Because these cases motivated the mechanism, their
improvement is disclosed consumed Development, not another fresh-source result.

Wrong-local controls retain their original purpose and source identity. Added
TP/FP are clean8/0, stress8/0, relation41/1 and distance14/1. Their two new FP
identities are exactly those of the normal cohorts. The restoration uses the
unchanged full-image RGB/ToF responsibility signal; it does not consume the
wrong local crop alignment. All prior positive score values remain unchanged.
Group summaries by family/site/group reconcile to these overall counters.

## False additions and the next bottleneck

The relation false addition is HEAD_NEAR on cabinet BODY_ONLY, frame249/global11449,
big05_site_142. RGB+1.808336 is wrong and ToF-1.901091 is correct; the mean is
-0.046377. Positive-branch confidence0.991493821 exceeds0.612323165. This is a
confident responsibility error, not merely a score at the decision boundary.

The distance false addition is BODY_FAR on hanging-sign HEAD_ONLY,
frame829/global14029, big05_site_107. RGB+0.288760 is wrong and ToF-1.747627 is
correct; the mean is-0.729433. Confidence0.660443425 exceeds0.604706168. The
calibration subset contained zero BODY_FAR false-restoration examples. Both
errors have original geometric support, so that guard does not resolve them.

This test establishes that positive information can recover missed obstacles
without disturbing existing positives, but does not establish sufficiently
selective restoration. The next separate question is whether existing TRAIN
examples cover these positive-action errors and provide a discriminating
responsibility or query-geometry signal. Do not raise cutoffs around these two
observed false additions, disable a failed query posthoc, or expand this run.

## Verification and disposition

Independent audit PASS replays32,800 attempted query bits,32,720 known bits,
all candidate values, calibration candidates and tie breaks, source sigmoid
confidence, preservation invariants and frozen hashes. All prior MZ35 positive
and addition values are unchanged. This was one CPU saved-scalar calibration
and replay (`TASK_NOT_GPU_SUITABLE`), with zero neural inference or training.

Keep MZ37 as a Development challenger with17 recovered noncalibration TP and2
new FP. The fixed no-new-FP gate fails. MZ35's old <=20FP gate also remains
unmet; no successful subcohort overrides it. No protected EVAL, physical sensor,
phone latency, App/default or safety promotion follows. Original MZ35 and MZ36
artifacts are unchanged, and the broad improvement goal remains active.

Artifacts: artifacts.local/work/mz37-positive-restoration-20260910/run-v1/
contains thresholds, calibration freeze, predictions, results, false-additions,
receipt and independent audit. group-summary.json retains per-source summaries.
