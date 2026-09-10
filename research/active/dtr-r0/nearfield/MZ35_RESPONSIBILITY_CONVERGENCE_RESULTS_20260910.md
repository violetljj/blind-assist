# MZ35: further fitting improves responsibility but does not remove transfer gaps

2026-09-10 EXPLORE, consumed Development. The fixed4800-step shared selector
reduces placement false bits30->26 versus MZ32, retaining all MZ28 true positives
on normal cohorts. Placement exact rises2908->2910/3000; FN stays67. The required
joint improvement gate of<=20FP still fails. Retain this endpoint as a CHALLENGER
with measured optimization response, not a complete baseline replacement.

[Protocol](MZ35_RESPONSIBILITY_CONVERGENCE_PROTOCOL_20260910.md),
[runner](mz35_train.py), [independent audit](mz35_audit.py),
[preceding gradient diagnosis](MZ33_QUERY_GRADIENT_RESULTS_20260910.md).

One continuous trajectory repeats the original1200x16 batch stream four times.
At1200 every model tensor and all saved loss/exposure rows match frozen MZ32
bit-for-bit; only then does training continue. Both optimizer endpoints are
saved. The model remains8641 parameters with identical normalization,1165
TRAIN disagreements,478/687 target classes, inverse-frequency weights and
Adam0.001. There is no architecture, feature, sample, loss or runtime-mask change.
The4800 steps contain4472 active batches and16120 eligible query exposures.

| Normal cohort | MZ28 exact | MZ32 exact | MZ35 exact | MZ35 FP | TP lost vs MZ28 |
|---|---:|---:|---:|---:|---:|
| oldDEV1000 |940|952|952|18|0|
| clean200 |176|179|182|7|0|
| stress200 |175|178|181|7|0|
| relationDEV2000 |1916|1936|1934|26|0|
| distanceDEV1000 |948|972|976|0|0|

All MZ28 additions remain byte-identical, including72 retained far additions;
trained pole remains49/49 clean and48/49 stress. The26 placementFP are24
baseline errors plus2 inherited MZ28 additions. Relative to MZ28,57FP are
removed with no TP loss. Relative to MZ32, distance removes4 HEAD_NEAR and1
HEAD_FAR FP; relation removes2 BODY_FAR and1 HEAD_FAR FP but adds4 HEAD_FAR
FP. OldDEV gains one FP while complete-frame count is unchanged. Thus better
fitting and aggregate placement benefit coexist with nonuniform transfer.

| Query | Step1200 TRAIN errors | Step4800 TRAIN errors |
|---|---:|---:|
| BODY_NEAR |7|8|
| BODY_FAR |13|6|
| HEAD_NEAR |22|5|
| HEAD_FAR |24|9|
| Total |66|28|

Weighted TRAIN BCE falls0.160144->0.063869 over the1165 disagreements. BODY_NEAR
error count nevertheless rises by one. MZ33's aligned aggregate gradients did
not guarantee uniform query accuracy or generalization; the fixed duration
comparison measures this distinction directly. No intermediate checkpoint was
selected for task performance, and this completed trajectory is not extended.

The prior crossbar case relation621/global11821 now has raw selector logit
+2.392551, correctly preferring RGB, with incorrect-negative-branch confidence
0.083742 below the frozen oldDEV-calibrated BODY_NEAR cutoff0.665512. MZ32 had
still preferred the incorrect ToF branch at confidence0.597725. This particular
responsibility classification is corrected at4800; one consumed case is not
independent confirmation. No threshold was selected from its outcome.

Artifacts are in artifacts.local/work/mz35-responsibility-convergence-20260910/run-v1/:
initial/prefix/final states, optimizer1200/4800, prefix-parity.json, full schedule,
losses, TRAIN scores, calibration, predictions, results and receipts. Independent
audit verifies the prefix and schedule, saved optimizer steps, weights/exposures,
TRAIN metrics, calibration and31,200 task bits. Fit time8.575s, total10.484s on
cached features with CUDA; neither measures backbone, phone or end-to-end latency.

One fixed trajectory and its audit are complete; no task process remains.
TRAIN predictions are in-sample and DEV is consumed. Preserve MZ30's lower-FP
candidate alongside MZ32/MZ35's observed retention tradeoffs. Further claims
require a separately defined mechanism or unchanged-method new-source check;
no protected EVAL, App/default promotion, device, temporal or safety claim follows.
