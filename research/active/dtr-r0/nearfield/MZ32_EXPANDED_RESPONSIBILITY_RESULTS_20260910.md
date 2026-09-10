# MZ32: expanded supervision restores retention with a false-alert tradeoff

2026-09-10 EXPLORE, consumed Development. MZ32 removes53 of MZ28's83 placement
false bits while retaining every MZ28 true positive on all normal cohorts.
Placement exact rises2864->2908/3000; FN remains67 and FP falls83->30.
Compared with MZ30 it restores one BODY_NEAR TP, but FP rises20->30. The
predeclared joint improvement requirement of at most20 FP therefore fails.
Retain this as a CHALLENGER with an explicit tradeoff, not a baseline promotion.

[Protocol](MZ32_EXPANDED_RESPONSIBILITY_PROTOCOL_20260910.md),
[fit](mz32_train.py), [independent audit](mz32_audit.py),
[preceding coverage diagnosis](MZ31_RESPONSIBILITY_COVERAGE_RESULTS_20260910.md).

The supervision expands from168 to1165 original TRAIN branch disagreements.
Architecture8641 parameters, seed123 initial tensors,264-feature normalization,
1200x16 batch stream, Adam0.001 and runtime correction eligibility remain fixed.
Inverse-frequency weights use the same formula with478/687 targets, giving
1.218619/0.847889 instead of0.7/1.75. The fit has1118 active batches and4030
supervised query exposures, versus565/760 in MZ30. Expanded supervision includes
supported and baseline-negative cases; runtime still changes only unsupported,
baseline-positive disagreements. No new observations or branch features are used.

| Normal cohort | MZ28 exact | MZ30 exact | MZ32 exact | MZ32 FP | TP lost vs MZ28 |
|---|---:|---:|---:|---:|---:|
| oldDEV1000 |940|957|952|17|0|
| clean200 |176|182|179|11|0|
| stress200 |175|181|178|11|0|
| relationDEV2000 |1916|1940|1936|25|0|
| distanceDEV1000 |948|975|972|5|0|

The placement30 FP comprise28 baseline errors and2 unchanged MZ28 additions.
All MZ28 additions remain byte-identical, including72 retained far additions;
trained pole coverage remains49/49 clean and48/49 stress. No new positive task
bit versus MZ28 is created in any normal or wrong-local-visual view. Counts are
query bits, not independent obstacles; clean/stress share sequence frames.

Relative to MZ30, relation restores1 BODY_NEAR TP, removes8 HEAD_FAR FP, but
adds2 BODY_NEAR,2 BODY_FAR and10 HEAD_NEAR FP. Distance adds4 HEAD_NEAR FP
and has no TP change. Thus the net10 FP increase hides opposite query effects;
it is not uniform improvement or uniform deterioration.

The restored crossbar case is relation621/global11821 at big05_site_153.
RGB+3.212352 is correct and ToF-0.555607 is incorrect. MZ30's raw selector logit
-1.456389 gives negative-branch confidence0.810980 above its BODY_NEAR cutoff
0.741196. MZ32's logit-0.395996 gives confidence0.597725, below its new oldDEV
calibrated cutoff0.993586. The raw selector STILL prefers the incorrect ToF
branch. Weaker confidence and more conservative calibration preserve the alarm;
the responsibility classification itself is not corrected. No threshold was
chosen from this case, and no posthoc rescue was run.

Independent audit PASS verifies exact supervision/targets, initial state and
normalization, inverse class weights,1200 batches and4030 exposures, first loss
within3.12e-8, calibration optimum,31,200 task bits and preservation invariants.
Independent NumPy selector replay also passes. A separate saved-case extension
records the restored-case interpretation without rerunning training or core audit.
Fit time3.125s and total runner6.066s use cached features on CUDA; these are not
RGB-backbone, phone, single-frame or end-to-end latency measurements.

Artifacts: artifacts.local/work/mz32-expanded-responsibility-20260910/run-v1/,
including supervision.npz, initial/final weights, scores, predictions, result.json,
receipt.json and audit.json. The fit and audits exited0. No second fit, new source,
protected EVAL, App/default change, temporal experiment or safety claim occurred.
TRAIN branch predictions are in-sample and all DEV views are consumed. Broader
label coverage changes the observed retention/FP tradeoff but does not establish
new-source responsibility transfer. Preserve MZ30 and MZ32 as separate candidates;
the next comparison must keep these frozen results and test the remaining
responsibility/calibration or independent-source gap explicitly.
