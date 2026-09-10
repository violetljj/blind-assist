# MZ18: BODY-only witness objective

2026-09-10 EXPLORE, consumed Development. Following the MZ17 result, the user
explicitly authorized doing the BODY-specific intervention. This is a new scoped
experiment; MZ17's aggregate10% diagnostic gate and false admission are preserved,
not altered or reinterpreted. There is no additional prevalence gate before fitting.

MZ17 observed known wrong positive winners at21/101 BODY_NEAR and22/153 BODY_FAR
TRAIN draws, with net upward candidate-logit gradients in every case. The HEAD
rates were much lower. This justifies testing a BODY-specific loss change, not
assuming that the conflict caused placement false positives or that it is fixed.

One candidate: replace only BODY query-loss terms in MZ16 HIGH_DETAIL. For each
BODY frame/query, reward the maximum known valid eligible actual query contributor
on full-query positives, and penalize the maximum known valid eligible false
candidate on either positive or negative frames. Missing positive support means
no witness reward. Unknown cells are not negative; multiple real contributors
may coexist. Keep original HEAD query BCE and the original global supervision
denominator exactly, so HEAD query-logit gradients match at identical inputs.
BODY now has up to two terms: this is an intentional gradient-weight redistribution,
not a pure loss-sign intervention. Keep the local BCE and query coefficient0.25.

Unchanged: native-detail224ROI input, frozen encoder,10740parameter QUERY network,
normalization, original initialization, saved1200x16 batches, Adam0.001 and inference
maximum over geometric candidates. Do not initialize from a completed checkpoint.
HEAD objective stays the same, but shared trainable layers mean HEAD predictions
can change; measure all four outputs. Native labels are training/evaluator-only,
never inference gates. No new data capture, EVAL, temporal module or App integration.

Reuse completed MZ16 HIGH_DETAIL as the matched objective comparator and MZ5 as
the composed baseline.10200-frame available training pool,7562 unique drawn frames,
19200 draws, including the200 already trained diagnostic frames. OldDEV1000 supplies
zero-added-MZ5-FP alert cutoffs and local1%FPR diagnostic cutoffs, then freeze both.
Replay consumed relationDEV2000, distanceDEV1000, clean/stress200 and wrong-zone
controls using unchanged stress contributor-slot alignment. OldDEV calibration
does not promise equal realized test-set false positives.

Primary effect: changes in BODY far TP and BODY near/far false positives relative
to MZ16, with the full four-query tradeoff reported. A retained MZ5 augmentation
must add no FP in any query on both placement sets or clean/stress, recover far
positives on both placement sets, preserve pole clean/stress at least48/49 and
retain baseline near positives. Also report exact frames, family/group/site counts,
true source at winning locations, local AP and P/R/FPR. AP explains the mechanism
but is not an independent veto on a useful task result. Wrong-zone disruption
tests correspondence sensitivity, not exact true-source localization.

Budget: one1200-step fit, no coefficient search, repeated seeds, threshold rescue
or fitting extension. Complete narrow gradient tests before fitting, independently
recount saved predictions/AP and verify hashes/batch/initial-state identity after.
Stop the experiment after that outcome; finish report, structured inheritance,
scoped commit/push and release task-owned processes and disposable resources.
