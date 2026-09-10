# MZ37: restore a confident positive modality without refitting

2026-09-10 EXPLORE, consumed Development throughout, including MZ36. MZ36 found
two far-crossbar misses where one modality and the frozen MZ35 responsibility
selector were correct, but the mean ensemble was negative and the selector's
alarm-removal scope could not act. Test a distinct additive responsibility rule;
do not fit to those two examples or relabel this as fresh confirmation.

Baseline is frozen MZ35, retaining its MZ5/MZ28 behavior and all old tradeoffs.
No weight, feature, bank, normalization, negative-selector cutoff, or source
change. Only restore an existing positive RGB/ToF branch when MZ5<0, MZ35<0,
the two branch signs disagree and original geometric support is present.
Use original support, before packet-availability restriction. Geometry alone
does not authorize restoration. Positive confidence is float32(1-negative
branch confidence) from the saved frozen MZ35 selector. The output is the
existing nonnegative branch score; never synthesize a new class label.

Calibrate exactly once on oldDEV1000 only. Per query, consider0.5, each eligible
confidence>=0.5, and nextafter(1,+infinity) as the disable threshold. Under zero
new FP maximize restored TP, breaking ties toward the highest threshold. If no
TP can be restored, disable that query. Freeze the four thresholds before
applying to clean200, stress200, relationDEV2000, distanceDEV1000, their saved
wrong-local-visual controls, and MZ36's380 admitted/400 attempted frames. No
other data choose a cutoff. OldDEV calibration is reported separately from
the remaining consumed Development cohorts; TRAIN responsibility is in-sample.

Save baseline/candidate logits, eligibility, confidence, positive branch,
restoration mask, labels and knownness for all attempted rows. MZ36's20 excluded
frames remain UNKNOWN in all four queries; do not infer or score those rows.
Report per-query and source-group confusion counts, complete-frame counts,
paired TP gained/lost and FP added/removed, all original positive preservation,
and the two previously identified MZ36 residuals without selecting by them.

Retain as a useful additive challenger only if at least one TP is restored
outside oldDEV calibration, no new FP occurs in any normal scored cohort,
all MZ35 positive judgments remain, and MZ28 additions remain. Report wrong
controls separately; no new positive depending on wrong local feature alignment
is justified by this RGB/ToF responsibility rule. A zero-effect or new-FP outcome
is a failed fixed test, not permission to append thresholds, masks or fits.

Budget: one oldDEV calibration and one saved-logit replay, zero model inference
and zero training steps. CPU scalar replay is TASK_NOT_GPU_SUITABLE. Bind source
receipts/model/cutoff/bank hashes and independently audit calibration, all query
changes and UNKNOWN. Preserve the completed MZ36, this run's failures and durable
outputs; no protected EVAL, device, App/default or safety claim. Stop this fixed
test after audit and scoped delivery; the broader improvement goal remains active.
