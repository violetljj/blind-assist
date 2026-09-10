# MZ20: separate true-witness strength from false-location suppression

2026-09-10 EXPLORE, consumed controlled Development. Authorized by the active
goal to improve obstacle algorithms. MZ18 remains a completed component result.

Hypothesis: independent absolute penalties on false maxima improve local ranking
without separating weak real-witness scores from hard negatives across frames.
An explicit cross-frame ranking term may preserve localization while restoring
useful BODY detection at the unchanged calibration rule. It may also fail if
the fixed features cannot separate those examples; the experiment tests this.

Primary-source motivation: Huang et al., *Hard Negative Sample Mining for Whole
Slide Image Classification*, MICCAI 2024,
https://papers.miccai.org/miccai-2024/paper/3822_paper.pdf . Read via Exa on
2026-09-10. The paper combines instance ranking and hard negatives in MIL.
Our geometry-supervised cross-query application is an adaptation, not a paper
reproduction or evidence that pathology results transfer to obstacle detection.

MZ19 saved-score diagnostic additionally shows the cutoff-driving negative
frames frequently lack known-local candidates. That explains why restricting
all negative supervision to local known masks misses the actual score tail.
Read-only evidence: artifacts.local/work/mz19-margin-diagnostic-20260910/result.json.

Change only BODY query objective from MZ18. Positive bags reward their highest
known actual eligible witness with softplus(-score); negative bags penalize the
original full inference maximum with softplus(score), under MZ16's original
task-negative supervision mask. Additionally compare each true BODY witness
maximum to every same-query competitor in its TRAIN batch: negative bags use
their full maximum; positive bags supply their highest known false location.
Use softplus(1 + false - true), averaged over competitors per positive.
Include same-frame pairs. Fixed ranking coefficient 1 and margin 1;
no sweep. Keep local BCE, outer query coefficient .25, original global query
denominator and HEAD terms. Unknown local candidates never receive new local
negative labels; task-level event absence keeps its existing non-CLEAR semantics. Geometry
and native labels remain training/evaluator-only; inference is unchanged max
over observable geometrically eligible candidates.

One 1200-step fit from identical original MZ16 HIGH_DETAIL initialization, same
1200x16 batches, Adam .001, native224 ROI cache and 10740-parameter network.
No warm start, encoder update, new capture or protected EVAL access. The 7562
unique sampled frames include the previously trained 200-frame diagnostic.
OldDEV1000 alone sets the same zero-added-MZ5-FP cutoff rule. Compare MZ5, saved
MZ16 HIGH_DETAIL and MZ18 BODY_WITNESS. Replay consumed relationDEV2000,
distanceDEV1000, clean/stress200 and wrong-zone controls, with unchanged stress
slot alignment. Reuse remains Development, never independent confirmation.

Report exact frames, per-query TP/FP, far additions, BODY local AP, true winning
contributors, pole BODY/HEAD detections and family/group/site results. A usable
augmentation must add no FP on these cohorts, gain far TP on both placements,
retain at least48/49 pole opportunities under clean and stress, and preserve
MZ5 near positives. Local AP alone cannot qualify. Failure preserves the
specific negative result; no within-run extension or threshold rescue. A
subsequent distinct experiment under the broad goal needs its own comparison.

Validate field gradients/unknown handling before the fit, then independently
recount saved outputs, AP, cutoffs, initialization and batch identity. All
artifacts use a fresh artifacts.local/work/mz20-rank-objective-20260910 child.
Mechanical failures retain receipts and may be corrected without hiding the
attempt. Release owned processes; report result and scope and deliver only
task-owned changes. No default-App, hardware or safety promotion.
