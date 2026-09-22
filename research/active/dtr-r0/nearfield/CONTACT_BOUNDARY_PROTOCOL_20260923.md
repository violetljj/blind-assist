# Counterfactual contact-boundary Development pilot

User-authorized EXPLORE after the 2026-09-23 proposal review. Reuse the consumed
1728-frame query-occupancy source; no new UE capture, original-recipe retry,
video, App replacement or protected confirmation. The new question is whether
explicit monotone spatial aggregation or direct query learning transfers to
unseen widths/horizons on layout-disjoint Development images.

## Task and data

Keep camera X-right/Y-down/Z-forward and BODY Y[.42,.9], HEAD Y[-.2,.42].
A query is a centred width w and absolute axial horizon h. Its contact is any
declared rendered cube intersecting X[-w/2,w/2], the height band and Z[.3,h].
This is a virtual cross-section swept forward from Z=.3, with travel h-.3;
it is not a reconstructed human body or a claim about the unobserved near field.
First-hit travel is first intersecting axial depth minus .3, not the ToF range.
Use all target and background AABBs, closed boundaries and the original level
camera. Independent world-space intersections audit the camera-space labels.
New prisms stay inside the old six-query visible-surface audit domain.

Use original whole-layout split24/8/16,864/288/576 frames. Every lateral paired
variant stays with its base layout. This is consumed same-generator Development,
not fresh confirmation. Preserve regional single-return simulator, RGB, ToF
validity and baseline UNKNOWN as recorded. Geometry, native depth, identities,
relation, family, baseline output and labels never enter prediction features.
Existing lateral variants are matched interventions; no paired texture-only
capture exists, so appearance invariance is not tested.

Training widths [.36,.60,.84,1.08]m, horizons [.6,.9,1.2,1.5,1.8,2.1,2.4,2.7,3]m,
both height bands:72 queries/image. Model/threshold selection uses only these
queries on dev layouts. Primary new queries: widths [.48,.72,.96]m and horizons
[.75,1.05,1.35,1.65,1.95,2.25,2.55,2.85]m:48/image. Report seen widths/horizons,
width-only and horizon-only interpolation, combined interpolation, and width
extrapolation [.24,1.20] separately. No extra image independence comes from queries.
Width boundaries use .20..1.20m in .02m steps at h=3; horizon boundaries use
.3..3m in .05m steps at w=.6. Report all censoring, nonmonotonic reversals,
finite-boundary coverage, conditional MAE and joint within-5cm success.

## Paired comparison and budget

Both arms use the same frozen local ImageNet MobileNetV3-small prefix [:7],
adaptive8x14 ordered spatial features plus ordered canonical64x6 ToF tokens.
Feature normalization is fitted on train rows only. Same initialized128D scene
trunk; direct head conditions on width/horizon/layer, geometry head predicts a
nonnegative |X| by Z intensity grid for each layer and aggregates fractional
query-cell overlap. Contact probability is1-exp(-summed intensity). This enforces
monotonicity but is a structured task projection, not supervised or validated
physical occupancy. Both receive exactly the same contact BCE supervision;
neither receives pixel masks, dense depth nor privileged geometric regression.
Report effective parameter differences; stored paired carriers alone do not
constitute equal capacity. The frozen encoder bounds cost and scope.

One fit/arm, seed202609231,100 epochs,batch32,AdamW lr1e-3,weight_decay1e-4;
identical shuffled image schedules. Equal-positive/negative training BCE uses
only the aggregate training fraction. Check dev BCE every10epochs, choose its
minimum (earlier ties). One dev-only atomic threshold per arm maximizes query
recall at FPR<=5%, with fewer FP breaking ties. Never select using novel-query
dev outcomes or evaluation labels. Seal checkpoints and every query prediction
before evaluation-label join. CPU/CUDA backend choice uses cloned real training
steps and tools/research_backend.py; report actual device and cost.

No tuning after the sealed evaluation, additional seeds, backbone changes or
automatic active-observation successor. Mechanical repairs preserve receipts
and cannot change the scientific cases or consume an additional fit budget.

## Decision

Report query TP/FP/FN/TN,precision,recall,FPR,Brier, by family and lateral relation,
train-layout/new-query versus evaluation-layout/new-query, width and horizon
boundary accuracy/censoring, and true-changing lateral paired correctness.
Compare arms on the same rows; whole-layout bootstrap intervals use1000resamples.
Do not interpret queries as independent trials or posed-frame metrics as alerts.

A useful contact-query component requires primary combined-interpolation
recall>=.80,FPR<=.05; width and horizon joint within-5cm success>=.80 over every
finite interior true boundary (missing predicted boundaries count as failures).
Relative advantage requires >=5points recall gain with FPR increase<=2points,
or >=30%relative FPR reduction with recall loss<=3points, without a family
recall loss>5points. Retain useful partial evidence with its actual costs even
when the full gate fails. If neither passes, close this exact cached-feature
readout package; do not infer that RGB lacks information or all contact learning
is impossible. Monotonicity alone is not a positive result. Finish focused tests,
independent output audit, result/lineage registration and scoped Git delivery.
