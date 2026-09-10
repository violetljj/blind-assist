# MZ60: direct positive query gradients to native witnesses

2026-09-11 EXPLORE. MZ59 matched diverse training found one new native-winning
BODY_NEAR heldout event, with extra false alerts and old-source losses. Its
shallow-awning training positives remain 16/100 TP. Code inspection shows a
specific conflict: positive query BCE increases the global spatial maximum
even when it is a known nonnative or UNKNOWN cell, while local negative BCE
is averaged over many cells. A known positive witness only enables that
query loss; it does not constrain which cell receives its positive gradient.

Hypothesis: restricting the positive-query training maximum to known native
witness cells improves localized near-body evidence under missing ToF without
the background reinforcement of the current objective. This changes the
positive query objective only. Native labels are loss-side supervision;
inference remains the unchanged global spatial maximum with no label input.

Use the exact MZ59 DIVERSE source, 600-step schedule, MZ56 GLOBAL initialization,
11020-parameter AnchorQuery, frozen RGB encoder/normalization, Adam0.001,
seed151, profile cycle and old negative replay. Reuse sealed MZ59 CONTROL and
DIVERSE predictions as comparators. Do not rerun a baseline fit or cohort.
One new 600-step WITNESS fit, not a continuation of the completed MZ59 weights.

The local balanced BCE is unchanged. The frame query mask is unchanged:
frame-known, some candidate cell known, and a native witness for positives.
For an eligible positive query, replace `OPEN.raw` in query BCE with the max
of `field` over native-positive known candidate cells. For negatives retain
the original `OPEN.raw`. Queries with no positive witness retain the original
skip semantics and finite arithmetic. Keep coefficients: local +0.25 query
+0.25 original selected-old-negative loss. No new margin, distillation,
hard-negative term, global near/far exclusivity or extra data selection.

Each new output receives one cutoff vector under the same zero-added rule
on original DEV1000 + MZ48cal256 DROP rows. No MZ55 calibration rows, threshold
search, checkpoint selection or tuned rejection rule. Report both candidate
and final fixed OR with unchanged OLD_NEG union, preserving MZ37 positives.

Primary retaining check under DROP, all required:

- New native-winning BODY_NEAR true events over OLD_NEG on MZ55 new-family
  heldout480 must exceed MZ59 DIVERSE's one.
- Final OR on all MZ55 heldout640 must retain at least454 TP while having
  no more than79 FP (DIVERSE TP and CONTROL FP respectively).
- Final OR on legacy noncalibration must have no more than45 FP and retain
  at least2722 TP, the unchanged OLD_NEG/CONTROL operating point.
- Final OR on MZ48 nonfit must retain at least514 TP and no more than23 FP,
  the stronger MZ59 CONTROL operating point on that source.

Report each clause separately, full paired gains/losses versus both MZ59
arms, old baseline costs, all three profiles, every cohort, MZ55 train/cal/
heldout/site/family/relation/context groups, shallow-awning positive counts,
native versus nonnative/UNKNOWN winners, and all original UNKNOWN attempts.
Failure rejects this all-criteria retaining claim; keep any measured tradeoff
and mechanism evidence without extending or recutting the run. A success
retains a consumed-Development challenger, not a default/hardware/safety claim.

Before fitting, test synthetic gradients: a nonnative/UNKNOWN global winner
must receive no positive-query gradient in the new loss; native witnesses
must receive it; negative queries, skipped positives and local loss retain
their old semantics; no-witness/all-unknown batches stay finite. Verify exact
loaded parameter/buffer identity and at most16 TRAIN-only frame arithmetic
comparisons to frozen MZ56 outputs at inherited atol2e-5/rtol1e-6. Check
all-missing observed context stays exactly zero. No native depth read in
predictor execution and no truth leakage through forward inputs.

Build one temporary FP32 feature cache under this task's canonical artifact
root, encode in fixed batches of16 and reuse training rows in evaluation.
Previously deleted caches remain deleted. Reuse MZ59's exact source/index
assembly and bind the new owner adapter; do not edit frozen source code.
Save all schedules, input hashes, actual backend/device, feature bytes and
stage times. Primary GPU is admitted only without primary UE or competing
heavy compute. Independent saved-output CPU scoring reconstructs the new
cutoff/candidate/OR/scalar/native counts and verifies sealed baseline arrays.

Budget: one600-step fit, one cutoff vector, one seven-cohort/three-profile
inference pass for the new head and one independent score. No new capture,
automatic successor fit or changes to real-sensor assumptions. Preserve any
mechanical failures and repair only their identified causes. After scoring,
release handles/processes and remove only this task's temporary feature file;
keep sources, feature indexes, checkpoints, predictions and receipts. Finish
interpretation, scoped registration disposition and delivery for this run.

All sites and comparisons are previously inspected/consumed Development.
The existing MERGE/DROP profiles remain controlled stresses, not calibrated
VL53L8CX probabilities. Sensor absence does not negate a native obstacle.
