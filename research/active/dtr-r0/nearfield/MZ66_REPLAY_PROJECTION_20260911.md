# MZ66: protect the replay loss during geometry learning

EXPLORE, consumed controlled Development. MZ64 learns useful new geometry but
increases old-scene false alerts. Its saved-output diagnosis finds 16 added old
DROP false events versus matched CONTROL: 14 are already explained by changed
raw scores under the old cutoff, two need both raw and cutoff movement, and none
is cutoff-only. All have some available global anchor. Three far-awning BODY_NEAR
errors have known native BODY_FAR evidence at the selected location. Other old
errors are not all range confusions. OLD_NEG also samples only 3,533 unique old
training frames; cycling 1,536 steps does not expand that memory. The semantic
negative types exist in training, but the present weighted loss does not impose
an update constraint. These observations motivate a learning-interference test,
not a claim of catastrophic forgetting or a complete diagnosis of every error.

Fit one PROJECT head for exactly 1,536 steps. Reuse the sealed MZ64 GEOMETRY arm
as the equal-budget comparator, with the exact same MZ62 CONTROL initialization,
Adam lr0.001/seed151/reset state, frozen encoder and normalization, MZ64 GEOMETRY
sequence, four MZ48 plus four MZ61 and eight selected OLD_NEG rows per step,
three original profiles, original local/query loss and 0.25 OLD_NEG softplus.
MZ61 TRAIN only is fitted; both source CAL and HELD roles stay out of fitting and
calibration. No new source, extra replay examples, ALL_INVALID fitting, cutoff
search or inference architecture change is included. Original comparators and
all original predictions remain immutable; a new matched comparator fit is not
needed to consume the same source/optimizer budget again.

The only learning change is a constraint on the actual Adam parameter delta.
Before Adam, compute g, the gradient of the existing OLD_NEG batch loss. Obtain
Adam's proposed delta d from its actual FP32 before/after parameter vectors.
If g dot d is positive, replace d by d - (g dot d)/(g dot g) g; otherwise keep
Adam's proposed parameters exactly. Zero g leaves the proposal unchanged.
Dot products/projection use float64 on the same device, with final parameters
stored in float32. Adam's moments/step counters follow its original proposal;
there is no optimizer-state projection, added reference loss, repeated projection,
line search, teacher fitting or second constraint.

This is an A-GEM-inspired halfspace projection of an actual Adam step, not an
exact implementation of its original gradient update. [A-GEM section4](https://arxiv.org/abs/1812.00420)
motivates protecting an average episodic-memory loss with one gradient constraint.
The returned method section and equations were inspected; its lifelong-classifier
results are not ToF evidence. MZ33's earlier absence of final query-gradient conflict
concerned a different selector and does not decide this cross-source step test.

Record all 1,536 proposed/post-projection dot products, gradient/step norms,
conflict frequency, correction magnitude, and the explicit FP32 rounding bound.
Check the assigned step against that arithmetic bound. Save full before/proposed/
reference-gradient/assigned vectors at 16 evenly spaced fixed step indices for
an independent NumPy closed-form check; no outcome selects those indices.
This only checks a local
linearized average over the sampled negatives. It does not guarantee actual
nonlinear loss reduction, per-query retention, held-source correctness or model
calibration. Record pre/post replay losses every step using the same already-loaded
eight replay inputs, without another encoder pass; these measurements never select
updates or extend the fit. No inference input receives labels or evaluator counts.

Use one shared temporary cache for the 6,676 fit-unique RGB frames (MZ48 1,095,
old 3,533, MZ61 2,048). MZ55 is evaluation-only and has no training cache entries.
Keep fixed encoder batch16 and original arithmetic. Before fitting, check warm-start
raw/support parity on 16 MZ48 plus 16 MZ61 TRAIN frames against the saved MZ62
outputs, and exact zero anchor on missing packets. This is not a full baseline
replay. Evaluate the new head once across the original eight cohorts and three
profiles, plus ALL_INVALID on MZ61; share each decoded image across profiles.
Derive one cutoff vector using only the original 1,256 DEV/MZ48 DROP rows and
the original zero-added rule. Score independently from saved arrays.

The primary question is Pareto improvement over MZ64 GEOMETRY under DROP.
Require TP not lower and FP not higher in each of legacy noncal, MZ48 nonfit and
MZ55 held640, with strictly fewer FP summed across those three groups. On MZ61
held geometry1024 require TP not lower, no newly false bits, native BODY_NEAR
additions beyond OLD_NEG not lower, and held-awning native BODY_NEAR additions
not lower. These are 11 clauses. Report complete event gains/losses, every profile,
all source roles, other retained comparators, known wrong and UNKNOWN winners,
and missing-input failure even if the primary gate fails. Native winners are
attribution, not causal proof. A failed gate retains only the measured tradeoff or
diagnostic; no outcome-dependent weight, threshold, extra steps or rerun is allowed.

Register before new model work. Budget: one 1,536-step candidate, the 32-TRAIN-frame
initialization check, one full candidate evaluation and one independent score.
Preserve mechanical failure evidence and permit only an input-identical repair.
Release task-owned GPU/process/feature handles after execution and reclaim only
the disposable cache after durable outputs are scored and verified. Keep source,
checkpoints, predictions and receipts. Primary UE cannot overlap this GPU run;
an independently scoped worker source task may proceed. No hardware, natural-scene,
Android, default-model, clearance or safety promotion follows from this experiment.
