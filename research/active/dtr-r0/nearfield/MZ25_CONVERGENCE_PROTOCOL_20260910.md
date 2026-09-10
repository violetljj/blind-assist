# MZ25: a fixed longer optimization budget for decisive availability

2026-09-10 EXPLORE, consumed Development. MZ24 reduces negative-availability
TRAIN bag activation85.37->29.43%, but leaves1855/6303 such bags active and4
placement addedFP. Its loss/gradient audit passes. Optimization sufficiency is
unresolved; residual TRAIN errors are not a transfer-only explanation.

Test one changed variable: optimization exposure. Start a new4800-step run from
the exact MZ24 seed123 initialization. Keep the10577-parameter AngularAvailability,
MZ24 dense1/negative0.25/positive0.25 objective, Adam0.001 without a new schedule,
features, normalization, labels, MZ20 and MZ5 unchanged. Concatenate four exact
copies of MZ24's1200x16 batch sequence:76800draws over the same7562uniqueTRAIN
frames. No new images/labels/sites or changed batch ordering. This is a new
registered convergence contrast, not an extension of the closed MZ24 artifact.

Save the step1200 checkpoint and compare every tensor to MZ24's final checkpoint
with predeclared atol=rtol=1e-4; record maximum absolute difference and per-tensor
results. If prefix parity fails, stop and investigate the mechanical discrepancy
before proceeding. Do not treat a mismatched prefix as a clean budget contrast.
Save availability-gradient/loss inputs at steps1,1200,4800 for independent audit.
The only new task candidate is the final4800 checkpoint. No early-stop selection,
checkpoint sweep, seed/weight/LR search, cutoff change or subsequent extension.

Inference stays predicted availability logit>=0, frozen MZ20 task cutoffs and
add-only MZ5 fallback. Native labels are training/evaluator-only; known0 means
missing native<=4m availability, not no obstacle, FREE_RAY or CLEAR. Every MZ5
positive must remain unchanged and no positive may be created beyond MZ20.

Evaluate the original oldDEV, clean/stress, relationDEV2000, distanceDEV1000 and
both-chain wrong-zone controls. Report TP additions/losses, addedFP, group/site
counts, availability and contributor rejection. Useful effect still requires
zero addedFP on normal cohorts, >=68/75 far additions with gains on both placement
cohorts, trained pole>=48/49 clean/stress and all baseline positives retained.
The pole contains no negative-availability bags, so pole retention alone cannot
establish hard-negative separation. All thresholds are experimental utility
criteria, not safety criteria.

After the final fit, compare MZ24 and MZ25 extrema on the same7562uniqueTRAIN
and disjointoldDEV1000/consumedplacementDEV3000. Distinguish reduced TRAIN overlap
from transfer and true-witness loss. No model updates or threshold selection in
this diagnostic. A plateau after one longer budget is not a mathematical capacity
ceiling. If errors fall only by deleting witnesses, reject the apparent gain.

Use fresh `artifacts.local/work/mz25-availability-convergence-20260910/` children.
Bind code/input/output hashes; independent audit checks prefix, batches, scalar
loss/gradients, task outputs, support and selected model inference. Finish the
report/terminal/scoped delivery and release processes/temporary resources after
the fixed run. No new capture, protected EVAL, App, temporal, device or safety
promotion is part of this experiment. The broader improvement goal remains active.
