# Native-depth routing oracle with fixed sensitivity checks

2026-09-09 EXPLORE. Zero fitting/inference; frozen B count probabilities and
original320 native arrays. Privileged oracle only, not RGB-only performance.
Native depth generates the labels, so exact ownership is not independent evidence
that absent metric depth caused the learned model's failures.

Read nearest native pixel at each valid lattice projection. Depth is camera axial
X in metres, not Euclidean ray length. Save observed-query signed/absolute
residuals. Compare seven fixed cell reliability gates: axial interval witness;
full XYZ cell witness; max Gaussian residual gate with sigma0.375m (one quarter
of1.5m bin width); full XYZ with observed axial depth scaled0.9/1.1 or offset
-0.2/+0.2m. No fitted tolerance or sweep. Geometry uses existing boundary rules.

For each cell, gate is maximum of valid sample gates. Native UNKNOWN samples
pass through with gate1, rather than serving as negative evidence; out-of-FOV
points are excluded. Record how often unknown alone forces passthrough. This
conservative convention intentionally differs from selecting only known truth.

Apply gate g to B's nonempty count probabilities, transfer removed mass to class0:
p'[1:]=g*p[1:]; p'[0]=1-g*(1-p[0]). Count classes retain their original units and
near_from_counts semantics. No sum of sampled witnesses becomes a native count.
Do not claim this cell-level veto exhausts point-feature depth injection methods.

For each gate persist original-policy DEV frame cutoffs before EVAL reporting;
also show original B cutoffs and fixed EVAL oracle TP at FP<=2 as diagnostics.
Report BODY/HEAD, query AUC, groups/conditions and matched deltas, all frames.
Keep native UNKNOWN counts and sparse coverage limitations. Stop after seven
predeclared comparisons; no automatic monocular-model download or new fit.
Large headroom would justify a separately scoped predicted-depth transfer test,
not prove that a monocular prior can realize it. CPU scalar sampled-ray audit is
TASK_NOT_GPU_SUITABLE; native full-grid memberships are not recomputed.
