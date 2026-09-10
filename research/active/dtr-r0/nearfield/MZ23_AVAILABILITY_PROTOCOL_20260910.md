# MZ23: learn angular in-domain evidence availability

2026-09-10 EXPLORE under the active obstacle-improvement goal. MZ21 found all
MZ20 new false additions outside known native angular support. Unlike MZ22's
task-output arbiter, explicitly supervise a distinct evidence-availability
variable and preserve the existing MZ20 witness scores.

Target: whether any native pixel in the angular subcell has finite axial depth,
0<axial<100m and radial<=4m. This is exactly `cell_known_counts>0`, independent
of selected return bin or BODY/HEAD query. Zero can mean out-of-range, invalid
native data or no eligible mapped pixel; it is not empty space, no obstacle,
FREE_RAY or CLEAR. Do not change unknown occupancy/query labels. Stress changes
selected returns, not this scene-level target. Native labels remain training
and evaluator-only. Pixel footprints/cell boundaries are approximate support,
not measured precise3D collision points.

First, a zero-fit saved-score known-availability oracle tests whether perfect
availability would retain at least68/75 MZ20 placement far additions, remove
all8 newFP and keep pole>=48/49 clean/stress. This is diagnostic upper-bound
intervention for this mask, not a deployable model. If it fails, do not launch
this availability-veto fit; diagnose the support/coverage limit instead.

If useful, freeze MZ20 and MZ5 and train one independent availability head:
Conv64->16(3x3), local+zone features, angular coordinates and valid/range packet,
MLP40->32->1. Output is64x49 angular scores with no query/echo-specific labels.
Use original native224 ROI cache/normalization and exact saved MZ20 1200x16
batches, new seed123, Adam.001, class-balanced per-frame availability BCE.
One1200-step fit only; no warm start or source/threshold/hyperparameter sweep.

Inference restricts MZ20 geometrically eligible candidates to predicted
availability logit>=0. Reuse original MZ20 per-query alert cutoffs exactly;
then compose add-only with unchanged MZ5. Neither oldDEV nor placement fits a
new cutoff. As a pure restriction, it cannot add a positive beyond MZ20; it can
lose true additions. Explicit support=false invokes unchanged MZ5, never CLEAR.
Run clean/stress and wrong-zone correspondence controls with both local chains
using the same visual shift. All observations are consumed Development.

Report per-query removed/new TP/FP, retained75 far additions, availability
precision/recall and missed real contributor cells, pole clean/stress, and
family/group/site results. The useful candidate criterion is0 addedFP over MZ5
on normal cohorts, >=68/75 placement far additions retained with gain on both
placements, baseline near retained, pole>=48/49 clean/stress. This90% recovery
retention is an experiment utility rule, not a safety threshold. Score separately
whether each original baseline positive survives (must be all).

Validate invalid-input/mask monotonicity first; independently replay saved
predictions and threshold identity afterward. Outputs use fresh children of
`artifacts.local/work/mz23-availability-20260910/`; bind code/inputs/labels/batches.
Record any mechanical retry and retain failed evidence. Stop the fit at1200,
complete report and structured inheritance, scoped delivery and resource release.
No capture, protected EVAL, end-to-end latency, physical sensor, temporal or
default-App/safety claim is part of this comparison.
