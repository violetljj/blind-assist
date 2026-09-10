# MZ9: source-contributor supervision and matched training controls

EXPLORE, 2026-09-10. MZ8's0.10m auxiliary rule rewards22 false region assignments.
Replace that surrogate with actual contributors to the simulator's selected
first/last supported radial bins. Preserve outside-query contributors and
multiple sources. Native contributor coordinates label only; inference cannot
read them. Do not change MZ0 packets, query boundaries, or the MZ8 max aggregator
in this first test. Keep MZ5 frozen as the retained reference; temporal stays off.

Reuse MZ8 cache-v5: old2500 TRAIN, old1000 DEV, and200 consumed MZ6 targeted
training/regression. Reconstruct each simulated packet from native data and
verify cached ranges/validity. For each zone/return/7x7 angular subcell record
contributor count and actual per-query contributor counts. A local query target
is positive iff that subcell has a contributor in that query. Ignore cells with
no known native samples; invalid returns never positive. No distance-tolerance
proxy or unique target identity. Query labels retain the original>=3-pixel rule.

Three bounded fits, seed109,1200 Adam steps each, lr0.001, batch16 using the same
eight old TRAIN/eight MZ6 sampled indices at each step. No backbone training.

1. MZ5_ADAPT: start from retained compact MZ5 weights, enable the two independent
   branch heads, minimize mean RGB/ToF BCE. Cheap additional-training alternative.
2. SOURCE_RGB: fresh small65->32->4 local evidence head, four query biases and
   existing angular eligibility/max. Query-specific outputs are necessary to
   express regional contributor supervision. Query BCE +0.25 balanced local
   contributor-support BCE. Same frozen dense RGB map and observed range input.
3. SOURCE_NO_RGB: same initialization/training/supervision/geometry, but set all
   visual tokens to zero. Same tensor architecture; report inactive input weights.

SOURCE arms differ only in RGB availability. MZ5 versus SOURCE comparison tests
practical alternatives with equal NEW sample exposure, not isolated architecture
causality: initialization, parameter count and auxiliary supervision differ.
Each arm fits four thresholds on old DEV to maximize TP under frozen MZ5 per-query
FP budgets, ties favor fewer FP then higher threshold. Unsupported SOURCE outputs
remain masked. Evaluate correct and four-column shifted RGB correspondence with
fixed weights/thresholds, plus existing artificial return-deletion packets.

Admission for a fresh confirmation: on MZ6 recover thin and far detections, no
per-query FP increase, no near TP loss; on old DEV no near TP loss and no decrease
in total far TP versus retained MZ5 under its FP budget. These are consumed/fitted
checks, not confirmation. Among qualifying arms choose higher MZ6 exact, then
total TP; ties prefer the existing MZ5 structure, then the RGB-free source arm.
RGB contribution
requires improvement versus SOURCE_NO_RGB plus sensitivity to wrong correspondence;
do not demand an RGB mechanism claim to retain a better simpler method.

Only a qualifying fixed candidate warrants a separately frozen up-to200-frame
paired confirmation with changed within-zone position/background/boundary layout,
whole configurations separate from training/DEV. Do not fit on confirmation.
If none qualifies, stop these three fits and report the remaining failure and
whether supervision or aggregation is implicated; no automatic pooling sweep.
Invalid observations remain UNKNOWN. No hardware, real-time walking, clearance
or safety claim; no temporal reopening in this experiment.
