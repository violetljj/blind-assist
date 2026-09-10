# MZ7 single-frame paired readout diagnostic

EXPLORE, 2026-09-10. Retain frozen MZ5; temporal inference stays closed.
First inspect the already consumed MZ6 capture-v3 (200 settled samples, four
target/empty clip pairs). No new collection or training in this diagnostic.
Verify admission, cache/checkpoint identity, identical paired cameras and
non-target objects; target attribution is evaluator-only. Export all four raw
RGB and ToF logits for clean and missing-return packets and target-minus-empty
deltas. Reproduce stored CURRENT predictions. Count opportunities, not independent
events; summarize by configuration and only use actual positive target bits
whose paired empty bit is negative for target-specific response analysis.

For each missed positive distinguish one branch positive from both negative.
Report continuous delta distributions and whether the correct bit increases
more than every currently false bit; do not invent a post-hoc small-delta cutoff.
These tests localize a readout failure, not prove information absent from RGB.

Decision: branch-positive losses justify a small scale calibration comparator;
both-negative but selective positive deltas justify a readout-boundary comparator;
nonselective responses direct future work to representation/return attribution.
Select only one modification after this diagnostic; write its parameters and
fitting source before running it. Existing 200 samples remain Development.
Any retained upgrade needs far/thin recovery with no larger false-positive budget
per query, including near false activations, plus new-configuration confirmation.
Spatial attribution additionally needs a wrong-zone correspondence control.
Do not collect confirmation for a comparator which fails the Development gate.

Temporal reopening requires complete-evidence detection, degradation on deletion,
and recovery from observable history. Native truth/pose/identity may score but
must never select a branch or enter model inference. Invalid packets stay UNKNOWN;
four negative flags are not a clearance or safety certificate.

## Selected comparator after paired-v2, before calibration outcomes

The clean thin clip has both branches negative on all 49 far opportunities;
all 14 approaching HEAD_FAR positives have correct ToF suppressed by RGB.
Run one small calibration comparator, not an attribution network: fit eight
positive inverse temperatures (one per branch/query) by binary cross entropy
on the old MZ1 DEV_ONLY 1000 frames, then choose four thresholds maximizing DEV
true positives with no more false positives per query than frozen MZ5 on that
same DEV set. Ties prefer fewer false positives, then the highest threshold.
Temperature optimization: double precision CPU LBFGS, initial log-scale zero,
100 iterations, strong-Wolfe search; scales bounded exp(-6)..exp(6).
Thresholds enumerate distinct calibrated DEV scores plus an all-negative option.
No MZ6 labels enter fitting; no EVAL_ONLY access is needed. This is a consumed
DEV fitted comparator, not an independent validation on DEV. Freeze parameters
before MZ6 scoring; apply unchanged to clean and stress. A failed thin/far or
per-query FP gate ends this comparator; no fresh confirmation collection then.
