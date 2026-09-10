# MZ10: fixed candidate-availability composition

EXPLORE, 2026-09-10. User-approved rule, frozen before replay outcomes: for
each query independently, use MZ9 SOURCE_RGB's existing thresholded logit when
its geometric candidate support exists; otherwise use retained MZ5's original
zero-threshold logit. Support uses observed range/validity and fixed ray/query
geometry only. It is not source-contributor truth, confidence, or observed
correctness. No training, calibration, threshold change, pooling variant or
oracle branch choice. Original independent alert outputs remain untouched.

Replay old DEV1000 and consumed MZ6 training/regression200, clean and stress.
Compare frozen MZ5, MZ9 MZ5_ADAPT, standalone SOURCE_RGB, and fixed composition.
Do not call these independent confirmation. Preserve the original scoring batch
size32 for MZ9 threshold-boundary numerical replay. Report branch selection,
candidate absence, recovered/lost TPs and new/removed FPs per query, including
the individual errors selected from both branches.

Admission to fresh confirmation: on DEV, no per-query FP increase versus MZ5,
no BODY_NEAR/HEAD_NEAR TP loss, no total far TP decrease; on MZ6, no per-query FP
increase, no near TP loss, total far TP improvement and at least the standalone
SOURCE_RGB thin48/49 retained. These tests apply to composition as a whole;
separate branch budgets do not imply the combined budget holds.

If all pass, freeze one new paired capture up to200 frames with changed thin
within-zone positions, backgrounds and boundary placements; whole configurations
separate from training/DEV. Compare all three fixed practical methods without
refitting. If admission fails, stop this rule and report the failure; do not
rescue it with retrospective thresholds or additional routing variants. Temporal
stays off. Invalid/no-return does not mean CLEAR. No deployment/hardware claim.
