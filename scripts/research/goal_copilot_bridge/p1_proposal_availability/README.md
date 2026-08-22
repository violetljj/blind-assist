# P1-PA0 target candidate availability

状态：`P1_PA0_TOP1_COLLAPSE_SIGNAL_ON_FAILURE_COHORT / AMRM_AND_VERIFIER_FROZEN / DEFAULT_APP_UNCHANGED`

P1-PA0 asks only whether a correct target candidate enters an ordered pool bounded at ten candidates when the target is
visible. Its seven cases are the post-outcome-selected visible first-poison frames from the sealed P1-AMRM0 canary, so
they can diagnose mechanics but cannot select a production generator or establish held-out/generalized performance.

The provider sees only the current RGB frame and the frozen frame-0 target exemplar/bbox. ADT visibility, future bbox,
category, instance name, and object UID remain private evaluator inputs. Memory, reacquisition, identity selection,
verifier, VLM, VIO/SLAM, geometry, and App integration are outside this experiment.

Primary reporting is target-visible candidate Recall@1/3/5/10 at the existing ADT proposal IoU threshold of 0.10, with
0.30 and 0.50 sensitivity, first-correct rank, background-only rate, shortest-side strata, proposal counts, latency,
and compute. If a provider does not expose pre-filter/pre-NMS proposals, generation-versus-retention attribution is
`NOT_EVALUABLE_PROVIDER_INTERFACE` rather than fabricated.

The frozen YOLOE-26n-seg visual-prompt arm produced Recall@1/3/5/10 of `0/7, 0/7, 0/7, 2/7` at IoU 0.10. The
two hits first appeared at ranks 9 and 10; all K were `0/7` at IoU 0.30 and 0.50. This is a weak top-1-collapse
mechanism signal, not adequate candidate availability or a generator-selection result. See the
[result](../../../../docs/research/goal-copilot/P1_PA0_TARGET_CANDIDATE_AVAILABILITY_RESULT_2026-08-22.md).
