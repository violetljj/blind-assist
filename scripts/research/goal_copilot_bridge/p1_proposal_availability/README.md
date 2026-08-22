# P1 proposal availability: PA0 and PA1

状态：`P1_PA1_FIXED_TILED_SCALE_RESCUE_NOT_SUPPORTED_ON_FAILURE_COHORT / AMRM_AND_VERIFIER_FROZEN / DEFAULT_APP_UNCHANGED`

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

## P1-PA1 fixed tiled rescue

PA1 reused the sealed PA0 cohort/provider and changed only the input search to fixed 2x2 tiles with 20% overlap, each
resized to 640. It retained per-tile postprocessed candidates, global dedup decisions, full rank, and bounded-K removal.
The preregistered IoU 0.30 Recall@10 endpoint remained `0/7`, including `0/7` over the full postprocessed rank. PA0's
five permissive-IoU absent cases were rescued `0/5`; 4x input images increased proposal volume without creating a new
adequate target candidate. See the [PA1 result](../../../../docs/research/goal-copilot/P1_PA1_TARGET_PROPOSAL_RESCUE_RESULT_2026-08-22.md).

## P1-PA2 oracle representation observability audit

PA2 is a consumed-Development autopsy, not a formal provider arm. It deliberately uses private GT to construct an exact
target crop and a fixed 3x target-centred ROI, then compares the frozen target-only visual prompt with one fixed 2x
exemplar-context prompt. The same YOLOE checkpoint, 640 input, 0.001 confidence floor and complete provider-postprocessed
rank are retained. Since the provider API exposes neither prompt-embedding similarity nor pre-NMS proposals, the audit
reports operational recognizability/localization and marks those internal attributions `NOT_EXPOSED_BY_PROVIDER_INTERFACE`.

Run the focused mechanics check with:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest scripts.research.goal_copilot_bridge.p1_proposal_availability.test_pa2
```

The result and exact command are recorded in
[`P1-PA2 result`](../../../../docs/research/goal-copilot/P1_PA2_TARGET_REPRESENTATION_OBSERVABILITY_AUDIT_RESULT_2026-08-22.md).
