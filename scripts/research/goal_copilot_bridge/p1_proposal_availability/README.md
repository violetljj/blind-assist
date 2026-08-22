# P1 proposal availability: PA0–PA3

状态：`PROSPECTIVE_PA3_DENOMINATOR_GATE_ENFORCED / PA3_INFERENCE_NOT_AUTHORIZED / AMRM_AND_VERIFIER_FROZEN / DEFAULT_APP_UNCHANGED`

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

## P1-PA3 goal-semantic proposal availability

PA3 has now run once on a legal consumed development cohort. It changes only the proposal
conditioning interface from a specific visual exemplar to the globally frozen semantic text prompt in a prospective,
pre-truth Goal Contract. `materialize_pa3_inputs.py` closes the C0 goal → capture → later private truth hash/time chain;
`run_yoloe_semantic_prompt.py` is GT-blind and keeps YOLOE-26n-seg, 640 input, 0.001 confidence floor, provider
`max_det=100`, and bounded K=10; `evaluate_pa3.py` opens truth afterward.

`UNIQUE` and `SET_VALUED` enter primary Recall@1/3/5/10 at IoU 0.30. `AMBIGUOUS` is proposal-set diagnostic only and
cannot be counted as specific-referent success/failure. Identity selection, visual exemplar fallback, per-episode prompt
override, AMRM/verifier, model/config sweep, and cross-cohort PA2-vs-PA3 superiority claims are forbidden.

Result: YOLOE semantic Recall@1/3/5/10 was `0/2` at IoU 0.30. A post-outcome anchor-containment diagnostic was
`1/2 @1` and `2/2 @3`, which does not relabel PA3 success. The pre-existing frozen Grounding DINO functional prompt
then produced FRG1 Recall@10 `1/2` on the same consumed cohort. See
[`P1-PA3 + FRG1 result`](../../../../docs/research/goal-copilot/P1_PA3_GOAL_SEMANTIC_AND_FUNCTIONAL_REGION_RESULT_2026-08-22.md).

Future PA3 runs first require `authorize_pa3.py` to bind the public/private inputs, one prediction path, and one dispatch
journal. It writes `pa3_inference_authorized=true` only at `>=5` visible episodes and `>=8` visible frames. The semantic
runner requires that receipt before reading provider assets or importing YOLOE, creates the journal before the first
prediction, journals every dispatched/completed case, and permanently refuses any existing journal or output. A failed
or interrupted run is sealed rather than retried. The private evaluator accepts only a prediction bound to the same
authorization and a completed journal.

The runner also requires `--text-encoder artifacts.local/models/mobileclip2_b.ts`; its SHA-256 is frozen and loaded
locally, preventing Ultralytics from downloading an unmanifested text encoder at runtime. Authorization permits only
the YOLOE semantic arm; FRG, identity, prompt/model/threshold/pool sweeps and replay remain forbidden.

The frozen execution sequence is:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.p1_proposal_availability.authorize_pa3 `
  --public artifacts.local/<run>/pa3/public_input.json `
  --private artifacts.local/<run>/pa3/private_eval_input.json `
  --prediction-output artifacts.local/<run>/pa3/prediction.json `
  --dispatch-journal artifacts.local/<run>/pa3/dispatch.json `
  --authorization-output artifacts.local/<run>/pa3/authorization.json

# Run only when authorization.json says pa3_inference_authorized=true.
E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.p1_proposal_availability.run_yoloe_semantic_prompt `
  --public artifacts.local/<run>/pa3/public_input.json `
  --prompt-map artifacts.local/<run>/prompt_map.json `
  --model artifacts.local/models/yoloe-26n-seg.pt `
  --text-encoder artifacts.local/models/mobileclip2_b.ts `
  --authorization artifacts.local/<run>/pa3/authorization.json `
  --dispatch-journal artifacts.local/<run>/pa3/dispatch.json `
  --output artifacts.local/<run>/pa3/prediction.json

E:\codex-tools\bin\blindassist-python.cmd -m scripts.research.goal_copilot_bridge.p1_proposal_availability.evaluate_pa3 `
  --public artifacts.local/<run>/pa3/public_input.json `
  --private artifacts.local/<run>/pa3/private_eval_input.json `
  --prediction artifacts.local/<run>/pa3/prediction.json `
  --authorization artifacts.local/<run>/pa3/authorization.json `
  --dispatch-journal artifacts.local/<run>/pa3/dispatch.json `
  --output artifacts.local/<run>/pa3/evaluation.json
```

Focused check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest `
  scripts.research.goal_copilot_bridge.p1_proposal_availability.test_pa3
```

Implementation status:
[`P1-PA3 implementation ready`](../../../../docs/research/goal-copilot/P1_PA3_GOAL_SEMANTIC_PROPOSAL_IMPLEMENTATION_READY_2026-08-22.md).

## P1-HRG2 global-local reranking

HRG2 is the frozen successor to the fresh HRG1 failure. It does not add a spatial hint because the current product
interface has no independently proven pre-truth route-endpoint contract; an OSM entrance bearing remains private truth.
Instead, HRG2 processes all ten already-bounded HRG0 coarse regions, keeps the same two crop-local Grounding DINO
proposals per parent, globally orders them by local provider score, applies the already frozen class-agnostic NMS IoU
of 0.50, and returns at most ten candidates. Parent rank and local rank are deterministic tie-breaks only.

The model, prompt, thresholds, HRG0 parent prediction, local proposal count, final K, and identity prohibition are
unchanged. The consumed HRG1 cohort must not be rerun with HRG2; execution requires a newly frozen Goal Contract,
acquisition, and pre-provider truth cohort.

The public-goal-anchor observation path now honors its frozen `selected_per_episode` field (bounded at three) instead
of silently forcing one frame. Selection remains metadata-only: camera position, heading, distance to the public named
place anchor, capture time, and viewpoint separation. It does not use an OSM entrance node, downloaded pixels, private
visibility, target boxes, or model outputs.
