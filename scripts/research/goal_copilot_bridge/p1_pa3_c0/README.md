# P1-PA3-C0 public Goal Contract cohort materialization

状态：`PROSPECTIVE_INTAKE_READY / EXISTING_ELIGIBLE_EPISODES=0 / PA3_INFERENCE_NOT_AUTHORIZED`

This surface records provider-public user/product task semantics before episode capture and before target truth exists.
It derives `canonical_prompt` only through one globally frozen exact `goal_type` mapping. Per-episode prompt overrides,
target/category/instance/bbox/mask/evaluator fields, historical backfill, model calls, PA3 inference, identity selection,
AMRM, verifier, and App integration are forbidden.

The existing P1-D0/PA0, Silver-B, and Last-10m assets are not admitted: they respectively lack public goal text, derive
goal text during truth-bearing annotation, or template it from a supplied target/site name without an original
pre-truth user-task receipt. The empty template is intentionally not a materializable cohort. A future prospective
intake must contain at least one real `USER_TASK_INPUT` or `PRODUCT_TASK_INPUT` record, with capture `NOT_STARTED` and
truth `NOT_CREATED` at goal recording. The output does not self-certify temporal precedence: future private truth must
bind the immutable goal-receipt body SHA-256 before an admission step may confirm `created_before_truth`. The materialized
receipt remains `pa3_inference_authorized=false`.

Focused mechanics check:

```powershell
E:\codex-tools\bin\blindassist-python.cmd -m unittest scripts.research.goal_copilot_bridge.p1_pa3_c0.test_materialize
```
