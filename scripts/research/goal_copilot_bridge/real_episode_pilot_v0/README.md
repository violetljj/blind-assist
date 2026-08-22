# Public-real episode mining + selective-guidance pilot V0

状态：`FROZEN_8X89_EXECUTED_AND_SEALED / TRUTH_SUBSTRATE_AUDITED / ABOTN_WEBGL_RENDER_TRANSPORT_PASS / ABOTN_ARRIVAL_TRUTH_ONLY / FUNCTIONAL_PIXEL_REGION_NOT_ESTABLISHED / NO_P1 / DEFAULT_APP_UNCHANGED`

This package automatically converts public real-world sequence metadata into goal-driven approach episodes, reuses
frozen current-frame provider output, applies Selective Guidance V0, and evaluates only truth-supported denominators.
The prospective entrypoint rejects a roster unless its public Goal Contract and entrance candidate set were frozen
before Mapillary metadata, pixels, model output, and evaluator truth.

Truth authority is ordered as `NATIVE_GT / MAP_TRAJECTORY_DERIVED / TEACHER_SUPPORTED / TEACHER_ONLY_WEAK / UNKNOWN`,
then manual annotation as a last resort. Missing exact frame-region visibility truth never becomes
a fabricated negative. `UNIQUE`, `SET_VALUED`, and `AMBIGUOUS` remain distinct.

Annotation V1 preserves five truth-authority tiers, raw independent `teacher_A/B/C` outputs, agreement/disagreement,
functional authority, and native/map authority sources. Teacher consensus alone cannot establish functional truth.
The evaluator refuses unfrozen truth and reports denominators plus failure attribution separately by authority tier;
UNKNOWN remains visible as cohort coverage and never enters accuracy.

```powershell
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.public_real_mining prospective `
  --goal-roster <frozen-public-goals.json> `
  --mapillary-metadata <sequence-metadata.json> `
  --output-dir <new-output-dir>

python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.baseline `
  --public-manifest <public.json> --provider-observations <provider.json> `
  --config scripts/research/goal_copilot_bridge/real_episode_pilot_v0/baseline_config.json `
  --output <prediction.json>

python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.evaluate `
  --annotation <annotation.json> --prediction <prediction.json> --output <evaluation.json>
```

`adapt-consumed-replay` is a smoke-only adapter for the sealed Last-10m Mapillary sequence. It labels the source
`PROJECT_CONSUMED_DEVELOPMENT_PIPELINE_SMOKE_ONLY`; it cannot establish freshness or performance. The current fresh
successor is automatic Mapillary + OSM/Overture mining. ADT is calibration/mechanism support, Ego4D is domain-realism
support, and Habitat is explicit-goal mechanics support. Physical capture is not a current blocker and is considered
only if public data cannot answer a separately stated, high-value question.

The executed prospective V1 froze four unused, venue-taxonomy-gated goals before Mapillary access, expanded 14 full
sequences from bbox-nearby metadata, and mined 8 episodes / 89 observations. It downloaded no pixels and made no model
calls. The next bounded stage may materialize only these frozen observations; it may not replace goals after outcome.

That bounded stage is now complete. `frozen_8x89_runner.py` froze the three local teacher identities, downloaded only
the 89 roster pixels, preserved all three raw outputs, froze region-based private truth before baseline candidate IDs
existed, then ran the unchanged Grounding DINO + Terra provider. The clean provider terminal is 89/89 with zero
`in_doubt`. Truth coverage is `strong native/map-only=0 / teacher-supported weak usable=4 / teacher-only weak=19 /
UNKNOWN=66`; all eight episodes remain `TRUTH_OR_CONTRACT_INSUFFICIENT`. No automatic algorithm successor follows.

The independent private regions are matched to later provider candidates at fixed IoU 0.30. This removes candidate-ID
dependence without allowing the baseline to run before truth freeze. Public map range shared with the provider is
explicitly non-claimable as independent range accuracy.

Before that stage, the per-frame contract was hardened to emit `NOT_VISIBLE`; only the episode interaction FSM may
derive `LOST` from `VISIBLE -> NOT_VISIBLE`. These are mechanical validity gates, not performance gates.

No component adds tracking, re-ID, persistence, world memory, VIO/SLAM, model/threshold search, completion authority,
or default-App integration. Public media cannot support a blind-user effectiveness claim.

`audit_abotn_poibench_truth_source.py` is the bounded post-8x89 substrate canary. It pins the public dataset and
evaluation repository revisions, downloads only the small task JSON files, and distinguishes named-POI metric-arrival
truth from frame-region truth. It must report metric endpoints without inventing entrance boxes. It also detects that
the official evaluator exposes `target_position` and `distance_to_goal` to an agent; any later BlindAssist visual
provider adapter must remove those evaluator-private fields.

```powershell
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.audit_abotn_poibench_truth_source `
  --cache-dir artifacts.local/datasets/abotn-poibench/fbb62cc3 `
  --output artifacts.local/evidence/abotn-poibench-truth-source-v0/audit.json
```

`audit_abotn_render_runtime.py` then checks the pinned official renderer requirements against the current host before
any scene payload is downloaded. A host below the official 24 GB VRAM minimum closes locally as `NOT_EVALUABLE` with
zero render, teacher, and provider calls.

The separate `abotn_webgl_canary` is an explicitly unofficial renderer-mechanics adapter. It pins the smallest public
ABotN PLY plus the lexicographically first task, maps only its initial camera pose into the rotated point-cloud frame,
and serves no goal, endpoint, distance-to-goal, teacher output, or private truth to the renderer. A successful receipt
establishes only real-RGB transport and arrival-substrate mechanics; it does not establish official-renderer pixel
equivalence or functional entrance-region truth.

```powershell
Push-Location scripts/research/goal_copilot_bridge/real_episode_pilot_v0/abotn_webgl_canary
npm install --ignore-scripts
node run.mjs --ply <point_cloud_rotated.ply> --annotation <traj_0.json> --output-dir <new-output-dir>
Pop-Location
```

```powershell
python -m unittest `
  scripts.research.goal_copilot_bridge.selective_guidance_v0.test_contract `
  scripts.research.goal_copilot_bridge.real_episode_pilot_v0.test_pilot
```
