# Public-real episode mining + selective-guidance pilot V0

状态：`FROZEN_8X89_EXECUTED_AND_SEALED / TRUTH_SUBSTRATE_AUDITED / ABOTN_OFFICIAL_RENDER_CANARY_PASS / ABOTN_SEALED_FAILURE_RENDERER_CONFOUNDED / ABOTN_ARRIVAL_TRUTH_ONLY / FUNCTIONAL_PIXEL_REGION_NOT_ESTABLISHED / NO_P1 / DEFAULT_APP_UNCHANGED`

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

`audit_abotn_official_pixels.py` closes a separate release-tree ambiguity. It inventories every file in the pinned
dataset revision and binds the sealed task to its same-name official PNG. The released PNGs are trajectory
visualizations or occupancy maps, not pre-rendered camera observations; they cannot substitute for an official render
server or be exposed to the provider.

```powershell
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.audit_abotn_official_pixels `
  --action-graph-receipt artifacts.local/evidence/abotn-v0-action-graph-v0/freeze-receipt.json `
  --output-dir artifacts.local/evidence/abotn-official-pixel-availability-v0
```

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

`abotn_arrival_provider_canary.py` freezes the same task's public RGB/goal envelope separately from evaluator-private
metric arrival truth, preflights the exact provider before creating its formal directory, and permits one provider
observation only. The completed canary used the frozen Grounding DINO + Terra V0 runtime and returned seven proposals
plus `ABSTAIN_NO_RELIABLE_EVIDENCE` in one brain attempt. Its prompt/JSONL audit found no private endpoint or distance
literal and no tool/command event. That establishes provider-firewall and interface mechanics only: absent functional
pixel-region truth and a closed control loop, selection, bearing, range, and arrival success remain `NOT_EVALUABLE`.

```powershell
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.abotn_arrival_provider_canary `
  --annotation <traj_0.json> --webgl-receipt <render-receipt.json> `
  --output-dir <new-output-dir> --grounding-dino <frozen-model-dir>
```

The same adapter's `--mode trajectory` freezes all source poses before rendering and materializes the complete
trajectory without provider, teacher, or baseline calls. The sealed task produced 89/89 nondegenerate 1280x720
frames with 89 unique hashes. `audit_abotn_trajectory_denominator.py` then verified every frame hash and evaluator-
private endpoint distance: 75 poses are outside 2 m, 14 are within 2 m, and the first within-threshold pose is index
75. This is a source demonstration, not a provider-controlled path, so it establishes a usable pixel/arrival
denominator but not control success. Additional provider calls remain closed until the existing V0 action semantics
have a renderer adapter; the expert trajectory cannot be credited to the algorithm.

```powershell
node scripts/research/goal_copilot_bridge/real_episode_pilot_v0/abotn_webgl_canary/run.mjs `
  --mode trajectory --ply <point_cloud_rotated.ply> --annotation <traj_0.json> --output-dir <new-output-dir>
python -m scripts.research.goal_copilot_bridge.real_episode_pilot_v0.audit_abotn_trajectory_denominator `
  --annotation <traj_0.json> --pixel-receipt <terminal-receipt.json> --output <arrival-denominator.json>
```

The closed-loop successor reuses, rather than redesigns, the sealed responsive V0 action contract:
`TURN_LEFT / TURN_RIGHT / FORWARD / RESCAN_HOLD`, five fixed yaw states at 12 degrees, a fixed approximately 2 m
source-path forward step, and the existing 12-instruction/three-unreliable-observation limits. It freezes a public
445-node graph separately from evaluator-private endpoint distances, then renders and hashes every node before any
provider call. All 445 frames were nondegenerate and unique. Fixed ORB checks on five preselected poses confirmed
that left/right actions move image features in the expected directions; endpoint/distance literals remained absent.

The one authorized V0 episode then stopped after exactly three fresh observations. Grounding DINO produced 4/4/5
proposals, but the unchanged Terra brain returned `ABSTAIN_NO_RELIABLE_EVIDENCE` three times because the blurred or
obscured views did not reliably identify the named POI. The FSM correctly entered `ABSTAIN` with zero reliable
groundings, zero instructions, zero arrival confirmations, zero false arrivals, and no `in_doubt`. The supported
engineering attribution is `CURRENT_FRAME_GROUNDING_BOTTLENECK`; range/bearing, control, and `LOST_AFTER_VISIBLE`
were never reached. Proposal count alone is not target recall, so `PROPOSAL_MISS` versus referent-selection cannot be
adjudicated without functional pixel truth. Official-renderer equivalence is also unavailable; consequently the result
cannot distinguish provider semantic-identity weakness from alternate-renderer fidelity and does not authorize an
algorithm rescue, P1, or a rerun.

The subsequent zero-model canary pinned the official ABotN renderer at commit `2a0aefb5`, verified the exact scene
PLY and CUDA-extension hashes, and rendered only source pose indices 0/1/2 with the official 720x640 front camera.
All 3/3 PNGs passed local hash/dimension/call-count audit; server POST count was exactly three and provider, teacher,
baseline, and sealed-episode-rerun counts remained zero. A read-only binding to the sealed WebGL inputs then confirmed
the same scene/poses/front-view semantics while the renderer and camera envelopes differ, and all three already-sealed
provider rationales independently cite obscured, indistinct, or blurred input. The supported conclusion is therefore
`ABOTN_SEALED_FAILURE_RENDERER_CONFOUNDED`: the prior current-frame failure cannot be assigned solely to provider
semantic selection. It does not establish provider success on official pixels, functional entrance-region truth,
proposal miss versus referent selection, or accuracy. Any later provider run must be a fresh prospective official-pixel
episode, never a replay of the sealed episode.

```powershell
python -m unittest `
  scripts.research.goal_copilot_bridge.selective_guidance_v0.test_contract `
  scripts.research.goal_copilot_bridge.real_episode_pilot_v0.test_pilot
```
