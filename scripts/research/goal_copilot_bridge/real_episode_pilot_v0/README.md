# Public-real episode mining + selective-guidance pilot V0

状态：`FROZEN_8X89_EXECUTED_AND_SEALED / TRUTH_SUBSTRATE_AUDITED / ABOTN_FRESH_COHORT_0_OF_8_COMPLETION_1_OF_8_METRIC_ARRIVAL / CMP_NATIVE_DOOR_89_SELECTION_COMMITMENT_DOMINANT / NO_P1 / DEFAULT_APP_UNCHANGED`

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

A fresh prospective successor then selected the lexicographically first unused task before seeing its pixels or any
provider output: `20260212121852/traj_0` (`麦当劳`). It froze 73 source poses, 365 action nodes, the unchanged provider
lock, native metric endpoint truth, and a maximum of 15 observations. The official renderer produced 365/365 unique
720x640 frames with exactly 365 successful server POSTs and zero provider calls before qualification. All five fixed
ORB turn-direction checks passed and the private endpoint/distance literals remained absent from the public graph and
provider inputs.

The one-shot unchanged V0 made five provider calls with zero `in_doubt`: `ABSTAIN -> GROUNDED -> ABSTAIN -> ABSTAIN ->
ABSTAIN`. Its single `FORWARD` instruction reduced native endpoint distance by 2.031 m; total terminal progress was
2.465 m after separating the simulator's non-instruction `RESCAN_HOLD -> next source pose` updates. The episode stopped
at 7.738 m, outside the 2 m arrival domain, with no completion or false arrival. The terminal is therefore
`ABOTN_OFFICIAL_V0_PARTIAL_METRIC_PROGRESS_NO_GOAL_SUCCESS`. Functional entrance-region truth is still absent, so the
one provider commitment cannot establish selection correctness, and the later abstentions cannot self-authorize a
`VISIBLE -> NOT_VISIBLE_AFTER_VISIBLE` transition. `PROPOSAL_MISS`, referent selection, wrong-confident guidance,
`LOST_AFTER_VISIBLE`, and range/bearing bottlenecks remain unsupported. No tuning, rerun, P1, or product claim follows;
dominance requires a separately frozen fresh cohort.

That cohort is now sealed. Before any new pixel or provider output, the executor fixed the first eight unused tasks
from scene `20260212121852` (`traj_1` through `traj_8`), the unchanged V0/provider lock, native metric truth, 3,190
official render calls, and at most 120 provider observations. All 3,190 official frames passed byte/hash,
nondegeneracy, per-episode uniqueness, 40/40 fixed ORB turn-direction checks, and private-literal firewalls. The
shared official server log contains exactly 3,190 successful render POSTs; render/provider `in_doubt` remained zero.

The eight sealed runs used 82 provider observations. Seven episodes had positive instruction-attributable metric
progress, one reached the native `<2 m` arrival domain, none emitted a completion confirmation, and false arrival was
`0/8`. Initial-to-terminal distances were `11.191→5.774`, `10.373→0.033`, `9.161→3.173`, `9.904→5.829`,
`8.743→2.874`, `8.665→2.620`, `12.401→12.118`, and `17.767→14.745` m. Under the precommitted aggregation rule,
current-frame reliability limited 5/8 episodes and is the supported single-scene majority failure. This does not
identify acquisition versus proposal versus referent selection: functional entrance-region truth remains absent,
selection accuracy and `LOST_AFTER_VISIBLE` remain `NOT_EVALUABLE`, and P1 is still unauthorized. The simultaneous
`0/8` completion and one arrival-without-confirmation also establish a control/termination problem; neither finding
may be rescued by replaying or tuning this consumed cohort.

The functional-region truth gap was then repaired with a separate, non-episodic CMP Facade cohort rather than by
relabeling or replaying ABotN. Before baseline execution, all 606 official RGB/XML/PNG triples were inventoried and
the 211 images with exactly one native XML `door` instance and non-empty native door pixels were ranked by a frozen
SHA-256 rule; the first 89 were sealed. The provider saw only RGB, the literal goal `the door`, and frozen Grounding
DINO boxes. Native PNG door pixels remained evaluator-private. The unchanged V0 completed 89 proposal observations
and 12 Terra batches with zero `in_doubt`, retry, teacher call, or rerun.

At IoU 0.50, usable proposals existed for `82/89`; final native-PNG Recall@1/3/5/10 was `44/67/73/82`.
Final outcomes were `45 CORRECT_GROUNDING / 7 PROPOSAL_MISS /
30 REFERENT_SELECTION_ABSTAINED_WITH_USABLE_PROPOSAL / 7 WRONG_CONFIDENT_GUIDANCE`. The Brain committed on 55
observations and was correct on 45, while 10/89 total observations produced wrong confident guidance after including
three commitments whose correct proposal was absent. Thus proposal availability is not the dominant first failure in
this constrained current-frame domain; referent selection and selective commitment account for 37 of 44 failures.

The initial post-run evaluator accidentally interpreted CMP XML `<x>` as horizontal. A read-only repair used the
already-frozen official PNG pixels as region truth and confirmed that CMP XML axes are transposed: mean normalized
PNG/XML edge difference is `0.001307` after swapping versus `0.434569` without swapping. No model output was replayed.
This result supports a fresh-cohort selective-commitment/CONTESTED investigation, but it is limited to rectified
facade stills with a generic door goal. Named-store identity, approach control, range, bearing, arrival, `LOST`, P1,
and product effectiveness remain unevaluated.

The [fresh selective-commitment confirmation](../../../../docs/research/goal-copilot/BLINDASSIST_CMP_SELECTIVE_COMMITMENT_V1_RESULT_2026-08-23.md)
is now sealed negative. The 122 unused native-door images were frozen
before provider output as `32 Development / 64 Confirmation / 26 Reserve`, with zero overlap against the consumed 89.
Development selected the predeclared `confidence >= 0.75 AND provider rank == 1` gate. On 64 untouched Confirmation
images, V0 made 43 commitments with 31 correct and 12 wrong over all observations; the offline V1 retained 16
commitments with 14 correct and 2 wrong. Commitment accuracy improved from `72.1%` to `87.5%`, but correct-grounding
retention was only `14/31 = 45.2%`, below the frozen 80% requirement. The verdict is therefore
`SELECTIVE_COMMITMENT_NOT_SUPPORTED`. Reserve remains untouched; no policy, threshold, provider, prompt, goal, or
cohort rescue is permitted. Provider rank is not referent authority, and this simple gate does not establish a V1 or
authorize P1.

The [GroundBench strong-referent route](../../../../docs/research/goal-copilot/BLINDASSIST_GROUNDBENCH_REFERENT_RESULT_2026-08-23.md)
then replaced generic-door truth with RefCOCO-family expressions and public-dataset-derived polygons. On the first 89
frozen static COCO observations, V0 had usable proposals on 77, correct grounding on 65, 12 proposal misses, and 12
selection/commitment failures with a usable proposal. This established a mixed proposal-and-selection bottleneck
rather than a teacher-derived accuracy claim.

A fixed 24-category public lexicon union was evaluated only on fresh cohorts. The first attempt was sealed
`NOT_EVALUABLE_TRANSPORT_RUNTIME` before any V1 process after Windows rejected a 46,284-byte argv; moving the identical
prompt to stdin was the sole transport repair. An uncapped union increased both correct grounding and wrong
commitments, so it was sealed negative. Its Development result selected the minimal above-V0 cutoff, Top-10, before
the final untouched 64-observation Confirmation. There V1 increased proposal availability `57 -> 59`, but correct
grounding fell `47 -> 46` and all wrong commitments rose `13 -> 16`; maximum V1 candidates was exactly 10. The frozen
verdict is `DOMAIN_LEXICON_PROPOSAL_UNION_NOT_SUPPORTED`. No reserve activation or same-cohort rescue is permitted;
static COCO evidence does not authorize approach, control, range/bearing, arrival, `LOST`, P1, safety, or product claims.

Two candidate-level successors were then gated without reusing a Confirmation outcome. Zero-shot CLIP exact-crop,
expanded-crop, focused-context, and dual-view reranking all failed to improve Rank@1 on the consumed first 89, so no
fresh call was authorized for that route. A separately frozen 21-feature relational ranker trained on consumed
positions 1--217 and improved held-out 218--281 Rank@1 `37 -> 41`, authorizing positions 282--345. On that fresh paired
64, V0/V1 used identical candidate sets. The ranker improved correct grounding `41 -> 42`, wrong-all `20 -> 18`, and
MRR `0.7736 -> 0.7877`, but fresh Rank@1 tied `33 -> 33`; therefore the frozen mechanism clause failed and the verdict
is `RELATIONAL_CANDIDATE_RANKER_NOT_SUPPORTED`. The final eight identities remain untouched; no gate/model/feature
rescue, P1, or approach/product claim follows.

A further representation-only Development check reused the 77 consumed GroundBench observations with usable
proposals. The same Brain saw the unchanged whole scene plus enlarged numbered crops of the same candidates; candidate
sets, order, scores, provider, and private truth were unchanged. Across ten successful one-shot batches with zero
`in_doubt` or teacher calls, correct grounding fell `65 -> 63` and wrong confident guidance rose `5 -> 7`. The result
is sealed `CANDIDATE_ZOOM_REPRESENTATION_DEVELOPMENT_NOT_PROMISING`; no fresh Confirmation call or ABotN integration
was authorized, and crop/prompt/provider/cohort tuning cannot rescue the consumed Development evidence.

```powershell
python -m unittest `
  scripts.research.goal_copilot_bridge.selective_guidance_v0.test_contract `
  scripts.research.goal_copilot_bridge.real_episode_pilot_v0.test_pilot
```
