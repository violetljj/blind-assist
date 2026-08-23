# ABotN official-waypoint HANDOFF_V1 result (2026-08-24)

## Headline

Replacing the bounded-yaw action graph with ABotN's pinned continuous waypoint evaluator removed the previously
observed action-edge exhaustion, but did not establish arrival or handoff success. On the fresh sealed `traj_3`
task, V0 reduced native goal distance from `12.5335 m` to `2.9600 m` and then exhausted its frozen 12-instruction
control budget while the current-frame provider was still returning `GROUNDED`.

The supported first-failure attribution is therefore:

`CONTROL_POLICY_BOTTLENECK_INSTRUCTION_BUDGET_AFTER_REPEATED_ALIGNMENT`

This is an engineering result from one public task. It is not a navigation, safety, product, selection-accuracy,
arrival, handoff, or completion claim.

## Frozen execution boundary

- BlindAssist adapter commit at the successful run freeze: `802f6062773bf7c1b63f8a4b2ebbe1405eb66bb7`.
- Official ABotN evaluator/renderer commit: `2a0aefb56f1e2d315bba924239e9e8ad9dca9d92`.
- Dataset revision: `fbb62cc3382d8ff84f7fe3b6a3e7d48e4c21e974`.
- Scene: `20260227163550`; fresh task: `traj_3`; public goal: `艺鑫造型`.
- Provider: the already frozen Grounding DINO Tiny proposal stream plus Codex Terra decision policy.
- Provider visibility: current true-front RGB and public POI name only.
- Withheld from provider: target position, distance, pose, rotation, heading, maps, history, and evaluator truth.
- Frozen controller: `2.0 m` forward step, `12 deg` in-place turns, 12 instructions, 15 evaluator steps.
- `HANDOFF_READY` remains distinct from metric arrival and completion.
- Occupancy and height maps were not cached; collision outcomes are `NOT_EVALUABLE_MAP_NOT_CACHED` despite the
  official result object's mechanical zero-collision fields.

The pinned renderer returns views as `left,right,front`, while the pinned evaluator assigns its input list as
`left,front,right`. The adapter repairs only that ordering at the renderer boundary so the public projection uses
the actual front view. No evaluator truth or metric is changed.

## Sealed adapter failure (`traj_2`)

The first fresh official-waypoint task completed three provider calls, all `RUN_SUCCESS`, before the controller
raised the pre-existing freshness contract:

`turn/advance/rescan requires a fresh observation and frame`

Cause: the initial adapter mapped `RESCAN_HOLD` to a zero-motion pose. The deterministic official renderer therefore
returned the same pixel SHA on observations 2 and 3. This task was sealed as
`ABOTN_OFFICIAL_WAYPOINT_RUN_FAILED`; it was not resumed or rerun.

- Freeze SHA-256: `6e75f66ad9b18ed56045a3a2505e5c9c5e852e748f5616ef0f0159f4e0e3624a`.
- Terminal receipt SHA-256: `47da5844db21717cd3596fce0e40c478ada8a72f61eb5f7a7dd0a73305d164eb`.
- Provider calls: `3/3` completed, `0` in doubt, `0` teacher calls.
- Algorithm outcome: not evaluable beyond the action-adapter freshness failure.

The minimum next-task repair mapped rescan to one frozen `12 deg` in-place sweep. It did not change the provider,
prompt, thresholds, goal, teacher policy, or any consumed output.

## Fresh official result (`traj_3`)

The repaired adapter produced 14 distinct current-front frame hashes and reached a normal sealed evaluator result:

| Measure | Result |
| --- | ---: |
| Provider observations | `14` |
| Brain attempts | `14` |
| Provider calls in doubt | `0` |
| Teacher calls | `0` |
| `GROUNDED` observations | `13/14` |
| Initial native distance | `12.5335 m` |
| Minimum/final native distance | `2.9600 m` |
| Net native progress | `9.5735 m` |
| Travel length | `10.0000 m` |
| Official arrival threshold | `2.0 m` |
| Official success | `false` |
| Handoff ready | `false` |
| Completion authority receipt | absent |

The action sequence was seven `TURN_LEFT` actions, five `FORWARD` actions, one fresh in-place rescan sweep, and one
terminal stop. The last observation was still `GROUNDED` with confidence `0.95`, center-x `0.49`, and candidate
height `0.30`. The state machine abstained because its instruction count had reached the frozen maximum, not because
the final current frame was unreliable.

- Freeze SHA-256: `e0832e6509ccb1cc527e11aabf63a675777a989207f8bb7c24afcd5dfdeed552`.
- Terminal receipt SHA-256: `310997551923325849aeeb105de37c3922f5d6b03ef643d1df39212f311d0083`.
- Provider private-field-name hits: none.
- Selection accuracy and wrong-target confirmation remain
  `NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING`.

## Interpretation

This run rules out the bounded-yaw graph as the immediate failure on this fresh task and provides positive evidence
that current-frame grounding plus the official continuous waypoint interface can make substantial metric progress.
It does not establish that the selected visual referent was functionally correct, because ABotN does not release a
functional entrance region for these pixels.

The run stopped at a declared control-policy boundary: repeated fixed-size alignment consumed most of the instruction
budget before the agent reached either the official 2 m arrival region or the visual `HANDOFF_READY` cue. A next
algorithm lane is legitimate only as a new, prospectively frozen control-policy study on unused tasks; changing the
budget or rerunning either consumed task would be a rescue and is forbidden.

## Bearing-aware control follow-up (`traj_4`)

Commit `166b849c9b51613afe5c4f08bfd1e7a3d74ecf63` introduced one prospectively declared control-only change for a new
unused task: turn magnitude was computed from the public current candidate center and the frozen camera intrinsics
(`width=720`, `cx=360`, `fx=252.075`), capped at `45 deg`. The instruction budget, 2 m forward step, provider,
prompt, thresholds, goal handling, handoff cue, evaluator, and truth firewall stayed fixed.

On `traj_4` (public goal `晓锋鸽子王`), the result was again negative:

| Measure | Result |
| --- | ---: |
| Provider observations / brain attempts | `15 / 15` |
| Provider calls in doubt | `0` |
| Distinct current-frame hashes | `15 / 15` |
| `GROUNDED` observations | `13 / 15` |
| Initial native distance | `18.3216 m` |
| Minimum/final native distance | `9.6973 m` |
| Net native progress | `8.6242 m` |
| Travel length | `10.0000 m` |
| Official success / handoff / completion | `false / false / false` |

The task still consumed seven alignment turns, five forward steps, and two rescan sweeps. The turn stream reversed
direction at high confidence (`+38.14 -> -25.22 deg` and `+37.77 -> -40.65 deg`) rather than converging monotonically.
The final current observation remained `GROUNDED` (confidence `0.90`, center-x `0.49`) and the controller again
stopped at its frozen budget boundary.

- Freeze SHA-256: `0128eb0de074ca6f1243c903d4fa1c4ae11ed891a4b3b21ecaa1dab07119821e`.
- Terminal receipt SHA-256: `8452a4073db8e1232ba93f8c8bc59297222af98cb7a02793ebde88aff7d556b1`.
- Provider private-field-name hits: none.
- Collision and functional selection outcomes remain not evaluable for the reasons above.

Because `traj_3` and `traj_4` are different tasks, their metric progress is not a causal comparison of the two turn
policies. The within-task evidence is sufficient only to reject the claim that calibrated turn magnitude by itself
removed the control bottleneck. It exposes high-confidence control-direction inconsistency, but cannot establish a
wrong referent without functional region truth. Further angle tuning would be outcome rescue. A future lane, if
explicitly authorized and prospectively frozen on unused tasks, should study selective commitment or an oscillation
guard rather than another turn-angle adjustment.

## Bearing-coupled servo follow-up (`traj_5`)

Commit `210679fcff3f1595f9931706cdf51d10c4b2005c` introduced one further control-only change before selecting a new
task: every reliable current-frame candidate produced a `2.0 m` local waypoint along its public image bearing, so
alignment and translation were coupled in one official evaluator action. The provider, prompt, candidate thresholds,
instruction budget, arrival threshold, goal handling, teacher policy, and private-truth firewall stayed fixed.

On fresh task `traj_5` (public goal `鲜牛羊肉`), this controller reached ABotN's native metric arrival region:

| Measure | Result |
| --- | ---: |
| Provider observations / brain attempts | `9 / 9` |
| Provider calls in doubt / teacher calls | `0 / 0` |
| Distinct current-front frame hashes | `9 / 9` |
| `GROUNDED` observations | `6 / 9` |
| Initial native distance | `11.2369 m` |
| Minimum/final native distance | `0.9246 m` |
| Travel length / shortest path | `12.0000 / 11.9788 m` |
| Official success / oracle success | `true / true` |
| Official SPL | `0.8316` |
| Visual handoff / completion | `false / false` |

The ninth current-frame decision was still `GROUNDED` (confidence `0.78`, center-x `0.599`, candidate height
`0.258`) and dispatched a bearing-coupled step of `-15.85 deg`. That action entered the official `2.0 m` arrival
region, where the pinned evaluator terminated before asking the agent to process step 9's newly rendered current
front. The pre-action image visibly contained the storefront; the unconsumed post-action image faced along its side.
The provider therefore never emitted `HANDOFF_READY`, and no post-hoc provider call was added.

- Freeze SHA-256: `fd736575aaeda70215977f6de2066f0dba6b6fe4b7736b6f1f425165f0497937`.
- Terminal receipt SHA-256: `e392edfab3e6a5a0df67ad220a39f56a4938fce12a631c2343ac0857dddba50e`.
- Provider private-field-name hits: none.
- Collision remains `NOT_EVALUABLE_MAP_NOT_CACHED`; functional selection accuracy remains
  `NOT_EVALUABLE_FUNCTIONAL_PIXEL_REGION_MISSING`.

This is positive evidence that current-frame bearing-coupled control can land the native point-goal metric on one
fresh public task. It is not evidence of a functional entrance selection, visual handoff, physical arrival,
collision safety, or episode completion. The sealed first-failure class is
`METRIC_ARRIVAL_WITHOUT_VISUAL_HANDOFF`. `traj_5` must not be rerun or retrospectively extended with the skipped
terminal observation.

## Terminal-frame reobservation bring-up (`traj_6` and `traj_7`)

The `traj_5` termination sequence exposed an adapter mismatch: the official evaluator rendered the post-action
arrival frame but stopped before the current-frame controller could process it. Commit
`990121aa5e31bd500a26475dc6f372037533c23c` added one prospectively selectable adapter rule for future tasks:
after a normal official terminal, process that already-rendered current front exactly once when the provider is
still non-terminal, and discard any resulting motion command. Provider input remains current front RGB plus public
POI name; the provider receives no metric-arrival bit, pose, distance, map, or history.

The first two fresh tasks selected for this adapter yielded setup-only failures and no algorithm evidence:

| Task | Formal failure | Provider / teacher / in-doubt | Scientific outcome |
| --- | --- | ---: | --- |
| `traj_6` (`云唐超市`) | local official evaluator import lacked `scipy` after formal start | `0 / 0 / 0` | `NOT_EVALUABLE_SETUP_RUNTIME` |
| `traj_7` (`五金建材`) | renderer base URL was supplied instead of the required `/render_gs` endpoint | `0 / 0 / 0` | `NOT_EVALUABLE_RENDER_ENDPOINT` |

Neither task was rerun. The runner was then tightened so official imports, scene loading, and evaluator construction
finish before formal output creation (`acf93213f9f7a73866061cd71c47c00bd52513d4`), and a non-`/render_gs` URL now
fails before the formal boundary while zero-provider official runtime failures seal cleanly
(`fc9c740e523609b1c76dbf19fe5cca6b6d58b925`).

- `traj_6` freeze / terminal receipt SHA-256:
  `7302bbc20316e2e959447a3c46de35d702d1522d6bafaec9cbf430287409299b` /
  `426342c90c52b0fd897bde3f3fc9d8d4760ccd0f97bda166a4cb81e0af328d39`.
- `traj_7` freeze / terminal receipt SHA-256:
  `b2d0acc32eced7014b72dabd2e96e9107aaec36f6458156c7992b7d448d31691` /
  `6740b91709c5d798aebe288424e17d9e225073c176891ce439af38b388f40f11`.

The terminal-frame hypothesis therefore remained unevaluated after these two setup failures; they must not be
counted as negative handoff outcomes.
