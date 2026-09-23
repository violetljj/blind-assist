2026-09-23 [frozen contact decomposition](../research/active/dtr-r0/nearfield/CONTACT_DECOMPOSITION_RESULTS_20260923.md):
Position transfer remains weak beyond gating: exact internal median72.14%train
versus28.41%held within5cm;134/165blocked cases also have inaccurate positions.
Read-only diagnostic retained; no new predictor,training or App change.

2026-09-23 [exact contact supervision result](../research/active/dtr-r0/nearfield/EXACT_CONTACT_RESULTS_20260923.md):
Adding exact train labels fails selected transfer: Z5cm33.24 ->7.67%,recall72.45
->49.53%. Train-only trajectory shows metric fitting capacity with false-crossing
costs; preserve this partial evidence and frozen negatives. App unchanged.

2026-09-23 [direct metric contact pilot](../research/active/dtr-r0/nearfield/METRIC_CONTACT_RESULTS_20260923.md):
One completed same-query CDF fit fails the joint boundary upgrade: Z5cm47.73 ->33.24%,
width25.38 ->10.80%,recall78.49 ->72.45%; fewer false Z crossings cost missing
finite contacts. Retain exact negative control and prior sampling gains; no App change.

2026-09-23 [boundary-aware sampling result](../research/active/dtr-r0/nearfield/CONTACT_SAMPLING_RESULTS_20260923.md):
Fixed72query Development: geometry evaluation width within5cm43/528 ->134/528,
horizon61/352 ->168/352; train horizon175/560 ->467/560. Recall86.88 ->78.49%,
FPR9.48 ->5.16%, horizon falsecross119/800 ->146/800. Retain sampling COMPONENT;
neither full package passes.13tests/777024world-label auditPASS. Old controls,
A/LOCAL/UNKNOWN preserved; two fits complete, no automatic successor.

2026-09-22 [single-frame spatial-learning pilot](../research/active/dtr-r0/nearfield/QUERY_OCCUPANCY_RESULTS_20260922.md)
completed as a scoped negative control. Joint occupancy supervision increases
matched-classifier event recall but exceeds false-alert costs and yields zero
maskIoU at the fixed cutoff. Retain A; no temporal or App successor was run.

2026-09-19 scope update: [camera-forward RGB + 8x8 ToF](../research/active/dtr-r0/nearfield/CAMERA_FORWARD_CONTRACT_20260919.md) is the adopted minimal research task. Optical axis defines forward; historical four-sensor/body-frame results below retain their original scope. This records the task definition, not a runtime change.

# Project state

Updated: 2026-09-07

BlindAssist is a runnable Android showcase research prototype and thesis project.
The primary goal is useful, measurable technical effect and a credible, stable
controlled demonstration. Innovation depth depends on the problem; major novelty
is not a requirement. Established methods, integration, and incremental improvements
are valid choices when their effect and cost justify them. Describe the actual
contribution honestly, and pursue new mechanisms when they address a concrete gap.
Natural-distribution and safety claims require their own evidence; a build or a narrow replay does not establish them.

Forward-alert effect version delivered,2026-09-18:
[standalone A* entry and full comparison](../research/active/dtr-r0/nearfield/corridor_fusion_v1/ASTAR_EFFECT_USAGE_20260917.md)
load one retained HGB; display, saved output and reminders share `alert=A*`.
All288 scores and event times exactly match frozen A*: clear96/3/12,F192.75%,
core18/18; strict107/6/37,24/30. The48-episode A/A* demo includes all costs.
Direct host decode/frontend/A* mean66.67ms,p5066.04,p9574.64; no speedup claim.
Old comparison semantics remain intact. Public positive and boundary OR remain
research controls. No training, capture, threshold or Android-default change;
this is engineering delivery of retained evidence, not new confirmation.

Forward-alert single-version confirmation,2026-09-17: [frozen public model and same-data A*](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_SINGLE_RESULTS_20260917.md)
are runnable with one model/threshold each. New clear A/OR both97/25/11; all11
misses lack returned corridor support, so clear rescue transfer is untested.
OR adds1 boundary TP and1FP. Same-data A* gives96/3/12,F192.75%, removes22 clear
FP but loses4 A TP/rescues3; core17/18to18/18, one rod onset delayed.5s, strict
events25/30to24/30. Retain head component and A* precision-oriented challenger
with costs; no App promotion or retuning. Whole algorithm63.95/66.15ms A/OR.
One288-frame capture complete, audited, resources released; ledger303 pending.

Forward-alert public-evidence v2, 2026-09-17: [pooled calibration result](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_V2_RESULTS_20260917.md)
reuses1,344 existing frames. The unchanged public1537-parameter BCE head recovers
8 changed clear misses with no new FP across1,152 reported frames:103/27/5,
F186.55%. Old/MZ146/MZ158 remain A-identical; changed strict128/45/16,
boundary unchanged. Peak loss adds1 old TP and3 boundary FP. Retain BCE plus
pooled calibration as a Development challenger; no single deployment model,
fresh confirmation or App change. Both arms independently audited.

Forward-alert public-evidence update, 2026-09-17: [trained public return readout](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_RESULTS_20260917.md)
uses 1,537 parameters and only public ToF geometry, with grouped existing data.
Selected OR stays identical to A. Posthoc logit0 recovers 8 changed clear misses
but adds 6 FP (103/33/5, F1 84.43%); boundary false alerts grow substantially.
Retain the implemented component and diagnose calibration separately from learned
signal. No fresh claim, final-method promotion, capture or default-App change.

Forward-alert tri-state update, 2026-09-17: [existing-evidence readout](../research/active/dtr-r0/nearfield/corridor_fusion_v1/TRISTATE_EVIDENCE_RESULTS_20260917.md)
gives privileged clear 103/27/5 for positive OR and 102/17/6 for joint veto,
F1 86.55% / 89.87%, with zero-return fallback preserved. Joint veto loses one
rod TP and delays that event by 0.75 s; outside returns do not prove clear space.
This supports a public spatial-readout research candidate, not a deployed result.
No training/capture; A and App defaults unchanged.

Forward-alert surface-oracle update,2026-09-17: [existing-return full-face audit](../research/active/dtr-r0/nearfield/corridor_fusion_v1/SURFACE_ORACLE_RESULTS_20260917.md)
leaves frozen A clear95/27/13 unchanged. Eight misses already have sampled
corridor support;five rod misses lack corresponding returns. Full extent adds
zero clear-support opportunities. A barely uses its support-endpoint fields,
so this rejects the fixed-entry replacement, not all spatial support learning.
No new capture/training or automatic successor; retained per-frame diagnosis.

Forward-alert task-definition update,2026-09-17: [existing-data re-evaluation](../research/active/dtr-r0/nearfield/corridor_fusion_v1/TOLERANCE_RESULTS_20260917.md)
uses576stored frames, no training/inference.5cm lateral tolerance retains75%:
old A107/22/1,F1 90.30%; changed A=S195/27/13,F1 82.61%. Boundary pressure
is separate;40clear transfer errors remain beyond10cm laterally. Main reporting
now separates clear-task accuracy/coverage, strict boundary and alert behaviour.
Intrusion training is deferred, its capture cancelled with resource release;
reuse existing data before any new acquisition. Online DA-V2 remains closed.

## Current research lines

Forward-alert research update,2026-09-17: user-adopted [balanced A](../research/active/dtr-r0/nearfield/corridor_fusion_v1/BALANCED_BASELINE_20260917.md)
supplies137/28/7,F1 88.67%,30/30events on consumed MZ170. The frozen
[conditional depth specialist](../research/active/dtr-r0/nearfield/corridor_fusion_v1/SPECIALIST_RESULTS_20260917.md)
removes4FP with no additional FN or event delay, invokes10/288frames and adds
2.86msmean online cost. Retain as a Development challenger; App defaults unchanged.
The [completed targeted confirmation](../research/active/dtr-r0/nearfield/corridor_fusion_v1/CONFIRMATION_FINAL_20260917.md)
shows A/S1 both120/45/24,F1 77.67%,29/30events:14C calls remove no FP.
Keep the old Development result but close this online DA-V2 recipe. Identical
source completed after correcting an arbitrary acquisition timeout; no models
or thresholds changed. No CNH successor; all owned processes released.

Latest forward-alert result,2026-09-16: [MZ146](../research/active/dtr-r0/nearfield/MZ146_RESULTS_20260916.md)
tests unchanged MZ145 on288 fresh same-generator controlled frames. MZ129
130TP/86FP/14FN becomes143/57/1; events29/30to30/30, false segments28to21,
with no incumbent true frame or event lost. The aggregate gain is substantial,
but rod FP25to29 fails the pre-outcome family condition. The fixed confirmation
is closed without tuning; MZ129 and the App default remain retained. The
[owning route](../research/active/dtr-r0/CURRENT.md) preserves MZ145's consumed
Development gain separately. No natural-scene, device or all-family benefit claim.

Forward-perception architecture correction (2026-09-13): the user-confirmed
simulation mainline is **one RGB camera + ToF + Radar + IMU**. Prioritize measurable
current-corridor alert benefit under the fixed cheap hardware budget. The
[MZ136 direct paired-training experiment](../research/active/dtr-r0/nearfield/MZ136_RESULTS_20260914.md)
does not improve the joint alert tradeoff; keep MZ129 and its independent native
evidence. Unique return attribution is not a required intermediate task.
The subsequent [TRAIN fit repair](../research/active/dtr-r0/nearfield/MZ136_TRAIN_FIT_REPAIR_20260914.md)
improves the same192 training frames from154 to191 correct with385new readout
parameters. Its frozen dev48 transfer needs20FP versus MZ12913 at matched recall
and timing, so retain it as a fitting diagnostic only. No new capture was used.
The [grouped readout follow-up](../research/active/dtr-r0/nearfield/MZ136_GROUPED_READOUT_20260914.md)
reduces head instability but still needs22FP versus13 at matched dev recall/timing.
The [boundary-input audit](../research/active/dtr-r0/nearfield/MZ136_BOUNDARY_INPUTS_20260915.md)
reduces edge MAE2.886 to1.227px on the same38 TRAIN frames, with fine edges
available38/48. Retain the geometry component only; MZ129 stays the alert baseline.
The [MZ137 end-to-end contrast](../research/active/dtr-r0/nearfield/MZ137_EDGE_CORRIDOR_20260915.md)
compares coarse/fine edges under one public-range plane on consumed dev48:
all arms24TP/13FP/0FN,5/5events unchanged. One corrected plane crossing never
changes the final alert; native support loss rejects this fixed integration.
MZ129 and the conditional MZ136 component remain; no tuning or test run follows.
The [MZ138 support ceiling](../research/active/dtr-r0/nearfield/MZ138_SUPPORT_CEILING_20260915.md)
finds true complete-surface oracle24TP/2FP/0FN versus24/13/0 on consumed dev48,
with all event times retained. Sampled-point supports miss unsampled surface
portions; splitting sampled ownership alone has no gain. Diagnostic headroom
only, with disclosed full-surface scope correction; MZ129 remains the system baseline.
The [MZ139 observable surface estimator](../research/active/dtr-r0/nearfield/MZ139_SURFACE_FIT_20260915.md)
fits public regional ToF and RGB jointly, but frozen consumed dev48 still gives
24TP/13FP/0FN with unchanged events.10/48accepted surfaces do not change alerts;
one of3unsampled intrusions is supported while native geometry still loses24
corridor contributors. Keep MZ129; exact estimator is a negative control,
registration/inheritance pending ledger303. No additional oracle or test run.

[MZ140 DEPTHOR Small](../research/active/dtr-r0/nearfield/MZ140_DEPTHOR_20260915.md)
now provides runnable full-RGB/ToF pretrained geometry. TRAIN48 improves BODY
and HEAD extent agreement over box-only RGB, but fails rod/shallow recovery.
The declared mechanism check stops before dev; no end-to-end gain or latency
claim. Keep MZ129, pause MZ139's exact manual-fit recipe; metadata pending303.

MZ101--106 are a separate stereo+ToF branch, not evidence about this four-sensor
system. Follow the [four-sensor mainline](../research/active/dtr-r0/nearfield/FOUR_SENSOR_MAINLINE.md)
and its paired ToF+Radar+IMU versus +RGB comparison. Architecture changes require
an explicit new user decision; historical experiment suggestions do not change it.

| Line | Capability and present emphasis | Owning current |
| --- | --- | --- |
| `L10_R0_ACTIVE` | Recover and retain the requested target with useful evidence and observation cost; distinguish missing support, identity contradiction and endpoint extent. | [L10 current](../research/active/l10-r0/CURRENT.md) |
| `DTR_R2_DYNAMIC_RETAINED` | Current work: cane-complementary, class-agnostic forward obstacle awareness. Prioritize walls, body/head, suspended hazards and poles; ultra-low obstacles are secondary. DTR motion findings remain historical. | [Perception current / DTR history](../research/active/dtr-r0/CURRENT.md) |

These lines have independent evidence, budgets and decisions. Existing experimental
versions and detailed results belong in the owning current/ledger; this page does
not duplicate their trajectories. Uncommitted candidates do not change authority.

## Start and proceed

1. Read [current cross-route decisions](CURRENT_DECISION.md) and the affected route
   current; follow result/protocol/code links only for the present question.
2. Use [the research workflow](../research/WORKFLOW.md) to choose exploration,
   confirmation or engineering and the smallest check that changes a decision.
3. Implement and evaluate against a credible baseline. Report task effect together
   with relevant errors, UNKNOWN/coverage and observation or runtime cost.
4. Decide whether to retain, change, integrate or stop, then finish the remaining
   authorized delivery. Preserve historical results and release task-owned capacity.

## Demonstration and engineering

Semantic Anchor to Marker Pose remains a separate live-device showcase closure;
it does not transfer evidence to L10 or DTR or change their integration priority.

- Workstation entrypoint: `tools/ba.ps1`.
- Android builds: `scripts/run_android_gradle.ps1`.
- [Code ownership](CODE_MAP.md), [CARLA integration](CARLA_PLAYBOOK.md),
  [artifact routing](LOCAL_ARTIFACTS.md), [device evidence](DEVICE_REGRESSION.md).

## Evidence boundaries

- `UNKNOWN` and `NOT_EVALUABLE` are neither negative method evidence nor known-safe.
- `referent != affordance != waypoint != arrival != handoff`.
- Synthetic, replay, curated Development, registered-source, live-device and natural
  evidence retain their actual scopes. Disclosed reuse never restores freshness.
- [Formal governance](formal/RESEARCH_GOVERNANCE.md) applies to protected claims;
  it does not turn nearby reversible engineering into a final evaluation.
- [History index](history-index.md), owning ledgers/results and Git preserve history.

The full previous project narrative is retained at Git
`daf5720064d98a93b75336469d18e9a2fe0023e5:docs/PROJECT_STATE.md`.
