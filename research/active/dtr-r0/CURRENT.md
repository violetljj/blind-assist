# Cane-complementary forward perception and DTR history

Updated: 2026-09-07

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).
Current work: cane-complementary, class-agnostic forward obstacle awareness.

Primary Development environment: **self-built UE5 StreetLabV4**, with 10 Hz
observations and measured conservative envelopes. The historical motion runner
is `tools/run_obstacle_research.py`; CARLA is retained for history/supplementary
checks. See [UE acceptance](unreal/UE_LAB_ACCEPTANCE_20260906.md).

The [Willow sample](unreal/UE_WILLOW_SAMPLE_20260907.md) provides native geometry
and sanitized RGB-D. Conservative tree AABBs still disagree with native sweeps;
its engineering checks are not dynamic algorithm scores.
Visual work is paused at the user's request. The saved sample is frozen at
`willow-finish-4k`; first-person acquisition uses an optical center **1.70 m
above the floor**, with `tools/run_sample_segment.py sensors`. The eleven-frame
`willow-eye170-first-person` passes replay and ground-depth optical-height checks.
This is an acquisition smoke test.

## Capability question and current probe

Goal: **盲杖互补的类别无关前视障碍感知** under limited compute. Prioritize walls,
large forward barriers, body/head protrusions and suspended hazards; then poles
and multi-height supports. Knee-height hazards remain relevant; ultra-low
obstacles are secondary compatibility evidence. Earlier warning and dynamic
coverage need future temporal tests. The wearer chooses movement, not a planner.
Past denominators stay frozen; no assumed cane-coverage labels or retroactive
headline selection. Base near evidence survives missing optional ground/height.

The [representation probe](nearfield/REPRESENTATION_PROBE_20260907.md) retained
162/210/240 of240 analytic positives with stride/dense/supported methods; FP0/0/1
retains the coherent3x3 artifact. This is constructed component evidence.
The fixed Hypersim Small RGB model initially recovered0/26 on11 consumed frames.
Simulator height/pitch remain privileged; native agreement is not object truth.
Original GPU inference/postprocess P50 was86.17/3.91 ms, not phone latency.

Saved-prediction diagnosis found all 119,721 native near-surface reference pixels
lost before aggregation: two invalid, 67,956 over-range and 51,763 still near but
below the height filter. The [cached ground probe](nearfield/GROUND_ANCHOR_20260907.md)
then recovered 1/26 with scale only and 21/26 with ground-relative height, with
or without scale; all four arms had 0 FP. Fits used predictions/calibration only.
Low-boundary matches123/807 remain secondary diagnosis, not the main objective.
[Support comparison](nearfield/SURFACE_SUPPORT_20260907.md): all arms 21/26,0 FP.
[18 distinct views](nearfield/DISTINCT_VIEWS_20260907.md): raw22/87 vs ground37/87,
0 FP, but failed ground fits suppress15 raw wall positives. [Fusion](nearfield/EVIDENCE_FUSION_20260907.md)
restores them:52/87 legacy hypothesis cells,0 FP; direction-only near output can
retain UNKNOWN height. Body/head subgroup16/47 is not verified-height recall.
[Frontend comparison](nearfield/FRONTEND_DOMAIN_20260907.md): VKITTI518 recovers0/87;
Hypersim1036 reaches70/87 with9 FP and925 ms P50, but still misses the1.96 m bar
(13.43 m predicted). Neither replaces the baseline; local structure/range evidence
is the next unresolved frontend question. No App promotion. Historical registry
fingerprint validation is repaired; see [registration record](nearfield/REGISTRY_REPAIR_20260907.md).

## Retained evidence and baseline

- **Public/JRDB X21:** same-live-track transport of an already authorized X13
  component; six consumed sequences reached `5/6` CONTACT, 11 false segments,
  45.45% Event F1, `3.061 s` median lead, and `8/18` dropout recovery. This is
  same-source Development; genuinely source-disjoint confirmation is pending.
- **CARLA X73:** the latest complete source-disjoint synthetic Development
  confirmation. On C35, parent-hull reconstruction improved X72 by
  `+6 TP / +0 FP / +2.24 pp` frame F1, reaching `132/18/40` TP/FP/FN and
  `88.00/76.74/81.99%` precision/recall/F1. The same-map, scripted-motion,
  detector and evaluator boundaries remain; successors lack its confirmation.
- **CARLA X94:** consumed post-hoc Development reference, `1478/84/417`
  TP/FP/FN and `94.62/77.99/85.51%` frame precision/recall/F1 across eleven
  cohorts. Its one-observation full-dropout bridge added six TP and no FP over
  X93; same-parent evidence and an unchanged valid issued plan remain required.
- **Credible simple comparator:** in the consumed C35 raw-input pilot, Kalman
  CV + route tube + `0.60 s` hold tied X94 at 66.67% Event F1 and four false
  segments, with two versus ten fragmentation runs. X94 retained better frame
  F1 (`84.62%` versus `81.52%`), median lead (`2.80` versus `2.65 s`) and CLEAR
  (`0.00` versus `0.20 s`). This tradeoff has no fresh eleven-arm adjudication.
- **X95 remains a challenger:**
  `DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_GATE_NOT_MET`. Its consumed replay
  gained `3.74 pp` Event F1 and removed five false segments versus X94, but lost
  `7.02 pp` frame F1 and added 30 fragment gaps. Simple `0.60 s` hysteresis
  reached 89.07% frame F1 on that replay; complexity has not established a win.

## Preserved source bottleneck

Frozen avoidance R1 remains `NOT_EVALUABLE_SOURCE_CAPTURE_INTERRUPTED`.
Subsequent startup engineering admitted a reused Development composite, but
FIT_ONLY S03 failed dropout credential readiness; no final method score followed.
See [startup and method diagnosis](CARLA_CAMERA_STARTUP_20260905.md). Source
engineering changes no algorithm inheritance. Old raw-input comparisons lack
sufficient retained payloads beyond C35 for fair reconstruction.

## Preserved controller decisions (parked during perception work)

1. **UE Development:** use fixed RGB-D replay for perception changes and V4
   closed loop for motion changes. The measured motion reference is DEPTH_ONLY;
   candidate DTR remains an optional challenger. The new 10 Hz Development
   bank completed 32 actual branches: both sets of 8 straight controls passed,
   candidate depth succeeded 5/8 and candidate DTR 6/8, with no success
   regression. The extra success was late crossing, with nine changed-action
   frames. This is rerendered Development, not exact-pixel pairing or promotion.
2. **Latest Development finding:** the separately identified
   [visible-surface clearance experiment](unreal/UE_VISIBLE_SURFACE_CLEARANCE_20260907.md)
   finished 16 actual branches. A fixed .40 m extra surface margin and +/-1.20 m
   bypass regressed depth success from 5/8 to 2/8, with no gains. Late-stop and
   continued-walking both halted before a static planter entered by the new
   bypass. Reject this fixed variant; keep the defaults and earlier evidence.
   Its structured disposition is pending normal registry publication because
   the pre-existing experiment-index fingerprint mismatch remains unresolved.
   The next useful comparison must cover static street geometry in contact
   evaluation, establish a traversable witness for its action space and test
   replanning when a bypass becomes blocked. Existing scenario-only contact
   scores do not cover the newly diagnosed planter. Preserve conservative
   contact, goal, time and missing-support reporting; do not shrink proxies or
   tune this consumed margin to rescue the result. The unchanged late-crossing
   DTR gain still awaits a scoped replication.
3. **CARLA preservation:** park the pending dropout-window continuation and
   eleven-arm execution during the UE migration. Do not restore missing CARLA
   assets or recapture by default. Keep complete source, frozen method snapshots,
   old failures and prior algorithm inheritance. UE results do not inherit CARLA
   confirmation authority. This switch starts no X97, learner or held-out run;
   JRDB source-disjoint confirmation remains a separate requirement.

## Boundaries

- Frozen R1 is not retried or reclassified as fresh confirmation. Preserve its
  seals, nine shards and crash evidence; no protected fit/final access follows
  from capture-engineering success.
- C35-C41 cannot be rerun as confirmation. C8-C11 admitted no evaluable X31
  occlusion source; N4 v1 cannot resume or retry. It is consumed incomplete,
  not a three-town result. Historical outcomes and source gates remain unchanged.
- Consumed diagnosis is Development, not fresh authority. Keep frozen thresholds,
  source gates, lifecycle, association, seeds and denominators with their original
  results; changes belong to a separately identified Development version.
- `UNKNOWN` and `NOT_EVALUABLE` are not `CLEAR`, negative evidence or safety.
  Historical DTR route conflict owns its event correctness; near-field alerts
  use their separate perception evaluator. Public replay and CARLA do not establish Android readiness,
  natural-distribution performance, user benefit, deployment or safety.
- Uncommitted candidates and outputs remain WIP. Existing structured inheritance
  roles and historical verdicts remain authoritative; this compaction changes none.

## Evidence links

- [X21 result](X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md), [X73 confirmation](carla/DTR_CARLA_C35_X73_FRESH_CONFIRMATION_20260901.md), [X94 result](carla/DTR_CARLA_X94_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md).
- [C35 raw-input pilot](DTR_BASELINE_RECKONING_C35_RAW_PILOT_20260905.md), [X95 result and simple controls](carla/DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_20260901.md).
- [Frozen comparison design](DTR_FINAL_RECKONING_ROSTER_R1_20260905.md), [latest source execution and crash](DTR_FINAL_SOURCE_EXECUTION_20260905.md); the later execution record owns the design's current execution status.
- [Completed RGB/depth and lossless-PNG probe](CARLA_RGBD_THROUGHPUT_20260905.md).
- [Failed Development composite and DX11 probe](CARLA_FAST_COMPOSITE_SOURCE_20260905.md).
- [Detailed ledger and reproduction](README.md), [formal research governance](../../../docs/formal/RESEARCH_GOVERNANCE.md).

The pre-scope-change current is preserved at Git
`fc70658e:research/active/dtr-r0/CURRENT.md`; linked reports retain the full history.
