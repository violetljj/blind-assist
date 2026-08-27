# Current decisions: L10-R0 Goal-Lock and Dynamic Travel Risk R0

Status: `L10_R0_ACTIVE / CONTROLLED_MECHANISM_POSITIVE` and
`DTR_R0_ACTIVE / REAL_INPUT_CANARY_PENDING / NO_RESULT`

## Parallel product lines

Ten-meter goal completion and dynamic obstacle/risk guidance are separate active
lines. L10-R0 does not wait for DTR-R0, and neither line supplies evidence for
the other.

L10-R0 deliberately starts with readable goals such as room signs, exits,
named entrances, elevator buttons, and service desks. It replaces the former
GRAIL dependency with a goal-conditioned evidence controller: long-term
text/appearance/structure identity memory, short-term target motion, explicit
LOST -> REACQUIRE search, and independent near/completion evidence.

In the first 250-episode controlled closed-loop Development benchmark, L10-R0
reached 87.5% task completion, 81.2% post-occlusion reacquisition among
prior-locked episodes, 93.1%
direction accuracy, 1.7% wrong-lock frames, and 0/50 false completions when the
target was absent. Against the sticky local tracker this is +8.5pp completion,
+13.3pp reacquisition, +13.6pp direction accuracy, and a reduction from 67.1%
to 1.7% wrong-lock frames. These are synthetic controller/mechanics numbers,
not real-camera or product evidence. The runnable route is
`research/active/l10-r0/`.

With the controller frozen, a second 250-episode seed reproduced 88.5% task
completion, 82.4% reacquisition, 91.9% direction accuracy, 1.8% wrong-lock
frames, and another 0/50 absent false completions.

## DTR-R0 strategic transition

GRAIL owner orientation is no longer the daily algorithm mainline. R1C-V,
R1C-P, R1C-L, G0, and G1 form a preserved negative-result chain: the available
RGB/masks, simple pose transport, and fixed multiview representation did not
reliably recover ProcTHOR owner-canonical sign. The terminal G1 evidence is
recoverable at commit `4db9a11964ff9af9b5b500d59a60d8bb6fc0213b`.

G1 specifically closed this claim:

> On fresh house-disjoint synthetic ProcTHOR Development data, a fixed
> camera-lateral anchor/left/right scan, with no view-role, scan-geometry,
> camera-pose, owner-pose, or canonical-sign input and a shared pair encoder plus
> permutation-invariant `mean+max` aggregation, did not recover more stable
> owner-canonical PRESERVE/FLIP authority than its matched single-view model.

The apparent accuracy gain was a stronger PRESERVE tendency under a 988/148
class imbalance. Balanced-accuracy uplift was -0.05pp and -0.49pp. Doorway FLIP
collapsed to `0/24` in both seeds, taking Doorway balanced accuracy from
53.09% to 46.18% and from 50.32% to 47.35%. Across the 17 owner groups that
contain both modes, macro balanced accuracy remained near chance:
52.57% to 53.64% and 50.68% to 51.29%.

This does not establish that every active multiview RGB formulation is
impossible. A future GRAIL successor would need either a different task
representation, such as a reference-anchored dense correspondence field, or a
genuinely different information source. Neither is active now.

## Active question

Can short causal target tracks, ego-motion compensation, and intersection of
future target occupancy with the wearer's short-horizon route reduce
non-actionable alerts while preserving truly crossing or oncoming events?

The active route is `research/active/dtr-r0/`. The first scientific cohort is
not yet admitted. The synthetic smoke verifies only coordinate, state-machine,
and metric mechanics; the truth-blind real RGB observation adapter is
implemented, but the 24-event input canary has not been captured or run.

## First comparison surface

All arms use the same causal observation ledger and the same
`ONSET / HOLD / CLEAR / UNKNOWN` lifecycle:

- B0: a tracked target is present;
- B1: target distance crosses a fixed near threshold;
- B2: radial time-to-collision crosses a fixed horizon;
- DTR-R0: a short ego-compensated constant-velocity track predicts target
  occupancy from now through a frozen horizon in 1.5--3.0 seconds and
  intersects it with the wearer route tube.

The primary comparison is frozen before real-input access as
`C_ROUTE_INTERSECTION vs B2_RADIAL_TTC`. Both arms consume causal short tracks,
so an increment can be attributed to route relevance rather than merely to
tracking. B0 and B1 remain fully reported explanatory baselines.

Before the scientific cohort, a 24-event real-input canary uses four staged
clips from each of the six scene classes. It can establish only that RGB,
person detections, causal identity tracks, flat-ground metric projection, and
time-aligned body/camera pose can materialize a stable observation ledger. It
has no advancement-gate authority.

If that source-admission canary succeeds under its frozen input contract, the
controlled Development cohort contains exactly 120 staged real-RGB events:
20 per scene class. Each clip is 8--12 seconds and includes the pre-event,
track formation, event, closest approach/crossing, exit, and clear phases.
Synthetic trajectories remain a separate mechanism-diagnostic stratum and
never enter the 120-event headline result.

Primary metrics are critical-event recall, false alerts per minute, first-alert
lead time, delivered alerts per event, event fragmentation, and CLEAR delay.

## Advancement line

DTR-R0 advances only if, relative to the frozen B2 radial-TTC baseline:

- critical-event recall does not decrease;
- non-actionable alerts decrease by at least 40%;
- median first-alert lead time is at least 1.0 second;
- mean fragments per event are at most 1.5;
- route exit produces stable CLEAR behavior.

The synthetic mechanism smoke cannot satisfy this line. A scored result needs
the controlled event cohort, source/episode separation, causal inputs, and
independent event intervals.

## Boundaries

- G1 remains `STOP_G1_ACTIVE_MULTIVIEW_APPEARANCE /
  DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`; no extra views, seed/epoch/loss,
  view selector, larger DINO, G0 fusion, or consumed-subset search is open.
- Prior USTRF route-target source searches, causal route-intrusion signals, and
  HFTF selected-box projection outcomes remain historically closed. DTR-R0
  changes the event representation and requires a new controlled cohort; it is
  not a renamed rerun.
- `UNKNOWN` is not CLEAR, and silence is never presented as evidence that the
  route is safe.
- All four arms share a frozen 0.50-second clear grace. A single known-negative
  frame cannot fragment an active event, and UNKNOWN cannot complete a clear.
- Evaluator-only side/overhead video or ground markers may define wearer and
  target 2-D trajectories and route entry/exit times, but none of that truth
  may enter the DTR observation ledger.
- No Android/default-App, natural-distribution, user-benefit, product, or safety
  claim follows from route scaffolding or synthetic trajectories.
- Semantic Anchor to Marker Pose remains a separate live-device demonstration
  closure, not DTR-R0 algorithm evidence.
- Probabilistic occupancy, learned trajectory prediction, Transformers, VLMs,
  and DTR-R1 remain closed unless deterministic R0 first produces a positive
  controlled real-RGB result.
