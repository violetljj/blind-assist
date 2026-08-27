# Current decision: advance Dynamic Travel Risk R0

Status: `DTR_R0_ACTIVE / CONTROLLED_EVENT_COHORT_PENDING / NO_RESULT`

## Strategic transition

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
not yet admitted; the current dependency-free synthetic smoke verifies only
coordinate, state-machine, and metric mechanics.

## First comparison surface

All arms use the same causal observation ledger and the same
`ONSET / HOLD / CLEAR / UNKNOWN` lifecycle:

- B0: a tracked target is present;
- B1: target distance crosses a fixed near threshold;
- B2: radial time-to-collision crosses a fixed horizon;
- DTR-R0: a short ego-compensated constant-velocity track predicts target
  occupancy from now through a frozen horizon in 1.5--3.0 seconds and
  intersects it with the wearer route tube.

The controlled cohort should contain 100--160 short events, balanced across
targets crossing into the route, oncoming targets, parallel passers outside the
route, static roadside targets, ego-turn pseudo-motion, and enter-then-exit
events. Staged real RGB and source-native synthetic trajectories must be
reported as separate evidence strata.

Primary metrics are critical-event recall, false alerts per minute, first-alert
lead time, delivered alerts per event, event fragmentation, and CLEAR delay.

## Advancement line

DTR-R0 advances only if, relative to the strongest credible baseline:

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
- No Android/default-App, natural-distribution, user-benefit, product, or safety
  claim follows from route scaffolding or synthetic trajectories.
- Semantic Anchor to Marker Pose remains a separate live-device demonstration
  closure, not DTR-R0 algorithm evidence.
