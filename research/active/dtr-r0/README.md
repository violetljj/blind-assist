# DTR-R0: Dynamic Travel Risk trajectory-to-route event mechanics

Status: `DTR_R0_ACTIVE / CONTROLLED_EVENT_COHORT_PENDING / NO_RESULT`

## Question and claim boundary

The R0 question is deliberately narrow: can a causal short-track predictor use
ego/head pose compensation and a time-aligned wearer route tube to keep critical
event recall while suppressing irrelevant detection reminders?

This directory is a dependency-free executable definition of that question. It
does **not** contain a scientific result. Its deterministic synthetic episodes
have the fixed claim ceiling `CONTROLLED_SYNTHETIC_MECHANICS_ONLY`; they check
coordinate transforms, causal access, lifecycle behavior, and metric plumbing
only. They are excluded from every advancement-gate judgment.

Existing USTRF and HFTF closed/consumed results remain closed and historically
true. DTR-R0 does not tune, resample, rerun, fuse, or otherwise reopen their
cohorts, thresholds, lifecycle windows, or outcome authority. DTR-R0 asks a new
dynamic route-event question with a future controlled event cohort.

## Matched arms

All four arms consume the same ordered causal frames and emit the same
`ONSET / HOLD / CLEAR / UNKNOWN` lifecycle:

| Arm | Decision rule |
| --- | --- |
| `B0_detection_reminder` | Any current tracked detection requests a reminder. |
| `B1_distance_gate` | The nearest current detection requests a reminder inside the fixed metric-distance gate. |
| `B2_radial_ttc` | A constant-velocity short track requests a reminder when radial closing time is inside the TTC gate. |
| `C_route_intersection` | Ego-compensated constant-velocity target occupancy from now through a frozen horizon in `1.5–3.0 s` intersects the time-aligned wearer route tube. |

The mechanics implementation freezes the horizon at `3.0 s` and evaluates the
whole causal interval from now through that horizon. It therefore does not go
silent merely because an intersection has become less than `1.5 s` away.

## Coordinates, causal access, and missingness

World coordinates are a metric 2-D ground plane. World yaw zero is `+x` and
positive yaw turns counter-clockwise toward `+y`. Detector coordinates are
`forward_m` and `left_m` in the sensor frame. `body_yaw_rad` owns wearer-route
direction; `sensor_yaw_rad` owns camera direction. They are intentionally
separate, so a head turn changes the camera transform without changing the
wearer's route.

Each current observation is transformed into world coordinates using only its
same-time ego/sensor pose. Constant target velocity is then fitted from a short
causal history. No arm receives a future sample or evaluator truth. A missing
current track, missing ego pose, or insufficient motion history is `UNKNOWN`,
never `CLEAR`. `UNKNOWN` also preserves an already-active lifecycle state; the
next evaluable risky frame remains `HOLD`, not a second `ONSET`.

## Six balanced scene classes

The planned controlled cohort uses these six classes:

1. `crossing_enters_route` — lateral motion enters the route;
2. `oncoming` — a target approaches head-on;
3. `parallel_outside_route` — parallel motion remains outside the route;
4. `static_roadside` — a stationary roadside target remains outside;
5. `ego_turn_pseudo_motion` — camera/head rotation creates apparent image motion;
6. `enter_then_exit` — a target enters, then leaves, exercising stable `CLEAR`.

The future controlled cohort target is `100–160` short-video events, balanced
across the six classes. The synthetic generator emits only a few deterministic
episodes per class and is not a substitute for that cohort.

## Event metrics and future advancement gate

Metrics are event-level rather than per-frame success claims:

- critical-event recall: at least one `ONSET/HOLD` in the complete-trajectory
  truth warning window;
- irrelevant alert segments: `ONSET` segments outside a relevant event window;
- first-alert lead time from truth event start;
- mean alert segments per critical event;
- false alerts per minute of evaluated episode time;
- stable post-exit `CLEAR` rate;
- median `CLEAR` delay after route exit;
- `UNKNOWN` count and known-frame coverage (never folded into negatives).

For the future controlled cohort, all four arms are reported and the comparator
selection rule must be frozen before outcome access. The synthetic smoke uses
B0 only for plumbing diagnostics; that comparison has no gate authority.
Advancement requires every user-specified condition against the frozen
strongest credible baseline:

- critical-event recall does not decrease;
- irrelevant reminders decrease by at least `40%`;
- median first-alert lead time is at least `1.0 s`;
- mean alert segments per event is at most `1.5`;
- targets that exit the wearer route become stably `CLEAR`.

The comparator, thresholds, horizon, missingness, and metric code must be frozen
before opening that controlled outcome. The evaluator intentionally labels the
current smoke `EXCLUDED_SYNTHETIC_MECHANICS_SMOKE` and never returns a gate pass.
It also rejects any episode not explicitly marked `mechanism_smoke_only`; the
future controlled cohort requires a separately frozen evaluator version.

## Run the mechanics smoke

From this directory, write all generated material to a temporary or ignored
path:

```powershell
python -m unittest -v test_dtr_r0.py
python generate_smoke.py --output $env:TEMP\dtr-r0-smoke.jsonl
python evaluate.py --input $env:TEMP\dtr-r0-smoke.jsonl --output $env:TEMP\dtr-r0-metrics.json
```

The JSON output includes per-arm diagnostics plus
`claim_ceiling=CONTROLLED_SYNTHETIC_MECHANICS_ONLY` and
`result_status=NO_SCIENTIFIC_RESULT`.
