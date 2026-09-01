# DTR CARLA C42 / X96 source result — 2026-09-01

## Terminal decision

`DTR_CARLA_C42_X96_SOURCE_NOT_EVALUABLE`

C42 produced zero durable sensor frames. No evaluator row, prediction metric,
dropout recovery rate, or X96-vs-X94 conclusion exists for this cohort. C42 may
not be presented as positive, negative, fresh-confirmation, or mechanism-not-
exercised evidence.

## Frozen objective

C42 preregistered one fresh CARLA source and a single joint score of four arms:

1. X94 one-frame full-dropout continuity;
2. recursive full-dropout forward-fill;
3. 0.60 s hysteresis;
4. X96 credentialed bounded dropout survival.

The intervention matrix fixed `2 / 3 / 6` frame detector-plus-metric dropouts
at four placements before capture:

- `A_ACTIVE_MIDDLE`, sample 22;
- `B_PRE_ONSET`, sample 2;
- `C_RELEASE_BOUNDARY`, sample 44;
- `D_PLAN_CONFLICT`, sample 22.

Protocol:
`dtr_carla_c42_x96_dropout_survival_protocol.json`

Protocol SHA-256:
`48BF56D34E0B433BB2FD82DB6DA748C2C4E565EE3BF94F320120793893B825D1`

The source changed to the previously unused `c17_*` actor trajectory bindings,
capture seed `421096`, four new weather assignments, and new pixels. Dropout
indices were not selected from truth or predictions.

## Capture attempts

Output root:
`E:\linnan\CARLA\experiments\dtr-carla-c42-x96-dropout-survival\evidence\c42-x96-source-20260901-231500`

Attempt 1 used the default `4.0 GB` free-physical-memory startup floor. The
CARLA wrapper terminated before server launch and before any durable frame when
capacity failed to recover; terminal free physical memory was about `3.25 GB`.

The frozen source contract permits one retry only when the failed shard has
zero durable frames. The retry preserved the same run id, protocol, seed,
trajectories, weather assignments, and algorithm hashes, used `-Resume`, and
lowered only the wrapper's operational memory floor to `3.0 GB` (inside its
declared supported range). It again terminated before CARLA launch and before
any durable frame; terminal free physical memory was about `2.20 GB`.

No third attempt is authorized for C42. Ports `2000 / 2001 / 2002` were free at
terminal inspection, no task-owned CARLA or capture Python process remained,
and the wrapper released its task-owned storage lease.

## Implementation-only falsifier

The X96 implementation and runner were syntax checked. A truth-free synthetic
state fixture directly exercised the X96 transition semantics:

- one real `MEASURED` rigid surface carrier with an existing X75 credential
  survived six 0.1 s full-dropout observations;
- the seventh dropout observation, at 0.7 s from the real anchor, did not
  survive;
- fill rows did not become anchors;
- a pre-onset all-absence sequence created zero X96 births;
- a controlled `route_mode_changed` frame carried no X96 risk.

This is implementation evidence only. It does not establish CARLA effect,
fresh-source transfer, detector robustness, event quality, or real-world value.

## Next admissible action

Keep X94 as the current cumulative main arm and X73 as the latest complete
source-disjoint authority. Retain X96 and the C42 protocol as a frozen pending
challenger. The next scored attempt must use a new cohort id and a new capture
protocol after sufficient free memory is available; C42 itself must not be
resumed, retuned, or rescored.
