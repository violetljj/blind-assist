# External RGB future occupancy capsule R0

Date: 2026-08-03

Status: `FROZEN_BEFORE_FRESH_WALKING_XYZ_OUTCOME`

## Question

Can a seven-frame relative metric person track produce a compact set-valued
one-second future occupancy region when a point constant-velocity forecast is
not reliable for stop, start, and turn motion?

## Data roles

- calibration: consumed TUM Freiburg 3 `walking_static` fixed windows;
- fresh evaluation: TUM Freiburg 3 `walking_xyz` fixed windows;
- model input: RGB only when evaluating Metric3D;
- registered sensor depth: future evaluation truth and oracle-track arm only.

The final external camera remains outside this proxy result.

## Frozen construction

- sampled rate: 10 FPS;
- history: seven contiguous observations;
- horizon: frame `+10`, approximately one second;
- horizontal axes: forward and lateral;
- current-static baseline: disk centred at current position;
- OLS baseline: disk centred at the seven-frame OLS endpoint;
- candidate: capsule around the segment joining current position to the OLS
  endpoint, explicitly representing both stopping and continued motion.

Each arm receives its own radius from the same calibration opportunities. For
target miscoverage `alpha=0.10`, sort the arm's future-to-set distances and use
rank `ceil((n+1)*(1-alpha))`, capped at `n`. No radius is readjusted on fresh
evaluation.

Disk area is `pi*r^2`. Capsule area is `pi*r^2 + 2*r*segment_length`.

## Fresh support gates

All must pass:

1. capsule coverage `>=0.85`;
2. capsule mean area reduction `>=20%` versus current-static disk;
3. capsule mean area reduction `>=20%` versus OLS-endpoint disk;
4. capsule median area reduction `>=20%` versus current-static disk;
5. capsule median area reduction `>=20%` versus OLS-endpoint disk;
6. mean uncovered excess distance no worse than current-static disk;
7. mean uncovered excess distance no worse than OLS-endpoint disk.

Pass:
`FUTURE_OCCUPANCY_CAPSULE_SUPPORTED_DEVELOPMENT_ONLY`.

Otherwise:
`FUTURE_OCCUPANCY_CAPSULE_NOT_SUPPORTED`.

No history, horizon, target coverage, geometric arm, anchor, tracker threshold,
window, or gate may be changed after reading `walking_xyz` outcomes. A failure
does not invalidate the already supported Metric3D depth-source result.
