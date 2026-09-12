# MZ88: uncertainty-aware geometric association

Mode: `EXPLORE`, zero training, fresh controlled analytic source.

## Question

Can an uncertainty-aware geometric association replace MZ85's hard stabilized
corridor boundary and retain useful ToF/radar decisions under a declared
small-to-medium IMU/calibration envelope, without recovering false alerts by
merely widening the corridor?

MZ85 is the hard-association baseline. MZ87 is consumed diagnostic evidence and
may explain the mechanism, but none of its outcomes, scenario rows, failure
locations or thresholds may be read by the MZ88 predictor or used to choose a
margin. MZ86 continuity work is paused. MZ88 uses no learned component, EKF,
threshold search or evaluator-visible family/target identity.

## Only mechanism change

The estimated ray is still rotated causally into the episode-initial frame.
The hard decision `abs(theta_hat) <= 12 deg` becomes a three-state association
computed from a declared one-standard-deviation angular uncertainty:

`sigma_theta^2 = sigma_gyro^2 + sigma_bias^2 + sigma_extrinsic^2 + sigma_sync^2 + sigma_dropout^2`.

For a return at range `r`, the horizontal quantities are:

- `y_hat = r * sin(theta_hat)`;
- `w(r) = r * sin(12 deg)`;
- `sigma_y = r * abs(cos(theta_hat)) * radians(sigma_theta)`;
- `z = (w(r) - abs(y_hat)) / max(sigma_y, 1e-6)`.

With fixed `k = 1.0`, `z >= k` is `CERTAIN_IN`, `z <= -k` is
`CERTAIN_OUT`, and the remainder is `UNCERTAIN`. Lateral one-second crossing
uses the same interval rule on the causal current/forecast segment, with
forecast uncertainty propagated from the current and previous angular
uncertainties. No corridor width or confidence multiplier is searched.

The predictor assumes the following fixed envelope, chosen before MZ88 source
materialization: `2 deg/s` bias, `2 deg/s RMS` gyro noise, `2 deg` extrinsic
uncertainty and `20 ms` synchronization uncertainty. Bias uncertainty grows
with time from the episode anchor; white-noise uncertainty grows as a random
walk. Synchronization uncertainty is the observed angular rate times `20 ms`.
When an IMU increment is absent, orientation is held and dropout uncertainty
grows causally from the last observed angular rate plus the bias envelope.
These are analytic pressure bounds, not a claimed device distribution.

## Responsibility rule

- `CERTAIN_IN`: ToF may create normal generic geometric evidence.
- `CERTAIN_OUT`: ToF creates no evidence and valid ToF continues to suppress
  coarse Radar evidence.
- `UNCERTAIN + Radar`: Radar may confirm a generic hazard; height remains
  `UNKNOWN` unless current ToF is `CERTAIN_IN`.
- `UNCERTAIN + previous credentialed hazard`: preserve that hazard for at most
  one frame. A held frame cannot reseed another hold.
- `UNCERTAIN` alone: output `UNKNOWN`, never `CLEAR` and never a new hazard.
- missing/invalid ToF retains the frozen MZ84 rule: Radar may produce a generic
  hazard and otherwise the frame is `UNKNOWN`.

The full arm is compared with two fixed attribution arms: tri-state geometry
with `UNCERTAIN -> UNKNOWN` only, and tri-state geometry with Radar confirmation
but without the one-frame hold. These are not candidates for margin selection.

## Fresh source

MZ88 materializes a new 10 Hz analytic source before predictor scoring. It does
not reuse MZ84/MZ87 frames, trajectories or failure instants. The source varies:

- six distinct yaw/pitch waveforms with different phase and peak rate;
- both bias/extrinsic polarities, nominal control and mixed medium stresses;
- one- and two-frame IMU gaps at variant-specific times;
- direct obstacles at multiple ranges and signed distances inside the boundary;
- outside-boundary clutter at multiple signed distances;
- lateral crossings with different direction, speed and closest approach;
- head-motion association swaps at different waveform phases;
- ToF gaps with different onset and obstacle range;
- centered weak-Radar obstacles that require certain ToF evidence.

The six families are `boundary_inside`, `boundary_outside`,
`lateral_crossing`, `head_motion`, `multi_target_gap` and `weak_radar`; each has
six variants of 30 frames. The analytic materializer seals observations and IMU
validity before predictor output. Predictor output is hashed before evaluator
truth and family labels are opened.

## Predeclared checks

Relative to hard MZ85 association on the same fresh observations, the full MZ88
arm passes only if all checks hold:

1. overall F1 is strictly higher, FP is at most half the hard baseline, and TP
   loss is at most two frames;
2. on nominal/no-gap variants, TP loss is zero and no new FP is added;
3. at least half of hard-baseline FP caused by stressed boundary or head-motion
   association are removed;
4. positive-event first-alert delay is at most `0.2 s` and no event gains more
   than one fragment;
5. `UNCERTAIN` ToF alone creates zero new hazards, every held hazard has an
   immediately preceding non-held credential, and height is never fabricated
   from Radar or hold evidence.

If the full arm fails but an attribution arm passes the performance checks,
retain only the demonstrated submechanism. If Radar confirmation increases FP
inside the uncertainty band, record that authority failure rather than tuning
Radar or `k`. If all arms fail, close this scalar uncertainty recipe and move
the next hypothesis to a richer occupancy/collision tube; do not fit MZ88.

## Evidence boundary

This is fresh only relative to the consumed MZ84/MZ87 analytic frames. It is
still constructed, source-aware controlled simulation, not source-disjoint
natural confirmation. It does not establish a real covariance model, sensor
calibration, timing tolerance, hardware performance, alert benefit, deployment
readiness or safety improvement. `UNKNOWN` is reported separately from `CLEAR`.
