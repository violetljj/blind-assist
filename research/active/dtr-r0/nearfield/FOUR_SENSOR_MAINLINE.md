# Four-sensor simulation mainline

User-confirmed architecture, 2026-09-13: **one RGB camera + ToF + Radar + IMU**.
The immediate priority is algorithmic benefit in class-agnostic forward obstacle
awareness. Work remains simulation-only. A second camera, stereo depth, removal
of Radar, or substitution of simulator truth for an observable input changes the
architecture and requires an explicit new user decision.

## What the next implementation must answer

1. Does a measured surface occupy the current walking/body/head corridor?
2. Which image region, ToF zone and Radar return can credibly refer to one target?

RGB supplies image extent and angular localization. ToF supplies zonal range and
validity, not exact per-pixel depth. Radar supplies uncertain range, bearing and
radial velocity, not object identity. IMU supplies rotation increments, not
accurate metric translation or the user's future intended path. Time and
extrinsic calibration are explicit inputs. Association ambiguity stays visible.

Use identical ToF/Radar/IMU packets, geometry, history and task labels in the
three-sensor baseline and the four-sensor candidate. Report individual small-pole
and HEAD retention, false alarms, missing support/UNKNOWN, association coverage
and cost. Preserve independently valid ToF evidence. Missing RGB detections or
missing range returns cannot establish clearance. Never suppress an independent
sensor merely because another sensor missed the object.

## Evidence routing and correction

- MZ90--100 are range/rotation sensor experiments; their predictors do not consume
  RGB. They do not establish a tested four-sensor fusion system.
- MZ101--106 are a **separate stereo RGB + ToF branch**, with no Radar in the
  evaluated pipeline. Preserve their negative results and baselines within that
  branch. They do not diagnose the four-sensor mainline's algorithmic ceiling.
- MZ107 starts a fresh, bounded, synchronized four-sensor simulation canary.
  Its simple rendered appearance is controlled Development, not a general visual
  detector benchmark. Native geometry and target identities are evaluator-only.

## Advancement and stop points

First verify input provenance and that actual RGB pixels affect association.
Then measure a paired task effect with an RGB-disabled control. A gain must
retain critical baseline positives; fewer reported points alone is not a gain.
If image proposals or associations fail, report that specific mechanism and stop
the bounded attempt. Do not jump to larger depth models or more sensors.

Only after a useful observable association mechanism is fixed should a complete
new scene split test it unchanged, including ambiguity, low contrast, occlusion,
time/calibration errors and sensor gaps. Current-corridor occupancy and future
collision prediction are different tasks; report them separately. Each experiment
ends at its stated budget, followed by evidence and code delivery.
