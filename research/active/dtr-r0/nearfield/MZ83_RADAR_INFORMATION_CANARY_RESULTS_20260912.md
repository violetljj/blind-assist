# MZ83 coarse radar information canary

Decision: `RADAR_INFORMATION_CANARY_PASSES_LOW_FP_RESCUE`.

The fixed radar-like generic hazard expert passes the predeclared gate in all
three MZ79 5-klux typical profiles. It rescues 47 / 53 / 59 frozen ToF
false-negative frames with zero added generic false-positive frames; the required
25% rescue counts were 12 / 14 / 15. This supports radar range, radial velocity,
and coarse azimuth as an information source worth carrying into one bounded
ToF+radar late-fusion successor.

## Fixed observable-only design

[Protocol](MZ83_RADAR_INFORMATION_CANARY_20260912.md): only the sensor forward
model reads 160 native scene-depth images. It never reads target IDs, actor names,
masks, source cases, BODY/HEAD labels, MZ79 predictions, or evaluator truth. The
sealed model-visible packet contains only range, radial velocity, coarse azimuth,
validity, clip boundary, and nominal time.

The deterministic uncalibrated forward model uses nine 10-degree azimuth cells,
two returns per cell, surface-support clustering, range and velocity noise and
quantization, range-dependent misses, 0.35 m return merging, a fixed vertical
aperture, and Poisson clutter. The fixed expert requires range <3.18 m, radial
velocity <=-0.35 m/s and |azimuth| <=20 degrees, with causal two-of-three
activation and two-empty-frame release hysteresis. Sensor packets and expert
predictions were separately hashed before evaluator access.

## Primary result

| MZ79 profile | ToF generic TP/FP/FN | Required rescue | Radar rescue / added FP | Fixed ToF OR radar TP/FP/FN |
| --- | ---: | ---: | ---: | ---: |
| 5 klux, 88% typical | 30 / 0 / 48 | 12 | **47 / 0** | 77 / 0 / 1 |
| 5 klux, 54% typical | 24 / 0 / 54 | 14 | **53 / 0** | 77 / 0 / 1 |
| 5 klux, 17% typical | 18 / 0 / 60 | 15 | **59 / 0** | 77 / 0 / 1 |

Radar-only generic confusion is 77 TP / 0 FP / 1 FN / 82 TN. All radar-only
rescues remain `GENERIC_FORWARD_HAZARD / HEIGHT_UNKNOWN`; none is silently
promoted to BODY or HEAD. Existing ToF query attribution is unchanged.

## Temporal behavior

| Clip | TP/FP/FN | First alert front distance | TTC MAE | Alert episodes in positive interval |
| --- | ---: | ---: | ---: | ---: |
| head bar | 26 / 0 / 0 | 3.1 m | 0.149 s | 1 |
| thin pole | 26 / 0 / 0 | 3.1 m | 0.115 s | 1 |
| wall | 25 / 0 / 1 | 3.0 m | 0.164 s | 1 |
| no-added-target control | 0 / 0 / 0 | none | n/a | 0 |

The packet has 1,799 valid returns over 156/160 frames and includes 29 injected
clutter returns. Hysteresis produces 77 hazard frames from 81 raw-candidate
frames. The single wall FN is consistent with causal activation delay.

## Interpretation and next boundary

This is a strong information-level contrast with MZ81/MZ82: coarse metric range
plus closing velocity and azimuth separate the simple MZ77 targets from its
control without needing appearance or precise height attribution. The fixed
late OR is already arithmetically useful on this source and preserves ToF output
semantics.

The near-perfect number is not evidence that real radar detects these objects
with 99% recall. MZ77 contains straight constant-speed approach, simple opaque
geometry, one scene, no ego rotation, and a control with no near central obstacle.
The simulator has no RCS/material response, multipath, antenna pattern,
range-Doppler ambiguity, interference, or measured calibration. Native depth
makes geometric surfaces available before the declared miss/clutter model, so
this is an information sufficiency result, not a radar performance estimate.

Retain the packet contract and fixed generic expert as a `COMPONENT`, not a
sensor baseline. The next radar experiment should challenge the same late-fusion
policy with new generic-hazard and hard-negative scenes containing weak/small
reflectors, near off-corridor clutter, walls/multipath proxies, and head motion;
it must keep radar-only rescues height-UNKNOWN. Do not train a radar network or
start a ToF+radar+IMU state estimator from this one easy source.

## Evidence and verification

Canonical evidence is
`artifacts.local/work/mz83-radar-information-canary-20260912/run-v2/`. It contains
sealed radar observations, sensor and predictor receipts, fixed predictions,
result, source snapshot, and a hash receipt. CPU execution took under four
seconds. Three focused tests pass for identity-free surface clustering, causal
clip-isolated hysteresis, and confusion/event accounting. Python compilation,
knowledge validation, and `git diff --check` pass.

This remains consumed controlled-simulation Development evidence, not calibrated
radar physics, natural-scene, measured-hardware, latency, alert, deployment,
user-benefit, or safety evidence.
