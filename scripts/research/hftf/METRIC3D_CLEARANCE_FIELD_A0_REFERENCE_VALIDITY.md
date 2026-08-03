# Metric3D clearance field A0 reference validity correction

Date: 2026-08-03

Status: `FROZEN_BEFORE_CORRECTED_REFERENCE_OUTCOME`

## Invalidity evidence

The registered-depth reference used the same independent per-frame ground
RANSAC as the candidate. On the nominally static-camera TUM sequence, its three
fixed windows produced median camera heights of 1.39, 0.79, and 0.77 m. That
physical contradiction shows that `sensor depth` alone did not make the
derived clearance field an oracle; the reference coordinate frame could jump.

The historical A0 fresh terminal remains recorded, but model attribution is
not clean until the reference ground frame is repaired.

## Single correction

- Reference obstacle geometry still comes exclusively from registered sensor
  depth.
- TUM motion-capture orientation supplies world-up to the reference arm only.
- Reference ground offset uses the already frozen 4 cm densest-bin and 8 cm
  support construction.
- The Metric3D candidate remains depth-only per-frame RANSAC and receives no
  pose, gravity, sensor depth, normal, future frame, or outcome.
- Bands, heights, percentile, horizons, unknown semantics, sampling, and all
  seven A0 gates remain unchanged.

First rerun the consumed Development and `walking_rpy` cohorts. This is an
evaluator-validity diagnostic, not fresh evidence. Only if the candidate passes
all seven gates against the corrected reference may a new fresh
`walking_halfsphere` confirmation be frozen.
