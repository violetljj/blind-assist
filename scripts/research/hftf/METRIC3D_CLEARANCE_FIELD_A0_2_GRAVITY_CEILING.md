# Metric3D clearance field A0.2 gravity ceiling

Date: 2026-08-03

Status: `FROZEN_DIAGNOSTIC_BEFORE_GRAVITY_CEILING_OUTCOME`

## Question

After predicted-normal guidance failed the original A0 gates, is unstable
ground orientation actually the limiting variable, or would the clearance
field still fail with an exact independent gravity direction?

The consumed `walking_rpy` 180-frame cohort is used only as a mechanism ceiling.
TUM motion-capture orientation supplies camera-frame world-up. Metric3D still
supplies all obstacle depth; registered depth remains reference-only. The
ground offset is the densest 4 cm bin in the unchanged 0.45--2.20 m interval,
with an 8 cm support neighbourhood. All collision geometry and the original
seven A0 gates remain unchanged.

An RGB frame without a motion-capture orientation within 30 ms remains
`UNKNOWN_GRAVITY`; it is not replaced and does not fall back to depth RANSAC.

If all seven gates pass, the result supports adding a camera-aligned gravity
source (for example a small calibrated IMU) to a new Development candidate. It
does not support using motion capture, does not make the final external camera
equivalent to an ARCore phone, and does not authorize A1 or App changes.

If any gate fails, stop ground-frame-only successors for this A0 construction.
