# Camera-conditioned scale student offline stress R0

Date: 2026-08-04

Decision: `KEEP_FROZEN_MECHANISM_CANDIDATE_WITH_STRICT_EXTERNAL_INPUT_GATES`.

The sealed `CAMERA_CONDITIONED_SCALE_STUDENT_R0_FINAL_5P` was stressed without
fit, retraining, feature search, threshold changes, or operating-point search.
The cached-depth layer covers 330 frames, 10 parents, and 47 fixed scenarios.
The RGB layer selects 5 outcome-blind frames from each parent and reruns the
exact frozen DA checkpoint for 25 scenarios.

## What it can and cannot correct

The student corrects a single global DA scale drift very well. Across the fixed
-40% to +40% range, clearance MAE stays approximately 0.103-0.107 m versus
0.105 m clean. This matches the sealed-weight algebra: the final calibrated
depth is almost invariant to one global multiplier.

It does not correct local geometry. A 20% left-to-right bandwise deformation
raises frame-level accepted-bad to 29.1%-51.5% while coverage remains
99.4%-100%. Horizontal 20% deformation reaches 23.3%-26.4%. Confidence and the
current plane gate therefore cannot distinguish scale-correctable drift from
accepted local shape error.

Height errors through +/-10 cm do not materially degrade metrics among accepted
records. The asymmetric coverage drop for negative errors comes from synthetic
parents at 0.801-0.804 m crossing the frozen 0.80 m receipt range, not an effect
failure. This does not authorize relaxing the existing real-phone receipt:
measured uncertainty remains at most 5 cm until real-phone confirmation.

Changing only fx/fy by +/-5% produces little change. This is not evidence that
digital zoom or a different CameraX crop is safe. RGB crop, roll, and pitch
canaries are mostly accepted even when coordinates are deliberately mismatched,
so exact camera, intrinsics, crop, rotation, and mount identity must be checked
outside the student.

## UNKNOWN and image quality

The current rejection path is not reliably fail-closed under missing ground.
Masking the full-width lower ROI by 50% leaves 97.6% coverage but raises MAE to
0.321 m and false-clear to 12.3%; at 75%, coverage is still 92.7% while MAE is
0.653 m and false-clear is 34.4%. The provisional capture contract therefore
requires at least 75% unoccluded full-width lower-ROI support and returns
`UNKNOWN` below that point, pending real-phone confirmation. A center-only mask
is not equivalent to full-width ground loss and must not rescue the rule.

Blur also exposes accepted-bad behavior. Gaussian sigma 3 at 640x640 leaves 86%
coverage but produces 0.378 m MAE and 16.1% false-clear. A 17-pixel horizontal
motion blur leaves 100% coverage and produces 5.3% false-clear. Plane residual
alone is therefore not a sufficient blur detector; an independent image-quality
gate is required before output. Gamma darkening through 2.4 and exposure through
0.25 were materially better on this synthetic subset, but do not establish a
real low-light allowance.

## Provisional phone capture contract

- measured camera-height uncertainty `<= 5 cm`; keep the reported height inside
  the frozen 0.80-2.20 m admission range;
- exact camera/intrinsics identity and exact CameraX crop, rotation, and
  coordinate convention; no silent digital zoom;
- no numeric pitch allowance is established: mount identity plus a gravity/IMU
  receipt is required, otherwise `UNKNOWN`;
- at least 75% unoccluded full-width lower-ROI support, pending real-phone
  confirmation;
- an independent blur/image-quality gate; `UNKNOWN` on quality failure;
- `UNKNOWN` on any identity mismatch, invalid height receipt, insufficient
  support, rejected plane, or out-of-range recovered/student scale.

The bounds are a contract draft, not phone measurements. Full ignored evidence
is under `artifacts.local/evidence/hftf/camera-conditioned-scale-student-offline-stress-r0-20260804/`
and `.../camera-conditioned-scale-student-rgb-offline-stress-r0-20260804/`.
Their result SHA-256 values are respectively
`50DD0991B8F9E2DB227155F8580FEFC92C099C48C6EBE6980B9424D6D270B40D`
and `D4D92C225BFF97C483AD845D366C2422FCB8D26B3FC63F1985D527679E82C72A`.
All evidence remains historically consumed synthetic Development diagnostics;
it does not authorize default-App integration, live assistance, safety, or
production behavior.
