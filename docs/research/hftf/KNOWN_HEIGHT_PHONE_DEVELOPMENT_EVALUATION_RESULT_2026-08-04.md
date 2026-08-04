# Known-Height Phone Development Evaluation Result

Date: 2026-08-04

Decision: `DEVELOPMENT_NOT_EVALUABLE_CAMERA_HEIGHT_OUT_OF_PROTOCOL`

The three fixed-mount phone sessions contain 25 RGB frames each and consumed
Samsung Quick Measure references of `0.29/0.63/0.45 m`. Their receipts recorded
the camera height as `1.43 m`, but the user corrected the physical height to
approximately `0.15 m` after the first evaluation. The original receipts remain
unchanged; the rerun records `0.15 ± 0.05 m` as an explicit metadata correction.

## Corrected result

The frozen known-height operator admits camera heights only in `0.80–2.20 m`.
All 75 corrected frames therefore failed closed as
`INVALID_HEIGHT_RECEIPT`: known-height coverage was `0/75`, and no adjusted
distance or effect metric was produced. The earlier run using the incorrect
`1.43 m` receipt value is superseded and must not be used to judge the route.

This is `NOT_EVALUABLE`, not evidence that known-height geometry works or fails.
The current three sessions cannot test the frozen route because the lens was
about 65 cm below its minimum admitted height. The next valid collection must
place the camera lens at least 80 cm above the ground and record the actual lens
height before capturing any RGB frames.

## Evidence and authority boundaries

- The 75 frames represent three sessions, not 75 independent samples.
- Samsung Quick Measure is a same-phone AR reference, not independent
  ruler/laser truth, so these sessions cannot satisfy formal P0/R2 in any case.
- No height range, scale gate, plane gate, model, ROI, or threshold was changed
  after observing the data.
- This result grants no product, alert, safety, default-app, or production
  authority.

The corrected frame-level evidence is retained under
`artifacts.local/evidence/hftf/known-height-phone-development-evaluation-height-corrected-r2-20260804/`.
The result SHA-256 is
`849B9C7CB95918D308175DE4DA8F5AA051E49CE1F1253C03B8667EE81E5BF44A`; the
frame ledger SHA-256 is
`CEBC73AA9F38FE01B651CB09E8E7440816197F3230186A2DE4B62C3063150994`.
