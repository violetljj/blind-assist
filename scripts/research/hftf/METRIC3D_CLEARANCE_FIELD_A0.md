# Metric3D clearance field A0

Date: 2026-08-03

Status: `DEVELOPMENT_MECHANISM_SCREEN`

## Question

Can dense monocular metric depth reproduce a sensor-depth-derived, class-free
three-dimensional clearance field closely enough to justify replacing target
distance regression with collision-space reasoning?

## Fixed construction

- Input frames: unique RGB frames from the already consumed TUM Freiburg 3
  `walking_static` and `walking_xyz` pose-torso windows.
- Model arm: Metric3D v2 ViT-S PyTorch CUDA.
- Reference arm: registered TUM depth, never provided to the model.
- Intrinsics: published Freiburg 3 values.
- Point cloud stride: 4 pixels.
- Ground: deterministic RANSAC plane from the lower 45% of the image, followed
  by an SVD refit; accepted camera height is 0.45--2.20 m.
- Obstacles: points 0.08--2.00 m above the recovered ground and 0.20--4.00 m
  forward.
- Lateral bands: left `[-1.20,-0.40]` m, centre `[-0.40,0.40]` m, right
  `[0.40,1.20]` m.
- Per-band clearance: robust 2nd percentile of obstacle forward distance.
- Collision envelope probes: each band at 1.0, 1.5, and 2.0 m forward.

Unavailable ground or fewer than 20 obstacle points in a band produces
`UNKNOWN`; it is not converted to clear or blocked.

## Development continuation gates

All must pass before opening a fresh sequence:

1. at least 90% of unique frames have valid fields in both arms;
2. pooled left/centre/right clearance MAE is at most 0.25 m;
3. known-pair collision-envelope agreement is at least 90%;
4. false-clear rate is at most 5%;
5. temporal clearance-delta MAE is at most 0.15 m.

This screen can justify a fresh A0 evaluation only. It cannot change the main
research route, Android behavior, reminders, or safety claims.
