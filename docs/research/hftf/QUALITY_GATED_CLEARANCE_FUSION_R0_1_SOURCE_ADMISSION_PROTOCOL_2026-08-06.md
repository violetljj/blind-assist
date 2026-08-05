# Quality-gated clearance fusion R0.1 source admission

R0.1 replaces the invalid R0 source roster before acquiring new media.

The finite roster is exactly:

- TUM `rgbd_dataset_freiburg3_sitting_halfsphere`;
- TUM `rgbd_dataset_freiburg3_sitting_rpy`;
- ARKitScenes Validation visit `381644`, videos `41069048`, `41069050`,
  `41069051`, selected by a deterministic identity-only rule.

The two TUM archives may be transported and audited through tar headers and
timestamp text files. RGB/depth image bodies, clearance, state, transitions,
model outputs and metrics remain forbidden.

ARKitScenes visit `381644` is identity-locked only. Its media must not be
downloaded until the existing reviewed ARKitScenes license authorization is
explicitly extended to this visit and its RGB, depth, confidence, intrinsics
and trajectory assets.
