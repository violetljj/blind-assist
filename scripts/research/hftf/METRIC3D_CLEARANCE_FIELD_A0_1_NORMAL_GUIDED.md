# Metric3D clearance field A0.1 normal-guided successor

Date: 2026-08-03

Status: `FROZEN_BEFORE_NORMAL_GUIDED_WALKING_RPY_OUTCOME`

## Single changed variable

A0 failed under fresh roll/pitch/yaw motion with a depth-only ground plane fit.
A0.1 keeps the depth model, point cloud, bands, obstacle-height interval,
clearance percentile, collision horizons, unknown semantics, and all seven A0
fresh gates unchanged. It changes only ground-frame estimation.

The official Metric3D output `prediction_normal` contains a camera-frame unit
surface normal and a positive `kappa` concentration per pixel. A0.1:

1. uses pixels in the lower 45% of the image;
2. keeps normals with absolute camera-y component at least 0.55;
3. keeps the upper half by per-frame `kappa` median;
4. sign-aligns candidate normals toward camera-up and takes a capped-kappa
   weighted consensus;
5. forms 4 cm bins of point-to-plane offset in the accepted 0.45--2.20 m
   camera-height interval;
6. refits the densest support with SVD using an 8 cm offset neighbourhood and
   a 15 degree normal-consistency limit.

No IMU, registered depth, camera pose, person box, class, or future frame is an
input to the candidate.

## Data roles and decision

- Development mechanism screen: consumed `walking_rpy`, exact frozen 180 RGB
  frames from A0.
- If and only if all original seven A0 gates pass, freeze and evaluate the
  official `walking_halfsphere` sequence as new fresh evidence.
- Otherwise stop this normal-guided successor. Do not tune normal thresholds,
  kappa selection, offset bins, or refit limits on `walking_rpy`.

This successor does not reopen the A0 fresh terminal or authorize A1, Android,
reminders, or safety claims.
