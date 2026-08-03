# Metric3D clearance field A0 fixed-world-floor reference

Date: 2026-08-03

Status: `FROZEN_BEFORE_FIXED_WORLD_FLOOR_OUTCOME`

The first gravity-oriented reference correction still selected two horizontal
surfaces at approximately 0.8 m and 1.5 m camera height. This proves that
orientation alone cannot identify the floor when a desk is also horizontal.

The final reference correction defines one fixed world floor from the separate,
consumed TUM `walking_static` 0-second calibration window:

- 30/30 frames yielded a gravity-guided sensor floor;
- median world-floor z: `0.003324743488025139 m`;
- median absolute deviation: `0.0027076152515574936 m`;
- observed range: `[-0.009355602634888438, 0.017514498706511894] m`.

For every reference frame, mocap camera pose transforms this fixed world plane
into the camera frame. Registered depth supplies obstacle points. The Metric3D
candidate remains the original RGB-only depth-RANSAC arm and receives none of
the floor calibration, pose, gravity, or sensor data.

All A0 geometry, samples, unknown handling, and seven effect gates remain
unchanged. The corrected `walking_rpy` evaluation is Development-only because
its outcomes have already been opened. Passing all gates would authorize one
new `walking_halfsphere` fresh confirmation; failure would cleanly close this
exact candidate.
