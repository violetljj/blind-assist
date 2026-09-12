# MZ85: rotation-compensated ToF state canary

Mode: `EXPLORE`, zero training, consumed MZ84 analytic source.

## Question

Can a fixed gyroscope-derived 3-D rotation remove the eight MZ84 head-motion
false positives without changing radar, fusion authority, thresholds, positive
event timing, or the two known gap-onset fragments?

This is a paired mechanism diagnostic, not a new-source confirmation. The only
predictor change is that every current ToF ray is rotated from the moving camera
frame into the episode's initial stabilized frame before temporal association
and corridor testing. The MZ84 radar packets, radar expert, ToF range/corridor
cutoffs, hysteresis, and fusion rule remain frozen:

`ToF_alert OR (ToF_UNKNOWN AND Radar_alert)`.

The IMU forward model exposes only frame-to-frame yaw/pitch angular increments
at the same 10 Hz timestamps. Orientation is recovered by causal integration.
No evaluator truth, family label, target identity, accelerometer, translation,
future sample, learned parameter, or threshold search enters the predictor.

## Predeclared gates

Relative to exact MZ84 fusion `160 TP / 8 FP / 11 FN`:

1. fusion FP is at most 2 and no more than half the baseline;
2. fusion TP is at least 158 and added FN is at most 2;
3. every positive event keeps its MZ84 first-correct-alert frame, so added delay
   is 0.0 s;
4. no event gains an alert fragment, the two existing multi-target gap-onset
   fragments remain untouched, and every non-head-motion prediction is bitwise
   identical to MZ84.

Failure of any gate closes this fixed rotation recipe. Passing retains the
rotation layer as a component and makes fresh-source or measured IMU evidence
the next requirement. It does not authorize hold-state tuning or learned state
estimation.

## Limits

The yaw/pitch trajectory and ToF association error are constructed proxies.
Gyro bias, clock skew, camera/IMU extrinsics, rolling acquisition, translation,
vibration, drift, packet loss, and real sensor noise are absent. A pass shows
only that a causal rotation transform can remove the specifically constructed
ego-rotation error without damaging frozen MZ84 behavior. It is not measured
IMU, hardware, alert, deployment, user-benefit, or safety evidence.
