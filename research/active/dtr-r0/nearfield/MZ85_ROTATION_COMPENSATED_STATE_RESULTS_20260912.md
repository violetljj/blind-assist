# MZ85 rotation-compensated ToF state canary

Decision: `ROTATION_COMPENSATION_MECHANISM_CANARY_PASSES`.

On the consumed MZ84 480-frame analytic source, causal gyro rotation removes
all eight head-motion false positives without changing any true prediction,
first-correct-alert frame, or event fragmentation. Fusion changes from
`160 TP / 8 FP / 11 FN`, F1 0.944 to `160 / 0 / 11`, F1 0.967. All four
predeclared mechanism gates pass.

## Fixed comparison

[Protocol](MZ85_ROTATION_COMPENSATED_STATE_20260912.md) freezes the MZ84 source,
radar forward model and expert, ToF range/corridor thresholds, observability,
fusion authority, and hysteresis. Recomputed baseline episode IDs, radar, ToF,
fusion, ToF height and fusion height arrays are bitwise identical to sealed
MZ84 run-v2 predictions.

The only new observable is causal frame-to-frame yaw/pitch rotation at 10 Hz.
The predictor integrates those increments independently per episode and applies
a 3-D rotation to each ToF ray before temporal association and corridor testing.
No accelerometer, translation, truth, family label, target identity, future
sample, fit or threshold search is used.

## Result

| Method | TP | FP | FN | TN | F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Frozen MZ84 fusion | 160 | 8 | 11 | 301 | 0.944 |
| **Rotation-compensated fusion** | **160** | **0** | **11** | **309** | **0.967** |

The exact delta is eight corrected negatives and no other decision changes:

- head-motion FP: `8 -> 0`;
- TP/FN: `160/11 -> 160/11`;
- every non-head-motion fusion bit is identical;
- radar predictions are unchanged;
- every positive event has 0.0 s added first-alert delay;
- no event gains or loses an alert fragment.

The two known multi-target gap-onset fragmentations remain exactly two segments
in both baseline and compensated predictions. MZ85 neither fixes nor worsens
them; observation continuity remains a separate mechanism question.

## Interpretation

This result isolates the MZ84 residual FP mechanism: the constructed failures
come from comparing rays in inconsistent rotating camera coordinates. A causal
rotation into one stabilized frame is sufficient to remove them without recall
or timing tradeoff. In the controlled architecture, IMU therefore has a clear
role as the coordinate transform supporting temporal ToF association, rather
than as another classifier vote.

Retain the rotation layer as a `COMPONENT_OR_CHALLENGER/COMPONENT`. Do not train
a state estimator or tune gap hold time on this consumed source. The next valid
IMU question must change the evidence source through fresh simulation with
independent head motion, or measured/device-calibrated gyro plus sensor traces.

## Evidence boundary

MZ85 is intentionally a mechanism canary on the already consumed MZ84 analytic
source. Its yaw/pitch path and association error were constructed. It omits gyro
bias and drift, timestamp skew, camera/IMU extrinsic error, rolling acquisition,
translation, vibration and real packet noise. The perfect correction therefore
does not estimate real IMU performance or generalization and is not hardware,
alert, deployment, user-benefit or safety evidence.

## Evidence and verification

Canonical evidence is
`artifacts.local/work/mz85-rotation-compensated-state-20260912/run-v2/`.
It contains observable gyro increments, integrated orientations, sealed
predictions, evaluator arrays, receipts, exact MZ84 prediction hash and result.
Three focused tests pass for episode-isolated causal integration, 3-D ray
rotation and non-head-motion prediction identity. Python compilation, knowledge
validation and `git diff --check` pass.
