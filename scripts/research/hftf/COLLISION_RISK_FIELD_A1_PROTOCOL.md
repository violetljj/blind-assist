# Collision risk field A1 protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_A1_ARM_EXECUTION`

## Question and data role

Does the frozen motion-conditioned collision probability discriminate current
fixed-path occupancy better than simpler 2D, detector, and deterministic 3D
rules?

The already consumed TUM Freiburg 3 `walking_halfsphere` seven-window cohort is
Development data for A1. This is an arm-comparison and visualization screen,
not another fresh confirmation.

Each known sensor-depth band x horizon opportunity remains one independent
scoring row. The fixed lateral bands are `[-1.2,-0.4)`, `[-0.4,0.4)`, and
`[0.4,1.2)` metres; horizons are 1.0, 1.5, and 2.0 metres.

## Frozen arms

1. `yolo_center_box`: class-agnostic COCO YOLO11n detections at confidence
   0.25 and image size 320. A detection centre in an image-third band marks
   every horizon risky; its maximum confidence is the score.
2. `bbox_unidepth`: the same detections, with median UniDepth in the central
   half of each box. The projected metric box centre must enter the lateral
   band and its depth must not exceed the horizon.
3. `unidepth_2d_corridor`: no ground separation; take the second percentile of
   valid UniDepth values in each projected lateral band between image rows 10%
   and 90%, then compare with the horizon.
4. `unidepth_3d_envelope`: the frozen ground-separated deterministic clearance
   decision already stored in the A0 report.
5. `motion_probability_field`: the frozen A0.1 18-feature model and threshold
   `P(occupied)>=0.50`.
6. `sensor_depth_oracle`: reference label only, reported as a ceiling.

No detector class allowlist, confidence change, depth statistic, ROI, band,
horizon, score transformation, model coefficient, or decision threshold may be
changed after the first complete arm report.

## Metrics and continuation rule

Report Brier score, precision, recall, false-positive rate, F1, balanced
accuracy, and Matthews correlation coefficient (MCC) for every arm.

A1 passes only if all are true:

- at least 1,500 known opportunities;
- probability-field Brier reduction is at least 15% versus the deterministic
  3D envelope;
- probability-field occupied recall is at least 85%;
- probability-field false-positive rate is at most 15%;
- probability-field MCC is strictly greater than every non-oracle comparator.

Pass authorizes an A1 visualization and later independent-camera replication,
not reminder logic, Android promotion, or safety claims.
