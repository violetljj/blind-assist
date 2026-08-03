# Camera-Conditioned Lightweight Scale Student R0 Protocol

Date: 2026-08-04

Status: `FROZEN_BEFORE_STUDENT_EFFECT_EXECUTION`

This is the single aggressive software successor allowed after the known-height R0 and causal R1 remained below the absolute gates. It is not another threshold or window rescue. It tests whether a tiny camera/geometry-conditioned student can learn a cross-parent correction for DA scale.

The five consumed TartanGround parents form five leave-one-parent-out folds. Each model trains on four parents and predicts only the fifth. Fold membership is by `parent_id`; frames from a test parent never enter its feature standardization, ridge fit, or scale labels.

The student is a fixed closed-form ridge regression (`alpha=1.0`) on log metric scale. Its ten runtime features are frozen in the adjacent JSON and use only known camera height, the R0 ground-plane receipt, and DA depth quantiles. The training target is the consumed synthetic sensor-depth/DA pixel-median aligned scale. That target is a training label, not independent actionability truth. Held-out sensor depth and held-out clearance truth are unavailable to the runtime prediction path.

The original R0 absolute gates are unchanged. No alpha, feature, clipping, fold, threshold, or model-family search is allowed after the result. Any failed absolute gate closes this student as `CAMERA_CONDITIONED_SCALE_STUDENT_R0_ABSOLUTE_GATES_FAIL_STOP`. Even an all-gate pass would remain consumed synthetic parent-disjoint Development evidence and would require a separately frozen fresh evaluation.
