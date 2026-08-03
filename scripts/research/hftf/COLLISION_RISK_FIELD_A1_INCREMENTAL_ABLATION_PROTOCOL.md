# Collision risk field A1 consumed incremental ablation protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_CONSUMED_ABLATION_EXECUTION`

## Question and authority

On the already consumed TUM `walking_halfsphere` A1 cohort, does the fixed
ten-feature causal motion block add diagnostic value beyond the fixed
eight-feature metric-geometry block?

This is a Development-only ablation. It opens no fresh source, cannot repair
or replace `COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL`, and cannot authorize a
new model, threshold, camera, Android, reminder, product, or safety claim.

## Frozen comparison

The two arms use the same sequence-level leave-one-window-out folds, standard
scaling fitted inside each training fold, logistic objective, L2 coefficient
`0.01`, positive weight `1.25`, and decision threshold `0.50`:

1. `geometry_only`: the first eight frozen A0.1 features;
2. `geometry_plus_motion`: those eight features plus the ten frozen RAFT,
   affine-camera-motion, residual-flow, and missing-motion features.

No feature regrouping, seed search, refit choice, regularization change,
threshold search, subgroup selection, or post-result rescue is allowed.

## Readout and terminal

Report pooled Brier, log loss, ECE, precision, recall, FPR, F1, balanced
accuracy, MCC, and per-window Brier. Motion increment is supported only if:

- pooled Brier and log loss are both lower;
- pooled F1 is higher without reducing recall; and
- Brier improves in a strict majority of held-out windows.

Otherwise the exact terminal is
`A1_CONSUMED_MOTION_INCREMENT_NOT_SUPPORTED`. A positive terminal remains
consumed-data diagnostic evidence, not independent confirmation.
