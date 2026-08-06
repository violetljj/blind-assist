# Clearance-Student Mobile S1 development result

Terminal: `S1_A_NOT_SUPPORTED_UNDEFINED_GEOMETRY`.

S1 deliberately differed from S0: MobileNetV3-Large, 5,375,482 parameters,
four decoder scales, fixed pooled four-layer Canonical feature distillation,
and geometry-first curriculum.  The frozen five-epoch S1-A run selected epoch
1 by validation loss.

Implementation caveat: this first executable S1-A run optimized metric/teacher
depth, gradients, scale, and four feature targets.  Although the model exposes
ground-plane and camera-height heads, independent differentiable targets for
those heads were not yet materialized in the A4 stream, so their named losses
were not active.  This run therefore closes this concrete S1-A implementation;
it must not be presented as a complete test of every proposed S1 mechanism.

The fixed 120-frame consumed development screen reported:

- scale-aligned AbsRel: `0.1710521598`;
- camera-height MAE: undefined;
- clearance MAE: undefined;
- collision agreement: undefined;
- false-clear: undefined;
- false-block: undefined;
- known collision decisions: `0`.

Undefined metrics fail closed.  The student did not recover a usable ground
plane on the fixed cohort, so this is worse than both the intended geometry
screen and the S0 relative-structure result (`0.1190`).  S1-B is therefore not
authorized.  No clearance/occupancy/confidence curriculum, QNN profile, QAT,
Android integration, holdout access, production replacement, or safety claim
may follow from this run.

The experiment remains `DEVELOPMENT_ONLY`.  Canonical DA V2 518 remains the
teacher and quality upper bound.  The consumed 120 frames must not be used to
retune this checkpoint, seed, loss weights, thresholds, or model selection.
