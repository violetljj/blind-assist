# Scale-Free Traversability R2 ARKitScenes RGB-D Result

Date: 2026-08-04

Terminal: `SCALE_FREE_TRAVERSABILITY_R2_NOT_EVALUABLE_SOURCE_SUPPORT`

The frozen R2 evaluator completed all 3,000 matched frames from 20 unique public
ARKitScenes visits. The independent validator recomputed the frame ledger,
per-visit summaries, gates, and terminal without importing the evaluator and
returned `PASS` with no discrepancies.

The source-support precondition nevertheless failed. Visit `472626` had truth
score coverage `111/150 = 74%`, below the frozen 80% gate, and only 19 truth
directions against the minimum 20. Visit `469455` supplied only 17 truth
directions. Therefore the round cannot admit an accuracy estimate or the
positive replication terminal.

Practical-use decision:
`USE_FOR_DEVELOPMENT_DIAGNOSTIC_REGRESSION_AND_NEXT_CANDIDATE`. The 3,000-frame
ledger, strong majority behavior, low opposite-error signal, abstention patterns,
and visit `484248` counterexample remain useful engineering evidence. They will
not be discarded merely because the formal source gate failed.

## Diagnostic observations below the failed precondition

- candidate score execution was 100% in all 20 visits;
- 19/20 visits met truth-score coverage and 18/20 met directional support;
- observed visit-macro directional agreement was 94.90%, with 636/660 pooled
  recommended directions matching the reconstructed sensor reference;
- visit-macro left/right opposite error was 1.01% (9 pooled opposite errors);
- only 16/20 visits met the 50% recommendation-coverage gate;
- visit `484248` was a clear counterexample: recommendation coverage 22.41% and
  directional agreement 38.46%, below both frozen gates;
- exact decision agreement including `AMBIGUOUS` averaged 75.56% by visit.

These values are source-characterization diagnostics, not admitted accuracy.
Even if the source-support precondition were ignored, the per-visit coverage and
worst-visit accuracy gates would still prevent a supported conclusion. No truth
fraction, reconstruction, visit, frame, model, or threshold was modified after
candidate outputs were read.

## Authority boundary

The 20 visits were consumed by an older spatial-calibration Development round.
This reuse is allowed under the disclosed roles `PROJECT_CONSUMED_DEVELOPMENT`
and `OPERATOR_UNSEEN_EXTERNAL_REPLICATION`; it is not globally fresh, sealed, or
Confirmation evidence. Nearest confidence-2 sensor-return reconstruction is a
derived dense reference, and ARKitScenes handheld indoor scans do not represent
fixed eyeglass navigation.

Immutable evidence is retained under
`artifacts.local/evidence/hftf/scale-free-traversability-r2-arkitscenes-rgbd-consumed-20260804/`.
Result SHA-256 is
`1026E3E3C2BABF805D99F4CCA481725EE1D745848662999D44D5F8960B5E81F6`;
frame-ledger SHA-256 is
`BD65AE6114786C7CF0C5F75D80BF6EE1B9F94DAA4F608F43B2C74F26F0A1174E`.
This result does not authorize clearance, distance, alerts, safety, App
integration, or production behavior.
