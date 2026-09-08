# BodyLift R0: frozen visual features, depth-supervised lifting

2026-09-09 EXPLORE, same consumed320 frames. No depth prior download. Native
export writes nonfinite/nonpositive/>=100m values to0, destroying no-hit versus
invalid distinctions. Keep0 ignored, not FREE. Predict16 axial depth bins:
15 bins of0.3m on(0,4.5), plus valid[4.5,100) overflow. Exact edge belongs to
the higher bin. Each18x32 feature tile receives a histogram over its20x20 valid
native pixels; ignore tiles with no valid depth, retain histogram mixtures.

Freeze B backbone, deep/detail projections and support branch. Cache their
64-D18x32 appearance once and verify reproduction. Warm-start query_point and
count readout from final B. Fit TWO new matched heads-only arms,2000 steps each,
same seed17/batch32 schedule, AdamW lr1e-4 weight_decay1e-4, near BCE+.25 count CE.
CONTROL reads frozen appearance through original projection/mean/count path.
LIFT adds a zero-initialized1x1 64->16 depth head, explicit histogram CE*.25,
and scales appearance at each image location by16*P(query-depth-bin) BEFORE
bilinear projection. Uniform initialization is identity. This is probability
relative to uniform prior, not a bounded veto; query XYZ/mean/count units remain.
No learned attention, new backbone fit, or added capture. Cached support is
unchanged and its loss has no trainable gradient, so omit that constant term.

Final2000 only; same original DEV FPR<=.10/min_count8 cutoff selection, persist
before EVAL. Report original B, CONTROL and LIFT, both heads, query recall,
groups/negative conditions, depth histogram CE and near/far depth diagnostics.
Useful signal requires HEAD>=10/16 at low comparable FP and no material BODY
regression versus BOTH original B and matched CONTROL. EVAL oracle FP<=2 is
diagnostic, not selection. No sweep or automatic promotion. Failure rejects this
frozen-appearance depth-bin implementation only, not all depth mechanisms.

All fitting and feature extraction use primary CUDA. Native depth is TRAIN
supervision only; runtime uses RGB-derived features plus fixed calibration.
