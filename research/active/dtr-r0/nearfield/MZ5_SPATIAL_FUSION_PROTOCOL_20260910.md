# MZ5: fixed decision fusion and fusion before spatial averaging

2026-09-10 EXPLORE, consumed controlled Development. Implements the first two
comparisons in [the literature note](idea.md). No temporal experiment in this run.
MZ1's completed fit stays frozen; these are new representation comparisons.

The fixed 0.5/0.5 RGB_ONLY/TOF_ONLY logit ensemble is computed first under its
own on-disk protocol. It has completed before this spatial protocol is written:
1378/1500 exact, 4 wrong-far and 8 cross-body errors, with reduced far recall.
Its method was specified in the literature note before computation. Keep it as
a strong baseline; do not select weights, thresholds or sources from this result.

## Question and controlled change

Does preserving within-query RGB positions until interaction with regional ToF
improve BODY/HEAD range judgments beyond an otherwise identical pooled model?
The alternative explanation is a better geometry/readout architecture or richer
RGB features alone. Include matched pooled-fusion and local RGB-only controls.

Use the same admitted5000 rows, source order, RGB144x256, frozen B/JOINT encoder,
old normalization, clean8x8 two-surface radial packet, labels, and original roles:
2500 TRAIN_ONLY,1000 DEV_ONLY,1500 EVAL_ONLY. These are already consumed data.
Recache only the original12x27x64 sampled RGB features before query averaging.
No new backbone, training labels, native-depth feature, masks, scene metadata,
augmentation or additional sensor input. Fixed calibration/query geometry is
allowed as before. Verify averaged recache against MZ1 visual features and all
5000 original alerts. A failed parity check is an engineering failure, not a fit.

Each query sample receives its normalized XYZ, the two distances and validity
bits of the angular zone containing its ray, two valid-only distance residuals
relative to its radial distance, and an in-ToF-FoV flag. A regional return is
provided as a possible surface explanation, never as an exact depth at that ray.
Outside-FoV and invalid target values are masked; fixed zone membership has no
access to native geometry or labels. The generic radial interface is not CX firmware.

All three models use a point MLP74->64->32 with ReLU, masked mean pooling per
query, then concatenate12x32 features with the original772 visual and256 ToF
features and read1412->88->4 with ReLU. There are131580 trainable parameters,
versus132228 in MZ1; extra per-point compute is reported separately.

- POOLED_FUSION: replace each valid RGB sample by its query's mean before the
  point MLP; keep all geometry and ToF paths identical to LOCAL_FUSION.
- LOCAL_FUSION: retain the individual RGB sample at each projected position.
- LOCAL_RGB_ONLY: retain individual samples but zero all observed ToF ranges
  and validity in both local and global paths. Fixed geometry stays available.

Initialize all three from the same seed53 state. Use the exact MZ1 seed59
300x128 TRAIN_ONLY schedule, AdamW lr0.001/weight_decay0.0001 and mean BCE.
Exactly300 steps per arm, final checkpoint only. No EVAL/DEV labels on the loss
device, no continuation, hyperparameter search or threshold fitting. GPU-first
execution uses a measured equivalent forward/backward workload probe.

One frozen diagnostic permutes valid RGB positions within each query using
seed83, leaving their multiset/mean, ToF, geometry and invalid positions unchanged.
Evaluate the unchanged LOCAL_FUSION checkpoint with this permutation. A decline
would show dependence on spatial assignment, not by itself prove useful fusion.

## Decision and evidence

Report exact rows and paired gains/losses versus MZ0, MZ1 FUSION, fixed ensemble,
POOLED_FUSION and LOCAL_RGB_ONLY. Include four event confusion tables, wrong-far,
cross-body, HEAD_ONLY/near, negative and region strata; preserve every row.
Predicted zeros do not assert clearance. Preserve original B alerts independently.

A within-query fusion contribution requires LOCAL_FUSION to beat both matched
controls on exact rows, improve HEAD_ONLY/near over POOLED_FUSION, and not raise
wrong-far/cross-body versus that control. Otherwise the proposed mechanism is
unsupported or mixed in this fixed test, even if some other new arm improves.
Compare the whole tradeoff and compute to the fixed ensemble before retaining
a practical research baseline; no claim of uniform dominance from total exact.

Stop after this one three-arm fit and fixed permutation evaluation, independent
checks, interpretation and scoped delivery. Preserve mechanical failed attempts
and allow their repair without changing scientific settings or inspected fits.
No temporal training, source expansion, device deployment or safety claim follows.
