# Expanded-source B: fixed architecture and training budget

EXPLORE, 2026-09-09. Test whether structured multi-region training coverage
improves query evidence transfer without another architecture modification.

Use accepted dataset-v1 from body-query-5000-20260909, index SHA256
88dd9687fbefe2432ac05ac57d54510ec1f0d0e65faac646f324365ae380166e.
Keep TRAIN 2500 / DEV 1000 / EVAL 1500 and complete site/group roles unchanged.
Shared CitySample assets and repeated geometry designs mean controlled
Development, not independent natural scenes or protected confirmation.

OLD is the original final B checkpoint (dd6fab0bddf436477e9076bb47a0df8914b4ddd21dcf0ea9897143643460726b),
with zero weight updates. NEW starts from the same G13 initialization and seed17
as original B, not from final B. Verify the initial tensor digest against the
original fit receipt. Fit NEW once for 2000 steps, batch32 uniform TRAIN sampling,
AdamW lr1e-5 / weight decay1e-4, all parameters with frozen BN running statistics.
Retain near BCE + .25 support BCE + .25 unweighted count CE, fixed camera,
query points, mean pooling and count-to-near path. No augmentation, depth head,
attention, balancing, checkpoint selection, extra seeds or budget extension.
Same number of presentations is not same epochs: 64000/2500=25.6 expected
passes versus historical 64000/240=266.7. This contrast tests the expanded source
package, not data quantity separately from backgrounds/geometry distribution.

After the final fit, independently calibrate OLD and NEW on new DEV using the
existing inclusive selector (max recall at FPR<=.10, lower FPR, higher cutoff;
minimum48 positives/negatives). Persist both cutoffs before EVAL inference.
Report all frames including the existing half-logit UNKNOWN heuristic; do not
reinterpret native invalid depth as FREE. RGB only enters model inference.

Report TRAIN/DEV/EVAL BODY/HEAD confusion, AUC, complete groups, condition,
region/family and near/far nonempty query recall. Include descriptive EVAL
recall at FPR<=5% and10%; never select deployment thresholds from EVAL. Compare
paired group correctness and sample-level gains/losses. Keep old B's historical
results intact. NEW may be retained as the expanded-source working baseline if
HEAD recall and complete-group correctness improve with no higher HEAD FP,
no lower BODY recall and no higher BODY FP at DEV-selected cutoffs. Otherwise
report mixed/no gain and retain OLD; no automatic rescue experiment follows.
This is not default-App promotion. Check source/cache identity, gradients,
unchanged fixed buffers, independent metric arithmetic and process release.
Mechanical failure retains its receipt; completed training is not repeated to
repair an evaluation-only defect.
