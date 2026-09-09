# Frozen context decoder distance-pair transfer

2026-09-09 EXPLORE. The completed context-decoder comparison finds JOINT near
1680/1788 and wrong-far24/600 but regressed alerts/groups. This separate diagnostic
tests unchanged LOCAL/JOINT decoders on the existing admitted5000-frame2500-pair
source, using original10k B as baseline. No fitting, normalization update, threshold
selection, new data capture or checkpoint choice. Both decoder arms are evaluated.

Hash-check historical distance receipt, metadata, probabilities, native evaluator
and every source RGB. Reproduce old10k B predictions in32-frame batches. Reuse
TRAIN-only normalization and final frozen decoder checkpoints. Native data enters
scoring only after RGB inference; no endpoint/region/site feeds the model.

Primary: EVAL_ONLY750 pairs/1500 frames, already consumed shared-site Development.
Also disclose all2500 pairs and region slices. Report near/far event confusion,
exact two-range correctness, near query TP/FP, paired far-score direction/magnitude,
count-derived alert hits/false BODY, and unchanged retained-BASE alerts separately.
Strong transfer signal: near-query recall>=50%, wrong far-event FP<=150/750,
and exact HEAD range>=1200/1500. This is a diagnostic threshold fixed before outputs,
not a protected confirmation claim. Failing preserves the source-specific result.

All accepted endpoints are HEAD-positive: no HEAD alert FPR can be estimated.
No complement of far becomes near; independent count convolution applies.
Existing source limitations/UNKNOWN persist. Retaining old alerts plus new range
evidence is a separate two-output research interface, never relabeled as a passed
full count-bottleneck replacement. Disagreement does not establish CLEAR.

Stop after one frozen inference pass per decoder, probability validation and report.
One backbone pass may serve both fixed decoders without changing their computations.
Output artifacts.local/work/body-query-context-distance-20260909/run-v1. No persistent
process/worker allocation; retain exception receipts and all source evidence.
