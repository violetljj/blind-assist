# Regional pairing helps query recall, not held-layout contact localization

Completed the fixed [two-arm protocol](REGIONAL_CONTACT_PROTOCOL_20260923.md).
Correct regional RGB/ToF pairing did not pass the localization criterion against
within-frame RGB derangement. Retain the exact comparison as NEGATIVE_CONTROL,
with query-recall improvement as a partial result. No App or A/LOCAL/UNKNOWN change.

Both arms used the same 20,163-parameter regional model, initial weights, 100 epochs,
2,700 updates, sampled binary queries and exact train-only contact supervision.
The consumed simulation split has 864/288/576 train/dev/evaluation images and
24/8/16 layouts. Only RGB-to-sensor pairing changed. Frozen 40x8x14 visual features
were pooled at nine locations per sensing region. The comparison changes both
RGB-to-position and RGB-to-range association; it cannot isolate return ownership.

## Fixed-selection results

Dev BCE selected aligned epoch50/cutoff0.6808404922 and misaligned
epoch20/cutoff0.7803325653. Identical selection rules need not select identical
epochs. No checkpoint or cutoff was chosen from evaluation contact accuracy.

| Evaluation metric | Misaligned | Aligned | Aligned minus control |
|---|---:|---:|---:|
| Actual contact-distance boundary within5cm |12/352 (3.41%)|28/352 (7.95%)|+4.55pp|
| Conditional-median distance within5cm |55/352 (15.63%)|51/352 (14.49%)|-1.14pp|
| Width boundary within5cm |48/528 (9.09%)|64/528 (12.12%)|+3.03pp|
| Unseen width+horizon query recall |58.01%|70.06%|+12.05pp|
| Same query false-positive rate |6.79%|7.74%|+0.95pp|
| False distance crossings on right-censored truth |122/800|119/800|-0.38pp|
| False width crossings on right-censored truth |98/624|176/624|+12.50pp|

Both required distance gains were below10pp; all other predeclared joint terms
passed. The original full component gate failed for both arms. The width false
crossing increase was not a joint criterion, but is a substantial separate cost.
Query counts are aligned TP3140/FP1793/FN1342/TN21373 and misaligned
TP2600/FP1574/FN1882/TN21592. Layout bootstrap95% intervals for aligned-minus-control
query recall are [+5.47,+19.58]pp and FPR [-0.19,+2.31]pp; one seed/corruption
does not quantify training or corruption uncertainty.

Train actual distance hits were aligned27/560 versus control16/560; conditional
median hits255/560 versus180/560. Thus the +13.39pp train internal-position gain
did not transfer to held layouts. This supports a transfer gap for this package,
not proof of memorization or impossibility of better regional representations.

## Missing cases, tails and context

Conditional median is the modeled location GIVEN contact by3m, and is defined even
when contact probability is tiny. It is not an observed or deployed obstacle.
All352 finite evaluation truths remain in its denominator. Aligned/control median
MAE21.77/23.96cm and p90 43.64/47.94cm; modest error reduction did not improve5cm hits.
Actual distance boundaries are missing on121/154 of352 finite cases; conditional
MAE on the resolved subset is29.61/37.84cm. Actual width boundaries are missing
on31/71 of528; conditional MAE22.17/27.45cm. Missing cases remain failures in the
headline within5cm rates. Both arms have zero sampled width probability reversals
and zero binary reversals, which does not establish geometric correctness.
Actual resolved-boundary p90 is aligned/control56.32/68.84cm for distance and
46.68/57.90cm for width. Train distance missing155/242, p90 34.01/52.73cm;
train width missing13/75, p90 42.82/56.76cm. Tail statistics exclude missing cases,
unlike the headline hit-rate denominators.

487/576 evaluation frames retain the baseline sensor UNKNOWN flag in both arms.
Predicted probabilities do not turn those frames into observed free space.

| Evaluation family | Control recall / FPR | Aligned recall / FPR |
|---|---:|---:|
| Body protruding plane |59.38% / 13.14%|79.94% / 13.19%|
| Body suspended solid |45.01% / 7.52%|55.06% / 7.96%|
| Head hanging plane |62.08% / 3.30%|69.81% / 3.83%|
| Head horizontal |73.64% / 3.86%|79.31% / 6.50%|

Outside-layout query FPR rose9.38% to10.33%. Lateral-pair ordering improved
74.00% to80.29%, while both decisions correct improved12.88% to17.87%.
Full family/layout/query metrics are retained in the result payload.

Historical references are descriptive, not parameter-matched: prior exact-global
CDF had27/352 actual distance hits and100/352 conditional-median hits. Sampled
geometry had168/352 actual distance hits,134/528 width hits,78.49% query recall
and5.16% FPR. The regional result does not establish a better boundary model.

## Evidence and disposition

Run payload: `artifacts.local/evidence/ba-regional-contact-20260923/`.
All predictions and width0.6 components were sealed before evaluation-target join.
Two fits completed in62.14s total; actual CUDA backend was selected by equivalent
32-image full-loss probes (CPU10.15ms, CUDA9.63ms median). Twelve focused tests passed,
including independent pooling and reduction-order regression checks.
Checkpoints, initialization, permutations, train normalization, ten dev logits per
arm, source snapshots and input hashes are retained. No training restart occurred.

Independent audit passed at `artifacts.local/evidence/ba-regional-contact-20260923-audit/result.json`:
4,423,680 pooled visual scalars and all1,728 derangements checked; sensor fields
unchanged; train-only channel moments verified in float64; all selected metric
arithmetic and ten dev checkpoints per arm checked; conditional medians independently
inverted by60-step bisection. First32rows per arm/split replayed92,544 probabilities
on CUDA with zero difference. This is representative checkpoint replay, not full
training/optimizer reconstruction or independent bootstrap verification.

The first audit attempt incorrectly required byte-equal float32 moments after a
saved-array layout change altered reduction order (maximum difference6.08e-6).
Restoring original strides reproduced the stored float32 moments exactly.
The corrected audit uses independent float64 moments (maximum discrepancy7.52e-7,
2e-6 absolute/relative numerical bounds) and exact saved-normalization feature
pairing checks. Failed audit journal retained; no trained or predicted value changed.

This fixed coarse regional model fails its localization role. Preserve the
classification/ordering partial signal without promoting it to precise contact
geometry. A successor would need a separately justified observation or readout
mechanism and a new explicit comparison; this consumed run is not reopened.
