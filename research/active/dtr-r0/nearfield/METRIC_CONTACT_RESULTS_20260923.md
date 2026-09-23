# Direct first-contact CDF: fewer false horizon crossings, worse useful coverage

2026-09-23. One fixed fit completed under the [protocol](METRIC_CONTACT_PROTOCOL_20260923.md).
The new direct current-status CDF head fails its joint improvement criterion.
Retain this exact package as NEGATIVE_CONTROL, with its limited false-horizon-crossing
reduction preserved. The boundary-sampling mechanism and A/LOCAL/UNKNOWN remain intact.
No replacement threshold, seed, loss, epochs or automatic follow-up fit was run.

## Matched inputs and changed mechanism

Reuse the exact72 sampled binary queries per864 training images, frozen4864D RGB/ToF
features, train normalization, shared trunk initialization,100epoch image schedule,
batch32, AdamW1e-3/decay1e-4 and positive weight50168/12040 from the sampling comparison.
The24/8/16 layout groups and864/288/576 images are unchanged. Source is already consumed
same-generator Development; layout separation is not fresh confirmation.

Replace the1200-cell nonnegative geometry head by a131->128->64->3 head conditioned
on width/layer, predicting contact-by3m probability q, logistic location mu and scale.
Horizon enters only the normalized analytic CDF. Total effective parameters664835,
shared639488/head25347. New head initialization is seeded, not matched to the geometry
head. This tests this parameterization and capacity together, not all metric predictors.
No exact-distance label or geometry file is read by this fit. Weighted binary BCE is
current-status supervision, not exact-distance regression. q is not clear-space authority.

Select lowest original72query devBCE every10epochs: epoch50,BCE.2705543, then original
devFPR<=5% rule gives cutoff.7639141083. Dev recall77.59%,FPR4.9895%.
All ten dev logits and selected checkpoint are retained. Different dev-selected cutoffs
do not imply equal realized evaluation FPR. Predictions were sealed before target join.

## Layout-held results against saved boundary-sampled geometry

| Metric | Sampled geometry control | Direct CDF |
|---|---:|---:|
| Width within5cm, all528 finite truths |134 (25.38%)|57 (10.80%)|
| First-contact Z within5cm, all352 finite truths |168 (47.73%)|117 (33.24%)|
| Missing Z crossings on finite truths |47|94|
| False Z crossings on800 right-censored truths |146|111|
| False width crossings on624 right-censored truths |77|101|
| Query TP / FP / FN / TN |3518 /1196 /964 /21970|3247 /1399 /1235 /21767|
| Query recall |78.49%|72.45%|
| Query FPR |5.16%|6.04%|
| Width conditional MAE |16.82cm|22.93cm|
| Z conditional MAE |20.02cm|20.09cm|
| Width conditional error p90 |37.13cm|51.21cm|
| Z conditional error p90 |52.57cm|51.87cm|

The35 fewer false Z crossings come with47 additional unresolved finite Z boundaries,
271 fewer true contact queries and203 more false queries. No useful joint upgrade:
horizon gain=-14.49points (required+10), width gain=-14.58points (floor-3),
recall=-6.05points (floor-3). FPR+0.88points passes its+2 guard; Z false crossings
pass their nonincrease guard. The original full component gate also fails.

All finite truths, including missing predictions, stay in within5cm denominators.
Conditional errors exclude missing crossings and must not be treated as total accuracy.
Analytic continuous Z crossings separately achieve103/352 within5cm,258resolved,
conditional MAE19.51cm; this is not a replacement headline for the inherited grid metric.
Original487/576 sensor-UNKNOWN frames remain metadata; the offline model predicts on
all frames, which does not validate abstention or certify clear space.

Train results also regress: Z within5cm467/560 ->286/560, missing2 ->116;
width318/840 ->208/840. Thus failure is not confined to held-layout transfer.
Training dev-based selection, finite-horizon probability and location/scale ambiguity
remain possible causes; this experiment does not isolate them. Sampled binary queries
need not identify q,mu,scale uniquely. Do not infer sensor impossibility.

Horizon probability/binary reversals are zero. Width probability reversals:8 on train,
zero on evaluation; binary reversals zero. Width monotonicity is not structural.
All query sets, families,relations,pair ordering and Brier remain in saved result.json.
Whole-layout bootstrap (16groups,1000resamples) gives recall difference95% interval
[-11.67,-0.20]points and FPR[-0.10,+1.84]points; one seed/generator limits generality.

## Execution and retention

Equivalent cloned-step backend probes selected CPU_FASTER_MEASURED: CPU median4.62ms,
CUDA6.30ms on the available RTX5060Laptop. Actual fit ran on CPU,17.22s/2700updates,
complete scientific run19.44s. These tiny probes are placement evidence for this run,
not a general hardware-speed claim. No paid worker,capture process or service started.

Seven model math/gradient/query tests pass. Independent saved-output/checkpoint audit
PASS: ten dev checkpoints and threshold, all train/evaluation counts, boundaries,
subgroups, tails, gates, continuous metrics and hashes;832896 probabilities replay
exactly on the actual CPU backend (maximum error0). Bootstrap is not independently
audited. First audit admission rejected an output inside an existing input tree;
the output moved to a separate audit tree before execution. The next audit reached
CUDA replay and found one tolerance violation6.05e-6; same-backend replay then passed
without relaxing tolerances or retraining. All attempt receipts remain. Audit receipt:
`artifacts.local/evidence/ba-metric-contact-20260923-audit/result.json`. Evidence root:
`artifacts.local/evidence/ba-metric-contact-20260923` contains protocol/source snapshot,
all input hashes, initialization, backend probe, training receipt, selected weights,
selection history, prediction seals, probability grids, continuous crossings and results.
No input/control bytes changed. Controlled cross-sections are not full human geometry,
hardware, natural-distribution, event-timing or safety evidence. App behavior unchanged.
