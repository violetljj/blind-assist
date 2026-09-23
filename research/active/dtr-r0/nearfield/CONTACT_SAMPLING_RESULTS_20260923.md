# Boundary-aware sampling: real metric gains with unresolved costs

The same geometry model learns substantially better contact boundaries when its
72 training questions per image include near-boundary pairs. On consumed,
layout-held Development, width within 5 cm improves 43/528 -> 134/528 and
first-contact axial position 61/352 -> 168/352. This supports the proposed
supervision mechanism within this fixed package. It does not establish precise
virtual walking: query recall falls, first-contact false crossings increase,
and neither new model passes the original complete component gate.

Disposition: retain boundary-aware sampling as a COMPONENT research mechanism;
keep the original fixed-query packages as NEGATIVE_CONTROL. Do not promote the
new checkpoint into the App or reinterpret it as measured occupancy. No further
fit, seed, threshold rescue or automatic distance-regression successor was run.

## Fixed comparison and what changed

[Protocol](CONTACT_SAMPLING_PROTOCOL_20260923.md) was sealed before training.
The [old control](CONTACT_BOUNDARY_RESULTS_20260923.md) stays byte-identical.
Both models reuse its frozen RGB + ordered 8x8 ToF features, train-only
normalization, initial weights, 100-epoch image schedule, batch size 32,
AdamW settings and original positive-class weight 50168/12040. Each arm performs
2,700 updates. A subclass supports per-image queries without changing parameters
or architecture; tests compare logits and gradients against the original model.

Only training queries and their exact controlled-geometry labels change. One
fixed set of 72 queries per training image is reused for all 100 epochs and
both arms. There are 864 training images, 288 dev images and 576 evaluation
images, grouped by 24/8/16 complete layouts. Dev checkpoint selection still uses
old 72-query BCE every 10 epochs; the cutoff maximizes dev recall at FPR <=5%.
New selected epochs are direct 40 / geometry 10; cutoffs 0.79612923 / 0.66567636.
Old epochs 30 / 10 and cutoffs 0.62292975 / 0.52819169 remain untouched.

Sampling requested 36 uniform plus 36 boundary queries per image. Exact
admissible boundaries yielded 5,618 of 15,552 requested pairs; 9,934 pairs fell
back to uniform after the bounded search. Thus the actual 62,208 queries include
11,236 boundary queries (18.1%) and 50,972 uniform/fallback queries. Positives
are 13,368 versus 12,040 in the original fixed queries. Missing admissible
boundaries were not fabricated or clipped. This tests the whole mixed sampling
package, not a clean separation of continuous coverage from boundary targeting.

Queries stay inside original training widths [0.36,1.08] m and axial horizons
[0.6,3.0] m. Old off-grid evaluation coordinates now lie inside a continuously
sampled training domain; they are not an unseen size distribution. Evaluation
layouts remain absent from fitting, but the data source has already been
consumed. Extrapolation widths 0.24 / 1.20 m remain outside the training range.

## Boundary results

Each entry is old -> new. The denominator includes every finite interior true
boundary, including missing predicted crossings as failures. There are zero
left-censored true boundaries here, so the inherited interior metric does not
exclude any finite in-range boundary. Right-censored means no true crossing
inside the evaluated range, not necessarily no obstacle anywhere.

| Layouts | Model | Critical width within 5 cm | First-contact Z within 5 cm |
|---|---|---|---|
| Train | Direct | 94/840 (11.2%) -> 176/840 (21.0%) | 34/560 (6.1%) -> 70/560 (12.5%) |
| Train | Geometry | 124/840 (14.8%) -> 318/840 (37.9%) | 175/560 (31.3%) -> 467/560 (83.4%) |
| Evaluation | Direct | 37/528 (7.0%) -> 45/528 (8.5%) | 19/352 (5.4%) -> 41/352 (11.6%) |
| Evaluation | Geometry | 43/528 (8.1%) -> 134/528 (25.4%) | 61/352 (17.3%) -> 168/352 (47.7%) |

| Evaluation model | Width coverage | Width conditional MAE | Z coverage | Z conditional MAE |
|---|---|---|---|---|
| Direct | 483/528 -> 494/528 | 22.1 -> 21.6 cm | 280/352 -> 270/352 | 47.8 -> 41.0 cm |
| Geometry | 486/528 -> 482/528 | 18.1 -> 16.8 cm | 294/352 -> 305/352 | 21.7 -> 20.0 cm |

Conditional errors do not include missing predictions; the first table does.
The large near-boundary hit gain alongside modest conditional MAE improvement
shows that substantial error tails remain. Train geometry Z conditional MAE
is 4.47 cm, versus 20.02 cm on evaluation layouts: transfer is still a material
limitation. Axial position Z and travel distance differ by the fixed 0.3 m
sweep start; this comparison does not change that convention.

| Right-censored false crossings | Width old -> new | Z old -> new |
|---|---|---|
| Train direct | 10/888 -> 0/888 | 138/1168 -> 93/1168 |
| Train geometry | 10/888 -> 29/888 | 135/1168 -> 197/1168 |
| Evaluation direct | 108/624 -> 114/624 | 110/800 -> 107/800 |
| Evaluation geometry | 87/624 -> 77/624 | 119/800 -> 146/800 |

Geometry evaluation meets the prespecified joint boundary gain rule: both axes
gain >=10 percentage points and right-censored false-crossing rate increases
stay <=5 points. Train geometry improves strongly on finite boundaries but its
Z false-crossing increase is 5.31 points, narrowly failing that cost guard.
Do not round it down into a pass. Direct fails the joint boundary gain rule.

## Contact decisions and threshold sensitivity

The same 27,648 fixed off-grid evaluation queries are used for every comparison.

| Model | TP | FP | FN | TN | Recall | FPR | Precision |
|---|---|---|---|---|---|---|---|
| Direct old | 3346 | 1162 | 1136 | 22004 | 74.65% | 5.02% | 74.22% |
| Direct new | 3155 | 1096 | 1327 | 22070 | 70.39% | 4.73% | 74.22% |
| Geometry old | 3894 | 2196 | 588 | 20970 | 86.88% | 9.48% | 63.94% |
| Geometry new | 3518 | 1196 | 964 | 21970 | 78.49% | 5.16% | 74.63% |

Both fail the query-cost guard because recall drops more than 3 points versus
their own controls. Geometry reduces false positives substantially but misses
376 additional positive queries. Neither reaches the old joint component gate
(recall >=80%, FPR <=5%, both boundary hit rates >=80%).

As a prespecified diagnostic, evaluate new checkpoints at old cutoffs without
choosing a replacement: geometry width hit rate is 99/528 (18.75%) and Z
173/352 (49.15%). Gains over the original 8.14% / 17.33% therefore remain with
the numeric cutoff held fixed. Corresponding false crossings are 87/624 and
176/800. This supports a learned boundary change beyond cutoff movement, while
showing the cost of the lower cutoff. Direct old-cutoff rates are 35/528 and
29/352. No diagnostic cutoff replaces the dev-selected primary results.

Full saved results retain per-family/relation counts, pair ranking and joint
correctness, Brier scores, extrapolation, sampled monotonicity and UNKNOWN
counts. No sampled probability or binary reversals occurred. Group bootstrap
uses complete layouts, not individual correlated queries; this single fixed
seed does not establish stability across training seeds or generators.

Geometry new-minus-old 95% layout-bootstrap intervals are recall -11.87 to
-5.16 points and FPR -5.39 to -3.20 points (1,000 resamples, 16 layouts).
Pair ordering is direct 2977/2988 (99.63%) / geometry 2968/2988 (99.33%);
both paired decisions correct is 1381/2988 (46.22%) / 1239/2988 (41.47%).
The 487/576 baseline-sensor UNKNOWN flags are retained as metadata; these
offline models still predict on all frames. This is not validated abstention.

## Mechanism, evidence boundary and disposition

The original linear integration operator has shape 72x1200 and rank 72. A
constructed redistribution preserves all old binary decisions but changes a
new query from probability 0.950215 to 0.000036. Inherited float32 cell edges
leave a maximum old probability difference of 8.90e-8; nominal decimal-grid
mass residual is 1.29e-14. This is an operator counterexample, not learned or
measured geometry. The proof artifact and reproducible test preserve the exact
construction; its intensity choice need not equal another numerical example.

The new fit supplies empirical support for the user's diagnosis: sparse fixed
queries were a consequential limitation for the geometry arm in this package.
It does not prove they explained every error, that 72 queries identify 1200
cells, or that the integration representation is the best solution. The direct
arm's weak gains and the geometry arm's remaining error/costs preclude that
broader conclusion. The old error audit also found violations of coarse
constraints, so boundary underspecification was never the sole known issue.

Retain boundary-aware query sampling when designing a separately scoped next
comparison. A useful next question is whether a direct metric first-contact
output reduces the remaining error tails and false crossings under the same
sampling and censoring contract. That would be a new mechanism comparison,
not an automatic rerun or a reason to enlarge the current model. This run ends
after its two fits and evidence delivery.

Geometry truth is used only to generate training queries/labels; inference uses
public cached RGB/ToF features and query parameters. All new predictions were
sealed before evaluation target arrays were joined. Full geometry was loaded
earlier for indexed training rows and input files were hashed: this is code-path
separation, not evaluator process isolation. Controlled cross-section sweeps
are not full human bodies, hardware sensing, natural-distribution or safety
evidence. Existing A/LOCAL/UNKNOWN and App behavior are unchanged.

Actual compute was RTX 5060 Laptop CUDA: direct 7.63 s, geometry 9.41 s, whole
scientific run 20.29 s, excluding wrapper, audit and plotting. No service,
capture process or paid worker was started. Payloads remain under canonical
`artifacts.local/evidence/ba-contact-sampling-20260923` as durable evidence;
the mathematical proof, governed run specification and console are adjacent.

13 focused sampler/model tests pass. Independent saved-output audit PASS
reproduces 777,024 world-space labels, including all 62,208 new training queries;
it checks exact boundaries, sample pairing/fallback, unchanged input/control
seals, dev-only cutoffs, metrics and bootstrap. Selected epochs are checked
against saved history; this audit does not retrain or rerun checkpoint inference.
The narrow documentation-link check passes. Existing unrelated working edits
are preserved, and no repository-wide performance claim follows from these checks.

Post-run descriptive layout-bootstrap intervals for geometry boundary hit-rate
gains are +13.37 to +21.40 points (width) and +16.48 to +42.83 points (Z).
These intervals describe the existing 16 layouts and were not used to select
the model or alter any gate. The independent audit retains its own code hash
and artifact hashes. Plotting only reads sealed results.
