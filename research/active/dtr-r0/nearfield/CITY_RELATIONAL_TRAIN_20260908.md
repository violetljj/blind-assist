# Frozen-B adaptation with additional relational TRAIN

Pre-outcome EXPLORE protocol. Keep G13-D and the previous B training strategy.
Question: does adding region/family/relation coverage improve BODY and HEAD
recall under the same DEV operating-point rule, without changing the network
or loss? Prior B has DEV recall40.625%/25%, joint2/32 and HEAD support IoU0.
This is a data-recipe intervention, not an isolation of region, family or class
prior effects from each other. B's prior fixed-budget win is not convergence
or causal evidence that backbone freezing solves regional overfitting.

Add three TRAIN sites, each32 quartets and128 frames, to the original TRAIN750:
1134 frames total. Use four supported fixture families, eight quartets per
site/family, with CLEAR/BODY_ONLY/HEAD_ONLY/BOTH geometry. Use newly generated
parent groups, excluding all DEV parent identities. Native visible-support
labels remain authoritative; preserve UNKNOWN and any assembly/visibility gap.
Do not force labels or drop difficult cases to match intended relations.
Require each site/family at least8 known positives and8 known negatives per head.
Canary all12 site/family pairs before the unique384-frame capture. Sites must
avoid the fixed DEV region, with at least75m camera XY separation from every
DEV camera and at least30m between TRAIN sites. These remain regions of the
same synthetic map, not three independent worlds. Record actual sites and
geometric/visibility admission before fitting; fix source issues before model
inference, never from DEV scores.

The entire existing128-frame DEV cache is immutable and evaluator-only. Its
labels/predictions have already informed development: subsequent scores are
consumed Development comparisons, not fresh tests. No fresh TEST is captured
or accessed. The previous B checkpoint and predictions remain unchanged.

One new fit: original G13 seed17 initialization, RepViT fully frozen, all BN
running buffers frozen, remaining parameters AdamW lr1e-5/weight_decay1e-4.
Near BCE plus0.25 existing globally class-balanced UNKNOWN-aware support BCE;
no new augmentation, group loss, architecture, replay or scheduling mechanism.
200 optimizer steps, batch32, seed17 uniform replacement over concatenated
old750+new384. The sampling procedure is unchanged, but numeric indices must
be regenerated for1134 entries; do not claim the old750 index sequence is reused.
Save the actual schedule and sampled source/positive counts. Use original
worker CUDA/Torch2.9.1+cu128 and frozen common module hashes. No warm start from
the prior B result, extra steps, early selection or rescue training.

Final step200 only. Per head choose DEV threshold with empirical FPR<=10%,
max recall, lower FPR, higher threshold, using the unchanged float64 inclusive
selector. Compare against the prior B's own frozen DEV operating points.
Primary: both head recalls and absolute FP/negative counts. Secondary: complete
quartet correctness, support IoU, peak hits, negative activation and UNKNOWN.
Save selected thresholds before consumed-plaza diagnostics; plaza cannot revise
thresholds or candidate choice. Also score old/new TRAIN partitions separately
at0.5 to distinguish limited fitting from transfer failure. No Willow regression
in this bounded data experiment; it remains a gap before any broader replacement.

Disposition: if recall and localization improve together, retain the data recipe
as a Development challenger; mixed outcomes identify the remaining failure.
No improvement retains the previous baseline and does not justify extra fits.
No automatic promotion, empirical-FPR population guarantee or fresh-test claim.
Stop after one200-step fit and these fixed evaluations, then record results,
deliver scoped source/report changes and release owned processes. Keep raw
captures, caches, checkpoints, protocols, schedules and receipts for reuse.

## Executed source

Sites are the north frontages at(-76,+15) and(-35,+15), and south frontage
at(-55,-15), with minimum distances to DEV130.894m,91.181m and107.079m.
Minimum inter-site camera distance is32.503m. The first site expands the
existing street vicinity; the three sites are not three new maps. Fresh seeds
20260909/10/11 give96 quartets and384 distinct physical conditions, excluding
DEV parents. Actual distance spans0.805–2.737m, pitch-7.894–5.920degrees,
lateral offset-0.098–0.098m; sun/skylight multipliers are0.8/1/1.2 and fixed
within each quartet. No separate occlusion intervention was added.

All12 site/family canaries pass native/RGB inspection; the root also inspected
the second and third site's four-family mosaics. The full384 capture and cache
pass. Actual new BODY counts are191positive/193negative; HEAD192/192. Every
site/family/head has at least15 positives and16 negatives. Frame209 was intended
BODY_ONLY but has no visible BODY query support; its actual native label remains
zero, with the geometric/visibility disagreement recorded, not deleted. This
is a visible-support task label, not a claim that its physical obstacle vanished.
Pooled support retains11.9753% UNKNOWN. Combined TRAIN1134 therefore has
BODY326positive/808negative and HEAD207positive/927negative. No DEV labels enter
training. DEV families are now represented in TRAIN; this is no longer an
unseen-family test, while the DEV region remains excluded.

Capture/editor wall time255.422s; complete capture/verification/cache job312.103s,
including4.930s CUDA cache construction. Main evidence/cache at
`artifacts.local/work/city-relational-train-20260908/evidence/` contains75
SHA-verified files (81,550,926bytes). Worker raw data remain under
`work/city-relational-train-20260908/capture-v1`. Existing DEV spec case arrays
match the captured DEV source exactly; their top-level map path differs only
between host and worker. The fixed DEV cache identity is checked again by the
trainer. This report appends source/results after the worker's pre-outcome
protocol snapshot was frozen.

## Result: no broad gain at this fixed training budget

One new fit completes. Each row uses its own DEV threshold selected by the
unchanged empirical FPR<=10% rule; both heads still have64 positives/64 negatives.

| Frozen-B TRAIN recipe | BODY TP / FP | BODY recall / FPR | HEAD TP / FP | HEAD recall / FPR | Joint |
| --- | --- | --- | --- | --- | --- |
| Previous750 | 26 / 6 | 40.625% / 9.375% | 16 / 5 | 25.000% / 7.8125% | 2/32 |
| Old750 + new384 | 24 / 5 | 37.500% / 7.8125% | 18 / 6 | 28.125% / 9.375% | 2/32 |

The changes amount to two fewer BODY true positives and two more HEAD true
positives, with opposite one-FP changes. This is a mixed tradeoff, not the
proposed large cross-region improvement. New BODY/HEAD thresholds are
`0.47925853729248047 / 0.5018537640571594`. At0.5, new DEV recall is
32.8125%/29.6875%, FPR3.125%/10.9375% and joint2/32.

DEV BODY/HEAD support IoU changes from0.06448/0 to0.06366/0.00260; peak hits
remain7/64 and2/64. Negative known-pixel activation is8.42%/0.0403%, compared
with previous11.03%/0.0034%. HEAD localization is still effectively absent;
the tiny positive IoU is not a localization recovery. UNKNOWN handling is
unchanged, and all32 DEV groups remain evaluable.

The predeclared separate TRAIN evaluation reveals a more basic limitation:

| TRAIN partition at0.5 | BODY TP / FP | BODY recall / FPR | HEAD TP / FP | HEAD recall / FPR | BODY / HEAD support IoU |
| --- | --- | --- | --- | --- | --- |
| Old750 | 6 / 6 | 4.44% / 0.98% | 2 / 406 | 13.33% / 55.24% | 0.06601 / 0 |
| New384 | 35 / 7 | 18.32% / 3.63% | 61 / 37 | 31.77% / 19.27% | 0.08986 / 0 |

The new TRAIN quartet joint score is0/96. Its BODY positive denominator is191,
not192, because the visible-support gap remains in the data. The final logged
batch loss is0.97019; this is one batch, not a convergence certificate or a
direct loss comparison across different data distributions.

Because fixed0.5 failure alone cannot diagnose lack of separation, one additional
**post-outcome, zero-training** descriptive check reused the cached TRAIN scores
with the existing tie-aware `city_score_separation.curve`. No model, selected
threshold or DEV decision changed. On new TRAIN, BODY/HEAD ROC-AUC is
0.64390/0.61575, and the label-derived recall envelope at FPR<=10% is only
30.89%/17.71%. Old TRAIN AUC is0.61407/0.20853 and corresponding envelope
25.93%/0%. These are descriptive TRAIN curves, not calibrated operating points;
no thresholds were exported. The result is saved separately in
`artifacts.local/work/city-relational-train-20260908/train-score-diagnostic.json`
with score/label/source hashes. This confirms weak TRAIN separation in addition
to the fixed-point failure; the evidence is not merely high TRAIN fit followed
by a DEV-only collapse.

Consumed plaza with the frozen DEV thresholds remains weak: BODY30/135 TP with
17/615 FP (22.22% recall,2.76% FPR); HEAD0/15 TP with16/735 FP (0% recall,
2.18% FPR). Joint124/250 cannot compensate for zero HEAD recall. No threshold
or candidate was reselected from this diagnostic.

Disposition: retain the admitted multi-site corpus and the diagnostic result;
do not promote the new weights or claim that more varied data solved transfer.
The fixed200-step frozen-B recipe has not established strong TRAIN fitting,
so this run also cannot establish that data coverage is ineffective or that a
new architecture is required. The next decision-changing check should establish
bounded optimization/fit adequacy on this retained corpus before attributing the
remaining failure to cross-region generalization. No extra fit was added here.
Fresh TEST and Willow regression remain unavailable for this challenger.

## Execution and evidence

Actual sampling uses4222 old and2178 new frame draws, with1856 BODY and1193
HEAD positive draws over6400 examples. Saved schedule SHA:
`b803a65415298299ba46b16caccc109aa17d41ba3a5dac260cbf0331317a067e`.
Original initialization/common modules, immutable DEV cache and prior B result
identities pass. Backbone and all BN buffers remain unchanged. The generator
and trainer parse; worker import preflight, full capture/cache integration and
the sole fit/evaluation pass. The unchanged selector retains its earlier eight
focused passing tests; no redundant rerun or new selection rule was introduced.

Worker RTX3060 Laptop, CUDA/Torch2.9.1+cu128: fit10.080s, full model
training/evaluation16.365s. Main results/logs/cached predictions are under
`artifacts.local/work/city-relational-train-20260908/model-run-v1/` (16 transferred
files,8,610,179bytes, each SHA checked; later release/transfer receipts are extra).
Checkpoint remains on worker at that work prefix, file`B-relational-step200.pt`,
SHA`5e4ea3b1435a2158517f8195fc7c8032d023b821d827713b98bca933c16b8f84`.
Result SHA`63ffe7ae8b1bd1a932599357f991370d64cbb90b0a6fea9f8d049daff28d9ddb`;
selection SHA`03c2b9c5613a330930bc25b3cac38f6d733906725e3348d7dedf2d65ee5e7f04`.
Task-owned worker processes are released; original inputs, new raw data, caches
and checkpoints are retained. No concurrent UE/native-map edits are owned here.
