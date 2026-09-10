# MZ26: longer optimization helps; hard negative tails remain

2026-09-10, consumed Development. One deterministic4800-step trajectory completed
with predeclared1200/4800 checkpoints. The longer endpoint improves placement
far additions69->71 of75 and added false positives7->3, preserving trained pole
49/48 of49 clean/stress and every MZ5 positive judgment. Both endpoints fail the
zero-added-FP condition. Retain the optimization-response evidence as a COMPONENT;
MZ5 baseline and MZ20 ranking challenger remain unchanged. No App promotion.

## Fixed comparison and task effect

The [protocol](MZ26_DETERMINISTIC_CONVERGENCE_PROTOCOL_20260910.md) fixes the
10577-parameter availability head, seed123 initialization, MZ24 objective,
Adam0.001, continuous optimizer state, MZ20 scores/cutoffs and MZ5 add-only fallback.
Four exact1200x16 batch cycles provide76800 draws over7562 unique TRAIN frames.
Both checkpoints use the MZ25 fixed bilinear sampler and recorded strict CUDA
determinism. All fitting finishes before either endpoint is evaluated.

| Candidate | Added placement farTP /75 | Added placement FP | Exact frames /3000 | Pole clean/stress /49 |
|---|---:|---:|---:|---:|
| MZ5 baseline | 0 | 0 | 2803 | historical baseline |
| MZ20 reference | 75 | 8 | 2862 | 49/48 |
| MZ24 historical | 70 | 4 | 2860 | 49/48 |
| MZ26 step1200 | 69 | 7 | 2855 | 49/48 |
| MZ26 step4800 | 71 | 3 | 2860 | 49/48 |

Only the two MZ26 endpoints isolate duration within an actual shared trajectory.
Comparisons to MZ24 are descriptive because sampler/backend settings changed.
The75 denominator is the MZ20 far-recovery opportunity, not all true obstacles.
Exact requires all four BODY/HEAD near/far bits correct in a frame.

RelationDEV exact1909->1913 and distanceDEV946->947. OldDEV stays939/1000 exact,
but loses two recovered farTP bits: BODY17->16 and HEAD16->15. Thus improvement
is not uniform across cohorts. The step4800 placement addedTP vector is
[2,27,21,44], addedFP[1,1,0,1], in BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR order.
Remaining false additions are relation frame473 HEAD_FAR (oblique rod), relation
1677 BODY_FAR (hanging sign), and distance778 BODY_NEAR (hanging sign).

AddedFP is not totalFP: placement baseline has81 false bits, so the final
candidate has84. The add-only design cannot remove original baseline errors.
Clean/stress200-frame exact remains176/175; these are consumed trained sequence
configurations, not independent temporal improvement. Wrong-zone relation gains
one true bit with no addedFP; wrong-zone distance retains one addedFP and no gain.
No missing support is converted to CLEAR.

## TRAIN versus DEV extrema

Frozen inference covers11562 unique frames with no further updates. Negative
bags are frame/query groups with at least one geometrically eligible location
whose native availability label is0. Activation means maximum availability
logit>=0; it is not itself a task false alert. A positive witness is the fixed
MZ20 highest-scoring eligible actual contributor, selected without task cutoffs.

| Slice | Negative active at1200 | At4800 | Negative maximum p99,1200->4800 | Positive witness retained,1200->4800 |
|---|---:|---:|---:|---:|
| TRAIN7562 | 1855/6303 (29.43%) | 1156/6303 (18.34%) | 2.40->2.57 | 5800/6239->5815/6239 |
| oldDEV1000 | 317/822 (38.56%) | 236/822 (28.71%) | 2.56->3.00 | 694/770->685/770 |
| relationDEV2000 | 612/1650 (37.09%) | 506/1650 (30.67%) | 3.01->4.32 | 1339/1539->1322/1539 |
| distanceDEV1000 | 346/964 (35.89%) | 274/964 (28.42%) | 2.31->3.13 | 951/1000->946/1000 |

Longer fitting reduces average negative loss and activation on all four slices.
However extreme negative scores rise, and positive witness retention declines
on every DEV slice. TRAIN positive retention improves92.96->93.20%. This is
evidence of a residual separation/generalization problem, not proof of a capacity
ceiling, complete convergence, or a unique cause of each error. Pole has zero
negative-availability bags and cannot test this difficult-negative mechanism.
Blindly extending this fit is not the next discriminating comparison. Inspect
observable inputs at unsupported versus supported locations before choosing an
objective or spatial-representation successor; preserve this completed run.

## Verification, cost and disposition

Independent audit passes62400 task bits across both endpoints and391372800
geometric candidate bits,27 input hashes, exact initialization/batches, recorded
strict backend settings and checkpoint ownership. Three saved loss batches at
steps1/1200/4800 agree with independent scalar loss within1.34e-7 and analytic
gradients within1.74e-9 over150528 elements. Selected inference replays58 frames
per endpoint,116 total, with identical availability signs/support/task bits.
Changed replay grouping causes at most1.81e-5 availability and5.73e-6 raw-score
differences, within predeclared tolerances; these are not bit-exact floats.

Fitting takes73.79s; complete run99.40s and extrema diagnostic21.85s on the recorded
CUDA backend. These are cached-feature offline costs, not camera-to-alert or phone
latency. The sampler uses9.8MB derived matrix storage in addition to parameters.

Durable outputs: `artifacts.local/work/mz26-deterministic-convergence-20260910/`
contains `run-v1/receipt.json`, both checkpoints/predictions, `run-v1/audit.json`,
and `extrema-v1/receipt.json` plus per-frame arrays. All three receipts/audit pass.
Run and diagnostic processes exited. No capture, temporal state, device session,
or ongoing allocation was created. The controlled optimization response remains
eligible as a COMPONENT; it does not replace the baseline or establish independent
source, real-device or safety performance. The broad obstacle-improvement goal
remains active.
