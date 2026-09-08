# BODY-QUERY cached diagnostic: useful ranking signal, evidence transfer failure

Completed2026-09-09; zero training, inference or capture runs. This supplements
the [matched comparison](BODY_QUERY_V1_RESULTS_20260909.md), without reopening
its budget, modifying its thresholds or changing the no-replacement disposition.
All TRAIN/DEV/EVAL outcomes are consumed same-world Development evidence.

## Decision

B's EVAL gain is not merely conservative thresholding: both heads improve their
low-FP empirical recall envelope. But the HEAD ordering advantage reverses on DEV.
The query layer fits TRAIN substantially better than the high-empty-count EVAL
accuracy suggests: B recovers559/576 nonempty TRAIN cells, then25/93 DEV and29/97
EVAL. Current evidence favors a failure of query-evidence transfer across the
scene/geometry configurations, not a demonstrated inability to learn TRAIN
positives under unweighted CE. It does not establish which background, geometry,
feature or projection mechanism causes that transfer failure.

Retain the positive ranking/body-region discrimination signal. Do not launch
balanced-query-CE training as an attributed repair from these diagnostics. Class
balancing remains an untested candidate that might affect generalization; it is
not ruled out, but the proposed TRAIN/DEV corroboration condition is not met.
The current terminal remains COMPONENT-only, with no model promotion.

## Equal false-positive budgets

Each DEV/EVAL head has16 positives and24 negatives. Numbers below are maximum TP
at **at most** the named FP count, scanning already consumed scores. These are
oracle descriptive envelopes, NOT operational thresholds. Tied scores can skip
exact FP counts; the JSON separately records exact attainable points. No EVAL
threshold is installed or substituted into the original selected results.

| Split / head | Arm | FP<=0 | FP<=1 | FP<=2 | AUC |
| --- | --- | --- | --- | --- | --- |
| EVAL BODY | A | 5 | 6 | 7 | .70833 |
| EVAL BODY | B | 10 | 11 | 13 | .89583 |
| EVAL HEAD | A | 1 | 1 | 6 | .67969 |
| EVAL HEAD | B | 3 | 3 | 8 | .78646 |
| DEV HEAD | A | 11 | 15 | 15 | .96354 |
| DEV HEAD | B | 5 | 5 | 5 | .79818 |

TRAIN both heads/arms reach96 TP at0 FP and AUC1. EVAL B is better at all three
listed low-FP budgets; this is not a claim of full ROC dominance or stable
cross-region superiority. The original selected EVAL HEAD counts remain A11 TP /
8 FP versus B8 TP /2 FP. Comparing those two operating points alone cannot isolate
ranking quality. The full score audit includes every FP budget0..24 for DEV/EVAL.

## Paired identities at the original DEV thresholds

EVAL HEAD positives:6 both hit,5 A-only hit,2 B-only hit,3 both miss. Negatives:
2 both false alarm,6 A-only false alarm,0 B-only false alarm,16 both correct.
Thus B loses five and recovers two positives, not simply three lost examples.

| Category | Sample | Family | Condition | True HEAD query range |
| --- | --- | --- | --- | --- |
| A-only hit | 282 | crossbar | HEAD_ONLY | far |
| A-only hit | 291 | cabinet | BOTH | far |
| A-only hit | 299 | hanging_sign | BOTH | near |
| A-only hit | 302 | crossbar | HEAD_ONLY | far |
| A-only hit | 311 | cabinet | BOTH | far |
| B-only hit | 290 | cabinet | HEAD_ONLY | far |
| B-only hit | 319 | hanging_sign | BOTH | far |
| Both miss | 294 | oblique_rod | HEAD_ONLY | far |
| Both miss | 295 | oblique_rod | BOTH | far |
| Both miss | 310 | cabinet | HEAD_ONLY | far |

Ranges are query forward-coordinate bins: HEAD near x=.13..1.63m, far
x=1.63..3.13m; they are not inferred Euclidean object distances. Most positives
are in far bins, so their predominance among failures is not a distance-specific
causal conclusion. Losses span multiple families and both HEAD_ONLY/BOTH.

The six removed false alarms are samples281/289/309 BODY_ONLY,285/305 ABOVE,
and300 CLEAR. The two remaining alarms are293 oblique_rod BODY_ONLY and301
crossbar BODY_ONLY. Full BODY/HEAD sample identities and family/condition strata
for all three splits are saved in `scores/score-audit.json`.

## Query evidence and the class-balance hypothesis

Signal means `1-P(count0)>=.5`, matching the earlier nonempty-query report.
This is a diagnostic convention, not the model's aggregation operation. Query
accuracy below is exact0/1/2/3+ classification; near accuracy is separate.

| B query metric | TRAIN | DEV | EVAL |
| --- | --- | --- | --- |
| Nonempty cells hit / total | 559/576 (97.05%) | 25/93 (26.88%) | 29/97 (29.90%) |
| False-positive empty cells | 21/2304 | 21/387 | 23/383 |
| Exact count accuracy | 98.61% | 81.25% | 81.04% |
| Nonempty-cell AUC | .99926 | .76308 | .80579 |
| BODY nonempty cells hit / total | 285/286 | 15/46 | 13/51 |
| HEAD nonempty cells hit / total | 274/290 | 10/47 | 16/46 |
| HEAD near-bin nonempty recall | 66/82 | 0/12 | 0/6 |
| HEAD far-bin nonempty recall | 208/208 | 10/35 | 16/40 |

TRAIN contains80% empty cells, but the final B query CE is.04920: mean nonempty
CE.12167, mean empty CE.03108. Nonempty cells contribute49.46% of this final
unweighted CE despite comprising20% of cells. This is final-loss accounting,
not a measurement of gradient contribution or the full training trajectory.
Together with97% nonempty TRAIN recall, it does not support the strong claim
that positive evidence was simply ignored during fitting. Poor DEV/EVAL positive
CE (4.286/4.082) instead exposes severe out-of-training confidence errors.

B's8 EVAL HEAD misses all have no active correct HEAD cell, and no active wrong
HEAD cell at this diagnostic cutoff. Maximum correct-cell nonempty probability
ranges from.0000094 to.17475; near scores range from.0000294 to.30583. Therefore
none is in the category "correct cell strongly active but final alert suppressed".
B's11 DEV HEAD misses likewise have no correct HEAD cell above.5 (maximum.23460);
two also contain an active wrong BODY cell. Such co-occurrence is not proof that
evidence physically moved from one head to another.

B BODY misses:DEV6 with no target-head signal; EVAL9 with no target-head signal,
of which3 also activate wrong HEAD cells. The JSON's exclusive category order is
correct target cell, wrong target-head cell, wrong other-head cell, then no
target-head signal. A correctly active other head is not counted as misplaced.
Low responses may still contain weak information; these categories do not prove
an estimator, feature-reader or aggregation defect. A's query branch is auxiliary,
so corresponding A categories cannot be interpreted as its causal decision path.

If balancing is later tested under a new hypothesis, define the candidate as
`.5*mean(CE_empty)+.5*mean(CE_nonempty)` with explicit handling of absent strata.
Using the sum without.5 also doubles the auxiliary loss scale. No such training,
weight selection or claimed benefit occurred here.

## Evidence and verification

Scripts: `body_query_score_audit.py`, `body_query_cell_audit.py`. Inputs are the
unchanged cache and final predictions under
`artifacts.local/work/body-query-v1-20260908`; outputs are under
`artifacts.local/work/body-query-diagnostic-20260909/scores` and `cells-v2`.
The first cell-output version used an ambiguous category name; v2 only clarifies
it to no-target-head signal and preserves all numeric results. Original output
remains available. Both scripts use small CPU arrays, without model construction.

Input hashes/sample ordering, paired denominator closure and two independent AUC
calculations pass. Per-cell totals are checked against the original near/report
denominators; scores are not refitted or overwritten. The source receipts bind
scripts and cached inputs. The linked diagnostic terminal supplements the original
BODY-QUERY result; it is not a new trained model result.
