# MZ66: replay projection is a scoped negative control

The [registered comparison](MZ66_REPLAY_PROJECTION_20260911.md) fails its
retaining criterion: **5/11 clauses pass**. PROJECT reduces old false alerts,
but loses old and new true alerts. Retain this optimizer recipe as
`NEGATIVE_CONTROL`; [MZ64 GEOMETRY](MZ64_GEOMETRY_LEARNING_RESULTS_20260911.md)
remains a `CHALLENGER`, with its explicit old-scene costs.

One PROJECT fit uses the same MZ62 CONTROL initialization, 1,536-step MZ64
GEOMETRY sequence, source roles, original loss and calibration rule. Only the
actual Adam parameter step receives an OLD_NEG gradient halfspace constraint.
The sealed MZ64 GEOMETRY run is the equal-update-budget comparator; no control
refit, cutoff search, source expansion or outcome-selected mode was used.

| DROP final OR | MZ64 GEOMETRY TP/FP | PROJECT TP/FP | TP gained/lost | FP added/removed |
| --- | ---: | ---: | ---: | ---: |
| Legacy noncal | 2748/62 | 2740/59 | 0/8 | 2/5 |
| MZ48 nonfit1024 | 554/25 | 547/24 | 0/7 | 0/1 |
| MZ55 held640 | 487/86 | 481/84 | 0/6 | 0/2 |
| MZ61 held1024 | 795/38 | 775/38 | 1/21 | 0/0 |

Old totals are FP173→167 and TP3789→3768. The six net fewer false alerts include
two newly false bits and eight removed bits. MZ61 held loses nine BODY_NEAR,
two HEAD_NEAR and ten HEAD_FAR TP, gaining one BODY_FAR TP. Native BODY_NEAR
additions beyond OLD_NEG fall30→20; held-awning native additions fall5→3.
All four old-FP clauses and the held no-new-FP clause pass; all six TP/native
retention clauses fail.

| MZ61 profile | All4096: GEOMETRY→PROJECT TP/FP | Held1024: GEOMETRY→PROJECT TP/FP |
| --- | --- | --- |
| IDEAL | 3739/351→3674/333 | 921/91→908/85 |
| MERGE_CLOSE | 3816/119→3707/118 | 938/31→914/31 |
| DROP_CLOSE | 3244/154→3168/154 | 795/38→775/38 |
| ALL_INVALID | 248/16→202/16 | 43/4→36/4 |

The full source contains4096 positive query bits among16384 known bits; held
contains1024 among4096. ALL_INVALID still misses988 held positive bits. Its
ranges/validity remain zero, with no valid anchor; no such stress profile was
added to fitting. Naturally missing DROP inputs remain possible.

Projection occurs on361/1536 steps. All assigned-step scalar bounds and the16
preselected vector reconstructions pass. Actual same-batch negative loss rises
on292 steps:260 projected and32 unprojected, maximum+0.017093. The average
change across all steps is−0.009975, versus+0.000568 across projected steps.
These are batch measurements, not held or epoch losses; a first-order average
constraint does not ensure nonlinear or per-query preservation.

Held-awning BODY_NEAR native winners increase54→57/64 while native winners
below cutoff increase3→6 and final OR TP falls61→59. Localization absence alone
does not explain the loss. Fixed two-cutoff algebra attributes21 old TP losses
to12 cutoff-only sufficient crossings, eight joint raw/cutoff crossings and one
where either change suffices; the21 held losses divide14/two/five. This is not
a replacement calibration: keeping the old cutoff would abandon the declared
candidate-specific1,256-row rule. No threshold was searched or changed.

Complete candidate/OR, retained OLD_NEG/MZ57/MZ62/MZ64, all-profile/query/role,
native/known-wrong/UNKNOWN and paired-ID results remain in the
[sealed score](../../../../artifacts.local/work/mz66-replay-projection-20260911/score-v1/result.json)
and [event audit](../../../../artifacts.local/work/mz66-replay-projection-20260911/score-v1/paired-events.json).
The [execution record](MZ66_EXECUTION_20260911.md) separates successful numeric
verification from the failed retaining criterion. Dense argmax was not rerun;
native winner labels do not prove causal feature use. This consumed Development
result neither establishes a trained ceiling nor hardware, natural-scene,
Android, clearance or safety performance. It does not rule out other replay
coverage, objectives or representations, and it does not establish catastrophic
forgetting as the sole cause.

Result SHA: `fae5cf791594545b8ec1a8ed5ce37daa3e8038a114080b4b5614f44236c54652`.
Score receipt SHA: `4f970634117493a3aee5dfe2e8a59be4744c19a82cc91564b2df214a39b5bbd9`.
