# MZ64 geometry learning: useful detection gain with old-source false alerts

The [registered source-adaptation comparison](MZ64_GEOMETRY_LEARNING_20260911.md)
completed both1536-step fits and the independent saved-output score. The full
retaining check fails:14 of19 clauses pass; the five failures are old-cohort
false-alert limits. Retain GEOMETRY as a bounded CHALLENGER, the matched CONTROL
and all frozen comparators. This is a useful data-learning result with costs,
not a replacement of the retained baseline.

On DROP MZ61 HELDOUT_GEOMETRY1024, GEOMETRY has795 TP/38 FP versus matched
CONTROL761/39 and frozen MZ62 CONTROL756/38. These are four-query event counts,
not numbers of correctly classified images. Native-winning BODY_NEAR additions
beyond OLD_NEG are30 versus12 and8; held awning contributes5 versus2 and0.
Against matched CONTROL, GEOMETRY gains35 true events and loses1, adds no false
bits and removes1. Against frozen CONTROL it gains42 and loses3, with no newly
false bits or removed false bits. Thus unchanged net FP does not conceal a false
bit exchange on this primary group.

| DROP group | OLD_NEG TP/FP | Frozen MZ62 CONTROL | Matched CONTROL | GEOMETRY |
| --- | ---: | ---: | ---: | ---: |
| MZ61 held1024 | 740/38 | 756/38 | 761/39 | 795/38 |
| Legacy noncal | 2722/45 | 2732/57 | 2733/55 | 2748/62 |
| MZ48 nonfit | 463/20 | 537/24 | 550/24 | 554/25 |
| MZ55 held640 | 451/79 | 480/88 | 478/83 | 487/86 |

All three old groups retain aggregate TP against both controls. GEOMETRY adds
net7 legacy FP,1 MZ48 FP and3 MZ55-held FP versus the matched control. The MZ55
held FP count is lower than frozen MZ62's88, but not matched CONTROL's83. This
is why the full retaining check fails despite the primary geometry improvement.
Counts for each query, exact frames, misses, gains/losses and false-bit IDs remain
in the complete score, including candidate-only readouts and all old comparators.

The old regression is localized further in saved event IDs. Versus matched
CONTROL, all12 newly false legacy bits are in relation10000, with5 other false
bits removed. The new MZ48 nonfit HEAD_NEAR false alert has an UNKNOWN winner
on a supported BODY_ONLY oblique rod. Three new MZ55 held BODY_NEAR false alerts
are unsupported far shallow awnings at rows406/426/428, with known non-native
winners. Improved new awning recall coexists with these old near/far mistakes;
the result does not justify a global near/far exclusivity rule.

| MZ61 profile | Group | OLD_NEG TP/FP | Frozen CONTROL | Matched CONTROL | GEOMETRY |
| --- | --- | ---: | ---: | ---: | ---: |
| IDEAL | all4096 | 3514/274 | 3535/313 | 3565/348 | 3739/351 |
| IDEAL | held1024 | 865/70 | 871/80 | 882/94 | 921/91 |
| MERGE_CLOSE | all4096 | 3513/118 | 3521/126 | 3551/125 | 3816/119 |
| MERGE_CLOSE | held1024 | 860/31 | 863/32 | 874/33 | 938/31 |
| DROP_CLOSE | all4096 | 2985/154 | 3034/155 | 3068/155 | 3244/154 |
| DROP_CLOSE | held1024 | 740/38 | 756/38 | 761/39 | 795/38 |
| ALL_INVALID | all4096 | 17/16 | 71/16 | 78/16 | 248/16 |
| ALL_INVALID | held1024 | 5/4 | 10/4 | 18/4 | 43/4 |

Training rows are included in the all4096 tables and separated in the full score.
No ALL_INVALID samples were deliberately introduced during fitting; it remains
an evaluation-only zero-range/zero-validity stress. GEOMETRY still misses3848
of4096 true events there. Its higher count does not establish a useful sensorless
backup. IDEAL has substantially more false alerts than OLD_NEG. No profile is
chosen for deployment from these inspected results.

On64 held awning BODY_NEAR-positive rows under DROP, native winners increase
from42 frozen/44 matched to54 GEOMETRY; native winners below the original-rule
cutoff decrease from14/14 to3. The5 GEOMETRY BODY_NEAR additions beyond OLD_NEG
are native; this attribution is not causal proof that the network used that
surface. Local UNKNOWN and wrong winners remain visible in the complete audit.
The30 held native BODY_NEAR additions span awning5, sign10, grille7 and rod8.
Of20 added HEAD_FAR events beyond OLD_NEG,10 have native winners,1 a known
non-native winner and9 UNKNOWN winners; those9 do not establish correct obstacle
localization. DROP support-pair disagreements fall459 to441 overall, but BODY_NEAR
disagreements rise54 to76. This is not support-context invariance.

The changed source contributes beyond another equal-budget fit on MZ55. The
comparison jointly changes source geometry, source size and per-ID multiplicity;
the common warm start already saw MZ55. It does not isolate an individual object
family, prove natural generalization or establish a fully trained ceiling.
MZ61 TRAIN2048 is now consumed for fitting; CAL1024 and HELD1024 were excluded
from fitting and cutoff estimation. Both cutoffs used only the original1256
DEV/MZ48 rows. Source role IDs and sealed source evidence remain unchanged.

[Execution and integrity](MZ64_EXECUTION_20260911.md) records the shared feature
cache, both complete runs and exact old-result preservation. The full score is
`artifacts.local/work/mz64-geometry-learning-20260911/score-v1/result.json`, SHA
`2e22f123768ec8fbbe91c0442338bda24c460a66e353e8289b18973f05742b38`.
Paired event evidence SHA is
`94ec5a957754e854536cdcc9fc46bdbca103e695c73405a8446bb0116678e71e`.
This is consumed controlled Development; no hardware, natural-scene, Android,
clearance or safety promotion follows.
