# MZ51: broader negative coverage lowers false alerts with a recall tradeoff

The [matched experiment](MZ51_TRAINING_COVERAGE_20260911.md) completes two
600-step continuations from the exact completed MZ50 checkpoint. OLD_NEG
reduces GATED's old noncalibration added false events from19 to5, versus8
for equal-budget NEW_NEG. Both continuations reach442 new nonfit true events,
below fixed MZ50's453. OLD_NEG has20 false events versus21 for MZ50 and23 for
NEW_NEG. The registered primary gate fails; this is a useful tradeoff, not a
replacement or evidence that more training solves the current bottleneck.

The two arms share each batch of8 new fit frames and the same native/query
loss. Each adds8 query-negative presentations per step, drawn respectively
from the new1,280 fit frames or original7,562 TRAIN IDs. Only the designated
negative query is penalized; other event queries and unknown cells are not
relabelled. The old source is disjoint from all three old DEV cohorts.
Negative coverage changes while architecture, step budget and observation
profiles stay fixed. Shared sampling uses1,250 distinct new frames; the extra
negative presentations use1,242 NEW_NEG or3,533 OLD_NEG frames, with1,200
presentations per query in each arm.

The new nonfit group remains640 held-out-site plus384 withheld-family frames,
with992 positive event bits. Neither enters fitting or the fixed calibration
rule, which uses original1,000 DEV plus256 rich calibration frames under
DROP_CLOSE. All sites are consumed controlled Development sources.

| Nonfit profile | Fixed MZ50 GATED TP / FP / FN | NEW_NEG GATED | OLD_NEG GATED |
|---|---:|---:|---:|
| IDEAL | 677 / 34 / 315 | 668 / 37 / 324 | 654 / 34 / 338 |
| MERGE_CLOSE | 672 / 19 / 320 | 666 / 21 / 326 | 668 / 18 / 324 |
| DROP_CLOSE | 453 / 21 / 539 | 442 / 23 / 550 | 442 / 20 / 550 |

| Old noncalibration added FP over MZ37 | Fixed MZ50 GATED | NEW_NEG GATED | OLD_NEG GATED |
|---|---:|---:|---:|
| IDEAL | 33 | 20 | 9 |
| MERGE_CLOSE | 33 | 18 | 10 |
| DROP_CLOSE | 19 | 8 | 5 |

The old aggregate comprises relation2,000, distance1,000, older44 rich frames
and400 MZ36 attempts. Under DROP_CLOSE its absolute TP/FP/FN counts are
MZ37=2710/29/238, fixed MZ50 GATED=2723/48/225, NEW_NEG GATED=2723/37/225,
OLD_NEG GATED=2720/34/228. OLD_NEG thus removes14 false events relative to
MZ50 while losing3 correct additions there. It retains all original MZ37
positives. Its new nonfit result gains8 and loses19 true events against fixed
MZ50, net−11, while removing1 false event. Against matched NEW_NEG it gains10
and loses10 true events, removing3 false events: equal totals conceal swaps.

OPEN improves with old-negative exposure, but has its own false-alert cost:

| Nonfit profile | Fixed MZ50 OPEN TP / FP / FN | NEW_NEG OPEN | OLD_NEG OPEN |
|---|---:|---:|---:|
| IDEAL | 566 / 31 / 426 | 568 / 31 / 424 | 588 / 33 / 404 |
| MERGE_CLOSE | 621 / 20 / 371 | 622 / 18 / 370 | 631 / 21 / 361 |
| DROP_CLOSE | 386 / 20 / 606 | 383 / 20 / 609 | 405 / 20 / 587 |

OLD_NEG OPEN adds23 nonfit DROP true events over MZ37, distributed
`[1,0,20,2]` across BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR. All23 have saved
native winning-cell evidence;9 have no GATED candidate. GATED adds60, with42
native winning-cell matches. These source-bound saved flags are not another
independent native-depth derivation. Correct frame decisions without a matching
winning cell remain attribution gaps. OLD_NEG OPEN adds11 old false events,
versus2 for fixed OPEN and4 for NEW_NEG OPEN. Its gain is not free transfer.
The separately registered MZ53 composition test checks the complementary
readouts; it does not change this experiment's gate or cutoffs.

A separate CPU coverage audit locates a more basic limitation. All352 MZ48
positive events without a45-degree native witness are BODY_NEAR:352/512,
or68.75% of that query. The80 predeclared native samples contain10 such gaps;
all10 also lie below the224-pixel crop, with485–4,018 corresponding pixels
in the original640×360 image. Its narrow892-pixel border beyond the exact
45-degree field rescues none. Three original RGB examples were actually
viewed. Bottom-edge surfaces are visible but truncated, so this proves a
learning opportunity, not that RGB already recovers metric occupancy.
The remaining2,480 native files were not read for this audit; their exact224
coverage is not inferred from the sample. Full-source45-degree statistics
come from the bound compact auxiliary. Removing echo gating cannot enlarge
the spatial domain. MZ52 therefore derives full-frame supervision from the
existing native files, keeping ToF's original field and missing status.

The CUDA pipeline took74.437 seconds, including22.721 seconds for required
new detail features,7.192 seconds for NEW_NEG and7.433 seconds for OLD_NEG
fitting. It decoded new RGB once, reused the existing2.32GB old detail mmap,
kept513.80MB new features only in RAM, and performed zero new baseline
inferences. The16-frame initial replay is exact for OPEN/GATED raw values
and support; both arms start with exactly the same checkpoint weights.
CPU scoring took0.869 seconds, checking921,888 known scalar bits,262 exact
original MZ50 arrays, the seed151 schedule and four fixed cutoffs. All400
MZ36 attempts/80UNKNOWN remain. A focused negative-loss check confirms only
the selected query receives gradients. No dense cache or threshold search
was added; completed model/scorer processes exited.

Retain OLD_NEG as a challenger with the measured recall tradeoff, NEW_NEG as
its matched continued-learning comparator, and fixed MZ50/MZ37 unchanged.
The observation proxies remain fixed limited-sensitivity tests; they do not
establish calibrated VL53L8CX, natural-scene or deployment performance.
Evidence is under`artifacts.local/work/mz51-training-coverage-20260911/`
in`run-v1/`, `score-v1/` and`coverage-audit/`. Their receipts preserve exact
inputs, outputs, schedules and source identities.
