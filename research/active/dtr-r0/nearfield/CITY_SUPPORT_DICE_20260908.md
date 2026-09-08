# Masked support Dice: one fixed-budget loss comparison

EXPLORE continuation after the matched64 coverage pilot. Hypothesis: the
unchanged globally balanced support BCE permits imprecise local activation;
an explicit per-image/per-head overlap term can improve support while retaining
near classification. The previous added64 TRAIN fits every near bit but HEAD
IoU is0.205511 with15/32 peak hits. This is not evidence that all localization
errors are spill, or that coverage is already sufficient.

A is the completed coverage64 B-step2000 checkpoint, SHA256
`d60c871fd0a292bfd5b30c625e5a4189632a7c42576fd6d951c71e40e6d8b7bc`.
Reuse its predictions/metrics, not another baseline fit. B starts from original
G13 seed17, not A. Keep1198 TRAIN (old750+relational384+added64), architecture,
preprocessing, original initialization, full-parameter AdamW1e-5/wd1e-4,
frozen BN running buffers, batch32 and2000 steps. Reuse the exact saved1198-frame
training_indices.npy, not merely the same seed. No capture, augmentation,
threshold search on EVAL, support-resolution change or Dice-weight sweep.

The only optimized change is:

`near_BCE + 0.25 * (existing_global_balanced_support_BCE + masked_soft_Dice)`.

The Dice coefficient is fixed at1 inside the support objective (absolute0.25),
chosen before outcomes. For each image/head map with at least one known positive,
use `1 - (2*sum(p*t)+1e-6)/(sum(p)+sum(t)+1e-6)`, with both sums restricted to
known pixels. Average eligible maps equally. UNKNOWN is neither foreground nor
background and receives zero Dice gradient. Maps without positive support have
zero Dice contribution; their known negatives remain under unchanged BCE.
This is soft foreground Dice on body-query support, not whole-object masking,
class-weighted Generalised Dice, or a new architecture. The user's cited
[Sudre et al. paper](https://arxiv.org/abs/1707.03237) supplies the general overlap
loss motivation, not an expected result or an exact reproduction of its loss.

Focused synthetic checks verify formula, per-map averaging, false-positive
gradient, UNKNOWN invariance/zero gradient, empty-positive behavior and spill
versus missed-support behavior. These are implementation checks, not extra fits.

Only final B-step2000 selects thresholds on the original DEV128, unchanged
FPR<=10% selector and tie rules. A retains its stored DEV cutoffs. Existing
EVAL64 is now consumed Development, not fresh confirmation. Compare saved A/B
TRAIN partitions, DEV and EVAL: near confusion, AUC/AP, group correctness,
support IoU/peak (UNKNOWN peaks separate), known positive pixel TP/FN/recall,
false-positive known pixels and negative-frame activation. Include positive-map
and negative-map false pixels separately so shrinking support cannot hide lost
recall. Use fixed support threshold0.5, no map-threshold tuning.

If TRAIN and Development localization improve with reduced false activation
and retained near utility, retain a loss challenger before any new coverage run.
If only TRAIN improves, distinguish fitting capacity from regional transfer.
If localization does not improve or gains reflect erased positives, record the
failure without sweeping weights or extending steps. One loss comparison cannot
exclude other losses, decoder limitations, optimization or data coverage.
Stop at one2000-step fit and its listed assessment; keep any mechanical failure
receipt and recover evaluation only from a completed checkpoint. Preserve inputs,
weights and outputs; release only this task's worker processes/temporary files.

## Outcome: training overlap improves, transfer utility does not

Exactly one fit completed. The baseline A below is coverage64, not the earlier
1134-frame F2000. B changes only the fixed loss recipe. At support threshold0.5,
new TRAIN HEAD keeps267/268 known positive pixels while removing much spill:
positive-map false pixels1035 to344, empty-map false pixels716 to517. TRAIN
localization improvement is real rather than wholesale deletion of positives.
However, the same intervention loses substantially more true support on DEV
and consumed EVAL. The initial retention condition is not satisfied.

| Partition / head | Positive-map IoU A→B | Known-positive pixel recall A→B | FP on positive maps A→B | FP on empty maps A→B | Peak hits A→B |
| --- | ---: | ---: | ---: | ---: | ---: |
| old750 BODY | 0.2394→0.5362 | 99.94%→98.08% | 5322→1498 | 2337→3885 | 102→117 |
| old750 HEAD | 0.1848→0.4818 | 100.00%→100.00% | 369→94 | 2732→4660 | 5→5 |
| relational384 BODY | 0.2475→0.5693 | 100.00%→98.46% | 7185→1756 | 2523→2488 | 127→139 |
| relational384 HEAD | 0.2278→0.5940 | 100.00%→99.43% | 5080→1039 | 2492→2580 | 80→75 |
| added64 BODY | 0.1950→0.4223 | 100.00%→99.21% | 1577→568 | 599→505 | 17→18 |
| added64 HEAD | 0.2055→0.4421 | 100.00%→99.63% | 1035→344 | 716→517 | 15→19 |
| dev BODY | 0.2143→0.2385 | 71.54%→46.76% | 1698→387 | 588→329 | 31→32 |
| dev HEAD | 0.1852→0.2309 | 86.28%→55.21% | 2185→584 | 1821→569 | 22→27 |
| eval BODY | 0.1957→0.2500 | 63.73%→55.67% | 775→397 | 594→450 | 13→21 |
| eval HEAD | 0.1852→0.1930 | 55.94%→36.71% | 603→162 | 570→159 | 12→15 |

Recall is micro TP/(TP+FN) over known positive support pixels; IoU is the
mean over positive maps, not pooled IoU. Empty maps have no known positive
support. Unknown pixels never become negatives. Full TP/FP/FN, macro recall
and UNKNOWN peak/mask accounting are saved in pixel-assessment.json.

HEAD EVAL TP pixels fall160 to105 of286 (FN126 to181); total known false
pixels fall1173 to321. Thus its small IoU gain0.18517 to0.19299 coexists with
positive support recall55.94% to36.71%. DEV HEAD similarly falls522 to334
of605 positive pixels (86.28% to55.21%), despite IoU0.18521 to0.23089.
These results match the pre-stated caution about narrowing masks while losing
legitimate support. They do not prove a particular decoder or background cause.

HEAD UNKNOWN peak misses A→B: old750: 5→9, relational384: 87→106, added64: 4→5,
DEV: 3→6, EVAL: 6→4. In particular, higher TRAIN IoU does not imply that the
global peak improves everywhere; relational HEAD hits80→75 while UNKNOWN
peaks rise. UNKNOWN remains outside both optimized support losses.

## Near classification under unchanged DEV selection

A retains cutoffs BODY0.6730185151/HEAD0.9902606606; B selects
BODY0.6003856659/HEAD0.9448509812 on original DEV128 only. No EVAL cutoff
is selected or support threshold adjusted. Both models fit all2396 TRAIN
near bits at0.5; B also fits every TRAIN bit at its DEV cutoffs.

| Metric | A: BCE baseline | B: BCE + Dice |
| --- | ---: | ---: |
| DEV BODY TP / FP (64 positive /64 negative) | 58 /6 | 46 /5 |
| DEV HEAD TP / FP (64 positive /64 negative) | 54 /6 | 42 /5 |
| DEV all-four/all-head group correct | 12/32 | 4/32 |
| EVAL BODY TP / FP (32 positive /32 negative) | 16 /1 | 18 /3 |
| EVAL HEAD TP / FP (32 positive /32 negative) | 12 /1 | 11 /3 |
| EVAL BODY AUC / AP | 0.753906 /0.810054 | 0.819336 /0.858536 |
| EVAL HEAD AUC / AP | 0.708984 /0.748386 | 0.674805 /0.711763 |
| EVAL BODY within-quartet strict order | 62/64 | 62/64 |
| EVAL HEAD within-quartet strict order | 51/64 | 46/64 |
| EVAL all-four/all-head group correct | 3/16 | 2/16 |

No pair ties occur. EVAL HEAD recall37.5%→34.375%, FPR3.125%→9.375%;
BODY recall50%→56.25% at the same FPR increase. These32-positive/32-negative
counts and64 correlated pairs arise from16 quartets of one same-world region,
not independent real-world trials. The assessment is consumed Development.

Empty-map HEAD activation improves on DEV6.113%→1.910%, EVAL4.114%→1.148%,
but worsens on old750 (2732→4660 false pixels); it is not uniformly suppressed.
Dice applies only to positive-support maps, so this recipe does not directly
add an empty-map loss. Positive/negative response changes arise jointly through
the shared learned representation and existing BCE.

## Disposition and execution

Retain the fixed Dice recipe as a NEGATIVE_CONTROL for this attempted transfer
repair; preserve its evidence that training overlap can be improved without
new data/architecture. Keep the existing G13-D/coverage64 baseline unchanged.
The experiment does not justify retaining Dice as the default, sweeping its
weight, extending2000 steps, or ruling out all overlap losses. The added term
also changes total support-gradient weighting; this one recipe comparison does
not isolate overlap shape from every optimization-scale effect.

The training-localization limitation responds to loss supervision, but better
TRAIN overlap does not transfer into reliable HEAD evidence or decision utility.
Any next intervention needs to preserve cross-region positive support recall,
not merely improve IoU or suppress negatives. No further fit was launched.

CPU/CUDA focused loss checks pass: formula, per-map mean, positive eligibility,
known-negative gradient, UNKNOWN zero gradient/invariance, empty maps and spill
versus miss ordering. Fixed common source, initial weight, data manifests,
cached A predictions and exact2000x32 schedule hashes pass. BN running buffers
are unchanged. Added64 exposure remains3364 draws,34..66 per frame.

Root independently recomputed EVAL confusion/AUC/IoU/peak/group/pair metrics
and all five partitions' support TP/FP/FN accounting from saved predictions.
Root viewed the two fixed-unit HEAD contact pages; no outcome-based example
selection was used. The unchanged fixed_losses helper reports original BCE-only
loss terms, explicitly excluding Dice; it is not B's full optimized objective.

Worker RTX3060 Laptop/Torch2.9.1+cu128: fit291.83s, script298.53s, job301.71s.
One new fit, no evaluation recovery or extra training.24 files/26,928,527bytes
were returned with SHA verification, including weights and all predictions.
Task-owned model processes and scheduled tasks are released and transfer
archives removed; native caches and worker optimizer/RNG remain durable.

Evidence: artifacts.local/work/city-support-dice-20260908/model-run-v1/.
Result SHA256:
`ad8ff31b4d938cda0c3cb658e7402b73c8763a7cbcf72fd051cfb4325adb2a0a`.
Checkpoint SHA256:
`4576711a13b047b21bef97a77ecb4d8349ec9af7046f0c7e60fac1a6ff2b7b64`.
Executed loss SHA256:
`8f35545db58f9e93dd6305ec652836fe8360577202c20f492e71613608284c10`.
Executed runner SHA256:
`d0e99da947ac375facf4efd800713d95c8fb4ea7fd33a7a08bd055f4e241b929`.
Pre-outcome protocol SHA256:
`ccfe1423df2cec4ee990663303bdec1e36068a0c413b5dda8a194ee51599d7fa`.
