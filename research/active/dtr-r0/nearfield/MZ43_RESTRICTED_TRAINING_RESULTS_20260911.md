# MZ43: matched restricted training improves missing-return decisions

2026-09-11 EXPLORE. [Fixed brief](MZ43_RESTRICTED_TRAINING_20260911.md).
Changing only the training observation improves the restricted-input tradeoff
without extra model capacity. Retain the mixed-trained ToF/ensemble as a
CHALLENGER; the strict no-regression criterion is not met, so the original MZ5
baseline remains. This is a gain with a measured cost, not a universal replacement.

Both ideal TOF_ONLY/FUSION reproductions match all old final model tensors and
all 300 loss values exactly. Each mixed fit has the same initialization, sample
schedule, architecture and 300 steps. Its 38,400 TRAIN exposures are split
equally among ideal, merged-close and missing-close observations. No current
scene labels, condition ID, dense depth or object identity enter the predictor.
All four fits and cached scoring took 8.11 seconds on CUDA; no backbone ran.

## Fixed equal-logit ensemble on MZ36

Each row uses the same 380 admitted frames from 400 attempts. All 80 UNKNOWN
event bits remain excluded from accuracy and explicit in coverage.

| Observation | Ideal-trained FP / FN / exact | Mixed-trained FP / FN / exact |
|---|---:|---:|
| Ideal comparator | 6 / 10 / 365 | 4 / 8 / 368 |
| Close returns merged | 7 / 14 / 362 | 7 / 11 / 365 |
| Close returns missing | 9 / 21 / 354 | 7 / 14 / 362 |

Across the two restricted conditions, mixed training removes 10 net misses and
two net false alerts. In this cohort, every previous true positive survives and
no new false positive is introduced in any of the three conditions. The two
condition rows reuse the same scenes and are not 760 independent observations.
This separates one training mismatch from fixed-model information loss; it does
not establish that all missing observations can be reconstructed.

The more elaborate frozen MZ28/MZ37 have different training and decision
histories. Under missing returns their FP/FN are 9/10 and 2/18 respectively,
versus mixed ensemble 7/14. There is no demonstrated dominance over all retained
comparators. Their full per-query results remain alongside the matched pairs.

## Existing consumed EVAL data and joint fusion

| Observation, 1,500 frames each | Ideal ensemble FP / FN / exact | Mixed ensemble FP / FN / exact |
|---|---:|---:|
| Ideal comparator | 50 / 114 / 1378 | 52 / 104 / 1381 |
| Close returns merged | 41 / 146 / 1357 | 46 / 97 / 1389 |
| Close returns missing | 53 / 135 / 1352 | 46 / 100 / 1383 |

Ideal-input false-alert totals rise by two (BODY_NEAR +1, HEAD_FAR +1), while
misses fall by ten. Merge-only false alerts rise by five. Thus robustness gain
does not mean every condition improves every error count. No cutoff was changed
to conceal these costs; the recorded strict criterion remains false.

Joint FUSION also benefits: on MZ36 the restricted pooled FP/FN fall by 6/16.
Its missing-return FP/FN improve from 22/21 to 17/11. However ideal MZ36 false
alerts rise from 14 to 16, and ideal old EVAL from 115 to 116. It remains an
additional matched result, not a selected winner or evidence of free robustness.

## Reusable artifact and checks

The compact ensemble contains the unchanged RGB head plus the mixed-trained ToF
head. Only first-layer columns whose inputs were always zero are removed:
132,872 parameters, 534,669 checkpoint bytes. Load it with
`CompactEnsemble.from_checkpoint(path)` from [the existing readout](mz5_ensemble_readout.py).
Inputs remain frozen visual `[B,772]` and ranges/4 plus validity `[B,256]`;
four logits use threshold zero. Original B alerts remain independent.

The independent audit reconstructs 640,000 TRAIN packet slots, recounts 463,200
task bits, verifies TRAIN-only exposure and all UNKNOWN/alert preservation, and
compares 16,140 compact readouts. Every decision is identical; maximum arithmetic
logit difference is 2.87e-6. Cached head CUDA compute totaled 0.216 seconds over
these validation batches; this excludes visual inference, transfer, I/O and
device integration and is not wearable latency.

Evidence: `artifacts.local/work/mz43-restricted-training-20260911/` contains
`run-v1/{receipt.json,result.json,predictions.npz}`, the four final fits and
unchanged schedules, plus `compact-v1/{compact.pt,receipt.json}`. The source
hashes bind the executed code and immutable brief. The run-base revision is
recorded separately from the later delivery commit. No source data was deleted.

Retain the fixed-mixture checkpoint and original ideal comparator. The next
decision needs richer object/distance evidence and observation-quality handling
that can explain the remaining tradeoff. New MZ42/MZ44 objects were not fitted
or used to select this candidate. These geometric restrictions are sensitivity
proxies, not calibrated VL53L8CX responses or real-world/safety evidence.
