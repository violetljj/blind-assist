# MZ24: decisive availability supervision halves additions' false alerts

2026-09-10, one completed1200-step matched fit, consumed Development. MZ24
reduces MZ20's8 placement addedFP to4 (MZ23 left7), retains70/75 far additions
and trained pole49/48 of49 clean/stress. The objective has useful scoped effect,
but the no-added-FP criterion still fails. Retain the objective as a COMPONENT;
MZ5 remains baseline and MZ20 the retained ranking challenger. No promotion.

## Controlled change

The [protocol](MZ24_DECISIVE_AVAILABILITY_PROTOCOL_20260910.md) freezes the MZ23
10577-parameter head, seed123 initialization,1200x16 batches, optimizer, inputs,
MZ20 scores/cutoffs and MZ5. Add weight0.25 softplus on the highest unavailable
geometric candidate per frame/query, plus weight0.25 softplus(-availability) at
the teacher's highest-scoring actual query contributor. Keep dense balanced
availability BCE at weight1. Empty bags contribute no target. The two new terms
change together, so this contrast does not isolate their individual effects.

Native knownness/contributors are training/evaluator-only. Negative supervision
means absent native<=4m availability, not CLEAR or negative UNKNOWN occupancy.
Positive selection uses frozen raw MZ20 scores within actual eligible contributors,
never a DEV cutoff. Inference remains availability logit0, restricted MZ20 maxima,
and add-only MZ5 fallback. Every baseline positive stays unchanged.

## Task effect

Exact requires all four BODY/HEAD x near/far outputs correct in a frame. Output
bits are correlated and are not independent obstacle encounters. Order BN/BF/HN/HF.

| Cohort | MZ5 exact | MZ23 exact | MZ24 exact | MZ24 addedTP | MZ24 addedFP |
|---|---:|---:|---:|---|---|
| oldDEV1000 |912|941|939|3/17/10/16|0/0/0/0|
| clean200 |143|176|176|7/30/0/25|0/0/0/0|
| stress200 |143|175|175|7/30/0/24|0/0/0/0|
| relationDEV2000 |1883|1915|1914|3/27/10/11|0/1/0/1|
| distanceDEV1000 |920|946|946|0/0/10/32|1/1/0/0|

Placement exact is2860/3000, versus2861 for MZ23 and2862 for MZ20. Against MZ20,
the candidate loses8 added true bits (1/4/2/1, all relation) and removes4 added
false bits. It loses none of the original MZ5 positives. Against MZ23 it loses
7 true additions and removes3 more false additions. Thus this is an error/coverage
tradeoff, not uniform dominance. The far-retention70>=68, both-placement gain,
baseline-positive retention and trained-pole criteria pass; no-added-FP fails.

The four remaining false additions are relation frame473 HEAD_FAR (oblique rod),
relation1677 BODY_FAR (sign), distance778 BODY_NEAR (sign), and distance939
BODY_FAR (sign). MZ24 removes relation338/1538 BODY_NEAR, relation1488 HEAD_FAR,
and distance819 BODY_FAR. The last was already removed by MZ23. All these locations
were selected from the previously recorded8 MZ20 false additions.

Availability precision rises71.30->74.43% on relation and75.00->78.13% on distance,
while recall falls93.52->83.77% and94.52->85.16%. Task witness retention protects
the pole despite this cell-level recall decline. Neither the availability scores
nor the resulting task outputs establish actual physical sensor reliability.

## Frozen-score diagnostic

Reusing the saved-model tail runner on MZ24, the four remaining false decisive
availability maxima are2.3704,2.0060,0.9598 and0.0593. A descriptive strict common
boundary above2.3704 would remove all8 old additions but retain only32/75 far
additions. Pole remains49/48 at that diagnostic boundary. Score separation is
better than MZ23's14/75 and pole40/38 at its required boundary, but far retention
still fails badly. No new cutoff is adopted; inference stays at logit0.

`tail-v1` binds the MZ24 run receipt and contains its scores. The reused
`mz23_availability_tail.py` keeps a historical MZ23 label in its interpretation
string; this does not change the bound MZ24 input or calculations. These are
post-hoc separability diagnostics, not independently selected candidates.

## Validation and limits

The zero-fit extrema diagnostic evaluates both frozen heads on all7562 unique
TRAIN IDs and disjoint oldDEV1000 plus placementDEV3000 (11562unique inputs).
For each frame/query, the negative statistic is the largest availability score
over eligible known0 cells; the positive statistic is the teacher-selected actual
contributor. These availability bags are not task false alarms and do not use
the MZ20 task cutoff or task-level truth to select negatives.

| Cohort | Negative bags nonnegative: MZ23 -> MZ24 | Positive witness retained: MZ23 -> MZ24 |
|---|---|---|
| TRAIN7562 |85.37% ->29.43% (1855/6303)|98.48% ->92.69%|
| oldDEV1000 |91.24% ->36.86% (303/822)|98.05% ->89.48%|
| relationDEV2000 |86.85% ->36.36% (600/1650)|95.39% ->85.71%|
| distanceDEV1000 |89.94% ->38.28% (369/964)|98.10% ->94.90%|

Residual negative extrema are substantial already on TRAIN, across all four
training sources. MZ24 TRAIN negative median=-0.700, p90=+0.931, max=+5.120,
mean softplus=0.560. Thus this is not a transfer-only failure. Current weighting,
representation and optimization remain plausible causes; these measurements do
not isolate one. A distinct bounded optimization/convergence comparison can test
whether the present model has learned the objective sufficiently before changing
capacity or collecting more of the same data. This completed run stays frozen.

The trained25frame pole subset has zero negative-availability bags and49/49
positive teacher witnesses retained by both heads. Pole retention is necessary
but does not alone test the difficult negative separation. The clean200/pole25
slices overlap TRAIN and are not counted as additional independent inputs.
`extrema-v1` saves per-frame/query scalars, IDs, checkpoint/input hashes and PASS
receipt for independent recount. The diagnostic made zero optimizer updates.

Three focused objective tests pass: hardest unavailable-cell gradient pressure,
protected true witness despite a higher teacher score at an unavailable location,
empty masks, and finite dense/extrema gradients. Independent NumPy scalar losses
at step1/1200 agree within1.34e-7;100352 analytic availability-gradient elements
agree within1.39e-9, including echo ties in the maximum and teacher argmax.
Initialization matches MZ23 exactly and batches/cutoffs match MZ20 byte-for-byte.

Independent audit covers31200 task bits and195686400 geometric candidate bits,
full support masks, group/site metrics and source/query contributor rejection.
All8 old false frames plus25pole frames in both conditions (58frames) reproduce
with independent NumPy masked maxima. Availability difference<=8.94e-6 and task
logit difference<=5.72e-6; support and decision signs match. No implementation
or label-routing defect was found. Full candidate-score inference replay is
limited to those58frames; all saved task decisions/support masks are audited.

CUDA fit time27.40s excludes frozen visual feature extraction and is not device
latency. Artifacts remain under `artifacts.local/work/mz24-decisive-availability-20260910/`.
The fit and audits are complete; no capture, temporal state or service was created.
Consumed synthetic Development and trained pole coverage do not supply independent
source, natural-distribution, physical-device or safety evidence. Baseline errors,
remaining false additions and future temporal/device work remain unresolved.
