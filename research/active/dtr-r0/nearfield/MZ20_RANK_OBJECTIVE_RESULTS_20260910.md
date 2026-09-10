# MZ20: cross-frame separation restores pole recall, residual false alerts remain

2026-09-10 EXPLORE, one completed1200-step fit, consumed Development. The new
BODY ranking objective restores pole BODY detection to25/25 in both clean and
stress, versus MZ18's4/25 and1/25. It also improves BODY local AP over MZ16.
However the candidate still adds8 placement false output bits. Retain the
ranking objective/checkpoint as a component/challenger; MZ5 remains baseline.

The [protocol](MZ20_RANK_OBJECTIVE_PROTOCOL_20260910.md) fixes original MZ16
native224 ROI features, initialization,10740 parameters,1200x16 batches and
oldDEV cutoff procedure. Positive BODY bags reward true witnesses. Cross-frame
ranking compares these to original task-negative inference maxima and known
wrong candidates in positive bags. Unlike MZ18, task-negative supervision still
reaches maxima outside the known-local mask. Local UNKNOWN labels are unchanged.
The comparison changes ranking and negative-bag coverage together; it does not
attribute the gain uniquely to one of those two coupled corrections.

| Cohort | MZ5 exact | MZ20 exact | Added TP BN/BF/HN/HF | Added FP BN/BF/HN/HF |
|---|---:|---:|---|---|
| oldDEV1000 |912|942|6/19/10/16|0/0/0/0|
| clean200 |143|176|7/30/0/25|0/0/0/0|
| stress200 |143|175|7/30/0/24|0/0/0/0|
| relationDEV2000 |1883|1916|4/31/12/12|2/1/0/2|
| distanceDEV1000 |920|946|0/0/10/32|1/2/0/0|

Placement exact rises2803->2862/3000, with75 added far TPs and8 addedFP.
For comparison MZ16 gives80/16 and2858 exact; MZ18 gives70/7 and2857 exact.
MZ20 has4 more exact than MZ16,8 fewer addedFP, and5 fewer recovered farTP.
It is a tradeoff, not uniform dominance. All8 false winners lack actual selected
return source at the chosen cell. All existing MZ5 positives, including its
premature near false alerts, remain because this experiment composes add-only.

| Quantity | MZ16 | MZ18 | MZ20 |
|---|---:|---:|---:|
| Relation BODY_NEAR local AP |0.7078|0.8375|0.8015|
| Relation BODY_FAR local AP |0.7032|0.8281|0.7848|
| Pole clean BODY/HEAD detections |25/24|4/24|25/24|
| Pole stress BODY/HEAD detections |25/23|1/23|25/23|

MZ20 retains part of MZ18's localization benefit, not its full AP. Local AP is
conditional on known eligible candidates, not whole-field localization. Pole
counts are25 BODY and24 HEAD opportunities from one trained25-frame configuration.
Wrong-zone inputs remove all pole detections; placement wrong-zone controls
still add1 true and3 false bits across both cohorts. Correspondence sensitivity
does not establish correct winning location for every correct task output.

The fixed no-added-FP criterion fails; both placement far gains, baseline-near
retention, and pole48/49 clean/stress criteria pass. OldDEV calibration is not a
transfer guarantee. A later mechanism must discriminate the remaining unsupported
additions and address the old baseline's errors separately from add-only recall.

Validation: four focused gradient/UNKNOWN tests pass. Independent scalar replay
checks31200 task bits and662031 local candidate bits, exact batches and initial
state, frozen cutoffs, sklearn AP and family/group/site metrics. The initial
test invocation used repository-root import resolution and failed before tests;
running from the owning nearfield directory passes. No fit was repeated.
Actual CUDA fit time24.54s; this excludes encoder extraction and is not deployment
latency. The process exited and all durable outputs remain at
`artifacts.local/work/mz20-rank-objective-20260910/run-v1/` (result, audit, receipt,
checkpoints, initial state, batches, predictions and local samples).

All inputs are consumed synthetic/controlled Development; no fresh confirmation,
hardware validation, temporal upgrade or default-App promotion follows.
