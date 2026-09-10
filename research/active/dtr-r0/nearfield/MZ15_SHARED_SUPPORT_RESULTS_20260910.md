# MZ15: shared support does not meet the fixed error budget

2026-09-10, EXPLORE, consumed Development. **Neither new model qualifies as an
upgrade.** The proposed SHARED support model adds64 far true output bits on the
existing3000-frame placement replay, but also17 false bits and retrieves only
12/49 clean pole opportunities. Matched QUERY adds37 far true bits and15 false
bits, with49/49 clean and48/49 stress pole retrieval. Both fail the no-added-FP
criterion. Keep MZ5 and the existing MZ9 component unchanged; stop these fits.

## Implemented comparison

[Protocol](MZ15_SHARED_SUPPORT_PROTOCOL_20260910.md) fixed both fits before results.
Frozen64-channel18x32 RGB features feed a small3x3 convolution and local/zone
context, explicit relative/absolute coordinates, and both observed returns and
validity bits. The SHARED10641-parameter readout predicts one support field for
each return, then applies fixed body-query geometry. QUERY10740 uses the same
structure but predicts four query-specific fields. Their common layers and
initial per-query predictions match; only99 parameters differ. This comparison
changes shared factorization and local supervision together, so it does not
isolate their individual contributions.

Each arm gets the same1200 Adam steps and identical batch IDs. The available
training pool is10200 frames: oldTRAIN2500, relationTRAIN5000, distanceTRAIN2500,
and consumed sequence200. Sampling four per cohort in each batch16 produces
19200 frame draws and7562 unique frames per arm:2132/3113/2117/200 respectively.
Thus this is not a complete epoch over every available training frame. The
frozen encoder is unchanged; no additional capture or protected EVAL was used.

Local targets are actual selected-return contributors (SHARED) or actual query
contributors (QUERY). Native supervision is never an inference input. Masking
the auxiliary query loss avoids forcing a positive outside available contributor
coverage. No guarantee of correct localization follows from this masking.

The final composition accepts model-supported additions above an oldDEV-fitted
cutoff and otherwise retains MZ5. All baseline positive scores, including existing
false and premature-near judgments, remain unchanged. Zero added oldDEV FP is
calibration by design, not validation. Cutoffs are frozen before newDEV replay.

## Task effects at those frozen cutoffs

Event order in all arrays: BODY_NEAR, BODY_FAR, HEAD_NEAR, HEAD_FAR.

| Cohort / method | Four outputs all correct | TP per query | FP per query |
| --- | --- | --- | --- |
| OldDEV1000 MZ5 |912 |187 /180 /183 /165 |6 /11 /10 /7 |
| OldDEV SHARED |935 |198 /193 /191 /176 |6 /11 /10 /7 |
| OldDEV QUERY |923 |197 /184 /183 /171 |6 /11 /10 /7 |
| RelationDEV2000 MZ5 |1883 |382 /357 /388 /375 |2 /5 /26 /14 |
| RelationDEV SHARED |1909 |393 /381 /398 /387 |8 /5 /29 /18 |
| RelationDEV QUERY |1888 |393 /363 /389 /382 |8 /5 /26 /19 |
| DistanceDEV1000 MZ5 |920 |0 /0 /489 /446 |0 /0 /25 /9 |
| DistanceDEV SHARED |939 |0 /0 /498 /474 |2 /2 /25 /9 |
| DistanceDEV QUERY |930 |0 /0 /495 /470 |4 /0 /25 /9 |

Across the new3000 replay, SHARED adds94 total TP bits,64 far, and17 FP bits;
QUERY adds55 TP bits,37 far, and15 FP bits. These are output opportunities, not
independent obstacle events. SHARED's17 errors split10 hanging-sign and7 cabinet
bits; QUERY's15 split10 sign and5 cabinet bits. **All added false winners still
have zero actual source at their chosen return/cell**, despite direct local
supervision. The MZ14 failure signature persists in this implemented recipe.

Group aggregation is also retained. On400 relation groups, all-frame exact
groups change325->337 SHARED or324 QUERY; exact sites47->57 or52 out of100.
On500 distance pairs,424->443 or433; exact sites57->62 or58 out of100.
Relation and distance DEV share100 sites, so they do not supply200 independent
site confirmations. The data were consumed previously and remain Development.

## Pole retention and local correspondence

| Model | Clean pole far TP | Stress pole far TP | Wrong correspondence clean / stress |
| --- | --- | --- | --- |
| MZ5 |0/49 |0/49 |not applicable |
| SHARED |12/49 |18/49 |0/49 /0/49 |
| QUERY |49/49 |48/49 |0/49 /0/49 |

The49 opportunities are25 BODY_FAR and24 HEAD_FAR labels from one trained pole
configuration. SHARED recovers no pole HEAD_FAR labels. QUERY meets the pole
regression target but still fails the placement FP budget. Clean sequence200
exact is152 SHARED versus175 QUERY; stress152 versus174, versus143 MZ5. Both
keep sequence FP5/3/19/0 because baseline errors are immutable in this composition.

Wrong-zone RGB keeps geometry and packet inputs fixed. New-placement far gains
fall64->48 for SHARED and37->5 for QUERY; wrong correspondence raises added FP
to57 and12 respectively. This shows correspondence sensitivity, not evidence
that SHARED is the better mechanism. QUERY is much stronger on the trained pole,
while SHARED retrieves more far positives on the placement replay at a slightly
worse error budget. Neither dominates all required outcomes.

SHARED's12->18 pole gain when foreground returns are removed is not temporal
recovery: this is single-frame inference and the removal changes the two-return
conditioning. It does not establish correct use of the deleted foreground signal.
Of the6 far additions that MZ14's source oracle discarded, SHARED recovers4 and
QUERY1. These small consumed case counts cannot establish generalization.

## Verification and corrected diagnostic alignment

New cache10500 (TRAIN7500 + DEV3000) completes PASS in229.15s on RTX5060 Laptop.
All RGB/native hashes and all native truth labels agree with prior receipts.
TRAIN/DEV first16-frame MZ5 and SOURCE margin checks are exact, and two original
packet reconstructions have zero distance error. Cache observations and evaluator
labels are separate. Original old caches remain unchanged.

SHARED and QUERY fits take20.61s and18.06s respectively; the full fit/replay/write
command takes50.34s. These timings exclude cache extraction and are not device
deployment latency. CUDA is used for fitting/inference; metadata audits use CPU.

Independent review identified a stress-diagnostic alignment error: MZ6 moves the
remaining background return from slot1 to slot0, so contributor labels must move
too. Training and TP/FP predictions were unaffected. The evaluator-only repair
replays frozen checkpoints for800 stress/wrong-stress frames, verifies3200 raw
scores exactly, and aligns39 moved zones. It changes20 winner-source flags,
not query-winner flags, best-true scores, cutoffs, TP/FP or admission decisions.
The original script and outputs are preserved; future inference diagnostics use
the corrected alignment. No optimizer step or threshold was repeated.

Focused tests pass3/3 (shared field/matched initialization, invalid coverage and
wrong correspondence, stress slot identity). The stored-array audit passes62400
output bits, cutoff calculation, composition, immutable baseline positives,
initial weights, sampled partitions, family/group/site metrics and admission.

Code: [model](mz15_shared_support.py), [cache](mz15_cache.py),
[training](mz15_train.py), [evaluator alignment](mz15_evaluator.py),
[repair replay](mz15_repair_stress.py), [audit](mz15_audit.py).
Evidence root: `artifacts.local/work/mz15-shared-support-20260910/`.
Use `run-v1/stress-repair-v1/{result.json,predictions.npz,receipt.json}` for
corrected diagnostic statistics, together with `run-v1/{start,receipt,audit}.json`.
Original run-v1 predictions/result remain valid for all task metrics; their
stress winner-source flags are superseded by the repair.

## Decision and limits

The MZ14 diagnostic correctly motivated testing localization, but this compact
shared-support formulation did not learn reliable enough source assignment to
meet the task budget. More coherent structure alone did not produce a retained
upgrade. Neither a generic shared field nor all spatial fusion is ruled out by
one seed/1200-step fit. Conversely, the trained-pole success of QUERY does not
establish independent thin-obstacle transfer.

Stop this exact pair without widening, longer training, cutoff rescue or new
data acquisition. Preserve the cache, matched comparator, failed checkpoints,
and contributor diagnostics as a negative control for a materially different
intervention. No new baseline, temporal restart, or application promotion.
