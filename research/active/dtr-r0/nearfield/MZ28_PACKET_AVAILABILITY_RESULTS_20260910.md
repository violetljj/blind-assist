# MZ28: fixed packet-coordinate support improves the controlled tradeoff

2026-09-10, consumed Development, no parameter updates. A fixed TRAIN-bank
availability predictor retains72/75 MZ20 placement far additions with2 addedFP,
versus MZ26's71/75 with3 addedFP. Placement exact rises2860->2864 of3000, while
trained pole49/48 of49 and every MZ5 positive judgment survive. Zero-added-FP
still fails. Retain packet-coordinate availability as a controlled COMPONENT;
MZ5 baseline and MZ20 ranking challenger remain unchanged. No App promotion.

## Implemented method and source identity

The [protocol](MZ28_PACKET_AVAILABILITY_PROTOCOL_20260910.md) fixes one comparator.
The immutable bank uses7362 distinct MZ20 TRAIN frames after excluding200 sequence
frames. It stores each zone's float32 four-vector (two cleaned ranges/4 and two
good bits) and49 native availability labels. The375 bank sites are disjoint from
oldDEV, relationDEV and distanceDEV; this is verified source metadata, not a
query-time input. These repeatedly consumed Development sets are not fresh tests.

At each physical zone, current packet4 is compared in float64 Euclidean distance
to all bank packets. Five nearest references, ties by ascending globalID, vote
known at every angular cell. Votes>=3 enables that cell. The forward interface
takes only current ranges/validity; no current image descriptor, native label,
query truth, family, site or group. Fixed TRAIN labels are supervised reference
data, not current-frame privileged truth. Known0 means absent in-domain native
support, not occupancy-negative or CLEAR. Invalid packets have empty geometric
support regardless of votes. Unanimity is not calibrated confidence.

The availability mask restricts unchanged frozen MZ20 candidate geometry/scores.
Its original cutoffs and MZ5 add-only fallback are preserved. Wrong visual-zone
controls change only the MZ20 visual correspondence; packet availability is exact
under that control. No k/metric/cutoff sweep or alternative fusion arm was run.

## Task results

All vectors use BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR order. The75 denominator is
MZ20's added placement farTP opportunities, not all positive obstacles.

| Candidate | Added farTP /75 | Added FP | Exact frames /3000 | Trained pole clean/stress /49 |
|---|---:|---:|---:|---:|
| MZ5 baseline | 0 | 0 | 2803 | historical baseline |
| MZ20 | 75 | 8 | 2862 | 49/48 |
| MZ26 step4800 | 71 | 3 | 2860 | 49/48 |
| MZ28 packet | 72 | 2 | 2864 | 49/48 |

RelationDEV has1916/2000 exact, addedTP[3,29,11,12], addedFP[0,1,0,1].
DistanceDEV has948/1000 exact, addedTP[0,0,10,31], zero addedFP. Relative to MZ26,
placements gain6 true bits and lose4 (net+2), remove2 false bits and add1. The
remaining false events are relation1488 HEAD_FAR (cabinet) and1677 BODY_FAR
(hanging sign). The cabinet error reappears; the former oblique-rod HEAD_FAR and
distance hanging-sign BODY_NEAR errors are removed. No uniform dominance claim.
Total placement false bits are83:81 baseline errors survive the add-only design.

OldDEV exact940/1000 versus MZ26's939; it gains3 true bits and loses1, without
addedFP. Clean/stress200 exact176/175 remain unchanged. Wrong-zone placement
controls have zero addedFP; relation recovers one true near bit and distance
recovers none. These controls do not establish phone, temporal or natural-scene
performance. The bank excludes sequence frames, but the MZ20 scorer already
trained on the pole configuration, so pole evidence remains trained Development.

## Coverage and actual witness losses

MZ27's both-class reference requirement served descriptor comparison only. This
method supplies votes at every cell, without claiming its previously unscored
anchors have become verified successes. Geometric support and missing observations
retain their separate meanings. Per-cell vote histograms and nearest-distance
quantiles are saved for all cohorts; no confidence threshold is selected.

| Cohort | Actual teacher witnesses available | Retained by packet availability |
|---|---|---|
| oldDEV | [170,200,200,200] | [167,192,200,194] |
| clean/stress | [9,36,9,38] | [9,35,5,35] |
| relationDEV | [339,400,400,400] | [293,314,352,324] |
| distanceDEV | [0,0,500,500] | [0,0,500,500] |

A witness is the highest frozen MZ20 score among eligible actual query
contributors, independent of the task cutoff. These counts are not taskTP counts.
All distance actual eligible contributors survive, yet one formerly recovered
far bit is lost. Therefore that lost alarm is not explained by deletion of its
actual contributor support alone; high-scoring noncontributor hypotheses can
drive a correct task label. This limits simplistic recall-only interpretation.
Relation still loses many teacher witnesses and retains two unsupported alerts.

## Verification and evaluator correction

Independent audit passes31200 task bits,195686400 geometric candidate bits,
24460800 vote sums, all saved neighbor distances, wrong-visual invariance, group
metrics, MZ26 change counters and bank provenance. A separate NumPy full-bank
lexicographic search matches all58 selected frames x64 zones. No new inference
is used by this auditor.

Review found a missing explicit stress echo-slot label alignment in the runner's
diagnostic-only witness calculation. Preserve the original run bytes and source.
`mz28_stress_witness_correction.py` applies the existing evaluator alignment to
the200 stress frames and recomputes only attribution with frozen MZ20 and saved
votes. It observes39 moved echo zones but zero changed elements in all five saved
witness/contributor arrays. Independent reconstruction using aligned labels,
geometry and frozen local scores confirms this. The omission is latent for this
cohort: no reported task or witness number changes. For reproduction, run the
alignment check after the evaluator; `correction-v1` records the explicit result.

## Cost, disposition and next question

The bank occupies30,684,816 uncompressed array bytes; CUDA model buffers occupy
38,223,504 bytes with float64 packet storage. Learned parameter count is0, but
reference storage and search are real model costs. Synchronized retrieval takes
2.859s for4400 frames (~0.650ms/frame amortized in batches), frozen scoring1.477s;
complete cached-feature run14.27s. These exclude backbone/camera acquisition and
do not establish phone latency or interactive single-frame performance.

Retain this packet-coordinate support component because of measured task gains,
with its remaining FP/witness losses explicit. Learned and packet availability
make complementary errors; a successor must evaluate any joint decision against
their actual lost witnesses and false events, not infer success from selected
anchor labels. The fixed MZ28 comparison is complete; no extra arm or tuning is
appended. Baseline false alerts, independent-source and device/temporal capability
remain unresolved parts of the broad goal.

Durable receipts/arrays are under
`artifacts.local/work/mz28-packet-availability-20260910/run-v1/` and `correction-v1/`.
Run, alignment check and independent audit PASS and exit; no fit, service, device
session or persistent allocation remains. Broad obstacle-improvement goal active.
