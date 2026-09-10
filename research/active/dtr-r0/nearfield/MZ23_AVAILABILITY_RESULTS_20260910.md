# MZ23: learned angular availability preserves recovery but leaves false tails

2026-09-10, consumed controlled Development. One1200-step fit completed.
The availability veto retains74/75 MZ20 placement far additions and all trained
pole detections49/48 of49 clean/stress, but removes only1 of8 added false bits.
The no-added-FP criterion fails. Keep MZ5 as baseline and MZ20 as challenger;
retain this exact availability recipe as a negative control, not an upgrade.

## What was tested

The [protocol](MZ23_AVAILABILITY_PROTOCOL_20260910.md) first admits a privileged
known-availability oracle, then one learned head. Oracle masking removes all8
added false bits, retains71/75 far additions and pole49/48. This establishes
diagnostic room for an availability mechanism, not learned inference performance.
Native depth remains evaluator/training-only and missing support remains UNKNOWN.

The10577-parameter head uses local and zone features from the existing28x28
native224ROI feature map, angular coordinates and range/valid packets. It predicts
64x49 query-independent angular availability scores. Labels mean any native
pixel in that angular cell has valid finite axial depth and radial distance<=4m;
they do not mean physical occupancy, sensor reliability, FREE_RAY or CLEAR.
Class-balanced per-frame availability BCE uses the same1200x16 batches as MZ20,
seed123, Adam0.001. MZ20 scores, per-query cutoffs and MZ5 are frozen.

At inference, availability logit>=0 restricts the original geometric candidates.
The output can only remove MZ20 additions; every MZ5-positive bit is preserved.
No native mask is available to the learned inference branch. Scene availability
labels remain unchanged under return-slot stress; contributor diagnostics use
the correct stress-slot alignment separately.

## Paired task effect

Exact means all four BODY/HEAD x near/far outputs match in a frame. TP/FP changes
are output bits, not independent obstacles or sites. Ordering below is BN/BF/HN/HF.

| Cohort | MZ5 exact | MZ20 exact | MZ23 exact | MZ23 added TP | MZ23 added FP |
|---|---:|---:|---:|---|---|
| oldDEV1000 |912|942|941|6/18/10/16|0/0/0/0|
| clean200 |143|176|176|7/30/0/25|0/0/0/0|
| stress200 |143|175|175|7/30/0/24|0/0/0/0|
| relationDEV2000 |1883|1916|1915|4/30/12/12|2/1/0/2|
| distanceDEV1000 |920|946|946|0/0/10/32|1/1/0/0|

Placement exact is2861/3000, versus2862 for MZ20 and2803 for MZ5. One relation
BODY_FAR true addition is lost; one distance BODY_FAR false addition is removed.
The removed false bit did not make its entire frame correct. Far recovery is
74/75, and baseline-positive and trained-pole retention pass. All old baseline
errors remain. Wrong-zone controls preserve baseline bits but still add three
false bits across placements; this shifts both visual chains together and does
not isolate the availability head's correspondence contribution.

| Availability quality | Precision | Recall |
|---|---:|---:|
| oldDEV |72.61%|91.65%|
| relationDEV |71.30%|93.52%|
| distanceDEV |75.00%|94.52%|
| pole-containing clean clip |96.39%|99.71%|

High aggregate availability recall does not certify the handful of maxima that
decide an alert. Actual eligible query-contributor cells rejected on relation are
2920/2778/4266/865, and distance0/0/3608/1384. These correlated cell counts are
not task misses; max pooling often retains another candidate in the same query.

## Why a stricter cutoff does not rescue this head

`mz23_availability_tail.py` replays the frozen models and considers only original
eligible MZ20 candidates already above their unchanged task cutoff. For each
frame/query it records the maximum predicted availability and its exact cell.
All8 original false additions select known-unavailable cells in this diagnostic.
The7 surviving false bits have availability logits1.658..5.507; the removed
one has-0.431. False confidence is not confined to the neighborhood of zero.

A descriptive common-cutoff separability calculation finds that removing all8
would require a strict boundary above5.506674. At that boundary only14/75 far
additions survive, with trained pole40/49 clean and38/49 stress. This is a
post-hoc diagnostic, not a chosen operating point, new candidate or threshold
change. The original MZ23 result remains frozen at availability logit0.

`mz23_boundary_diagnostic.py` reconstructs the native angular-cell geometry and
checks the8 diagnostic winners against existing native depth. Six of the seven
surviving tails are5..6.32 angular cells from the nearest known cell, about
19.2..24.3 native pixels between cell centers and18.9..25.0 pixels to an actual
valid native point. That is roughly2.4..3.0 positions on the28x28 feature map.
Only relation frame1677 is one angular cell away (4.11pixels to actual support).
Thus most residual errors cannot be explained as a single adjacent-cell boundary
ambiguity. Broad contextual activation or insufficient training pressure on
decisive negative locations remain hypotheses; a resolution increase alone is
not selected by this diagnostic. It does not establish a causal feature defect.

The actionable gap is availability precision at task-decisive locations. The
oracle does not prove that RGB/range inputs or the present feature resolution
can recover the required boundary. A successor must change the representation
or training objective and show that the false tail separates from true witnesses;
extending this fit or moving its cutoff is not the next experiment.

## Verification and disposition

The two focused availability tests cover finite invalid-packet behavior and
monotone restriction with explicit missing support. Independent audit checks
31200 task bits,195686400 geometric candidate bits, seed123 initialization,
byte-identical MZ20 batches/cutoffs, all baseline-positive retention, group/site
metrics and availability precision/recall. CUDA replay of all8 prior false frames
and25pole frames in each condition covers58frames; independent NumPy masked
maxima reproduce saved support and decisions. Maximum availability difference
is1.36e-5 and task-logit difference5.72e-6; decision signs match. Full candidate
score replay is limited to those58frames, while all saved task decisions and
geometry/support masks are audited. No implementation defect was identified.

Fit time15.19s on CUDA excludes frozen feature extraction and is not end-to-end
device latency. The fit and diagnostics exited; no service/capture was created.
Durable artifacts live under `artifacts.local/work/mz23-availability-20260910/`:
`oracle-v1`, `run-v1` (including audit), `tail-v1`, and `boundary-v1`. Source and output hashes,
initial/final weights, batches, predictions and negative results are retained.
All evidence is consumed Development, including the trained pole configuration.
No default-App, fresh-source, temporal, physical-device or safety promotion.
