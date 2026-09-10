# MZ10: availability fallback restores recall but accumulates branch errors

2026-09-10 EXPLORE, zero training and zero threshold changes. **Do not retain
the fixed availability composition.** It restores old DEV near recall but adds
30 false-positive bits while preserving all34 existing MZ5 false positives.
DEV exact912->900 and FP6/11/10/7->11/20/19/14 fail the predeclared budget.
No new-placement confirmation capture was warranted; MZ5 remains the baseline.

## Rule and evidence scope

[Protocol](MZ10_AVAILABILITY_PROTOCOL_20260910.md),
[replay/composition](mz10_availability.py), [scalar audit](mz10_audit.py).
For each of the four queries, observed range/validity and the unchanged fixed
ray/query geometry determine whether SOURCE_RGB has any eligible candidate.
If present, use its existing MZ9 thresholded score; otherwise use frozen MZ5's
original zero-threshold score. No confidence, source-contributor label, evaluator
truth or correctness chooses the branch. Support is availability, not reliability.
MZ9's adapted MZ5 is scored unchanged as the simple alternative.

Old1000 DEV and MZ6's consumed200 training/regression frames are replayed. Neither
is fresh confirmation. MZ5/SOURCE/ADAPT checkpoint and cache hashes are checked;
DEV numerical replay matches the original MZ9 scoring batch size32 and results.
Independent original alert outputs remain untouched. No new calibration or fit.

## Old DEV: the missing near detections return, along with errors

| /1000 DEV | Frozen MZ5 | Adapted MZ5 | SOURCE_RGB | Composition |
| --- | ---: | ---: | ---: | ---: |
| Exact |912|917|899|900|
| TP BN/BF/HN/HF, each /200 |187/180/183/165|180/180/185/173|169/189/196/170|197/189/196/170|
| FP BN/BF/HN/HF |6/11/10/7|5/10/7/6|5/9/10/7|11/20/19/14|

The composition restores all28 MZ9-lost BODY_NEAR TPs that had no candidate.
Relative to MZ5 it gains TP10/14/13/13 and loses0/5/0/8. Thus having a candidate
does not ensure SOURCE is preferable:13 formerly correct far detections are
still lost in the selected-source portion.

Its FP decomposition is source5/9/10/7 plus fallback6/11/9/7. All34 original MZ5
FPs survive (33 via fallback and one via SOURCE), and SOURCE adds30 new FP bits.
No original FP is removed. Each branch's standalone budget does not carry over
to this partitioned combination. This is a measured error-allocation failure,
not a routing implementation defect or evidence that geometry is never useful.

Source is selected on178/238/228/278 DEV queries, fallback on822/762/772/722.
The candidate mask restores unsupported judgments without distinguishing helpful
and harmful fallback outputs. Availability alone is insufficient arbitration.

## Consumed sequence regression

| /200 MZ6 | Frozen MZ5 | Adapted MZ5 | SOURCE_RGB | Composition |
| --- | ---: | ---: | ---: | ---: |
| Exact |143|199|198|177|
| TP BN/BF/HN/HF |2/6/9/0|9/36/9/38|9/35/9/38|9/35/9/38|
| FP BN/BF/HN/HF |5/3/19/0|0/0/0/1|0/1/0/0|5/1/19/0|

Composition keeps thin48/49 and recovers far detections, with no new FP versus
MZ5 on this cohort. However fallback restores its5 BODY_NEAR and19 HEAD_NEAR
false activations, including the approaching bar's premature near outputs that
SOURCE had removed. Source contributes one BODY_FAR FP. Both correct detections
and wrong range decisions must be reported; do not claim full distance recovery.
Clean/stress output signs are identical. These results do not justify temporal
work or an independent generalization claim.

Only the DEV FP condition fails the seven declared admission checks, but that
condition is essential. Do not drop it because the near recall and targeted
sequence results improved. Stop this rule, retain MZ5 and the prior SOURCE
research component separately; no posterior threshold or alternate gate tested.
Any future combination must budget the errors of the combined system, rather
than assume separately calibrated branches compose safely. That is an unresolved
next question, not an implemented or validated upgrade.

## Validation and durable outputs

`artifacts.local/work/mz10-availability-20260910/run-v2/` contains result,
predictions, selection masks, original DEV frame IDs, changed-bit rows, receipts
and audit. Scalar audit verifies5600 query branch selections and44 metric/clip
groups; adversarial logit magnitudes cannot override the availability predicate.
All selected fallback scores equal the original baseline scores.

The initial replay reused clean MZ5 logits for stress, whose signs were already
identical. Final replay loads the actual frozen stress logits; every outcome
remains identical, with no training/threshold/routing change. Both versions are
preserved. CUDA only recomputes cached DEV readout; scalar work runs on CPU.
No UE/worker/port was started; processes exit and temporary validation resources
are released. Invalid/no-return remains UNKNOWN, never clearance. No App,
hardware, deployment or safety claim.

Post-hoc follow-up: [error and budget diagnosis](MZ10_ERROR_DIAGNOSTIC_20260910.md)
audits these frozen outputs without changing this run or its terminal. A SOURCE
threshold-only repair cannot preserve old far TPs at the complete composition's
FP budget; candidate extent is informative on consumed subgroups, while a
baseline-confidence veto would also suppress the intended thin-pole recovery.
