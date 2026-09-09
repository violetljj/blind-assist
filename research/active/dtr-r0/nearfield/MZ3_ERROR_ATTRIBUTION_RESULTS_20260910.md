# MZ3: fusion gain hides substantial row-level replacement

2026-09-10 EXPLORE. [Frozen diagnostic protocol](MZ3_ERROR_ATTRIBUTION_PROTOCOL_20260910.md).
MZ1 fusion gains 148 exact frames over MZ0 while losing 105 previously correct
frames. Against matched ToF-only it gains 113 and loses 87. The net improvements
therefore do not establish uniform dominance or justify replacing MZ0. All
1500 EVAL_ONLY rows remain included; this is consumed controlled Development.

## Paired exact changes

Each cell is gained / lost exact frames against the named unchanged baseline.
The candidate and baseline can both be wrong with different predictions; those
rows are retained separately in the artifact, rather than counted as recoveries.

| Candidate | MZ0 | Matched ToF-only | Matched RGB-only |
| --- | --- | --- | --- |
| MZ1 8x8 fusion |148 / 105|113 / 87|67 / 24|
| MZ2 1x1 fusion |140 / 116|120 / 113|58 / 34|
| MZ2 4x4 fusion |148 / 106|115 / 90|67 / 25|

For the frozen 8x8 stress arms, noise gains 5 and loses 4; dropout gains 8 and
loses 21; the column shift gains 21 and loses 63. Shift loses 48 correct frames
within the 450 BODY_ONLY/far, BOTH/far and HEAD_ONLY/far rows (16 + 19 + 13),
while gaining 4 there. Thus the earlier net 42-frame decline is concentrated
in far positive conditions. This is a fixed packet alignment intervention,
not measured walking, hardware noise, or evidence for selecting a corruption.

## Where fusion helps and hurts

| Region, 500 each | Fusion vs MZ0 gained / lost | Fusion vs ToF gained / lost |
| --- | --- | --- |
| big06 |40 / 20|23 / 22|
| big07 |64 / 39|61 / 32|
| dense02 |44 / 46|29 / 33|

| Physical condition | Rows | Fusion vs MZ0 gained / lost | Fusion vs ToF gained / lost |
| --- | --- | --- | --- |
| BODY_ONLY |300|45 / 20|60 / 9|
| HEAD_ONLY |300|33 / 18|21 / 20|
| BOTH |300|60 / 12|13 / 14|
| CLEAR |300|5 / 19|3 / 14|
| ABOVE |75|1 / 14|3 / 12|
| LOW |75|2 / 16|4 / 15|
| LATERAL_OUT |75|1 / 3|5 / 2|
| FAR_OUT |75|1 / 3|4 / 1|

Range matters within those conditions. Versus ToF-only, BODY_ONLY/near gains
47 and loses 3 out of 150; HEAD_ONLY/near gains 5 and loses 14 out of 150.
HEAD_ONLY/far gains 16 and loses 6. Against MZ0, BODY_ONLY/far gains none and
loses 16. The declared range on a negative fixture is a construction attribute,
not evidence that a target actually occupies that range.

| Fixture family | Rows | Fusion vs MZ0 gained / lost | Fusion vs ToF gained / lost |
| --- | --- | --- | --- |
| cabinet |300|10 / 10|15 / 9|
| crossbar |600|98 / 63|64 / 54|
| hanging_sign |300|21 / 10|15 / 8|
| oblique_rod |300|19 / 22|19 / 16|

Full tables include all 48 original fixture-group IDs, every condition/range
combination, and row IDs. These reused geometries/backgrounds are not independent
natural trials. The tables locate observed errors; they do not isolate fixture,
geography or modality as a unique cause and do not define a predictive router.

## Error identity and geometric support

| Fusion error comparison | Introduced | Removed | Retained |
| --- | --- | --- | --- |
| Wrong-far vs MZ0 |12|7|0|
| Cross-body vs MZ0 |17|13|2|
| Wrong-far vs ToF-only |6|3|6|
| Cross-body vs ToF-only |16|16|3|

All 12 fusion wrong-far errors differ from MZ0's seven; the counts are not a
simple accumulation of the same failures. Of fusion's 19 cross-body errors,
17 also occur in matched RGB-only, three also occur in ToF-only, and two also
occur in MZ0. These are overlaps between separately fitted outputs, not proof
of an internal weighting mechanism. The 12 wrong-far events occur in BODY_ONLY
(1), BOTH (7) and HEAD_ONLY (4).

Fusion has 115 false-positive event bits on 101 rows. All 115 have zero full
native and zero same-FoV native event-support counts; none has even 1-2 native
witnesses. Only one has clean 8x8 zone-center approximation support. Specifically,
none of the 12 wrong-far bits and one of the 19 cross-body bits has center support.
MZ0 has 33 false-positive bits, all with center support but zero native counts,
including all seven wrong-far and 15 cross-body bits. This distinguishes the
fixed decoder's angular approximation errors from learned positive outputs
without that approximation's support. It does not prove no useful depth cues
exist in other zones or justify a center-support veto.

Native >=3 support defines the reference, so its absence on false positives is
not independent validation. The stronger observed detail is zero witnesses,
rather than merely falling below three. Full native support uses the original
camera FoV and fixed radial domain; no actual return-to-object association is
available. Clean support for the stress arms is counterfactual source support,
not support measured from their corrupted packets. No missing support is CLEAR.

## Modality disagreement

Matched RGB and ToF predictions disagree on 281/1500 frames and 391/6000 event
bits: BODY_NEAR 117, BODY_FAR 84, HEAD_NEAR 116, HEAD_FAR 74.

| Descriptive evaluator category | Rows | Fusion exact |
| --- | --- | --- |
| Both unimodal models correct |1192|1180|
| Only RGB correct |118|106|
| Only ToF correct |135|60|
| Both unimodal models wrong |55|7|

Fusion preserves substantially more RGB-only successes than ToF-only successes
in these consumed rows. It also loses 12 rows where both unimodal predictions
are correct. These are concrete targets for investigating evidence conflict;
truth-dependent category membership cannot be used at inference. Of the 55
both-wrong rows, 27 have identical wrong unimodal outputs and 28 disagree, as
implied by 281 disagreements minus the 253 one-correct rows.

## Decision and reproducibility

Retain MZ0 as the information component and MZ1 as a challenger with its recorded
tradeoff. The next useful question is how observation correspondence and evidence
conflict behave on separately declared cases, especially HEAD-near and controlled
negative cases. Increasing zone count or extending training is not justified by
this diagnostic. No replacement gate, truth-selected router, threshold change,
new fit, model inference or capture was performed.

Run from the checkout using:

```powershell
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' research/active/dtr-r0/nearfield/mz3_error_attribution.py --root 'E:\linnan\linnan' --output 'artifacts.local/work/mz3-error-attribution-20260910/run-v1'
```

The existing run is immutable; a reproduction requires a fresh child directory.
Artifacts are at `artifacts.local/work/mz3-error-attribution-20260910/run-v1/`:
`rows.jsonl` binds every prediction/truth/support vector to source identity;
`false-events.jsonl` records each false event and its error/support categories;
`result.json` contains nine-arm metrics, 12 paired comparisons, complete strata,
disagreement and support counts. Start/final receipts bind protocol, script,
source index and original artifacts by SHA-256; original input hashes were
checked again after computation and remain unchanged.

CPU NumPy execution took 0.57 s (`TASK_NOT_GPU_SUITABLE`). All 35 original metric
groups were recomputed with an independent NumPy expression. Scalar reconstruction
from the exported rows verifies all 12 paired counts and nine-arm exact/wrong-far/
cross-body totals; `independent-audit.json` records PASS. Original alert arrays
retain 5000/5000 parity, including 1500/1500 against MZ0. No task-owned process,
port, accelerator lease or temporary resource remains allocated. This verifies
scoring and provenance only; no new generalization or product-safety claim.

A separate `artifacts.local/work/mz3-error-attribution-20260910/verify_export.py`
also reconstructs all 974 false-event records, support categories, disagreement,
error transitions and every gained/lost stratum using scalar row iteration.
`support-strata-audit.json` records PASS and binds those exports by hash. Invoke
it with the same Python executable; it performs no model inference.
