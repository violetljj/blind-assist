# Frozen contact decomposition: position transfer remains the main evidence gap

Completed the [read-only protocol](CONTACT_DECOMPOSITION_PROTOCOL_20260923.md).
No training, new cutoff, checkpoint promotion or App change. Extracted only frozen
selected binary-CDF(epoch50,cutoff.7639141083) and exact-supervision CDF(epoch10,
cutoff.8067280650), on864train/576evaluation images and51widths x2layers.
Both are consumed same-generator Development; old NEGATIVE_CONTROLs remain intact.
Retain this decomposition as COMPONENT diagnostic evidence, not a new predictor.

## Gate removal does not reveal broadly accurate held-layout positions

At width.6m, all560train and352evaluation finite interior truths stay in the position
denominator. Conditional median means the median position GIVEN modeled contact by3m;
it has a location even when q is tiny. It is not a deployed obstacle prediction.

| Metric | Binary train | Binary evaluation | Exact train | Exact evaluation |
|---|---:|---:|---:|---:|
| Conditional median within5cm |322/560 (57.50%)|100/352 (28.41%)|404/560 (72.14%)|100/352 (28.41%)|
| Conditional median MAE |12.77cm|23.79cm|4.78cm|22.35cm|
| Raw mu within5cm |352/560|117/352|404/560|104/352|
| Raw mu MAE |6.10cm|15.84cm|4.17cm|14.63cm|
| Actual continuous crossing within5cm |317/560|103/352|60/560|35/352|
| Actual missing crossings |116|94|264|165|
| False admissions on right-censored truth |96/1168|111/800|124/1168|110/800|

Continuous values differ from earlier sampled-grid headline counts by quantization.
The original grid results are unchanged. Position MAE above uses all finite truths
for conditional median/raw mu, not only admitted cases. Actual missing cases remain
failures in actual within5cm, while actual conditional MAE remains a separate field.

| Evaluation partition | Binary | Exact |
|---|---:|---:|
| Blocked by q, median accurate |21|31|
| Blocked by q, median inaccurate |73|134|
| Admitted, median accurate |79|69|
| Admitted, median inaccurate |179|118|
| Total finite truths |352|352|

Only21/94(22.34%) and31/165(18.79%)blocked cases have an accurate conditional median.
Both models have252/352inaccurate conditional positions even after ignoring admission.
The raw location parameter is also accurate in only117/352 or104/352cases, so the
gap is not only an artifact of choosing conditional median. Exact supervision makes
train conditional positions much better, but held positions remain poor. Different
layout distributions can contribute; these measurements do not prove memorization
or uniquely identify the causal feature/optimization failure.

Probability and distance are nevertheless coupled: exact evaluation has54admitted
cases with accurate median but inaccurate actual crossing. Actual crossing uses
G(z)=cutoff/q, which differs from the conditional median. Conversely20admitted exact
cases have inaccurate median but accurate actual crossing. Binary counterparts1/25.
Thus decoupling is a testable hypothesis, not a guaranteed lossless repair.

## Forcing contact probability is only an intervention diagnostic

Setting q=1 while keeping each original cutoff yields binary122/352 and exact90/352
accurate positions (34.66%/25.57%). It creates800/800false crossings on evaluation
right-censored cases, by construction. It must not be deployed, selected as a new
threshold or described as obstacle detection. Train counterparts361/560 and182/560,
with1168/1168false crossings. No q-unity prediction is promoted.

## Width consistency is not sufficient for correct geometry

Truth has zero wrong-direction finite-distance changes and zero disappearing contacts
as width increases. Evaluation covers57600adjacent pairs,16416with both true contacts
finite; these are correlated measurements, not independent trials.

- Both selected models have zero q-decrease pairs, zero admitted-contact disappearances
  and zero actual finite-crossing increases greater than1cm on evaluation widths.
- Binary conditional median has184wrong-direction increases on12curves, maximum7.85cm.
- Exact conditional median has zero increases greater than1cm, yet position accuracy
  remains28.41%. Correct monotone ordering therefore does not establish correct metric
  placement. Finer increases below1cm are not claimed absent.
- Binary train has8q-decreases across2curves and5conditional-median increases over1cm;
  exact train has neither. This is not a proof of structural monotonicity.

All-width conditional hit rates are58.36% ->27.84%for binary and72.69% ->29.21%for exact
(train27424/evaluation16944finite correlated width/layer truths), consistent with the
anchor's transfer limitation. Evaluation BODY/HEAD conditional hits are40/128 and60/224
for BOTH models; exact training138/192 and266/368, binary109/192 and213/368.

## Spread and authority boundary

Exact conditional10-90%span medians are21.62cmtrain/21.65cmevaluation, while conditional
position MAE rises4.78 ->22.35cm. Similar model spread does not track the observed
transfer error. Binary evaluation has184finite cases with span<=10cm but median error
>5cm; exact has13. These are descriptive diagnostics, not confidence calibration.
Large logistic scale and clipping to the.3mleft-censored boundary can also collapse
reported conditional spans; a narrow span cannot certify accuracy.
Original sensor UNKNOWN counts725/864train and487/576evaluation remain metadata.
Evaluator-partitioned position diagnostics do not provide online clear/UNKNOWN authority.

The next research priority is transferable position evidence, with probability/position
decoupling as a narrower question. This result does not establish that changing model
selection, lowering a threshold, or adding monotonicity will fix held-layout geometry.
A regional RGB/ToF correspondence mechanism would require its own justified experiment;
none was started here.

## Integrity, recovery and verification

Eight diagnostic math tests pass. Original extraction used actual CPU,
FROZEN_PROTOCOL_CPU_ONLY, matching both source checkpoints' recorded backend. Components
were sealed before geometry join. All4component arrays in final v3 are byte-identical
to that original extraction; no second extraction or training was needed.

First summary stopped because original truth retains finite coordinates beyond3m,
whereas this diagnostic records them as right-censored infinity. The comparison now
explicitly maps >3m to infinity, preserving all denominators. The failed extraction
tree was zero-copy registered through asset_catalog before governed reuse; the rejected
admission attempt and registration receipt remain. The v2 summary then stopped on a
31micrometre float32/float64 crossing difference. V3 records a1mm numerical bound PLUS
exact equality of finite masks and per-point5cmdecisions, without changing scientific
cutoffs. Maximum observed crossing difference.000030952m; qdifference1.1921e-7.
All failed receipts are retained, not presented as algorithm failures or extra fits.

Final evidence `artifacts.local/evidence/ba-contact-decomposition-20260923-v3` contains
source/input seals, unchanged copied components, queries, world-target arrays, full
partitions/distributions/curves/layers and parity. Original extraction is retained at
`ba-contact-decomposition-20260923`; partial summary at`ba-contact-decomposition-20260923-v2`.
These are owned durable diagnostic evidence, not live resources. No capture,paid worker,
service or background allocation was started. Independent audit PASS:146880world-space
targets; independent60stepCDF bisection reproduces all partitions, distributions,
layer/width/error counts;39168representative component scalars replay exactly on CPU.
Only first32rows permodel/split were component-replayed; all other rows received
arithmetic and saved-q parity checks. Receipt:
`artifacts.local/evidence/ba-contact-decomposition-20260923-audit/result.json`.
No hardware,natural-distribution or safety claim.
