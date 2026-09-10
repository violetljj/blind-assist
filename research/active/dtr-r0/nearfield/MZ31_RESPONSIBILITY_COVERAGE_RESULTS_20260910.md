# MZ31: wider TRAIN disagreements supply both responsibility classes

2026-09-10 EXPLORE, consumed Development, no fit or new inference. **The exact
7562 original TRAIN frames contain1165 RGB/ToF-sign disagreements, versus168
in MZ30's narrow inference-eligible subset.** Every query now has at least20
examples and5 explicit sites for each correct-branch class. This passes the
coverage gate for testing broader supervision; it is not a task-effect result.

[Protocol](MZ31_RESPONSIBILITY_COVERAGE_PROTOCOL_20260910.md),
[saved-data diagnostic](mz31_responsibility_coverage.py).

| Query | Original RGB/ToF examples | Expanded RGB examples/sites | Expanded ToF examples/sites |
|---|---:|---:|---:|
| BODY_NEAR |118/5|152/138|46/25|
| BODY_FAR |2/1|69/43|81/45|
| HEAD_NEAR |0/30|65/50|239/110|
| HEAD_FAR |0/12|192/130|321/153|

The original subset requires baseline-positive, geometrically unsupported
queries. Expanded supervision candidates require only branch-sign disagreement;
the correct branch is determined by the unchanged native event truth. Under
disagreement exactly one branch is correct. A responsibility class is not an
event-positive/event-negative class.

| Baseline sign | Geometric support | Disagreement query examples |
|---|---|---:|
| Positive | Present |469|
| Positive | Absent |168|
| Negative | Present |163|
| Negative | Absent |365|

The997 additional examples come from existing original TRAIN observations;
there are no new frames, DEV samples or protected EVAL inputs. Source counts
are229 old5000,351 relation TRAIN,511 distance TRAIN and74 MZ6 sequence query
examples. Global responsibility totals are478 RGB-correct and687 ToF-correct.
Sequence site/group identities remain UNKNOWN and do not contribute to explicit
site counts. Exact source-index/cache identities are used elsewhere; shared
physical site names are not made distinct by dataset names. Counts are query
examples, not independent frames, sites or obstacle events.

The BODY_NEAR ToF-correct class expands from five examples at one explicit site
plus three sequence examples to46 examples at25 explicit sites. Both previously
absent HEAD RGB-correct classes are present in the wider TRAIN disagreements.
Thus the narrow destructive-action eligibility was also restricting the
available responsibility supervision. This does not establish that labels from
supported or baseline-negative queries transfer to unsupported baseline alarms.

The decision-changing next contrast is broader responsibility supervision with
the original narrow runtime modification rule unchanged. Architecture, initial
state, feature normalization, batch sequence, optimization budget and calibration
rule should remain fixed; recompute inverse-frequency weights by the same formula
for the new supervision mask. Measure the resulting FP correction and baseline
TP retention directly. Coverage alone neither repairs MZ30's lost TP nor justifies
changing its frozen thresholds or predictions.

Artifacts are under `artifacts.local/work/mz31-responsibility-coverage-20260910/run-v1/`.
`result.json` retains all1165 rows, both responsibility classes, source/family/site
counts and all four baseline/support partitions. Independent `audit.json` PASS
recounts every row against saved branches, exact TRAIN indices and explicit
source identity, verifies input/output/code hashes, and reproduces all eight
coverage gates. The original receipt and its outputs remain unchanged. The audit
used scalar NumPy CPU work and exited0; no neural model ran.

The frozen branch predictions are in-sample, and these are reused controlled
Development sources. Coverage is not independent generalization, calibrated
responsibility, product behavior or a safety claim. No App/device/temporal
change, new candidate or baseline promotion occurs in MZ31.
