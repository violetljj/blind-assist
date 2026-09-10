# MZ14: local return attribution is the next concrete target

2026-09-10, consumed Development diagnostic. **No model upgrade was trained or
admitted.** MZ5 remains the baseline; MZ9 remains a coverage-bounded component.
The frozen trace locates a specific failure: all17 MZ13-added false output bits
on the existing3000-frame placement replay select angular cells with **zero
actual source pixels for the selected return**. This includes16 hanging-sign
bits across15 groups/12 sites and1 cabinet bit. The failure is an unsupported
return-location interpretation, not merely choosing the wrong BODY/HEAD query
for a correctly localized return in these17 cases.

## Scope and reproduction

- Old DEV1000 + MZ6 clean200 + relationDEV2000 + distanceDEV1000 =4200 frames.
- Frozen MZ9 weights, normalization and thresholds; no training, collection,
  cutoff search, protected access, temporal restart or default-App change.
- All4200 SOURCE outputs reproduce their previous margins exactly, not only
  signs. The scalar audit checks16800 output bits, exhaustive FP/FN partitions,
  mask monotonicity, coverage hierarchy and case completeness.
- Native pixels and contributor masks are evaluator-only. MZ13 predictions are
  read unchanged to identify its accepted additions. No oracle is an inference
  method. Existing data are consumed, including the already trained pole clip.
- No valid packet occurs in70/84/184/0 frames respectively across the four
  cohorts above. The no-support sentinel is retained for score comparisons;
  neither it nor missing packet coverage is a CLEAR observation.
- CUDA trace on RTX5060 Laptop:60.70s including input checks, extraction on3000
  images, cached replay on1200, native reconstruction, tracing and serialization.
  This is not per-frame deployment latency. Small paired descriptor statistics
  use CPU with `TASK_NOT_GPU_SUITABLE`; no backbone inference there.
- Mechanical attempt run-v1 stopped at the first cached-label comparison because
  CUDA does not implement UInt16 comparison. run-v2 explicitly promotes counts
  to int64 and completes; failed directory/log retained. No outcome rule changed.

Protocol: [bounded diagnostic](MZ14_EVIDENCE_TRACE_PROTOCOL_20260910.md).
Code: [trace](mz14_evidence_trace.py), [paired features](mz14_paired_features.py),
[audit](mz14_audit.py), [case figures](mz14_plot_cases.py).
Artifacts: `artifacts.local/work/mz14-evidence-trace-20260910/`:
`run-v2/{receipt,result,audit,cases,selected}.json`, `run-v2/traces.npz`,
`paired-v1/{receipt,result}.json`, and `figures-v2/` (v1 layout retained).

## What source-location masking explains, and what it cannot solve

Hold scores and thresholds fixed, but restrict the maximum to cells containing
at least one actual pixel contributing to that return. This source-presence
oracle does **not** require that the pixel belongs to the queried body region.
A second oracle additionally requires actual query membership. Their final
signs coincide in this replay, though51 scalar scores differ; the masks themselves
are not equivalent. Contributor presence is a cell-level test, not exact
per-pixel ownership or a guarantee that every point in the cell is occupied.

| Reused cohort | SOURCE TP before / after source oracle | FP before / after |
| --- | --- | --- |
| Old DEV1000 |724 /718 |31 /0 |
| MZ6 clean200 |91 /91 |1 /0 |
| Relation DEV2000 |1496 /1474 |57 /0 |
| Distance DEV1000 |956 /953 |17 /0 |

The oracle removes all106 SOURCE false output bits, but also removes31 true
output bits. Among the MZ13 additions on the new3000 replay it removes all17
false bits **and six recovered far true bits**. Therefore even perfect source
location filtering under the current scores would not preserve all useful
recall. Learning correct positive evidence must accompany suppressing wrong
locations. This table is a diagnostic intervention, not achievable model metrics,
and it does not remove the false positives preserved from MZ5 by an add-only rule.

Max pooling can pick a strong unsupported location. However, changing max to
another smaller aggregation cannot make an already negative correct candidate
positive with weights and thresholds fixed. Across these4200 frames, all130
SOURCE far false negatives have actual selected-return query contributors and
at least one geometrically eligible correctly supported candidate; their scores
remain below threshold. Local evidence scoring/representation needs attention,
not merely a different final pooling operator.

Coverage remains a separate issue. Selected returns contain no actual BODY_NEAR
query contributor for30/200 old-DEV positives and61/400 relation-DEV positives.
Of these91 opportunities, SOURCE misses85 and happens to predict6 positive from
other locations. This trace does not further divide absence into sensor crop
versus selected-bin omission. An attribution module confined to these packets
cannot be treated as a complete near-field replacement.

## The pole is visible to the representation, but localization is unstable

All49 pole far-query opportunities have actual selected-return contributors and
correct eligible candidates. MZ9 SOURCE has above-threshold correct candidates
in48/49; the source-presence oracle retains all48. MZ13 retained47/49 because
its separate gate rejected an additional HEAD_FAR opportunity. One BODY_FAR
opportunity is already below threshold at the SOURCE stage.

Nevertheless34 of the48 SOURCE true-positive winning locations have no actual
contributor to that query (24 BODY_FAR and10 HEAD_FAR). All34 have strictly higher
scores than the best correctly supported location; this is not an arbitrary
argmax tie. Correct output signs alone do not establish correct spatial reasoning.

The independent paired-feature diagnostic checks25 same-camera pole-present /
pole-absent pairs. Native far contributors define foreground angular cells; known
same-zone cells without those contributors define background. It measures the
L2 change of normalized64-channel frozen local features:

| Descriptive statistic across25 pairs | Result |
| --- | --- |
| Median localization AUC, foreground vs same-zone background |0.7535 |
| Pairs above chance ordering |22/25 |
| Median AUC against immediately adjacent left/right background |0.6266 |
| Pairs above chance against adjacent background |14/25 |
| Median foreground/background centroid distance, pole present / absent |2.781 /0.504 |

There is measurable pole-related feature response, while fine localization is
not stable across the25 views. These oracle groups and paired differences are
not inference inputs or a learned classifier. The1316 foreground and5642
background cell opportunities are not independent samples. This does not prove
the backbone has lost the pole, that current features are sufficient for general
attribution, or that increasing resolution alone will fix it.

## Decision

Retain the diagnostic and the actual-contributor supervision machinery. The next
small structural trial should predict **where each return is supported within
its angular zone**, shared across the four queries, then apply body-query geometry
to that support. Give it explicit relative location and local visual context,
retain more than one supported location/surface, and represent unsupported regions
as unknown. Supervise source location directly, with query labels as a separate
task check. This targets the identified17 wrong-location errors and avoids hiding
their origin behind four final scores.

Current results justify this target, not a particular architecture or a dense
depth network. Frozen visual features are a reasonable first comparator because
some useful response survives; resolution changes should be isolated if a learned
attribution readout fails to localize. Any future trained comparison must recover
far positives at the stated FP budget, account for the six oracle-lost additions,
preserve coverage outside the return-supported region, and test wrong local
correspondence. A mature geometry comparator must honor regional return semantics.

This4200-frame diagnostic is complete. No successor fit was started. No additional
large data collection is indicated by these findings, and temporal work stays
closed until current-frame evidence is used reliably.
