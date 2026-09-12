# MZ105: residual matching scores do not preserve useful alert benefit

2026-09-12. EXPLORE, 576 consumed rendered Development frames. **Retain unchanged
SGBM+ToF; do not integrate these fixed local matching gates.** No TAO replay,
training, new capture, alert-state change or automatic successor was run.

## Residual errors differ from MZ104

Exact replay reproduces every cached SGBM depth array (including invalid masks)
and original stereo/ToF support. Thus these SGBM pixels already passed its
uniqueness setting, explicit <=1px left/right consistency and >=8px connected
valid-mask support. The connected components do not enforce disparity coherence.

| Raw false query-frame origin | MZ101 | MZ102 | Total |
| --- | ---: | ---: | ---: |
| All original union FP | 43 | 19 | 62 |
| All support pixels native >4m | 26 | 6 | 32 |
| Mixed native >4m and eligible near outside query | 13 | 6 | 19 |
| All support pixels eligible near outside query | 4 | 7 | 11 |
| Independent ToF support present | 0 | 0 | 0 |

Only **32/62 (51.6%)** are exclusively far-surface underestimation, compared
with TAO's249/273 (91.2%).51/62 contain at least one far pixel. The10,322 stereo
support pixels behind these errors split into4,015 native-far and6,307 eligible
near but outside the original body query; none are native-in-query, below0.5m
or unavailable. These are evaluator-only source attributions, not runtime
detectors of those errors. All62 FP are stereo-only; ToF protection does not
explain the failed separability in this particular panel.

## Fixed observable features and retention envelope

The [brief](MZ105_RESIDUAL_MATCHING_DIAGNOSTIC_20260912.md) fixes one5x5 grayscale
ZNCC matcher over integer disparities0..95. Features: assigned-match similarity,
advantage over the best >4m candidate, and advantage over the best competitor
outside the assigned +/-1px basin. Every query-support pixel is included;
query score is the maximum, reflecting the current existential readout. Independent
ToF support always survives. The578 feature/observation files were sealed before
native depth, task truth and family annotations were read for evaluation.

The following is an **optimistic posthoc single-score threshold envelope on
consumed data**, using one shared cutoff across both panels. It is not a selected
method or a fresh test. Strict retention preserves all435 original raw TP,
including every thin/small critical-slice TP, separately on each panel.

| Strict-retention signal | MZ101 raw TP/FP/FN | MZ102 raw TP/FP/FN | FP removed |
| --- | --- | --- | ---: |
| Unchanged baseline | 219/43/5 | 216/19/6 | 0 |
| Assigned ZNCC | 219/42/5 | 216/19/6 | 1 |
| Far-match margin | 219/43/5 | 216/19/6 | 0 |
| Competing-match margin | 219/43/5 | 216/19/6 | 0 |

Assigned ZNCC removes only MZ101`head_turn_flat_02/HEAD`, whose7 support pixels
all come from eligible near surfaces outside the query. Yet it removes **zero
final FP**: the unchanged two-on/two-off state now loses2 final TP and adds up
to1.25s first-correct delay. Pooled final402/40/44 becomes400/40/46. All56 events
remain detected; this does not establish useful alert improvement. Raw UNKNOWN
increases655 to656 query-frames. The two margin envelopes leave predictions
unchanged. Merely requiring a finite score also leaves all predictions unchanged.

Relaxing to95% overall and95% per critical family, separately on each panel,
shows the tradeoff rather than an accepted rescue:

| Signal | Raw FP removed / TP lost | Final FP removed / TP lost |
| --- | ---: | ---: |
| Assigned ZNCC | 29 / 5 | 16 / 15 |
| Far-match margin | 29 / 6 | 16 / 14 |
| Competing-match margin | 13 / 5 | 5 / 13 |

All three relaxed examples add up to1.25s paired delay, despite retaining all
56 detected events. Assigned/competing gates lose1/34 MZ101 thin_left raw TP;
the far-margin gate loses2/40 MZ102 occluded_thin raw TP. MZ102 small_head stays
15/15 in these envelopes. Full per-panel final/critical/UNKNOWN/event details
and evaluator-derived cutoffs are in the linked summary; none is a production
threshold. No multifeature fit or threshold/window rescue followed.

## Why the overlap cannot be dismissed as only false-support TP

The lowest true query score is MZ102`head_bar_flat_05/HEAD`: assigned ZNCC0.223,
far/competitor margin-0.774, although **all42 candidate pixels have native depth
inside the actual HEAD query**. Adjacent low-score true frames similarly have
86/86 and42/42 native-in-query pixels. MZ101`head_turn_flat_03/HEAD` has59/59.
Thus globally demanding better local scores really can suppress correct near
surface evidence. Native depth explains this only after scoring; it cannot
choose an inference-time exemption.

Conversely MZ101`thin_left_flat_03/BODY` is a task TP supported by51/51 native-far
pixels. A query TP is not proof that its pixels match the true obstacle. Both
levels must remain visible; preserving baseline task TP is a conservative task
constraint, not a pixel-accuracy label.

![Consumed stereo-only query-score overlap](../../../../artifacts.local/work/mz105-residual-matching-20260912/score-overlap.png)

These results reject only the fixed5x5/integer/maximum-query scalar gates for
lossless residual suppression. They do not reject all confidence methods.
Rounding eligible depth near4m to disparity11 can place the assigned match
inside the far candidate set; nonpositive margin is not by itself proof of a
wrong far match. Patch mixtures, subpixel effects and other representations
remain untested explanations, not announced successors or established causes.

## Verification, cost and disposition

- Independent audit checks2,880 source RGB/ToF/depth hashes,578 sealed output
  hashes,1,152 query rows, strict-envelope retention and removed-query identity.
- All576 SGBM arrays and stereo/ToF support counts reproduce exactly. Synthetic
  tests check disparity direction against independently computed Pearson
  correlation, known shifted matches, flat patches and full-window borders.
  Source review found no blocking geometry/union/authority error.
- Initial float32 variance cancellation failed CPU/GPU parity before features.
  Preserved correction uses float64 and passes first-pair1e-7 equivalence. Actual
  CUDA ZNCC is faster than CPU on the same probe; mean complete diagnostic
  frame0.0959s, P950.1565s includes SGBM verification/hash/I/O and feature output.
  This is an offline cached diagnostic cost, not an integrated runtime claim.
- Task-owned processes exited. Inputs, features, audit, failed engineering logs
  and the figure remain under the canonical artifact junction for reproduction.

Intended terminal: `MZ105_LOCAL_MATCHING_NO_USEFUL_LOSSLESS_SEPARATION`,
NEGATIVE_CONTROL for these fixed local score gates. Registration still returns
the historical`experiments/index.jsonl:303`input fingerprint error; structured
terminal assignment remains pending, with desired disposition recorded locally.
No old ledger or validator was changed. A changed method would first need a
useful Development result; an unchanged selected method then needs complete new
scenes before any freshness claim. This run does not authorize that continuation.

Evidence: [summary](../../../../artifacts.local/work/mz105-residual-matching-20260912/evaluation-v1/summary.json),
[query rows](../../../../artifacts.local/work/mz105-residual-matching-20260912/evaluation-v1/queries.json),
[feature seal](../../../../artifacts.local/work/mz105-residual-matching-20260912/features-v3/seal.json),
[independent audit](../../../../artifacts.local/work/mz105-residual-matching-20260912/independent-audit.json),
[backend](../../../../artifacts.local/work/mz105-residual-matching-20260912/backend.json),
[pending registration](../../../../artifacts.local/work/mz105-residual-matching-20260912/pending-registration.json),
[implementation](mz105_residual_matching.py), [checks](test_mz105_residual_matching.py).
