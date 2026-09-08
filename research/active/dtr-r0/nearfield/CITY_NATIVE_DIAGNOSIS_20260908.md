# Native City inference replay and cached failure diagnosis

2026-09-08. Engineering follow-up to
[the completed native route](CITY_NATIVE_VALIDATION_20260908.md), authorized to
check inference and decompose existing misses. No training, new capture, gate
intervention, threshold selection, or model promotion. Original evidence is
unchanged. All outputs are in
`artifacts.local/nearfield/city-native-diagnosis-20260908/`.

## Inference chain

Frozen G10/G13 checkpoints (three seeds each), model source hashes, original
dataset identity and selected original RGB hashes were checked. The replay uses
the first 16 original G10 VAL samples in saved order, plus the six previously
reported target-positive City views (6,7,14,15,28,30). Both methods use full-frame
BOX resize to 256x144, RGB in [0,1], eval mode and one sigmoid; G13 applies its
saved ImageNet normalization internally. There is no evaluator input to models.

The predeclared absolute parity tolerance was 1e-4. `inference-parity-v1` failed
that strict check with smaller batches. `v2` matched historical batch context
(G10 64, G13 32; City 16): City scores and support became bit-exact, and G10
historical errors dropped below 6e-7. G13 historical error remained up to
0.000611. `v3` additionally matched the original uint8-to-CUDA then float/div255
order for historical images: all six weights reproduced all 16 historical
validation scores with **zero maximum absolute error**. The original City
adapter was also independently reproduced with zero score/support error on
the six selected views. Failed strict checks are retained, not relabeled PASS.

The City adapter previously converted/divided on CPU before CUDA transfer.
`tools/evaluate_city_native_route.py` now matches the historical GPU conversion.
The corrected adapter was run once on the same complete 40 frames as
`aligned-city-v1`, using the same weights, labels and thresholds. Maximum
ensemble near-probability change was 0.0000070632 for G10 and 0.0000225008 for
G13; **zero alert flips** for either method (80 head decisions each). This
numerical correction does not explain or fix the observed misses.

This is selected-sample numerical replay, not recomputation of the entire
historical benchmark. Actual inference ran on CUDA / RTX 5060 Laptop GPU;
backend observations and elapsed time are in each replay receipt/result.

## Cached route diagnosis

`cached-analysis` reads the original prediction cache, with no inference.
It reproduces the original frozen-threshold confusion matrices, checks pairwise
rank AUC against ROC integration and preserves frame29 as UNKNOWN. CPU scoring
and plotting took about 4.3 s (small cached arrays, TASK_NOT_GPU_SUITABLE).

| Method/head | Rank AUC | Positive median | Negative median |
| --- | ---: | ---: | ---: |
| G10 BODY | 0.7219 | 0.6890 | 0.3127 |
| G10 HEAD | 0.6500 | 0.2845 | 0.2360 |
| G13 BODY | 0.5267 | 0.5080 | 0.5076 |
| G13 HEAD | 0.3143 | 0.5112 | 0.5119 |

Denominators: BODY17 positive/22 negative, HEAD4 positive/35 negative, one
UNKNOWN for both. These are correlated consumed spatial frames, and the broad
reference describes visible in-query geometry; it does not certify free space.
Four HEAD positives provide very limited ranking evidence. AUC is diagnostic,
not new benchmark authority. All threshold breakpoints are saved as
POSTHOC_DIAGNOSTIC_ONLY with selected_threshold=null.

G10 has partial ranking signal but diffuse support. Bollard frame6 BODY score
0.1154 is below all 22 known negative scores (minimum0.1650). A monotone threshold
low enough to alert on this frame therefore also alerts on all 22 negatives.
This concrete failure cannot be resolved by simply lowering the global cutoff.
Some meter scores are closer to the historical threshold, so individual misses
need not have identical causes. Support overlap alone is weak when the map is
diffuse and its strongest point misses the checked target.

G13 has weak rank separation as well as low support. In observed activations on
the six City views, seed17/29 gated-feature mean absolute magnitude is about
0.0036/0.0032 versus 0.479/0.427 on the 16 original VAL views; their probabilities
lie within about0.0017/0.0011 of the learned bias-only probabilities. Seed43
retains more signal (gated magnitude about0.041 versus0.484), but the ensemble
still gives weak, nearly constant route scores. These activation observations
are consistent with the support gate suppressing the signal on this source;
they do not independently establish the root cause of that support failure.
No oracle-gate replacement was performed in this task.

Inspected figures: `cached-analysis/target-heatmaps.png` (three original RGB
views, independent target outlines, G10/G13 BODY/HEAD maps on a common0-1 scale)
and `scores-and-ranking.png`. Enlarged maps retain the original18x32 resolution;
display interpolation is not added model detail.

## Decision and reuse

The checkpoint/adapter chain reproduces historical behavior. A small conversion
ordering inconsistency is corrected and verified not to change City alerts.
Do not spend the next step on threshold tuning or larger-scale acquisition.
G10 retains some useful signal, while G13 support/source transfer warrants a
focused follow-up. Any future source adaptation should use different street
segments/instances and retain this consumed route as diagnostic regression;
improvements here alone cannot establish fresh generalization. No adaptation
or additional collection was executed in this engineering task.

Reproduce with `tools/audit_city_native_inference.py --match-batches
--historical-transfer --output <fresh canonical artifact directory>` and
`tools/analyze_city_native_scores.py --predictions <fixed-models-v1>
--labels <final-labels-v1> --capture <final-route-v1> --output <fresh directory>`.
Four affected evaluator tests pass. Processes ended after each bounded replay.
