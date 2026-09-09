# 10000-source B distance diagnostic

2026-09-09 `EXPLORE`. After the fixed 10000-frame B fit, run one RGB-only inference pass for each fixed checkpoint on the finalized 5000-frame distance source. This is a controlled shared-site Development diagnostic; it does not fit a model, select thresholds, or establish deployment, natural-scene generalization, or safety.

## Frozen inputs

Use the finalized distance source (`5000` accepted frames, `2500` complete near/far pairs, all intended native labels BODY=0 and HEAD=1) and the same 256x144 BOX RGB preprocessing as BodyQuery B. Compare OLD `c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776` with NEW `body-query-10000-b` step2000. Use each arm's thresholds already selected on the 10000-source DEV partition. Native counts, events, UNKNOWN masks and pair identity remain evaluator-only.

## Metrics

For each pair compute `S_far = P(sum of the three HEAD-far capped query counts >= 3)` and `delta = S_far(far)-S_far(near)`. Report far-higher (`delta > 1e-6`), near-higher, ties, positive raw deltas, both-endpoint HEAD alerts, HEAD hits at near/far endpoints, BODY false alerts, mean/median delta, native near/far event confusion at threshold 0.5, and HEAD-near nonempty-query recall at 0.5. Preserve all pairs and UNKNOWN coverage.

Report the primary `EVAL_ONLY` distance subset and all 2500 pairs as descriptive shared-site Development, with slices by source partition, region and family. Pair direction is a ranking diagnostic and must not be described as calibrated distance capability. The accepted distance source has no HEAD-negative endpoint denominator, so HEAD false-positive rate is not estimable from it.

Stop after this one two-arm inference and independent analysis. Keep the existing expanded B baseline and retain NEW only for measured alert-scope evidence; do not add distance losses, threshold rescue, extra seeds or additional capture automatically.
