# Frozen visual context decoder comparison

2026-09-09 EXPLORE. New experiment under the broad breakthrough goal. Historical
range-untying remains frozen. On its completed EVAL it improves HEAD-near115 to277
but adds query false activations332 to570. Scalar readout untying is insufficient.

Hypothesis: native-count attribution needs direct visual-feature decoding and
cross-query context, rather than the existing independently pooled query_point
embeddings. NDC-Scene discusses ambiguity when projecting a shared image feature
along different depths; MonoScene introduces context reasoning after projection.
These motivate a small context test, not a reproduction or application of either
full architecture: [NDC-Scene](https://arxiv.org/abs/2309.14616),
[MonoScene](https://arxiv.org/abs/2112.00726). No hidden-space completion is inferred.

Freeze original10k B SHA db39ccfcb9f3d1c8f3722711bab314ff0cf2da708e00d91f7fb263e6fec0ece0.
Extract raw64-channel masked-mean features before query_point for12 queries with
unchanged RGB preprocessing, projection and valid masks; reproduce baseline
probabilities in the same32-frame batches. Mean/std normalization uses TRAIN only.
No depth, condition, group, region, native mask or evaluator metadata enters a model.

Three predeclared final-step fits, each seed17, original2000x32 TRAIN schedule,
AdamW lr1e-3 wd1e-4, native count CE only; final weights zero, original count prior:

- LOCAL: shared MLP67->128->4 on one raw query plus fixed mean XYZ.
- JOINT: MLP768->128->48 on all12 raw query descriptors, reshaped12x4.
- PRIOR: identical LOCAL but all RGB-derived channels zero. It tests static
  coordinate/label priors; no image data reaches its decoder.

This is a new-from-scratch decoder regime, not a matched continuation against
the old learning rate. LOCAL versus JOINT also changes parameter count and output
sharing; a gain motivates context/capacity attribution, not unique causal proof.
All three fits finish before new DEV/EVAL feature extraction. Sources remain
already consumed shared-asset Development; no fresh confirmation claim.

Final alerts continue to use exactly the same count convolution. Support maps
remain unchanged. Freeze independent DEV thresholds before EVAL scoring. Report
all query/range confusions, near-only wrong-far, final alert TP/FP, complete groups,
all controls, heuristic UNKNOWN/coverage and actual backend/cost. Primary counts
include every frame. No selection on EVAL and no replacement of UNKNOWN with FREE.

Strong spatial gain requires EVAL HEAD-near>=894/1788 (50%), wrong-far<=300/600,
and improvement over PRIOR. Full replacement additionally requires groups>=508/600,
BODY TP>=1187 FP<=69, HEAD TP>=1150 FP<=35, HEAD_ONLY BODY FP<38. Report query false
activations alongside recall. Failure of full replacement preserves original B;
partial gain can remain a geometry component with explicit error costs.

Stop after3 fits, final metrics and one independent probability validation; no
seed/width/lr sweep or capture in this run. New mechanisms need a separate brief.
Outputs: artifacts.local/work/body-query-context-decoder-20260909/run-v1.
No persistent resource; exceptions retain stage and completed steps in failure.json.
