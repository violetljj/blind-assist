# Near/far readout untying: partial recovery, replacement gate unmet

2026-09-09 consumed controlled Development. Retain original10k B as full baseline;
retain range-untying as a component/challenger, not a full replacement.
[Protocol](BODY_QUERY_RANGE_UNTIED_PROTOCOL_20260909.md),
[operator](body_query_10000_range_readout_train.py),
[independent analyzer](body_query_range_untied_analysis.py).

One valid2000-step count-only fit on CUDA completed with initial prediction parity
exactly0. All non-readout tensors/buffers remain byte-identical and the sampling
schedule matches original10k B. New parameter shapes are2x4x32 and2x4, with each
range initialized as an exact copy. No backbone, query_point, pooling or loss change.

| EVAL metric | Original10k B | Range untied | Historical shared count-only |
| --- | ---: | ---: | ---: |
| HEAD-near query TP /1788 |115|277|12|
| HEAD-near query FP /7212 |332|570|173|
| Wrong-far on near-only /600 |564|549|562|
| Native HEAD-near event TP /600 |282|396|288|
| HEAD_ONLY BODY FP /600 |38|32|40|
| Complete groups /600 |508|517|511|
| BODY TP / FP |1187 /69|1187 /65|1187 /72|
| HEAD TP / FP |1150 /35|1152 /31|1152 /35|

The replacement gate passes7/8 named criteria but fails its predeclared near
minimum294. Do not lower that criterion to admit277. TRAIN near query TP improves
191/2978 to477/2978; shared count-only is11/2978. The effect is not confined to
EVAL. However, BODY-far query recall falls1495 to1002/1750 and HEAD-far falls1615
to1490/1762. Near FP increases; wrong-far remains549/600. This is partial spatial
redistribution and modest alert improvement, not solved distance attribution.

Independent float64 probability analysis verifies all split confusion/group/query
counts, DEV selections, native3-cell count convolution, historical shared-control
outputs, SHA256 identities and all unchanged state tensors. Validation PASS is
execution/scoring validity; replacement gate remains FAIL.

Complete run cost91.43s on NVIDIA GeForce RTX5060 Laptop GPU. Artifacts and hashes:
`artifacts.local/work/body-query-range-untied-20260909/run-v1/`, including
`range-untied-analysis.json` and `validation.json`. Process exited; no persistent
worker/service allocated. Existing support maps and heuristic UNKNOWN remain
reported separately; zero count does not certify free hidden space.

The outcome supports testing direct visual-feature/context decoding under a new
brief. It does not authorize additional steps, seeds or threshold selection in
this completed run. All sources are already consumed shared-asset Development;
no natural-camera, calibrated distance, user-safety or deployment claim follows.
