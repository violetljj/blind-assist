# Direct frozen visual decoding recovers spatial attribution

2026-09-09 EXPLORE, consumed shared-asset Development. Both visual decoders show
a large native-count attribution gain; neither passes the full alert replacement
gate. Retain original10k B alerts, and LOCAL/JOINT as separate geometry components.
[Protocol](BODY_QUERY_CONTEXT_DECODER_PROTOCOL_20260909.md),
[operator](body_query_context_decoder.py),
[independent analyzer](body_query_context_decoder_analysis.py).

All model inputs are RGB-derived features with fixed calibration. The baseline
encoder, projection, support and query_point weights remain frozen. Extract64
raw channels before query_point; fit exactly three final2000-step decoders with
the same original TRAIN sampling schedule. TRAIN-only channel normalization and
new-from-scratch AdamW lr1e-3 are explicitly different from the earlier low-rate
linear continuation. No new captures, seeds, checkpoint selection or backbone fit.

| EVAL metric | Original B | LOCAL | JOINT | No-image PRIOR |
| --- | ---: | ---: | ---: | ---: |
| HEAD-near query TP /1788 |115 (6.43%)|1470 (82.21%)|1680 (93.96%)|0|
| HEAD-near query FP /7212 |332|152|112|0|
| Wrong-far on near-only /600 |564|138|24|0|
| BODY TP / FP |1187 /69|1188 /56|1191 /85|0 /0|
| HEAD TP / FP |1150 /35|1119 /26|1110 /46|0 /0|
| HEAD_ONLY BODY FP /600 |38|35|41|0|
| Complete groups /600 |508|497|477|0|

The two explicit spatial thresholds (HEAD-near>=894, wrong-far<=300) pass for
LOCAL/JOINT. The image-free control produces exactly identical probabilities on
every image and partition, and has no near true hits. Its zero wrong-far is not
useful attribution: it detects nothing. The protocol's unspecific “improvement
over PRIOR” is described by near TP and balanced accuracy after the run, not
represented as an additional preregistered numeric gate. Near-query balanced
accuracy is LOCAL0.90054, JOINT0.96203, PRIOR0.5.

TRAIN near TP rises191/2978 to2529 LOCAL and2973 JOINT. EVAL near false activations
fall simultaneously with increasing hits, distinguishing this from the previous
range-untying bias shift. However, LOCAL already recovers much of the signal;
the experiment cannot attribute all gain to cross-query context. JOINT also has
104624 trainable parameters versus LOCAL/PRIOR9220 and different output sharing.
The evidence establishes a useful decoder regime, not a unique pooling cause,
metric-depth mechanism, novel architecture, or natural-image generalization.

Full replacement remains false. LOCAL fails group and HEAD-TP preservation.
JOINT fails group, BODY-FP, HEAD-TP/FP and HEAD_ONLY BODY-FP criteria. Retaining
original alerts plus new range evidence is an explicitly separate two-output
research interface; it cannot retroactively pass the original count-bottleneck
gate. [Interface](body_query_context_evidence.py) keeps independent outputs,
returns UNKNOWN for no asserted range and exposes alert/range disagreement.

Runtime: complete extraction,3 fits and scoring73.73s on CUDA/RTX5060 Laptop GPU.
Fit times4.61s LOCAL,4.26s JOINT,4.36s PRIOR. All frozen inference reproduces
baseline probabilities within the declared tolerance, with original weights and
support unchanged. Source/checkpoint/feature hashes, sample order, exact sampling
schedule, TRAIN normalization, DEV selections, every split confusion/group/range
metric and6-cell alert convolution pass independent validation. Convolution
maximum difference is5.03e-7. First independent XYZ equality check failed on a
2.48e-9 CPU/GPU summation difference; arithmetic comparison uses1e-7 tolerance
while checkpoint XYZ tensors must remain mutually exact. No scientific gate changed.

Artifacts: `artifacts.local/work/body-query-context-decoder-20260909/run-v1/`,
including final checkpoints, frozen features, predictions, fits, selection,
context-decoder-analysis.json and validation.json. Task process exited; no worker
or persistent service allocated. The separate unchanged-decoder
[distance check](BODY_QUERY_CONTEXT_DISTANCE_PROTOCOL_20260909.md) addresses
whether this large gain survives rigid same-object near/far translation.
