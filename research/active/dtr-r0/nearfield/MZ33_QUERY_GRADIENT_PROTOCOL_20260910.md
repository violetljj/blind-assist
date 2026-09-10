# MZ33: query gradient conflict at frozen MZ32 endpoints

2026-09-10, EXPLORE, consumed Development; zero optimizer updates.
Question: do the four responsibility queries have conflicting local gradients
in the shared MZ32 selector? This diagnostic does not measure negative transfer.

Use only MZ32 initial and final checkpoints and their bound MZ30 branch features,
normalization and exact 7,562 TRAIN IDs. Reconstruct the 1,165 RGB/ToF sign
disagreements and RGB-correct targets; verify saved MZ32 supervision. Keep global
inverse-frequency weights 1165/(2*478) and 1165/(2*687). For each endpoint, compute
each query's weighted BCE mean over its own disagreement examples, raw sign
responsibility errors, and full parameter gradient on CUDA. Each query receives
one full TRAIN gradient evaluation. No DEV scores, images, threshold selection,
new fit, parameter update or runtime decision change is included.

Report query counts, BCE, errors, gradient norms and all six pairwise dot products
and cosines. Separate the first-layer 264 feature columns plus bias, final shared
readout, their combined shared parameters, and the four one-hot input columns.
Also report full vectors. One-hot columns are disjoint query parameter supports;
their cross-query zero dot products do not establish absence of shared conflict.
A shared-gradient cosine <= -0.1 is a predeclared local conflict signal. A signal
does not prove negative transfer; no signal does not prove sharing harmless.

Independently differentiate the direct global weighted mean and check it equals
the count-weighted sum of query gradients (absolute 1e-5 plus relative 1e-4).
Assert all model state tensors are bit-identical before/after and checkpoint
bytes unchanged. Save gradient arrays, parameter layout, numerical checks,
input/code hashes, runtime flags, start/result/receipt under
artifacts.local/work/mz33-query-gradient-20260910/run-v1. Register active before
execution. Preserve any mechanical failure; do not expand the two-endpoint
budget. Finish with a short report. TRAIN accuracy is in-sample; neither this
diagnostic nor the consumed evidence establishes fresh-source or device benefit.
