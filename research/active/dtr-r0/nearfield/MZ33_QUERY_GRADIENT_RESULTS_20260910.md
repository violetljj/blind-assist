# MZ33 query gradient diagnostic

2026-09-10, consumed Development, zero optimizer updates. Run and numerical
checks PASS. The final MZ32 endpoint has no predeclared combined-shared gradient
conflict signal: minimum pair cosine **0.077761**, above -0.1. Final raw TRAIN
responsibility errors remain **66/1165**. This does not support launching a
parameter-splitting fit on the proposed final-gradient-conflict gate.

The diagnostic uses exact 7,562 original TRAIN IDs, 1,165 branch disagreements,
and unchanged global RGB/ToF class weights 1.218619/0.847889. Each query loss is
its own disagreement mean, without class rebalancing within that query. Both
frozen MZ32 checkpoints use the same bound MZ30 features and normalization.

| Query | Examples | Initial / final weighted BCE | Initial / final errors | Initial / final shared gradient norm |
| --- | ---: | ---: | ---: | ---: |
| BODY_NEAR | 198 | 0.745572 / 0.123414 | 68 / 7 | 2.354733 / 0.185788 |
| BODY_FAR | 150 | 0.760125 / 0.256315 | 104 / 13 | 1.161202 / 0.713153 |
| HEAD_NEAR | 304 | 0.679448 / 0.168579 | 219 / 22 | 0.963312 / 0.396084 |
| HEAD_FAR | 513 | 0.719370 / 0.141202 | 335 / 24 | 0.719722 / 0.245315 |

Combined shared means 8,513 parameters: first-layer feature columns and bias,
plus the final readout. It excludes the 128 query one-hot column parameters.

| Query pair | Initial cosine | Final cosine | Initial dot product | Final dot product |
| --- | ---: | ---: | ---: | ---: |
| BODY_NEAR / BODY_FAR | 0.281755 | 0.645946 | 0.770408 | 0.085585 |
| BODY_NEAR / HEAD_NEAR | 0.211250 | 0.180839 | 0.479188 | 0.013308 |
| BODY_NEAR / HEAD_FAR | 0.117897 | 0.077761 | 0.199806 | 0.003544 |
| BODY_FAR / HEAD_NEAR | 0.290743 | 0.437755 | 0.325225 | 0.123652 |
| BODY_FAR / HEAD_FAR | 0.415799 | 0.406569 | 0.347500 | 0.071128 |
| HEAD_NEAR / HEAD_FAR | 0.719995 | 0.271853 | 0.499185 | 0.026415 |

The saved result also separates the 8,480 first-layer shared parameters and
33 readout parameters. The final readout HEAD_NEAR/HEAD_FAR cosine is -0.077581,
below zero but above the -0.1 signal criterion. At initialization, readout-only
BODY_NEAR/HEAD_NEAR is -0.253142; the combined shared vector remains positive.
Thus this is not a claim that every layer is always aligned. One-hot gradients
have disjoint columns, so all cross-query dot products are exactly zero by
construction. Their final norms are 0.009099, 0.024855, 0.021586, 0.008734.

CUDA forward/backward evaluated both endpoints without an optimizer. The
count-weighted sum of four query gradients matches the independently reduced
global-objective gradient: maximum absolute errors 7.89e-9 initial and 1.11e-8
final. Both state tensor comparisons are bit-identical; state hashes and every
bound input file hash remain unchanged. Syntax check and the single diagnostic
execution passed. No image inference, DEV threshold selection or fit occurred.

These are in-sample responsibility errors and local full-TRAIN mean gradients,
not task false alarms or mini-batch optimizer trajectory measurements. Positive
aggregate cosines cannot rule out capacity limits, conditional conflicts or
negative transfer; the initial readout-only conflict cannot establish a final
task benefit from splitting parameters. No fresh-source or device claim follows.
The proposed final-endpoint conflict gate fails; retain this evidence and stop
the diagnostic without a posthoc threshold or expanded sweep.

Evidence: `artifacts.local/work/mz33-query-gradient-20260910/run-v1/` contains
`start.json`, `result.json`, `gradients.npz`, and `receipt.json`. The receipt binds
code, protocol, frozen inputs and output hashes; `result.json` provides the
machine-readable `summary` and all partition norms, dot products and cosines.
Runner: `mz33_query_gradient.py`; protocol:
`MZ33_QUERY_GRADIENT_PROTOCOL_20260910.md`.
