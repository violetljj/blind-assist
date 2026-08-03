# Depth Anything V2 QAIRT HTP R0

Date: 2026-08-03

Terminal: `HTP_EXECUTION_SUPPORTED_HIGH_FREQUENCY_NOT_SUPPORTED_RELATIVE_ONLY`

## Decision

The off-the-shelf Qualcomm Depth Anything V2 Small asset executes entirely on
the SM8650 HTP, but this local canary does not support it as the high-frequency,
low-cost endpoint of the Metric3D/fast-observer Pareto design. The fastest
tested arm was the cached float DLC at about `174.32 ms` mean NetRun execution
(`171.01 ms` accelerator time). W8A16 was substantially slower at `451.63 ms`.
Explicit `burst` mode did not rescue the float arm (`177.19 ms`).

This result does not ask the light observer to beat Metric3D accuracy. It asks
whether the observer saves enough execution cost to justify its quality loss.
On the PC GPU screen, DA V2 Small Metric FP16 occupied the efficiency endpoint
at `54.27 ms`, while Metric3D FP16 occupied the quality/balance endpoint at
`142.33 ms`. The current phone HTP asset does not reproduce that efficiency
ordering and therefore does not solve the deployment-performance problem.
Cross-platform timing is diagnostic rather than a controlled hardware
comparison.

## Frozen role boundary

The Qualcomm model is the standard relative-depth Depth Anything V2 Small
checkpoint, not the separately trained Hypersim/VKITTI metric checkpoint used
in the PC quality comparison. Its NPU output may only be treated as relative
structure. It cannot directly produce metre-valued clearance, drive an alert,
or replace a metric anchor.

The deterministic gradient tensor was repeated six times to isolate model and
runtime execution. This proves that the graph runs on HTP with four HVX threads
and no observed CPU fallback; it does not evaluate real-image depth quality,
temporal stability, thermals, energy, or end-to-end camera latency.

## Local measurements

| Asset / mode | Mean NetRun execute | Min–max | Accelerator mean | NetRun IPS incl. I/O | Decision |
|---|---:|---:|---:|---:|---|
| W8A16 cached / default | 451.63 ms | 449.98–454.04 ms | 449.41 ms | about 2.0 | reject as fast arm |
| Float cached / default | 174.32 ms | 173.02–178.51 ms | 171.01 ms | about 4.69 | fastest local arm, still not high-frequency |
| Float cached / burst | 177.19 ms | 175.96–181.29 ms | 173.50 ms | 4.32 | no rescue |

The default and `burst` float outputs had the same SHA-256,
`E80D6A65C5583863B3513FB65CC58AB6735FA778DFAF0F6DC1B9A64681A1EFBD`.
The result is therefore a performance difference, not an output difference.

QAIRT reported the cached record as compatible with `HTP_V75_SM8650_4MB`, but
also warned that the record VTCM size and DSP architecture did not exactly
match the target device. That warning prevents attributing the gap to the
model alone. It does not invalidate the observed local runtime.

## Evidence and next action

The model/input hashes and exact measurements are recorded in
`DEPTH_ANYTHING_V2_QAIRT_HTP_R0_RESULT.json`. The ignored raw inputs and local
receipts remain under
`artifacts.local/evidence/hftf/depth-anything-v2-qairt-htp-r0/`.

No fresh image, quality threshold, anchor period, or alert policy was consumed.
The next deployment candidate must change the implementation boundary rather
than tune this consumed timing outcome: convert the actual metric checkpoint,
reduce input/decoder cost, or test a purpose-built smaller metric observer.
Any such successor requires its own frozen quality/cost comparison. Metric3D
remains the present PC single-model quality/balance reference, not an assumed
final mobile model.
