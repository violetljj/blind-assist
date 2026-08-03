# Metric depth Android dual-arm R0 result

Terminal: `DUALARM_ANDROID_CPU_AND_NNAPI_EXECUTION_SUPPORTED`

Deployment decision: `DIRECT_FULL_PRECISION_ORT_ROUTE_NOT_REALTIME`.

Metric3Dv2-S is not excluded. It remains the stronger frozen metric-depth
quality reference, but its current full-precision ONNX artifact is not a viable
mobile real-time source. UniDepthV2-S is the lower-cost deployment reference,
not an accuracy winner. The next Metric3D work must target deployment
compression/acceleration or teacher-to-student transfer rather than silently
substituting UniDepth in the existing A0.1 quality evidence.

## Bound inputs and host export gate

- No fresh RGB or outcome labels were read.
- Metric3D stayed at its existing canonical `1x3x616x1064` input, opset 11,
  SHA-256
  `674A665052F01BB2B64687200182F3380F0A462F4B6EF2E51FEFD06C84D8EE75`.
- UniDepth stayed at the A0.1-compatible `1x3x434x574` image plus camera-ray
  input, opset 14, SHA-256
  `2BA1FE9F9D8F050FBE83C164C5B5D01234119EE44273AC825F233702415AB958`.
- The UniDepth export passed PyTorch-CUDA versus ONNX Runtime CPU parity on one
  already-consumed Bonn RGB row. Point output P99 absolute difference was
  `0.0004365`, maximum relative difference `0.0009067`; confidence P99 absolute
  difference was `0.001154`, maximum relative difference `0.010653`; all
  outputs were finite.

This gate establishes export fidelity only. It does not compare depth quality.

## Formal target-device result

Device: Samsung `SM-S9280`, Android 16 / SDK 36, `arm64-v8a`.

Runtime: `onnxruntime-android:1.26.0`, graph optimization `ALL_OPT`, four
intra-op threads, one inter-op thread. Every arm used one warm-up and three
measured runs with deterministic neutral tensors; each model SHA was rechecked
on device before loading.

| Arm / backend | Load ms | P50 ms | P95 ms | PSS delta | Native heap delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Metric3D / ORT CPU | 418.57 | 5,039.89 | 5,367.79 | 832.13 MiB | 817.04 MiB |
| UniDepth / ORT CPU | 802.87 | 1,699.91 | 1,721.97 | 260.54 MiB | 564.73 MiB |
| Metric3D / ORT NNAPI | 632.36 | 7,179.16 | 7,180.26 | 579.63 MiB | 816.22 MiB |
| UniDepth / ORT NNAPI | 983.42 | 1,937.12 | 1,964.69 | 325.75 MiB | 696.96 MiB |

All four sessions loaded and completed all runs with finite sampled outputs.
Formal Gradle instrumentation finished `PASSED`.

On the required common CPU backend, Metric3D P95 was `3.1172x` UniDepth P95.
Registering NNAPI made Metric3D P95 `33.77%` slower and UniDepth P95 `14.10%`
slower. Session success does not prove accelerator-only graph coverage; the
observed slowdown is consistent with substantial CPU fallback and/or partition
overhead, but profiling would be required to attribute it.

The PSS deltas are sequential-process snapshots, not isolated peak-RSS
measurements. They are sufficient to expose a large current cost, but should
not be used as precise production memory budgets.

## Interpretation

The result separates quality and deployability:

- existing consumed quality evidence still favors Metric3D over UniDepth;
- current Android deployability evidence favors UniDepth by a wide margin;
- neither current artifact is real-time on this ORT CPU/NNAPI route;
- therefore neither a silent source swap nor direct Metric3D App integration is
  authorized.

Metric3D should now be retained as an offline quality teacher and as the first
candidate for a separately frozen deployment optimization screen. The useful
next screen is static INT8/FP16 or a smaller distilled metric-depth student,
with host quality-retention gates before another target-device run. If that
screen cannot retain the metric-distance advantage, the side lane should use
UniDepth only as a latency baseline and investigate sparse metric anchors.

## Evidence and rerun boundary

Machine-readable evidence is materialized under ignored
`artifacts.local/evidence/hftf/metric-depth-android-dualarm-r0/`:

- `formal-result.json`;
- `unidepth-export-parity.json`;
- `sm-s9280-formal/instrumentation-logcat.txt`;
- `sm-s9280-formal/test-result.textproto`.

The final formal logcat SHA-256 is
`00EF5CCE9B299ADCB752BC7975E69601AA7A8A785C5DE3D24840B3F1B42401A2`;
the formal test-result SHA-256 is
`AC39A4F20BD9C649935FA1B61CCB6098404C7A20D016100117FF168B463BB597`.

Three preliminary executions completed the frozen model runs but failed after
inference while attempting to persist a result file from the test-only APK.
No model, tensor, thread, backend, warm-up, or run-count setting was changed.
The final implementation emits one bounded JSON line per arm to the Gradle-
captured logcat, removing that control-plane failure, and the final formal test
passed.
