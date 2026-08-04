# DA V2 QNN native cached context R0 result

Decision: `QNN_NATIVE_CACHED_CONTEXT_R0_SUPPORTED_DEVICE_ONLY`; `CANONICAL_NATIVE_STRICT_FP16_DEPTH_PARITY_SUPPORTED_DEVICE_ONLY`. This is an Android device-deployment result, not accuracy, safety, metric-geometry, or production authority.

## Frozen implementation

The canary APK now follows the QAIRT Native SampleApp lifecycle: load the HTP/System runtime, create backend and device once, select the optimal SM8650 cached DLC record, create the context from its binary, retrieve the single graph, and reuse one FP16 input and one FP16 output direct buffer. The device, context, graph, tensors, and sustained-high-performance vote remain alive for the session and are released in reverse order.

The canary build copies `libQnnHtp.so`, `libQnnSystem.so`, the V75 stub, and V75 skel from the configured `QAIRT_ROOT` into a generated JNI directory. No proprietary binary is committed. The packaged HTP runtime SHA-256 was `C0488F2DF87932A42CA0A563883E6FBA190896BCA439AD0FDAA2428358AB5092`, identical to the CLI runtime. Omitting `deviceCreate` was independently shown to leave RPC polling unavailable and double execute time to about 274 ms; the final wrapper passes the device handle to `contextCreateFromBinary`.

## USB device result

Device: `SM-S9280 / SM8650 / Android 16`, USB serial `R5CX10M8Y8X`. Frozen graph: `dav2_metric_hypersim_vits_518x686_htp_fp16`, input `[1,3,518,686]` FP16, output `[1,518,686]` FP16. Ten measured repetitions followed one warm-up.

| Measure | Result |
|---|---:|
| context initialization | 293.13 ms |
| graph execute P50 / P95 / max | 81.01 / 84.89 / 85.50 ms |
| canonical Native FP16 preprocess + graph P50 / P95 / max | 90.75 / 92.62 / 93.18 ms |
| thermal status before / after | 0 / 0 |
| App vs CLI, identical FP16 input mean / P95 / max | 0 / 0 / 0 m |

The App/CLI gate passes exactly. The 74.45 ms host wall time also matches the earlier CLI HTP accelerator average of 74.05 ms and improves on the CLI graph-execute average of 134.62 ms. This supports a persistent in-process ordinary client-buffer route; it does not yet measure CameraX capture, YUV conversion, scheduling, or ten-minute sustained behavior.

## Canonical strict-FP16 closure

The Kotlin FP32-to-FP16 helper previously rounded half-way cases upward. Replacing it with Android's IEEE 754 ties-to-even `Half` conversion made the official FP16 App input exactly reproduce the CLI output.

The old fast OpenCV/NEON FP32 route was close but not bit-exact. On the frozen corpus its FP32 max error `1.70e-6` crossed 610 half bins and produced depth mean/P95/max drift `0.001617/0.0078125/0.0390625 m`; it remains a diagnostic control only.

The promoted native canonical path explicitly reproduces the official OpenCV cubic coefficient precision and separable evaluation order, performs the official float64 intermediate arithmetic and final FP32 cast, then applies the strict integer converter. It produced exact FP32, `0/1,066,044` FP16 bit mismatches, equal FP16 SHA-256, and App/CLI depth mean/P95/max `0/0/0 m`. Preprocess plus cached-context execute P50/P95/max was `90.75/92.62/93.18 ms`, with thermal status `0/0`. No threshold, crop, rotation, normalization, geometry, or model was changed.

## Evidence

- Closure bundle: `artifacts.local/evidence/hftf/qnn-native-cached-context-r0-20260804-205756/result.json`
- Cached DLC SHA-256: `2BB02F37FEF177FF4B02B8EE0C416EE9FF998BCEEF9786B92959E1F682EBAA24`
- Host runner: `scripts/research/hftf/run_qnn_native_cached_context_r0.ps1`

Next gate: bind this persistent runtime to a `STRATEGY_KEEP_ONLY_LATEST` CameraX pipeline with at most one in-flight depth job, prompt `ImageProxy.close()`, timestamp/TTL, and thermal/background fail-closed behavior.
