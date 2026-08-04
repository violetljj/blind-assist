# DA V2 CameraX layered depth parity R0 result

Decision: `CANONICAL_NATIVE_FP32_STRICT_FP16_QNN_PARITY_SUPPORTED_DEVICE_ONLY`; `CAMERAX_LAYERED_PARITY_GATE_SUPPORTED`. Production, metric-geometry, accuracy, and safety promotion remain unauthorized.

## P0 result

The admitted low-latency OpenCV/NEON path was retained as a diagnostic control, but it was not bit-exact: Android OpenCV cubic operation order crossed FP16 quantization boundaries. The promoted native path now implements the frozen official OpenCV cubic contract explicitly: float32 interpolation coefficients, separable four-tap horizontal/vertical evaluation in the official order, float64 pixel arithmetic and normalization, final FP32 cast, followed by an integer IEEE-754 round-to-nearest-ties-to-even FP32-to-FP16 conversion. Native compilation disables fast math and FP contraction.

The converter conformance test covers every positive finite FP16 value, adjacent exact midpoints with both signs, signed zero, subnormal/overflow boundaries, infinities, and NaN. On the frozen static corpus, the promoted path is exact:

| Gate | Result |
|---|---:|
| canonical Native FP32 vs official FP32 | max `0`, all elements exact |
| canonical strict FP16 vs official FP16 | `0/1,066,044` bit mismatches; SHA-256 equal |
| App cached-context depth vs official-input depth | mean/P95/max `0/0/0 m` |
| preprocess + cached QNN P50/P95/max | `90.75/92.62/93.18 ms` |
| thermal before/after | `0/0` |

This closes P0 below the `250 ms` ceiling. The old fast route remains visible as a diagnostic: FP32 max error `1.70e-6` caused 610 FP16 bit mismatches on the same corpus. An FP32 tolerance pass was therefore insufficient authority for strict FP16 identity.

## P1 same-frame CameraX gate

The gate freezes one non-degenerate real `YUV_420_888` CameraX frame after warm-up, owns and saves tight Y/U/V bytes, and closes every `ImageProxy`. It saves RGB crop, official FP32/FP16 tensors, fast diagnostic and canonical Native tensors, App and CLI QNN depth, aligned depth, geometry JSON, buffer hashes, strides, rotation, timestamp, and file hashes.

The final valid front-camera bundle used RGB range `0..255`, mean `56.95`, and standard deviation `34.71`. Results:

| Layer | Result |
|---|---:|
| host YUV replay vs App RGB | exact, `0/921,600` mismatches |
| old fast host/Android cubic diagnostic | max `9.98e-6` |
| old fast strict FP16 diagnostic | `1,227/1,066,044` bit mismatches |
| canonical Native FP32 vs official FP32 | exact, max `0` |
| canonical strict FP16 vs official FP16 | `0/1,066,044` bit mismatches |
| Kotlin buffer hash vs JNI pointer immediately before QNN | exact |
| App cached-context depth vs CLI native-output depth | exact, max `0` |
| geometry artifact | saved; status `UNKNOWN` |

The complete CameraX gate passes. The former failure is localized specifically to the old Android OpenCV fast cubic evaluation, not YUV conversion, normalization policy, strict half rounding, QNN buffer transport, or QNN execution. The canonical operator fixes that layer without threshold tuning, biasing, relabeling, or downstream rescue.

Front-camera geometry uses canary intrinsics and is retained only to prove serialization and full-chain execution; it has no metric or safety authority. A prior all-black rear-camera capture remains rejected as degenerate.

## Evidence

- P0 bundle: `artifacts.local/evidence/hftf/qnn-native-cached-context-r0-20260804-205756/result.json`
- P1 bundle: `artifacts.local/evidence/hftf/camerax-depth-parity-r0-20260804-205846/capture/parity.json`
- Rejected all-black capture: `artifacts.local/evidence/hftf/camerax-depth-parity-r0-20260804-202405/`
