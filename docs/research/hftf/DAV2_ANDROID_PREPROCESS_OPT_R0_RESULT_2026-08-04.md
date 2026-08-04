# DA V2 Android preprocess optimization R0 result

Decision: `PREPROCESS_KOTLIN_TABLE_R0_ADMITTED`; `PREPROCESS_NATIVE_OPENCV_R0_ADMITTED`; `GPU_SHARED_BUFFER_PREPROCESS_R0_NOT_TRIGGERED_BY_CPU_GATE`.

This is a reversible platform-engineering benchmark on `SM-S9280 / SM8650 / Android 16` over USB ADB. It preserves the frozen `640x480 RGB -> float/255 -> OpenCV INTER_CUBIC 686x518 -> ImageNet normalization -> NCHW` contract. It does not change the model, crop, rotation, interpolation, normalization, geometry, student, App route, or production authority.

## Implemented arms

- Reference decomposition: resize, RGB-to-NCHW, normalization, Float32-to-Float16, and allocation/copy are measured separately.
- Kotlin table arm: precomputed four-tap X/Y indices and Float cubic weights, fused resize/normalize/CHW packing, and a reused direct Float32 buffer.
- Native arm: reusable OpenCV Float32 Mats, exact `INTER_CUBIC`, four OpenCV threads, ARM NEON interleaved RGB load plus normalized CHW stores, and reusable direct Float32/Float16 buffers.

The test records per-iteration wall, instrumentation-thread CPU, and process CPU time. Allocation and GC counters are sampled once around each 100-iteration stage so the allocation probe does not contaminate every iteration.

## Frozen parity

Against `clean/normalized_nchw_fp32_1x3x518x686.npy` from the official `DepthAnythingV2.image2tensor` corpus:

| Arm | Mean abs | P95 abs | Max abs |
| --- | ---: | ---: | ---: |
| Double reference | `8.17e-7` | `3.10e-6` | `3.74e-5` |
| Kotlin Float table | `7.24e-7` | `2.26e-6` | `8.20e-5` |
| Native OpenCV/NEON FP32 | `1.13e-7` | `2.68e-7` | `1.74e-6` |
| Native OpenCV/NEON FP16 round-trip | `2.04e-4` | `4.61e-4` | `9.77e-4` |

All frozen preprocess parity gates passed. FP16 tensor parity does not by itself grant model-output or downstream parity; those remain separate QNN gates.

## USB device microbenchmark, 100 repetitions per state

| Stage / arm | Awake lockscreen P50/P95 | Dozing lockscreen P50/P95 |
| --- | ---: | ---: |
| Reference Double resize | `1212.64 / 1216.54 ms` | `1214.53 / 1228.46 ms` |
| RGB-to-NCHW | `0.43 / 0.51 ms` | `0.43 / 0.48 ms` |
| normalization | `1.39 / 1.46 ms` | `1.38 / 1.43 ms` |
| Float32-to-Float16, Kotlin scalar | `20.59 / 20.64 ms` | `20.19 / 20.26 ms` |
| allocation plus two 4.26 MB copies | `3.27 / 5.03 ms` | `3.20 / 4.34 ms` |
| Kotlin Float table fused/reused | `60.88 / 61.03 ms` | `60.86 / 61.10 ms` |
| Native OpenCV/NEON FP32 reused | `1.29 / 1.78 ms` | `5.40 / 8.12 ms` |
| Native OpenCV/NEON FP16 reused | `1.25 / 1.58 ms` | `1.33 / 6.72 ms` |

Thermal status was 0 in both state receipts. The allocation/copy control allocated about `8.55 MB/iteration` and caused 19/14 GC cycles. Native FP32/FP16 observed zero Java allocation and zero GC in both states. Kotlin observed zero allocation/GC awake; the dozing stage observed 64 KiB and one GC, so the honest claim is no explicit per-frame allocation in the implementation, not a universal zero-process-allocation guarantee.

## Routing consequence

- The low-risk Kotlin arm clears `<100 ms` with about a 20x median reduction versus reference resize.
- The Native arm clears `<40 ms` in both states, including RGB byte-array JNI copy and Float32 conversion/resize/packing.
- The frozen CPU gate therefore does not authorize a GPU preprocessing arm. GPU/shared-buffer work stays deferred until a measured transfer or integration reason exists.
- The next route is the ordinary-copy App-native cached QNN context. Shared buffers and GPU fencing stay out of the first JNI integration.

Raw bundle: `artifacts.local/evidence/hftf/cpu-boundary-microbench-r0-20260804-182905/result.json`, SHA-256 `CC69AD1AB2C0EA40F3110FCAD303D8161285B4B6C0233D58FF6513DE420F9ECE`.

Claim ceiling: preprocessing implementation parity and same-device performance only. No App CameraX end-to-end, QNN output, depth quality, energy, safety, or production claim follows.
