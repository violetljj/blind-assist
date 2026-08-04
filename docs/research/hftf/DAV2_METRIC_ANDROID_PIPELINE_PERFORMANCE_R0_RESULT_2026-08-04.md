# DA V2 Metric Android Pipeline Performance R0 Result

Date: 2026-08-04

Device: Samsung SM-S9280 / SM8650 / Android 16

Protocol: `DAV2_METRIC_ANDROID_PIPELINE_PERFORMANCE_R0_PROTOCOL_2026-08-04.json`

## Decision

The exact 518x686 CPU routes are rejected for a real-time CameraX runtime. The 607.8-second fixed-corpus pipeline completed 82 valid frames at 0.135 FPS; total latency was 5.768 s P50 and 14.863 s P95. Thermal status remained 0, so the failure is computational and memory-cost dominated rather than an observed Android thermal-status throttle.

The existing QAIRT HTP result remains a separate deployment candidate: exact-resolution DA-only cached burst mean was 277.917 ms. It is not a matched full-pipeline, bright-screen, CameraX, or sustained thermal result and is not substituted for this CPU experiment.

## Frozen model and export identity

| Artifact | SHA-256 |
|---|---|
| PyTorch checkpoint | `B782898D8A3E8BE1F639DE33837ED85E9B4B73E40F8F5E5CD99067588D722545` |
| ONNX FP32, NCHW 1x3x518x686 | `870339770E21675830F7E2020983DDA058752D237C8B86951ED1E6F9A6243D01` |
| TFLite FP32, NCHW 1x3x518x686 | `0277FBC74C73D95433B43BEE9D61DD08F1E79B67A2F64A6DA871F3A23FBED8E3` |

Host ONNX-to-TFLite parity was 0.000000849 m mean absolute error, 0.000002027 m P95, and 0.000006318 m maximum.

## Android parity

All frozen parity gates passed, without changing margins, perturbations, percentiles, RANSAC gates, or student weights after observing results.

| Neural scenario | Mean abs. error | P95 abs. error | Max abs. error |
|---|---:|---:|---:|
| Clean | 0.000277 m | 0.000707 m | 0.005646 m |
| Gaussian blur sigma 3 | 0.000237 m | 0.000587 m | 0.001451 m |
| Horizontal motion blur length 17 | 0.000223 m | 0.000568 m | 0.001533 m |

| Downstream scenario | Android status | Relative-height abs. error | Student-scale abs. error |
|---|---|---:|---:|
| Clean | VALID | 0.001024 | 0.000148 |
| Bottom 50% ground ROI masked | VALID | 0.014869 | 0.004947 |
| Horizontal local deformation, 20% | VALID | 0.001347 | 0.000882 |

These are implementation-parity checks, not evidence that the perturbations should be accepted by a production refusal policy.

## Short DA runtime arms

| Runtime | DA P50 | DA P95 | Processing FPS | PSS after runs | Terminal |
|---|---:|---:|---:|---:|---|
| ONNX Runtime 1.26 CPU, 4 threads | 1.333 s | 1.768 s | 0.695 | 750,387 KiB | REJECT_REAL_TIME_RUNTIME |
| LiteRT 1.4.2 FP32 CPU, 4 threads | 6.239 s | 22.466 s | 0.110 | 570,918 KiB | REJECT_REAL_TIME_RUNTIME |
| NNAPI exact FP32 attempt | — | — | — | — | NOT_COMPARABLE |

The NNAPI attempt did not finish within the bounded setup/run attempt, so no fabricated or partial latency value is reported.

## Sustained exact CPU pipeline

Runtime: ONNX Runtime Android 1.26, CPU, 4 threads. Input was a fixed 640x480 RGB corpus; this run did not include CameraX capture. The user reported that the phone was locked during the run, and post-run `dumpsys` showed `Wakefulness=Dozing` and keyguard showing. Display state was not sampled per frame, so the condition is conservatively labeled locked/screen-off rather than treated as a bright-screen result.

| Stage | P50 | P95 | Max |
|---|---:|---:|---:|
| Preprocess | 2.383 s | 8.140 s | 9.812 s |
| DA | 2.331 s | 7.221 s | 7.548 s |
| Postprocess | 3.141 ms | 4.222 ms | 19.955 ms |
| RANSAC + features | 179.999 ms | 1.500 s | 1.548 s |
| Scale student | 0.022 ms | 0.087 ms | 0.209 ms |
| Total | 5.768 s | 14.863 s | 17.710 s |

- Target duration: 600 s; actual measured pipeline duration: 607.797 s.
- Completed/valid frames: 82/82; full-pipeline throughput: 0.1349 FPS.
- Instrumentation PSS: 180,009 KiB before, 745,961 KiB after.
- Host-observed maximum PSS: 769,781 KiB; final native heap allocation: 742,413,328 bytes.

## Thermal and battery envelope

The host collected 109 samples over 629.546 seconds. Android thermal status was 0 in every sample and in all device-side 30-second samples.

| HAL sensor | First | Last | Maximum |
|---|---:|---:|---:|
| AP | 39.7 C | 40.8 C | 49.5 C |
| BAT | 36.1 C | 36.4 C | 36.7 C |
| SKIN | 37.0 C | 37.1 C | 39.0 C |

The phone was unplugged; battery level changed from 100% to 98%. This is not an energy measurement because charge percentage is too coarse and the preceding short arms are adjacent to the sustained interval.

## Interpretation and next routing

- CPU is useful as an exact-output parity and diagnostic baseline, but must not be used for the live camera path.
- The scale student itself is negligible; optimization effort belongs in preprocessing, DA acceleration, and the CPU RANSAC/features implementation.
- Exact-resolution HTP DA at about 278 ms implies roughly 3.6 DA invocations/s before preprocessing and downstream cost. It is more plausible as a periodic observer or disagreement detector than a per-frame live-depth loop.
- A promotion decision requires a matched QNN full-pipeline run with CameraX preprocessing, fixed bright-screen state, and at least 10 minutes of thermal sampling. This result does not authorize accuracy, alerting, safety, or production claims.

## Evidence

Raw local bundle: `artifacts.local/evidence/hftf/dav2-android-performance-r0-20260804-155524`

| Evidence file | SHA-256 |
|---|---|
| `result.json` | `EEEC2787841ED2F26A75FF1BA876618CEF6C2D107AB7DB9B202E4770610601F2` |
| `instrument-parity.txt` | `9343CCBD9D56C0B45F24A0DDD73128F2AF23865BA83DF2D3AF21CD8BFC1A9BFF` |
| `instrument-onnx-short.txt` | `94802F3C0C4BCC13FDDA7964B0DB38AE2CE9C10BF7D135A0DD29D43843492B1A` |
| `instrument-tflite-short.txt` | `F665DC0B89979F10B5B250515F8C37B2732DA572313D13CBDD2F9ACCDD1832B6` |
| `instrument-sustained.txt` | `D9C929B4DD0F383156F23E1CEB2C8A4F3AE0E5AD30845FD018F06941238A7B4D` |
| `host-thermal-memory-5s.jsonl` | `89F1B4A8347F39C465D2EB2EB37AF995E21CBDCE8FE68412483FB2699FF6E2DC` |

Claim ceiling: device deployment and performance diagnostic only; no CameraX capture, energy, accuracy, safety, or production authority.
