# DA V2 CameraX latest-only R0 result

Decision: `CAMERAX_LATEST_ONLY_R0_SUPPORTED_DEVICE_CANARY_ONLY`. This proves the real-camera scheduling and resource contract on one device. It is not model-accuracy, safety, ten-minute sustained, background-service, or production-promotion authority.

## Frozen camera contract

The isolated canary uses CameraX `YUV_420_888`, requested `640x480`, and `STRATEGY_KEEP_ONLY_LATEST`. Each `ImageProxy` is copied into one of three reusable planar YUV slots and closed in the analyzer. Native OpenCV then applies `imageInfo.rotationDegrees` clockwise, center-crops to 4:3, and resizes to `640x480` RGB with `INTER_LINEAR`. The already-frozen cubic/normalization/NCHW preprocessor writes FP16 for the persistent cached QNN context.

The worker permits one running depth task and one replaceable pending input. A third slot lets a new camera frame replace the pending slot without allocation. Results carry the capture timestamp and a 750 ms TTL. Severe thermal status fails closed. The lifecycle owner is bound only while resumed and teardown unbinds CameraX, clears the analyzer, closes pending inputs, drains the worker, and destroys QNN/native resources.

## USB short-run result

Device: `SM-S9280 / SM8650 / Android 16`, serial `R5CX10M8Y8X`. A 20-second run used a five-second saturation arm followed by a 2 Hz paced arm.

| Measure | Result |
|---|---:|
| Camera frames / ImageProxy closed | 291 / 291 |
| stress / paced submissions | 64 / 29 |
| processed / pending replaced | 87 / 6 |
| maximum concurrent depth tasks | 1 |
| stale results / thermal fail-closed | 0 / 0 |
| YUV copy P50 / P95 / max | 5.47 / 18.64 / 22.04 ms |
| YUV-to-FP16-plus-QNN P50 / P95 / max | 75.93 / 84.44 / 86.00 ms |
| fresh result age P50 / P95 / max | 96.94 / 141.14 / 153.27 ms |

All three owned YUV slots were returned and thermal status remained 0. The overload arm demonstrated actual pending replacement, while the paced arm delivered 29 submissions over about 15 seconds. The observed frame rotation was 90 degrees and the requested 640x480 YUV size was honored.

The earlier official tensor started from a frozen 640x480 RGB file, so this is a new camera contract rather than a parity claim for unseen crops or orientations. Its crop, rotation, color conversion, and interpolation are frozen here and must not be silently changed.

## Evidence

- Host runner: `scripts/research/hftf/run_camerax_latest_only_r0.ps1`
- Bundle: `artifacts.local/evidence/hftf/camerax-latest-only-r0-20260804-192838/result.json`
- Bundle SHA-256: `564D10EC1912983C099C4700490EA250015022F7451C850F47203DD4032755E3`

Next gate: optimize geometry without changing its output, then run the complete bright-screen CameraX pipeline continuously for ten minutes.
