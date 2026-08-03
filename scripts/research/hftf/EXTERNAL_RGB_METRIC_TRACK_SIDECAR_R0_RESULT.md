# External RGB metric-track sidecar R0 result

## Terminal

`CANONICAL_VITS_WINDOWS_GPU_EXTERNAL_RGB_METRIC_TRACK_SIDECAR_EXECUTED_CONSUMED_REPLAY_ONLY`

The viable reference lane is now a Windows CUDA sidecar:

`external RGB -> YOLO11n + ByteTrack -> canonical Metric3D ViT-S -> robust torso depth -> seven-frame metric track -> D44 OLS`.

It emits research observations and one-second forecasts only. It does not emit an alert or make a safety decision.

## Frozen scope

- Only already-consumed Bonn and Tokyo material was read; no fresh cohort was opened.
- No operating point, numeric parameter, threshold, seed, or gate was tuned after outcomes.
- `rebased_torso_history` was proposed after the first three scheduler arms were read. It is a parameter-free structural follow-up on the same consumed sequence, so it is reported as a diagnostic rescue attempt and receives no fresh or promotion authority.
- The asynchronous-anchor gates were fixed before result inspection: source depth MAE `<= 0.25 m`, mean relative absolute error `<= 15%`, and mean D44 difference from the full-rate source `<= 0.50 m`.
- A failed gate remains failed; the `0.50 m` D44 gate was not relaxed.

## Mobile source and scheduling terminals

On the 30-frame consumed Bonn RGB-D person sequence, the `392x672` RAFT-2 HTP source reached depth MAE `0.297071 m` and mean relative absolute error `15.9104%`. Full four-iteration HTP improved to `0.280168 m` and `15.0098%`, but still failed both fixed source gates. This identifies input resolution, not only early exit, as a material accuracy limit.

For canonical ViT-S at the measured `1.500794 s` HTP service time, 130 consumed Tokyo frames were replayed causally. The strongest parameter-free scheduler rebased the full seven-frame torso history onto the latest completed anchor. It covered 115 frames and 106 D44 opportunities, with mean current-depth difference `0.198011 m`, but mean D44 position difference `0.524513 m`; the frozen `0.50 m` gate failed by `0.024513 m`. Terminal: `CANONICAL_VITS_1P5S_ASYNC_ANCHOR_CONTINUITY_NOT_SUPPORTED`.

UniDepthV2-S was also converted from the exact formal camera-input ONNX. A bounded analytic replacement for the converter-unsupported `Acos` had end-to-end ONNX `pts_3d` max absolute difference `0.00005102 m`, so graph surgery was not the failure. On SM8650 HTP, one-frame accelerator time was about `720.905 ms`, but `pts_3d` mean relative error versus the patched ONNX was `52.577%`; the GPU DLC failed graph finalization. No quantization rescue was attempted. Terminal: `UNIDEPTH_QAIRT_HTP_NUMERIC_AND_LATENCY_NOT_SUPPORTED_GPU_FINALIZE_FAILED`.

## Windows GPU sidecar execution

The committed runner supports calibrated camera input and deterministic manifest replay. A 12-frame frozen-box replay produced one metric observation per frame and six D44 forecasts, beginning exactly at frame 6. Excluding the cold first frame, canonical Metric3D median latency was `148.673 ms`. Its torso depths were bit-for-bit identical to the previously materialized canonical observations (`max absolute difference = 0 m`).

The same 12 frames were then rerun through YOLO11n and ByteTrack, without using frozen boxes. It produced 109 track observations across 21 track IDs; seven IDs accumulated seven valid observations and generated 30 D44 forecasts. Steady medians were `13.596 ms` for detection/tracking, `156.752 ms` for metric depth, and `170.348 ms` for their sum. This validates every software stage except the physical capture driver and final-camera calibration/domain behavior.

### CUDA FP16 admission

Before reading candidate outcomes, the precision canary required finite output, mean/max torso-depth difference from FP32 `<= 0.05/0.10 m`, mean D44 difference `<= 0.10 m`, steady median latency ratio `<= 0.90`, truth depth-MAE increment `<= 0.02 m`, truth D44-error increment `<= 0.05 m`, and at least eight truth D44 windows.

Across five already-consumed Bonn sequences (`150` frames), FP16 passed every gate. Steady Metric3D median latency fell from `150.344 ms` to `125.950 ms` (`0.83774x`). Mean/max torso-depth difference from FP32 was `0.000500/0.001995 m`; mean D44 difference was `0.000964 m`. Against registered RGB-D truth, depth MAE was `0.087694 m` for FP32 and `0.087624 m` for FP16. Across 70 truth-paired future windows, D44 mean 3D error was `0.429896/0.430078 m`. The 150-frame sidecar produced exactly 120 D44 opportunities, or 24 per sequence, after sequence-keyed history isolation. A missing processed frame now clears that track's history, so reacquired IDs cannot masquerade as seven consecutive observations.

Terminal: `METRIC3D_VITS_CUDA_FP16_PRECISION_AND_LATENCY_SUPPORTED_CONSUMED_BONN_RGBD`. FP16 is now the Windows sidecar default; `--metric3d-precision fp32` remains available as the reference control.

### VideoCapture full-chain canary

The runner now accepts `--video` in addition to manifest and camera sources. Twelve already-consumed Tokyo frames were encoded as a `1280x720`, `7.5 FPS` MP4, then reread exclusively through OpenCV `VideoCapture`; frozen boxes and manifest timestamps were not used. YOLO11n and ByteTrack produced 102 track observations across 21 IDs. Five IDs accumulated seven consecutive observations and emitted 28 D44 forecasts. Video timestamps were strictly increasing with a median interval of `133333333 ns`.

With default FP16, steady medians were `9.781 ms` for detection/tracking, `135.241 ms` for Metric3D, and `145.238 ms` for their sum. Camera and video modes now require both calibrated intrinsics and the calibration resolution. Any actual frame-size mismatch, non-finite/non-positive focal length, or principal point outside the frame fails closed before metric coordinates are emitted.

Terminal: `WINDOWS_GPU_VIDEO_CAPTURE_FULL_CHAIN_EXECUTED_CONSUMED_REPLAY_ONLY`. This covers the same capture API and downstream software stages as an external camera, but not the final camera's optics, driver latency, exposure, distortion, or metric truth.

## Claim ceiling and next admission

This result supports an executable Windows GPU research reference, not a live-camera or product result. It does not establish final-camera metric accuracy, cross-camera generalization, phone/NPU continuity, alert utility, safety, research-mainline promotion, or default-App authority.

The next admissible step, when recording becomes convenient, is a short calibrated external-camera canary with known distances and motion directions. Camera/video execution requires `--intrinsics fx fy cx cy --calibration-size width height`; this prevents silently applying calibration at the wrong capture resolution. Until then, this sidecar can generate offline teacher tracks and exercise the D44 integration without ARCore.

Machine-readable summary: [EXTERNAL_RGB_METRIC_TRACK_SIDECAR_R0_RESULT.json](EXTERNAL_RGB_METRIC_TRACK_SIDECAR_R0_RESULT.json).

Runner: [run_external_rgb_metric_track_sidecar.py](run_external_rgb_metric_track_sidecar.py).
