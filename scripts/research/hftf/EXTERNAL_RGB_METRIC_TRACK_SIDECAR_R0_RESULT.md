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

## Claim ceiling and next admission

This result supports an executable Windows GPU research reference, not a live-camera or product result. It does not establish final-camera metric accuracy, cross-camera generalization, phone/NPU continuity, alert utility, safety, research-mainline promotion, or default-App authority.

The next admissible step, when recording becomes convenient, is a short calibrated external-camera canary with known distances and motion directions. Until then, this sidecar can generate offline teacher tracks and exercise the D44 integration without ARCore.

Machine-readable summary: [EXTERNAL_RGB_METRIC_TRACK_SIDECAR_R0_RESULT.json](EXTERNAL_RGB_METRIC_TRACK_SIDECAR_R0_RESULT.json).

Runner: [run_external_rgb_metric_track_sidecar.py](run_external_rgb_metric_track_sidecar.py).
