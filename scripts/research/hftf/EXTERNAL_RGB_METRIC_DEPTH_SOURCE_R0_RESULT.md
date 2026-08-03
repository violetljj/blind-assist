# External RGB metric-depth source R0 result

Date: 2026-08-03

Terminals:

- `METRIC3D_V2_S_RGB_METRIC_DEPTH_SOURCE_SUPPORTED_DEVELOPMENT_ONLY`
- `UNCONDITIONAL_D44_RGBD_PERSON_SURFACE_TRACK_NOT_SUPPORTED`
- `RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## Question and boundary

Can an ordinary RGB camera provide a same-person, seven-frame metric track for
the D44 successor? Public registered RGB-D video was used because the final
external camera was unavailable. RGB was the only model input; sensor depth
was read only by the evaluator as metric truth.

This is a public-proxy Development result. It does not establish performance
for the final lens, mobile deployment, alerts, products, or safety.

## Sources and fixed sampling

- [Bonn RGB-D Dynamic Dataset](https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/):
  `person_tracking` and `person_tracking2`, registered depth and published
  intrinsics. Five predeclared 3-second windows were evaluable: 150 frames.
- [TUM RGB-D walking_static](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download):
  a different Kinect with a static camera. RGB/depth format, 5000 depth scale,
  registration, and Freiburg 3 intrinsics follow the official
  [file-format page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).
  Fixed windows at 0, 4, 8, 12, 16, and 20 seconds were attempted. Only tracks
  present in every one of 30 sampled frames and with at least 80% valid torso
  depth in every frame were admitted.

No window, model threshold, history length, or one-second horizon was changed
after reading model outcomes.

## Three-arm Bonn result

All arms produced valid depth for 150/150 frames and 100% usable seven-frame
windows.

| Source | Mean absolute error | Mean relative error | Direction | CUDA steady mean | CUDA P95 |
|---|---:|---:|---:|---:|---:|
| Metric3D v2 ViT-S PyTorch | 0.08769 m | 4.87% | 5/5 | 164.74 ms | 173.80 ms |
| UniDepthV2 ViT-S | 0.14691 m | 8.10% | 3/5 | 30.41 ms | 33.86 ms |
| Video Depth Anything Metric-S | 0.45484 m | 24.82% | 4/5 | 41.70 ms | 46.13 ms |

Metric3D is the quality candidate. UniDepth is the latency candidate. Video
Depth Anything is not supported as the preferred source by this cohort.

## Metric3D GPU correction and cross-dataset check

The initial ONNX arm ran on CPU because ONNX Runtime 1.23 could not execute the
model on the RTX 5060 compute-capability 12.0 GPU. This was a runtime artifact,
not a Metric3D limitation. The official PyTorch ViT-S path runs on CUDA:

- PyTorch versus ONNX person-depth mean absolute difference: 0.001415 m;
- output correlation: 0.999975;
- CUDA steady mean: about 163 ms instead of about 3947 ms on ONNX CPU.

On 150 TUM pose-torso frames from five complete tracks, Metric3D achieved:

- 150/150 valid predictions;
- mean absolute error 0.08036 m;
- mean relative error 4.16%;
- 5/5 track directions correct;
- CUDA steady mean 156.23 ms, P95 163.94 ms;
- CUDA peak allocation 709.16 MiB.

The similar Bonn and TUM errors support continued Development of Metric3D as a
metric-depth source, while its current latency is not yet a mobile admission.

## D44 bridge result

The fixed bridge used seven causal observations and predicted frame `+10` at
the sampled 10 FPS rate. Camera coordinates were mapped to forward, lateral,
and vertical axes before timestamp-aware OLS.

| Proxy track | Opportunities | Current-static mean horizontal error | Sensor-depth oracle OLS | Result |
|---|---:|---:|---:|---|
| Bonn fixed box torso | 70 | 0.25164 m | 0.38434 m | worse |
| TUM tracked box torso | 84 | 0.19650 m | 0.32764 m | worse |
| TUM tracked pose torso | 70 | 0.09829 m | 0.19913 m | worse |

For the TUM pose-torso Metric3D track itself, current-static mean horizontal
error was 0.15031 m and D44 OLS was 0.32166 m. The sensor-depth oracle also
worsened, so this terminal is not evidence that Metric3D caused the forecast
failure.

Two bounded diagnostics did not rescue the mechanism: a person-mask 3D point
centre still worsened the oracle forecast, and a history-only unanimous-motion
gate still worsened the static baseline. They do not authorize threshold or
horizon search on these consumed outcomes.

## Decision

Continue the external-RGB metric-depth branch with Metric3D as the quality
reference and UniDepth as the real-time reference. Stop direct promotion of the
unconditional D44 constant-velocity backend for RGB-D person-surface tracks.

The next independent candidate should ask a new question: whether a causal
motion-state estimator can distinguish stationary, persistent translation,
turning, and start/stop motion while exposing forecast uncertainty. It must be
tested on fresh sequences. The final external camera still requires its own
calibration and domain check when available.

## Reproducible code

- `prepare_bonn_rgbd_metric_depth_manifest.py`
- `prepare_tum_rgbd_tracked_metric_depth_manifest.py`
- `produce_external_rgb_metric_depth_observations.py`
- `evaluate_external_rgb_metric_depth_source.py`
- `evaluate_external_rgb_metric_track_d44.py`
- `run_external_rgb_metric_depth_triarm.ps1`
