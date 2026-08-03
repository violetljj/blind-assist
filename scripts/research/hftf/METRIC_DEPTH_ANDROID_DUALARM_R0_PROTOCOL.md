# Metric depth Android dual-arm R0 protocol

Frozen: 2026-08-03, before target-device execution.

## Question and authority

Can the existing Metric3Dv2-S and frozen A0.1 UniDepthV2-S camera-input models
load and execute through the same Android runtime on the current target phone,
and what are their latency and process-memory costs?

This is a deployment-feasibility experiment. It reads no fresh RGB, depth,
event, collision, occupancy, reminder, or safety labels; it cannot change the
existing A0.1 quality terminal or promote either model. A model-source change
still requires a separately frozen quality experiment.

## Frozen arms

| Arm | Model artifact | SHA-256 | Input |
| --- | --- | --- | --- |
| M | Metric3Dv2 ViT-S ONNX, opset 11 | `674A665052F01BB2B64687200182F3380F0A462F4B6EF2E51FEFD06C84D8EE75` | `pixel_values`, float32 `1x3x616x1064` |
| U | UniDepthV2 ViT-S camera ONNX, opset 14 | `2BA1FE9F9D8F050FBE83C164C5B5D01234119EE44273AC825F233702415AB958` | `rgbs` and normalized unit `rays`, float32 `1x3x434x574` each |

The UniDepth artifact must first pass the committed host export/parity canary.
Its shape and camera-ray input reproduce the `resolution_level=0` route used by
A0.1; replacing this with the easier self-estimated-camera export is forbidden.
Metric3D keeps its existing canonical `616x1064` input rather than being
downscaled to make the comparison look faster.

Large model binaries stay under ignored `artifacts.local/` and are transferred
to a fixed device-local canary directory. They are never committed or bundled
into the production App.

## Frozen runtime and tensors

- Device: the currently connected `SM-S9280`; record serial, Android release,
  SDK, ABI, manufacturer, model, and battery temperature if available.
- Runtime: `com.microsoft.onnxruntime:onnxruntime-android:1.26.0`.
- Required common backend: ORT CPU execution provider, graph optimization ALL,
  sequential arm execution in one instrumentation process, four intra-op
  threads, one inter-op thread.
- Optional accelerator screen: repeat with the ORT NNAPI execution provider
  after both CPU arms finish. Provider/session rejection is a recorded
  `NOT_SUPPORTED` outcome, never silently replaced with a different runtime.
- M input is a deterministic neutral padded image: channel constants
  `123.675 / 116.28 / 103.53`.
- U RGB input is deterministic ImageNet-normalized zero; its rays are generated
  from the already-consumed Bonn intrinsics and the frozen resize factor
  `0.8838834764831844`. No outcome data are read.

The tensors are intentionally not used for a quality comparison. Different
canonical resolutions and output heads are reported as part of the cost rather
than hidden by forcing equal shapes.

## Measurement

For each arm/backend, in the frozen order `M` then `U`:

1. verify the on-device file SHA-256 before model loading;
2. record process PSS, Java heap, and native heap before and after session load;
3. require session creation and record input/output names and shapes;
4. run one unmeasured warm-up followed by three measured inferences;
5. after every run require expected output count and finite sampled float
   outputs, close all tensors/results, and record latency;
6. record P50/P95/max latency plus post-run process memory.

No arm-specific thread count, graph optimization, input resolution, precision,
quantization, operator rewrite, warm-up count, or run count may be selected
after observing results.

## Terminals

- `DUALARM_ANDROID_CPU_EXECUTION_SUPPORTED`: both CPU sessions load and finish
  all runs with finite outputs. This supports comparison of measured device
  costs only.
- `METRIC3D_ANDROID_CPU_NOT_SUPPORTED` or
  `UNIDEPTH_ANDROID_CPU_NOT_SUPPORTED`: the named frozen model cannot complete
  the common CPU contract. Preserve the exception and completed opposing arm.
- `DUALARM_ANDROID_NNAPI_EXECUTION_SUPPORTED`: both optional NNAPI sessions
  complete the same contract. This still does not prove accelerator-only graph
  coverage unless profiling separately establishes it.
- `DUALARM_ANDROID_NNAPI_NOT_COMPARABLE`: either NNAPI session is rejected or
  fails; retain CPU evidence and do not substitute LiteRT/QNN in this protocol.

No latency threshold is a quality or safety gate. A practical real-time route
requires a later end-to-end camera experiment including preprocessing,
clearance geometry, tracking/flow, thermal steady state, and the final external
camera.
