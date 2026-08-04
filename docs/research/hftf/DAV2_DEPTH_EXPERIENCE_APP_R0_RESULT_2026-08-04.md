# DAV2 Depth Experience App R0 result

Date: 2026-08-04 (Asia/Hong_Kong)

Executor: Codex

Scope: device-only visual experience; no production or safety promotion

## Outcome

`DEPTH_EXPERIENCE_APP_R0_AVAILABLE_DEVICE_ONLY`

The independently launchable `:hftf-depth-demo-app` packages the frozen local
SM8650 cached DLC, QNN HTP runtime, CameraX preview, and the same canonical native
FP32 preprocessing plus strict IEEE ties-to-even FP32-to-FP16 implementation used
by the parity canary. The runtime Kotlin and C++ sources remain shared with
`:hftf-device-canary`; the demo does not contain an alternate preprocessing path.

The debug APK was installed and cold-started on physical device `SM-S9280 / SM8650`.
The rear camera reached active streaming, QNN HTP produced a live metric-depth
heatmap, and the foreground activity remained resumed without a fatal exception.
The captured live frame reported:

- center depth: approximately `1.67 m`;
- near-depth percentile: approximately `0.70 m`;
- full YUV-to-depth pipeline: `92.8 ms`;
- displayed depth update rate: `2.1 Hz`;
- thermal status: `0`.

Local visual receipt:
`artifacts.local/evidence/hftf/depth-demo-split.png`.

## UX and runtime contract

- CameraX analysis is fixed to `640x480 YUV_420_888` with
  `STRATEGY_KEEP_ONLY_LATEST`; every `ImageProxy` is closed in `finally`.
- Only one depth inference may be in flight; submissions are rate-limited to a
  nominal `2 Hz`.
- The overlay renders a `343x259` depth map and maps each frame's valid 5th-to-95th
  depth percentiles across a red-to-yellow-to-cyan-to-dark-blue scale. This improves
  visual contrast without changing the metric depth values used by the numeric
  center/near readouts. The panel also shows full-pipeline latency, refresh rate,
  HTP status, and thermal status.
- Severe thermal status stops camera inference and reports the failure.
- The default view places unmodified RGB on the left and an opaque heatmap on the
  right, matching the desktop comparison logic. The button cycles through split,
  full-frame translucent overlay, and RGB-only modes.
- The UI states that this is an algorithm experience and cannot replace mobility
  guidance or safety judgement.

## Boundary

This result establishes an installable visual demo on the tested device only. It
does not establish cross-device support, metric-depth accuracy in uncontrolled
scenes, accessibility readiness, production integration, release signing, or any
safety authority. The default BlindAssist App and its decision path are unchanged.
