# Motion occupancy A0.1 Android probability-head canary protocol

Date: 2026-08-03

Status: `FROZEN_BEFORE_ANDROID_DEVICE_EXECUTION`

## Question and scope

Can the exact frozen A0.1 standardization and logistic probability head execute
with numerical parity and negligible cost on the connected SM-S9280?

This canary replays feature rows from the already consumed A1
`walking_halfsphere` report. It reads no labels on device, opens no fresh data,
and performs no fitting or threshold selection. It covers only the downstream
probability head; it does not claim that UniDepthV2-S or RAFT-small already runs
on Android, and it grants no App, alert, product, or safety authority.

## Frozen input and implementation

- Source report SHA-256:
  `51018CA9576E4728EE76716C716F229DE231C27A9C32705BD9E12E98D953E3B2`.
- RAFT checkpoint SHA-256:
  `01064C6DBA73B0FC9FC8EDF772248560A00A3ACFD62AC6677E9EEEBAD9680E27`.
- Frozen model:
  `MOTION_CONDITIONED_OCCUPANCY_A0_1_FROZEN_MODEL.json`.
- Asset rows: all 1,716 valid band x horizon feature rows, with host-computed
  expected probability and no outcome label.
- Device arithmetic: Kotlin `Double`, exact frozen mean, scale, intercept, and
  18 weights; sigmoid clipping remains `[-40, 40]`.
- Decision parity readout uses the already frozen `P>=0.50` threshold.

The canary runs in the isolated `:hftf-device-canary` instrumentation APK
against the `dualLoopShadow` App variant and writes one atomic JSON report.

## Gates and terminal

All must pass:

1. device is Samsung SM-S9280;
2. raw decompressed asset SHA-256 and row count match the readiness receipt;
3. maximum absolute probability error `<=1e-12`;
4. probability mismatch count at that tolerance is zero;
5. `P>=0.50` decision mismatch count is zero;
6. measured per-row head P95 `<=0.10 ms`;
7. the canary remains non-actuating and explicitly reports heavy inference as
   uncovered.

Pass terminal:
`A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_SUPPORTED`.

Fail terminal:
`A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_NOT_SUPPORTED`.
