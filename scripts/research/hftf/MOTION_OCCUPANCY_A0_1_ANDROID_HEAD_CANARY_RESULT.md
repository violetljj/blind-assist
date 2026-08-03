# Motion occupancy A0.1 Android probability-head canary result

Date: 2026-08-03

Terminal: `A0_1_ANDROID_PROBABILITY_HEAD_PARITY_RUNTIME_SUPPORTED`

## Result

The isolated `:hftf-device-canary` instrumentation class ran 1/1 on a Samsung
SM-S9280 with Android SDK 36. It replayed all 1,716 hash-bound consumed feature
rows for 20 measured passes.

| Measure | Result | Gate |
|---|---:|---:|
| Probability mismatches | 0 | 0 |
| `P>=0.50` decision mismatches | 0 | 0 |
| Maximum absolute probability error | `2.220446e-16` | `<=1e-12` |
| Head call P50 | `0.000573 ms` | descriptive |
| Head call P95 | `0.001615 ms` | `<=0.10 ms` |
| Head call P99 | `0.005260 ms` | descriptive |

The App variant remained `dualLoopShadow=true`, `dualLoopActive=false`; the
canary was non-actuating, opened no fresh data, and explicitly reported
`heavy_inference_covered=false`.

## Infrastructure repair

The first two launches reached zero tests because the instrumentation APK
contained `lifecycle-common 2.3.1` while the target App used Lifecycle 2.8.7.
That shared-classloader ABI mismatch crashed AndroidX Startup before JUnit.
The isolated test module now explicitly uses the repository's existing
Lifecycle 2.8.7 catalog entry. Dependency insight confirmed 2.3.1 is upgraded
to 2.8.7; the named instrumentation class then completed successfully.

This was an execution-control repair, not an algorithm or threshold change.

## Claim boundary and next bottleneck

The exact frozen A0.1 standardization, sigmoid, and 0.50 decision can execute
on the target phone with effectively negligible cost. The downstream head is
therefore not an Android deployment blocker.

This result does not cover RGB decoding, UniDepthV2-S, clearance geometry,
RAFT-small, memory, sustained thermal behavior, or the full candidate pipeline.
The next deployment bottleneck is heavy feature production, especially mobile
metric-depth inference. This canary does not promote the route into the App or
change `COLLISION_RISK_FIELD_A1_DEVELOPMENT_FAIL`.

Ignored instrumentation logcat SHA-256:
`E3AC1E8E474B70329C19F1FE55176CB6C1F4A664082899A8B3248363512C4B30`.
