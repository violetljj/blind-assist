# Motion-conditioned occupancy A0.1 PC runtime result

Date: 2026-08-03

Status: `PC_COMPONENT_BENCHMARK_COMPLETE_NOT_DEVICE_EVIDENCE`

The supported current-occupancy route was measured over 210 frozen
`walking_halfsphere` frames on an NVIDIA GeForce RTX 5060 Laptop GPU with
PyTorch 2.11.0 CUDA 12.8. The first ten depth frames and first RAFT batch were
excluded from steady-state distributions.

| Component | Mean | P95 | Role |
|---|---:|---:|---|
| UniDepthV2-S inference | 28.44 ms | 31.52 ms | metric depth + confidence |
| 3D clearance CPU | 8.63 ms | 10.00 ms | ground and metric bands |
| RAFT-small pair pipeline | 13.19 ms | 14.89 ms | causal motion evidence |
| Logistic probability head | 0.0078 us/opportunity | - | occupancy probability |

Sequential mean component sum for the supported A0.1 route is approximately
50.25 ms per frame, or 19.9 FPS; the sum of component P95 values is about
56.42 ms. These are not a measured asynchronous end-to-end pipeline, and the
RAFT number includes file decode and resize. Geometry was valid on 94.5% of
post-warm-up frames.

The separately measured 2D corridor cost was 9.53 ms mean and 10.72 ms P95.
It belongs to the A2.1 successor whose fresh future transfer failed, so it is
not added to the supported current A0.1 runtime claim.

## Architecture implication

The classifier is not the deployment problem. Depth, ground recovery, and
optical flow account for effectively all compute. The next device canary should
therefore test a two-rate architecture:

```text
low-rate metric-depth anchor -> 3D clearance state
high-rate causal motion update -> frozen occupancy head
stale/invalid anchor -> UNKNOWN, never silently clear
```

No mobile rates or staleness limits are fixed by this PC result. They must be
measured on the actual target processor and final external camera. In
particular, 19.9 FPS on an RTX 5060 is not evidence that UniDepth or RAFT will
run acceptably on Android or A568.

Ignored machine report SHA-256:
`7449070257ED371146F4871B53A75990D911682270B9AFAA064EA974FFC5F4F9`.
