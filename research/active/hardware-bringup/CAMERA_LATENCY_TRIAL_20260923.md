# Camera transport coalescing trial — 2026-09-23

## Decision

Retain `atom-camera-wifi-v1`, Android v10.12.1 and fixed 8×8 ToF 10 Hz.
The candidate passed correctness and a first performance screen, but this
bounded A/B/A check did not establish a stable substantial latency reduction.
The original firmware was restored and its direct-phone transport rechecked.
No new APK, camera resolution, JPEG quality, risk algorithm or ToF rate was promoted.

## Mechanism and scope

Baseline sends multipart headers and JPEG in separate HTTP chunks, holding the
camera framebuffer until send completes. Candidate `v2-coalesced` allocates one
256 KiB JPEG staging buffer plus 512-byte header prefix per active handler,
copies JPEG, returns the camera buffer, then sends header and JPEG together.
VGA, Q12, two camera buffers, LATEST and original capture timestamps are unchanged.
Oversized JPEGs fail explicitly rather than being truncated. Review found no
buffer lifetime, cleanup or multipart defect. This is a transport engineering
trial, not a new obstacle algorithm or accuracy experiment.

The previous archived asynchronous producer/consumer approach had not shown a
clear benefit in its different SVGA/Q10 setup. This trial therefore tested a
smaller synchronous coalescing/early-return change, without adding a frame queue.

## Same-session measurements

One Samsung S24 Ultra hotspot; existing `HardwareWifiLiveTest`, 8-second warmup
and 45-second sampling per run. PC used ADB to launch and retrieve measurements,
not to relay sensor data. Runs were sequential, not randomized.

| Run | Camera age upper bound median / P95 / max (ms) | Usable snapshots | Distinct observed camera frames | ToF median / P95 (ms) |
| --- | --- | --- | --- | --- |
| A: original | 93 / 200 / 356 | 1300/1300 | 897 | 70 / 117 |
| B: candidate | 87 / 161 / 295 | 1303/1303 | 1032 | 68 / 117 |
| A: original restored | 90 / 175 / 342 | 1302/1302 | 986 | 71 / 128 |

The candidate met the initial screen of P95 ≤170 ms, median ≤102.3 ms, at least
99% usable snapshots and at least 807 observed camera frames. However, restoring
the unchanged original also improved its P95 by 12.5%. Candidate advantage was
19.5% against first A but only 8% against restored A. This does not prove zero
benefit; it does not justify a stable large-improvement claim or adding staging
complexity to the retained demonstration.

Camera acquisition-to-JPEG-ready P95 was 72.446 / 53.534 / 72.385 ms (A/B/A),
consistent with a possible buffer-return benefit. Transfer upper-bound P95 was
36.829 / 61.376 / 42.121 ms, so that segment did not improve. Independent segment
percentiles cannot be summed or used as a single-frame causal decomposition.

## Evidence limits and restoration

These are client-observed age upper bounds using synchronized device timestamps,
not physical event-to-screen or screen-photon latency. No synchronized optical
record of the scene and phone display was captured. App polling observations
are correlated and frame counts are observed unique frames, not sensor output
rates or a formal statistical confidence interval. No image-quality or physical
obstacle-accuracy improvement is claimed.

Local evidence: `artifacts.local/hardware-bringup/camera-latency-20260923/`.
Retain `trial-plan.json`, all three test logs and JSON receipts, comparisons,
candidate source/binary/build manifest, flash receipts and original backup.
Original application SHA256:
`17BBB17A3B819CBF418DB7014336664FBC01C0082DB9D085ED5B28F4F4A26201`.
Candidate SHA256:
`11DE499777CF177C7EA7D44A33C23D62A8131DB5DCC87F50778BFB7E49F95300`.
Only app0 was written; NVS and hotspot credentials were preserved. No second
camera mechanism or image-quality tradeoff was started after this trial.
