# DA V2 Metric Android QNN full-pipeline R0 result

Decision: `REJECT_LITERT_QNN_DIRECT_EXACT_GRAPH_FOR_APP_RUNTIME`; `REJECT_CURRENT_SEQUENTIAL_FULL_CHAIN_FOR_REAL_TIME_CAMERA_RUNTIME`; retain the precompiled QAIRT HTP route only as a periodic auxiliary/disagreement candidate.

This experiment did not alter scale-free margins, percentiles, smoothing windows, models, or training. It is a device deployment/performance diagnostic on `SM-S9280 / SM8650 / Android 16` over wireless ADB.

## What actually ran on the NPU

The SM8650-specific cached DLC has SHA-256 `2BB02...AA24`. QAIRT 2.47.0 found the compatible `HTP_V75_SM8650_4MB` record and executed the exact `1x3x518x686` DA V2 Metric graph with four HVX threads. A simultaneous VTCM/DSP-architecture mismatch warning remains part of the evidence.

Over the requested 600-second `sustained_high_performance` run:

- 1,342 inferences completed.
- NetRun throughput including tensor file I/O and miscellaneous work was 2.2353 FPS, or 447.37 ms/effective inference.
- QNN graph execute averaged 134.62 ms; HTP accelerator execute averaged 74.05 ms.
- Thermal status remained 0 in all 49 samples. Battery moved from 95% to 93%; sampled battery temperature was 32.2–33.9°C.

This proves real HTP execution. It does not prove an in-App zero-copy service time because `qnn-net-run` is a separate CLI process.

## Why the App route is still rejected

The LiteRT 1.4.2 QNN delegate genuinely entered HTP graph preparation; it did not silently fall back to CPU. The bounded host attempt produced zero frames in 604 seconds. The process continued preparing subgraphs in the background from approximately 16:27 to 16:59, logged nine successful `QnnGraph_finalize` events, and reached an individual `prepare_ms=143505`, but it still did not leave a usable compiled cache.

That cold-start path is operationally unacceptable even though the precompiled DLC route itself is healthy.

## Matched clean downstream check

The long NPU performance run reused the historical gradient canary input. Its output is valid for shape and performance but intentionally has no trustworthy floor geometry; therefore its earlier `0/N` downstream-valid count is not an algorithm failure.

A frozen clean corpus tensor was then run through the same cached HTP DLC. Its output SHA-256 is `5A3AC...28F4`. The frozen ground geometry and scale student accepted 12/12 awake-lockscreen frames and 15/15 dozing-lockscreen frames.

The existing Android CPU boundary remains the bottleneck:

| Condition | CPU-boundary P50 | CPU-boundary P95 | Valid |
|---|---:|---:|---:|
| Lockscreen awake | 1,349.49 ms | 12,710.07 ms | 12/12 |
| Lockscreen dozing | 1,275.29 ms | 12,668.75 ms | 15/15 |

Most median cost is the current Kotlin cubic preprocessing (1,193.39 ms awake; 1,142.36 ms dozing). RANSAC/features add about 125–127 ms; the frozen student head is negligible.

Adding the measured CPU-boundary median to QAIRT's effective CLI time gives only a staged sequential estimate:

- Lockscreen awake: 1,796.85 ms, about 0.557 FPS.
- Lockscreen dozing: 1,722.66 ms, about 0.580 FPS.

The repeated matched check did not establish a deterministic awake-versus-dozing median difference. Both conditions had roughly 12.7-second P95 tails. Locking the phone did not disconnect wireless ADB or stop HTP, but it is not a guaranteed low-latency service condition.

## Routing consequence

- Per-frame primary depth: reject.
- Current synchronous fallback: reject.
- Periodic auxiliary observation: retain as a deployment candidate.
- Disagreement detector: retain as a deployment candidate.

The next useful engineering experiment is an in-App precompiled/cached QNN integration with production image preprocessing and buffer reuse. It is not another depth model search and does not authorize rescuing the observed ARKitScenes visit by threshold tuning.

Claim ceiling: device deployment and performance diagnostic only. No CameraX capture-to-result, energy, accuracy, safety, scale-free role, or production authority.
