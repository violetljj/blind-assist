# DA V2 CameraX sustained ten-minute R0 result

Decision: `CAMERAX_FULL_PIPELINE_SUSTAINED_R0_PERFORMANCE_SUPPORTED_DEVICE_ONLY`; `PRODUCTION_PROMOTION_NOT_AUTHORIZED`. The full camera/QNN/geometry runtime passed its ten-minute resource, latency, scheduling, screen, TTL, and thermal gates. It used the non-promoted fused FP16 preprocessing arm whose strict depth parity already failed, so this result cannot rescue that arm or establish accuracy/safety semantics.

## Frozen run

The USB-connected `SM-S9280 / SM8650 / Android 16` remained interactive for 600 seconds. CameraX continuously produced real `640x480 YUV_420_888` frames with `KEEP_ONLY_LATEST`. The first five seconds saturated the one-running/one-pending scheduler; the remaining 595 seconds requested precise depth every 500 ms. Each processed frame ran orientation/crop/YUV conversion, FP16 preprocessing, persistent cached-context QNN, FP16 depth decode, align-corners resize to 640x480, and the equivalent frozen geometry.

USB stay-awake was enabled temporarily and restored from `7` to `7` afterward. Severe thermal status fails closed, expired results return explicit `UNKNOWN(EXPIRED)`, and teardown unbound CameraX and closed the scheduler, native preprocessor, QNN context, and buffers.

| Resource/scheduling measure | Result |
|---|---:|
| frames seen / ImageProxy closed | 8,993 / 8,993 |
| submitted / processed / pending replaced | 1,181 / 1,144 / 37 |
| paced submissions after stress arm | 1,117 |
| maximum concurrent depth tasks | 1 |
| YUV slots available after close | 3 / 3 |
| geometry VALID / UNKNOWN | 1,144 / 0 |
| fresh / stale-suppressed results | 1,143 / 1 |
| noninteractive observations | 0 |
| thermal before / maximum / after | 0 / 0 / 0 |

| Latency | P50 | P95 | Max |
|---|---:|---:|---:|
| YUV copy | 6.45 ms | 20.75 ms | 23.81 ms |
| YUV-to-FP16-plus-QNN | 81.33 ms | 86.88 ms | 96.95 ms |
| FP16 decode, resize, geometry | 91.58 ms | 118.39 ms | 140.16 ms |
| full depth plus geometry | 174.70 ms | 202.69 ms | 220.58 ms |
| result age | 184.83 ms | 218.47 ms | 255.49 ms |

PSS changed from 236,977 to 256,186 KiB; Java heap from 14.36 to 20.20 MB and native allocated heap from 9.63 to 23.17 MB. ART recorded 81.17 MB allocated, 14 GCs, and 321 ms GC time across the run. These endpoints do not prove a leak slope, but no owned CameraX/YUV/native resource remained open and there was no latency or thermal gate failure.

The paced arm delivered about 1.88 submissions/s, consistent with the intended 1-2 Hz precise-depth role. The full compute P95 is below the earlier 250-350 ms planning envelope, but that is a performance result only. The camera scene happened to yield 1,144 valid geometries; it is not an accuracy corpus and carries no safety authority.

## Evidence

- Runner: `scripts/research/hftf/run_camerax_sustained_10min_r0.ps1`
- Bundle: `artifacts.local/evidence/hftf/camerax-sustained-10min-r0-20260804-194526/result.json`
- Bundle SHA-256: `D9864887D51D19DF9623655D0376B1256CF11EF74EA567162B82A184CE8A5ABC`

GPU preprocessing remains untriggered: Native CPU preprocessing passed the frozen CPU gate, and no AHardwareBuffer/dma-buf handoff measurement was needed. Shared buffer and GPU fencing remain optional later experiments, not prerequisites for this result.
