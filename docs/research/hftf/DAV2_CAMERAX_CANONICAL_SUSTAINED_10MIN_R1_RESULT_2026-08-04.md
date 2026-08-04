# DA V2 canonical CameraX sustained ten-minute R1 result

Decision: `CANONICAL_CAMERAX_FULL_PIPELINE_SUSTAINED_R1_SUPPORTED_DEVICE_ONLY`;
`PRODUCTION_PROMOTION_NOT_AUTHORIZED`. The bit-exact canonical preprocessing
route passed the frozen ten-minute CameraX scheduling, latency, resource, TTL,
screen, and thermal gates on the supported device. This closes the deployment
evidence gap left by R0, which used the old fast/fused diagnostic route. It does
not change the historical R0 result or establish accuracy, metric geometry,
cross-device generalization, safety, or default-App authority.

## Frozen execution

The run used commit `5f73f54d8c6810310b51bdb336d61342f5673cac` on
`SM-S9280 / SM8650 / Android 16`. The test explicitly invoked
`preprocessFp16CanonicalStrict()` and reported route
`canonical_native_official_fp32_then_integer_rnte_fp16_v1`. The cached DLC SHA-256
was `2BB02F37FEF177FF4B02B8EE0C416EE9FF998BCEEF9786B92959E1F682EBAA24`.
The App and test APK SHA-256 values were respectively
`BD6223C5164FF30593B5D36C72D179C7875C483FDE2B44F8332B096BDEB418A1` and
`74C813043096FD7830FD634D7AA8D545585F06ED2C1DB8CC5A9CEA538667EB48`.

The same R0 scheduling contract was retained: real `640x480 YUV_420_888`
CameraX with `KEEP_ONLY_LATEST`, five seconds of scheduler stress, then 595
seconds requesting depth every 500 ms, TTL 750 ms, maximum one in-flight depth
task, three owned YUV slots, geometry enabled, screen continuously interactive,
and severe thermal status fail-closed. Before execution, R1 additionally froze
canonical route identity, preprocess-plus-QNN P95 `<=250 ms`, full-pipeline P95
`<=350 ms`, and fresh-result-age P95 `<=750 ms`.

| Resource or scheduling measure | Result |
|---|---:|
| frames seen / `ImageProxy` closed | 8,993 / 8,993 |
| submitted / processed / pending replaced | 1,181 / 1,143 / 38 |
| paced submissions after stress arm | 1,117 |
| maximum concurrent depth tasks | 1 |
| YUV slots available after close | 3 / 3 |
| geometry VALID / UNKNOWN | 1,143 / 0 |
| fresh / stale-suppressed results | 1,142 / 1 |
| TTL expiry probe | `UNKNOWN(EXPIRED)` |
| noninteractive observations | 0 |
| thermal before / maximum / after | 0 / 0 / 0 |

| Latency | P50 | P95 | Max | Frozen P95 gate |
|---|---:|---:|---:|---:|
| canonical preprocess + cached QNN | 95.87 ms | 99.00 ms | 104.22 ms | <=250 ms |
| full depth + geometry | 188.30 ms | 195.23 ms | 223.79 ms | <=350 ms |
| fresh result age | 206.51 ms | 215.71 ms | 277.52 ms | <=750 ms |

PSS changed from 244,669 to 277,112 KiB; Java heap from 18.89 to 26.78 MB
and native allocated heap from 17.60 to 52.97 MB. ART recorded 81.72 MB allocated,
12 GCs, and 314 ms GC time. These are endpoint observations rather than a leak
slope, but every `ImageProxy` closed, all three owned YUV slots returned, native
resources completed teardown, and no base or R1 gate failed.

## Evidence

- Runner: `scripts/research/hftf/run_camerax_canonical_sustained_10min_r1.ps1`
- Bundle: `artifacts.local/evidence/hftf/camerax-canonical-sustained-10min-r1-20260804-222958/result.json`
- Result SHA-256: `CB4AB3BB31DAD65BD62C6A53D8DD2C5411135DA6A3F3B9AAAF28BD293C2BF16E`
- Gate: `artifacts.local/evidence/hftf/camerax-canonical-sustained-10min-r1-20260804-222958/canonical-sustained-gate.json`
- Gate SHA-256: `17C8FECE8F54A7DF25B06BFC32E599C2F22D769295BEB844C04C13780DFBD1E0`

The canonical deployment loop is therefore complete only for this supported
device and fixed runtime. The old fast/fused path remains a diagnostic control;
its R0 performance evidence is not relabeled or promoted.
