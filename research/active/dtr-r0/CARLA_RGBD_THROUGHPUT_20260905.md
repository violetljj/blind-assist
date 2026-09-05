# CARLA 720p RGB/depth engineering probe

Status: short synchronized capture and lossless encoder checks PASS.

The user's next action was a small capture stability/throughput check after the
R1 source interruption. This uses two Town10HD_Opt camera paths, 50 samples each,
1280x720, 0.1 s timestep, DX12 Low, and a persistent server. No R1 source seed,
method prediction, fitting, final truth, or algorithm score is used.

## Observed result

| Comparison | Reference capture | Candidate capture | Interpretation |
| --- | ---: | ---: | --- |
| Serial RGB then depth versus synchronized RGB/depth | 57.23 s | 59.13 s | Synchronization alone did not improve throughput |
| Native PNG versus fast lossless PNG, both synchronized | 63.28 s | 22.00 s | 2.88x throughput, 65.23% less capture time |

Each side writes 200 images representing 100 RGB/depth pairs. Capture time
includes tick/receive, PNG encoding/writing, hashing and per-frame checks, and
excludes camera warmup. The encoder comparison uses one server and the same
camera paths, with native first and fast second. Native versus fast warmup was
0.47 versus 0.51 s; tick/receive was 2.30 versus 2.25 s; PNG write/hash was 55.03
versus 13.73 s. RPC/world readiness took 16.86 s in that run. This is one ordered
small comparison, not a statistical throughput claim or long-run reliability
proof. It does not quantify the old runner's restart overhead under matched load.

The new encoder preserves native BGRA values as an RGBA PNG, using lossless
zlib level 1 and no per-row filtering. An independent Pillow decoder verified
all 400 actual output images against hashes of their own capture-time raw RGBA
bytes: zero pixel mismatches. Every paired frame ID, timestamp and camera pose
matched. Native and fast camera paths matched. Encoded output grew from
259,205,974 to 282,435,489 bytes, an 8.96% increase. Depth byte values are preserved;
this does not validate downstream metric-depth decoding or avoidance accuracy.

## Implementation and evidence

- `carla/fast_sensor_png.py`: reusable lossless encoder.
- `carla/probe_carla_sync_rgbd.py`: bounded persistent-server comparison, with
  exclusive output, finite startup/frame waits, owned-server cleanup and port
  release verification. Use `--comparison png` for native/fast synchronized A/B.
- `carla/validate_carla_png_probe.py`: full inventory, independent pixel decode,
  paired frame/time/pose and cross-run camera-path verification.
- Two focused codec tests pass, including independent decoding of randomized
  channel/alpha values and rejection of malformed input.

Payloads are under `artifacts.local/runtime/carla-asset-library/experiments/`:
`rgbd-sync-probe-warmup-v3-20260905` owns the serial/sync result;
`rgbd-fast-png-probe-20260905` owns the encoder comparison and
`pixel-validation.json`. Probe source snapshots and hashes are retained.
Task-owned servers, ports and capacity leases were released after every attempt.

The first DX11 and DX12 attempts timed out in a faulty readiness loop that
reused a client created before RPC startup. They cannot establish backend
failure or lack of DX11 support. Fresh client creation after the port opens
fixed readiness. A subsequent camera warmup attempt timed out before measurement;
the retained asynchronous warmup then synchronous capture sequence resolved it.
All three failed probe directories remain retained. No new native engine crash
was observed; short success does not establish a fix for historical shader crashes.

The fast encoder is available as an engineering component. Frozen R1 capture
code and evidence remain unchanged, and R1 remains not evaluable. Experiment
index registration remains blocked by the existing line-252 input fingerprint
mismatch; this is a working engineering record, not a new high-authority terminal.
