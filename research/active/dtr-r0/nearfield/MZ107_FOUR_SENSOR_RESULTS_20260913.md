# MZ107: four-sensor path restored; no qualified algorithm gain

2026-09-13, EXPLORE, `NO_QUALIFIED_GAIN`. The actual input is **one RGB camera +
ToF + Radar + IMU**. Preserve the three-sensor comparator; do not promote the
fixed image-association rule. This is a completed controlled canary, not a tested
general four-sensor obstacle system or an algorithmic breakthrough.

## Paired result

12 new controlled UE scenes, 8 synchronized frames each, 96 total. Both arms use
the same full8x8 ToF packets, Radar horizontal range/bearing and integrated IMU
yaw. The added mechanism consumes actual640x360 RGB pixels. Radar velocity is
recorded but unused in this current spatial-occupancy readout. No stereo, learned
metric depth, native-depth input or second camera is present.

| Current spatial support | ToF+Radar+IMU | Same inputs + RGB association |
| --- | ---: | ---: |
| TP / FP / FN | 61 / 8 / 3 | 61 / 8 / 3 |
| No support / UNKNOWN | 27 | 27 |
| HEAD TP / FN | 16 / 0 | 16 / 0 |
| Pole TP / FN | 14 / 2 | 14 / 2 |
| False-alarm episodes | 2 | 2 |
| Completely missed positive episodes | 0 | 0 |

No old TP lost, no new TP gained, no FP removed. RGB-disabled reproduces every
baseline decision exactly. Independent ToF support is preserved. These are spatial
frame/episode summaries, not final alarm-state metrics or a comparison against
every historical temporal policy. A missing-support negative is UNKNOWN, not CLEAR.

## What this run actually diagnosed

The fixed brightness/connected-component proposal finds regions in57/96 frames:
BODY16/16, HEAD5/16, pole16/16, offroute16/16, clear0/16, turning4/16. The low HEAD
and turning coverage exposes a weakness of this simple visual extractor even in
controlled scenes. It does not prove that RGB lacks the required information.

Of31 accepted Radar--image--ToF associations,29 have matching native-surface
identity evidence;2 wrongly attach a simulated persistent Radar ghost to a real
object (`turning_1_00`, `turning_1_01`). The evaluator reconstructs exact native
ToF ray endpoints and requires unique membership in the same native bounds as
the Radar origin. This information is unavailable to the predictor. A compatible
angle and distance, even with only one eligible region, do not prove identity.

All8 baseline FP are persistent simulated Radar ghosts:4 in an offroute scene,
4 in a clear scene. No baseline FP in this panel is caused by a real offroute
object's coarse bearing. Therefore this panel has **no observed real-object FP
opportunity for the proposed angular refinement**; its zero FP gain cannot rule
out that mechanism on suitable scenes. Missing RGB never vetoes Radar, so those
unsupported ghosts remain. The sole Radar support change correctly adds a pole
return already covered by independent ToF; there is no frame-level gain.

## Boundaries and corrected next decision

The engineering closure is real: fresh synchronized four-sensor input, actual
image-derived regions, explicit UNKNOWN, paired evaluation and source-identity
audit now exist. The tested matching rule remains an intended NEGATIVE_CONTROL
for promoting unique angle/range matching to reliable identity. Keep the source
and comparator as Development diagnostic infrastructure.

Any next bounded attempt should first address **observable visual-region coverage
and competing association explanations**, and include real objects near corridor
boundaries as well as ghost/multiple-object cases. The current panel is consumed
Development and can diagnose/regress; a fixed useful method still needs a complete
new-scene check. Do not turn this result into another depth-model replacement,
silently remove Radar, infer accurate translation from IMU, or claim that all
four-sensor approaches failed. No successor, threshold sweep or training was run.

## Execution and reproducibility

- [Protocol](MZ107_FOUR_SENSOR_PROTOCOL_20260913.md),
  [source](mz107_source_spec.py), [sensor proxy](mz107_sensors.py),
  [single-camera collector](mz107_single_rgb_capture.py),
  [observable predictor](mz107_rgb_association.py),
  [paired runner](run_mz107_four_sensor.py).
- Canonical evidence:
  `artifacts.local/work/mz107-rgb-tof-radar-imu-20260913/capture-v1` and
  `analysis-v1`; raw packets, RGB hashes, separate native/provenance files,
  sealed predictions, identity audit, summary and posthoc diagnosis retained.
- Source spec SHA256:
  `578de9513e0b17c14ab0ad3706267f55adc21ec4be01a78cd995eb85cf292e57`.
- Actual UE5.8.2 D3D12 rendering;96-frame capture63.672s excluding cold shader
  startup. Hypothetical Radar noise/dropout/ghost proxy, not RF or hardware.
- OpenCV4.10 CPU proposal/association averages0.728ms/frame with RGB preloaded;
  excludes image IO, sensor transport and rendering. Backend receipt uses
  `GPU_BACKEND_UNAVAILABLE` (installed OpenCV CUDA device count0); no CUDA claim.
- Six focused tests pass, including192 source/predictor ray comparisons, true
  pixel-driven refinement, RGB-off fallback, range conflict, independent ToF and
  UNKNOWN. An independent focused audit also checked200 randomized rows and133
  ToF-positive preservation cases. ToF pinhole-angle mismatch was fixed before
  predictions; maximum audited ray error2.50e-16. No post-score model change.
- Structured registration/inheritance disposition is pending the pre-existing
  knowledge-ledger fingerprint error; the exact attempt and intended role are
  retained locally. This is a metadata gap, not a successful registry update.

```powershell
& 'E:/codex-tools/tools/venvs/blindassist-torch-gpu/Scripts/python.exe' research/active/dtr-r0/nearfield/run_mz107_four_sensor.py --capture artifacts.local/work/mz107-rgb-tof-radar-imu-20260913/capture-v1 --output artifacts.local/work/mz107-rgb-tof-radar-imu-20260913/analysis-replay-new
```

Replay is consumed Development; choose a fresh output path and preserve the
original sealed result. Task-owned capture processes are released; the ignored
DDC is retained as a reusable shader cache, owned by this local simulation task.
