# Motion occupancy A0.1 candidate CLI

Date: 2026-08-03

This is the first candidate-only runnable form of the supported current
occupancy branch. It accepts ordinary RGB frames and calibrated intrinsics; it
does not need ARCore, phone depth, object classes, boxes, or sensor truth.

For a video, first extract the frozen 10 FPS manifest:

```powershell
python scripts/research/hftf/prepare_external_rgb_video_manifest.py `
  --video INPUT.mp4 `
  --frames-dir artifacts.local/candidate-input/frames `
  --manifest artifacts.local/candidate-input/manifest.jsonl `
  --sequence-id external-camera-001 `
  --target-fps 10 `
  --intrinsics-fx-fy-cx-cy FX FY CX CY
```

Then run the frozen candidate:

```powershell
python scripts/research/hftf/run_motion_occupancy_a0_candidate.py `
  --manifest artifacts.local/candidate-input/manifest.jsonl `
  --model scripts/research/hftf/MOTION_CONDITIONED_OCCUPANCY_A0_1_FROZEN_MODEL.json `
  --raft-weights artifacts.local/models/hftf/torch/optical-flow/raft_small_C_T_V2-01064c6d.pth `
  --unidepth-repo artifacts.local/vendor/UniDepth `
  --output-jsonl artifacts.local/candidate-output/occupancy.jsonl `
  --summary-output artifacts.local/candidate-output/summary.json `
  --output-video artifacts.local/candidate-output/occupancy.mp4
```

The MP4 band strip is schematic. Every output is explicitly current occupancy,
not the unsupported 0.5-second future field. A candidate run has no labels and
cannot by itself establish accuracy; the final external camera still needs
controlled distance and geometry reference capture.
