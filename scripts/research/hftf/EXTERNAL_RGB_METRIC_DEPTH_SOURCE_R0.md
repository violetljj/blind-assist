# External RGB metric-depth source R0

This is the depth-only admission path for the D44 successor. It deliberately
uses one shared person torso ROI for all three depth models so detector and
tracker errors do not decide the depth-source comparison.

## Capture

Calibrate the exact RGB camera first and retain `fx fy cx cy` for the recorded
resolution. Record one controlled person with an unobstructed torso:

- static camera, static person at 1, 2, 3, and 5 metres;
- static camera, approach, recede, and lateral motion;
- only after those work, repeat the motion cases with a walking camera.

For static cases, measure optical-axis depth from the camera optical centre to
the torso plane and pass `--truth-depth-m`. For motion cases where per-frame
metric truth is unavailable, pass only the observed direction with
`--truth-direction`; those rows contribute to direction accuracy, not metric
distance error.

Extract 10 FPS frames and shared torso ROIs:

```powershell
$env:PYTHONPATH = 'E:\linnan\linnan\artifacts.local\vendor\python-packages-hftf-metric-depth-r0'
E:\codex-tools\venvs\riskseg-r0-py311\Scripts\python.exe `
  E:\linnan\linnan\scripts\research\hftf\prepare_external_rgb_metric_depth_manifest.py `
  --input-video <video.mp4> `
  --output-dir <artifacts.local/output/sequence> `
  --weights E:\linnan\linnan\artifacts.local\models\yolo11n.pt `
  --sequence-id <unique-id> `
  --scenario static `
  --camera-motion static `
  --truth-depth-m 2 `
  --intrinsics-fx-fy-cx-cy <fx> <fy> <cx> <cy> `
  --target-fps 10 `
  --device 0
```

The controlled scene must contain one intended person. The preparer selects the
largest COCO person in each sampled frame and fails immediately on a missed
detection instead of silently changing targets.

## Three-arm run

One command accepts one or more prepared manifests:

```powershell
pwsh E:\linnan\linnan\scripts\research\hftf\run_external_rgb_metric_depth_triarm.ps1 `
  -Manifest <static-1m/manifest.jsonl>,<static-2m/manifest.jsonl> `
  -OutputDir <artifacts.local/output/triarm>
```

The output `triarm-report.json` contains valid fraction, metric error where
truth exists, direction accuracy, static jitter, seven-frame availability,
cold/steady-state latency, process RSS, and CUDA peak allocation. Metric3D is
currently run through FP32 ONNX on CPU because the installed ONNX Runtime CUDA
wheel cannot execute on this RTX 5060; its latency is therefore not comparable
to the two PyTorch CUDA arms.
