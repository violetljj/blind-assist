# UE becomes the primary avoidance Development laboratory

Date: 2026-09-06. User-directed migration; no algorithm accuracy promotion.

## Decision and operating scope

Use the self-built UE5 StreetLabV4 for new pedestrian-avoidance Development.
CARLA is retained for historical evidence and explicitly justified supplemental
checks. Do not spend the next research cycle restoring its missing assets or
continuing the parked R1 dropout-window experiment. Preserve existing CARLA
payloads, frozen failures and component results; do not delete or rename them.

This replaces the default research workflow. It does not claim complete CARLA
feature parity, transfer its historical scores, or finish porting every arm of
the pending eleven-method comparison. UE currently has an incremental X73
perception executor and measured `DEPTH_ONLY` motion reference; X94/X95 retain
their existing recorded authority. Candidate DTR motion remains a challenger
because its last paired comparison showed no incremental success gain.

## One entrypoint

Use the project Python with CUDA PyTorch, Ultralytics, NumPy, Pillow and psutil.
`tools/run_obstacle_research.py` supports:

```powershell
python tools/run_obstacle_research.py status
python tools/run_obstacle_research.py replay --output artifacts.local/unreal/<new-replay>
python tools/run_obstacle_research.py closed-loop --output artifacts.local/unreal/<new-loop>
python tools/run_obstacle_research.py compare --output artifacts.local/unreal/<new-comparison>
python tools/run_obstacle_research.py calibrate --output artifacts.local/unreal/<new-calibration>
```

Replay uses the existing 733-frame sensory dataset and incremental prediction.
It tests perception on recorded movement, with no counterfactual motion score.
Closed-loop uses V4 and `DEPTH_ONLY`; `--controller-mode` selects other existing
research controllers. `--split development` selects the existing Development
bank. `--case` performs a subset smoke check while the evaluator retains the
full denominator. Compare runs the existing paired candidate-action experiment.

The engine is resolved from `--engine`, then `UE_ENGINE_ROOT`, then the Epic
installation matching the project's EngineAssociation. There is no fallback to
another simulator or older map. Status checks local file existence only; actual
run entrypoints retain their source, model and input validation. The underlying
closed-loop launcher's default map has also changed from V2 to V4.

All outputs are exclusive and routed through `artifacts.local`. Existing live
launchers own process cleanup and resume identity checks. The wrapper invokes
their existing CLI code in-process, preserving those rules. Held-out cases are
not exposed by the default wrapper and were not consumed during migration.

## Actual migration checks

- Full fixed-input replay: **733 frames, 52.65 s**, CUDA detection 20.42 s and
  incremental updates 16.69 s. All 733 outputs match the previous same-input
  replay on shared fields. This is regression evidence, not a speed comparison
  with CARLA and not a new accuracy claim.
- Live V4 smoke: `occluded_crossing_collision`, two newly executed branches,
  96 RGB-D frames. Straight control contacted the proxy at an 8.0 s trajectory;
  the depth-controlled branch reached its goal without contact in 10.8 s.
  Only one of eight cases was run; the evaluator correctly reports the suite
  as `INCOMPLETE`. This verifies routing and execution, not an 8/8 result.
- Native depth check: six known fronto-parallel planes, 1/3/6 m at pitch
  0/-10 degrees, 640x360 and 100-degree horizontal FOV. The exact SceneCapture
  depth path used by the live capture passed center and off-axis patches with
  **maximum sampled error 0.000276 m**, below the preset 0.02 m limit. All six
  serialized float32 payloads independently passed hash, shape, validity and
  numeric checks. This tests ideal-plane axial depth and units; it does not
  establish that error bound on arbitrary materials or natural surfaces.
- Three CLI routing tests passed. The initial in-process replay failed before
  creating output because its import directory was absent; adding the target
  directory fixed it, and the full replay above exercised the correction.
- Both owned editor runs and the sensor worker exited, the worker port closed,
  and the V4 map hash remained unchanged from the live-run identity.

Evidence under `artifacts.local/unreal/`:
`ue-primary-replay-20260906`, `ue-primary-live-smoke-20260906`, and
`ue-primary-depth-calibration-20260906`. The latter uses a disposable blank world
and never saves the V4 map or its temporary geometry.

## Remaining replacement work, in decision order

1. Verify RGB/depth edge alignment and visual-mesh versus declared contact-proxy
   agreement. Current success labels use swept discs/boxes and vertical zones;
   mesh collisions are disabled and animated limbs are not injury truth.
2. Match each algorithm's required sampling cadence to the UE source, then run
   the smallest useful baseline/challenger contrast. Do not assume historical
   CARLA arms are ported because shared Python functions can be imported.
3. Expand discriminating Development conditions before enlarging the scene for
   appearance alone. Current evidence covers four scripted families, with
   limited repeated runs and limited unconsumed variation. Preserve the simple
   depth controller when added complexity brings no measured task benefit.

The default switch is complete; broad simulator parity and the above validation
gaps remain open. No new learner, held-out evaluation or fresh confirmation was
started. Existing experiment-index registration remains independently blocked by
the line-252 input-fingerprint mismatch; it does not prevent this reversible
workflow migration or grant a new algorithm terminal.
