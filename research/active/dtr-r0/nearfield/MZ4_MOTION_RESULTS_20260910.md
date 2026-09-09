# MZ4: most motion gain comes from additional views; joint constraints add14 cases

2026-09-10 EXPLORE. [Fixed protocol](MZ4_MOTION_PROTOCOL_20260910.md).
Known-pose moving8x8 observations resolve both BODY/HEAD memberships in495/500
finite-family cases, versus224 stationary and481 when moving frames are decoded
separately. Retain the conditional information mechanism, not a deployable method.
No RGB network, MZ1 checkpoint, original alert, or previous experiment was changed.

## Same observation budget, different use of motion

The500 cases reuse the previous single-zone study's exact zero/one/two-patch
catalog selection, including300 HEAD positives and122 BODY positives. BODY and
HEAD can overlap; positive denominators are different. Known absence below means
absence within this finite scene family only, never a clearance assertion.

| Method | Both memberships known and correct /500 | BODY positive resolved /122 | HEAD positive resolved /300 | Any UNKNOWN /500 | Empty feasible set |
| --- | --- | --- | --- | --- | --- |
| Initial8x8 frame |89|49|126|411|0|
|25 stationary repeats, joint |224|75|178|276|0|
|25 moving frames, independent query conclusions accumulated |481|112|295|19|0|
|25 moving frames, exact-pose joint scene constraints |495|120|297|5|0|
|Same moving inputs, assume first pose throughout |5|0|2|495|494|
|Same moving inputs, fixed+0.5degree yaw error |154|25|80|346|338|

Matched single/stationary/framewise/joint methods make zero false-positive or
false-negative assertions, because the true scene remains among feasible scenes.
This is a verified consistency property of the matched family, not independently
measured reliability. The two mismatched-pose arms also made no wrong assertions
in this sample, but their collapsed coverage is not successful inference.

Stationary repeats recover missing zones under the shared20percent dropout.
Moving framewise conclusions resolve257 more whole cases than stationary;
scene-level joint inference adds another14 cases, including8 additional BODY
positive and2 additional HEAD positive resolutions. Joint inference loses no
previously resolved query; median feasible scene count falls from200 stationary
to2 joint. Framewise median90 is the minimum per-frame scene count, not a jointly
retained scene set. Most observed benefit therefore comes from sampling new
views, not uniquely from the joint solver. HEAD-only extra benefit of joint
inference is particularly small here:295 to297 of300.

## What is being measured

Each zone integrates solid-angle coverage of foreground patches over a45x45degree
FoV. A foreground return occurs when its total coverage is>=2percent. Returned
distances are reduced to foreground/background range bands. This intentionally
discards within-band metric detail and retains no true object identity as input.
Successive observations may originate from different patches; inference retains
all compatible zero/one/two-patch scenes rather than forcing one tracked object.

The distinct background-only return law resolves0/500 cases in all three tested
single/stationary/moving-joint arms. Motion cannot recover foreground information
when the assumed output stream never carries it. These are two hypothetical laws,
not a fitted sensor response or a claim about VL53L8CX target selection.

The0.5degree bias leaves338/500 sequences inconsistent with every permitted scene.
This exposes a brittle exact compatibility assumption. The finite integer-grid
catalog itself contributes to that brittleness; it is not a measured permissible
pose-error threshold for real hardware. Free head/body motion, uncertain pose,
translation, arbitrary objects, dynamic occlusion and real return statistics are
not included. Background X changes from3.5m in the older narrow-cone study to3.25m
here to keep the enlarged footprint below4m radial range; query geometry is
unchanged. This is generic12Hz simultaneous8x8 sampling, not a sensor timing model.

## Decision in combination with MZ3

[MZ3](MZ3_ERROR_ATTRIBUTION_RESULTS_20260910.md) shows that learned fusion gains
148 exact rows over MZ0 while losing105; all12 fusion wrong-far errors are new
relative to MZ0. Retain MZ0/component and MZ1/challenger. No truth-derived router,
center-support veto, training extension, or automatic model replacement follows.

MZ4 supports retaining motion as a source of additional angular observations,
with a small measured extra value from joint feasible-scene reasoning in this
family. Do not begin with a large temporal network. A future mechanism should
first represent uncertain correspondence/pose and conflicting evidence, and
justify added complexity against the strong moving-framewise control. This
experiment ends here: no amplitude sweep, robustness rescue, or new fit.

## Reproduction and verification

```powershell
$env:CUBLAS_WORKSPACE_CONFIG=':4096:8'
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' research/active/dtr-r0/nearfield/mz4_multizone_motion.py --output artifacts.local/work/mz4-multizone-motion-20260910/fresh-reproduction
& 'E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe' research/active/dtr-r0/nearfield/audit_mz4_multizone_motion.py --run artifacts.local/work/mz4-multizone-motion-20260910/fresh-reproduction
```

Successful evidence: `artifacts.local/work/mz4-multizone-motion-20260910/run-v2/`.
`observations.npz` retains all catalog codes, selected scenes, noise/dropout and
per-case/per-frame decisions; `geometry.npz` retains footprints and coverage.
Protocol, input, operator and artifact hashes are recorded. Tensor placement
uses the actual mismatch operation: CPU median12.49ms versus CUDA0.315ms for
the declared candidate-block probe. Complete measured run4.35s on RTX5060 Laptop;
this is offline batched finite-catalog computation, not end-device latency.

Independent NumPy geometry checks49,159,680 code entries over6 poses for each
orientation model, with maximum coverage difference3.89e-16. Python integer
bitsets independently reproduce all4000 sequence rows and12500 individual frame
rows; all128020 physical membership labels and aggregate changes agree. These
checks establish implementation consistency, not scene or sensor validity.

The first launch failed before scientific outputs because deterministic CUDA
does not implement weighted `bincount`. It was replaced by fixed per-zone sums;
scientific settings were unchanged. `run-v1/` preserves its launch/source/backend
receipt and `launch-v1.log` preserves the error. Do not overwrite that attempt.
Both commands exited; no task-owned process, device connection or worker remains.
