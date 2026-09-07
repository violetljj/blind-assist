# Native depth and batched settled-pose rendering

Engineering follow-up to the [EXR transport upgrade](UE_CAPTURE_THROUGHPUT_20260907.md).
This upgrades controlled UE acquisition; it does not train or score a model.

## Result

Same48-frame G9 specification, same640x360 RGB/depth, same number of settling
renders, same saved Willow map:

| Mode | Steady frames/s | 47 inter-frame intervals | Median frame interval | Process elapsed |
| --- | ---: | ---: | ---: | ---: |
| Previous single-access Python, tick | 2.772 | 16.953s | 359ms | 78.407s |
| First upgrade EXR, tick | 3.984 | 11.797s | 250ms | 74.078s |
| Native NPY, burst | **9.342** | **5.031s** | **109ms** | **63.562s** |

The new mode is **2.35x** the first upgrade and **3.37x** the Python comparator.
Steady acquisition time falls57.4% versus the first upgrade. Startup, map loading
and shutdown still dominate this small batch; full process elapsed falls14.2%.
One run per production mode, with earlier comparator records explicitly reused.
No endurance or hundreds-of-thousands-frame run was performed. Linear steady-rate
extrapolation gives about2.97hours per100,000frames for this workload; it is not a
measured long-run guarantee.

Native export median was5.681ms. A separate24-frame same-target probe produced
**byte-identical NPY files** versus Python single-access conversion for all
5,529,600 depth pixels. Probe medians were167.607ms Python versus5.845ms native.
That probe intentionally exports twice and is not a production throughput run.

Across the48 production frames, comparison to the prior EXR/tick RGB gives
global RGB MAE median0.1145/255, maximum1.0143/255. Maximum object-region MAE is
0.09935/255. Every tested projected object pixel agrees with the analytic native
surface within3cm (minimum per-object coverage1.0). Same-target depth conversion
is byte exact; cross-run RGB is approximately consistent, not claimed identical.

## What changed

- [Native plugin](native_capture/BlindAssistCapture/BlindAssistCapture.uplugin)
  calls the same raw render-target read, applies the same finite/0<cm<10000 rule,
  divides in double precision then casts to float32, and writes the same NPY
  header and row order. It closes a new partial file and renames it only after
  successful writing; existing outputs and partial files are not overwritten.
- `burst` issues the original number of capture renders in one callback. The
  unused editor viewport's realtime override is disabled for that owned process.
  It does not reduce resolution, remove warmup renders or change postprocessing.
- TAA view sample indices still advance per render, but global engine/world time
  does not advance as it would across separate ticks. This is a **settled-pose
  acquisition profile**, not a real-time physics/video capture equivalence claim.
  For time-dependent materials, continuous simulation or a frozen historical
  protocol, explicitly select `--cadence tick` unless separately validated.
- Existing legacy launchers keep their original behavior. The new launcher
  auto-selects burst for grounding/factorial poses and keeps tick for whisker
  motion; an explicit `--cadence` overrides that selection. Auto export selects native when a plugin is supplied and
  otherwise uses the existing EXR path. EXR's eight-frame host queue bound and
  failure handling remain; native writes finish synchronously with no host queue.

The32-frame cadence experiment used two groups of all four object-presence
variants, with reference A / viewport-off / burst / reference B. At26 settling
renders, median frame cost was497.069 /492.522 /237.784 /493.718ms. Viewport-only
change did not materially improve throughput. Burst object-region MAE versus A
was at most1.027/255, comparable to repeated-reference1.058/255; near depth matched
exactly. This motivated the complete48-frame native/burst validation above.

An additional32-frame lower-settling diagnostic is **not adopted**. Two-extra-render
and zero-extra-render candidates did not pass its complete quality criterion.
The zero-extra-render candidate also exceeded the5/255 object/removed-object ROI
limit (7.354). Whole-near-scene exactness failed even for its repeated reference;
that part of the diagnostic is not evidence of candidate-specific geometry loss.
The delivered profile keeps the original settling counts. Failed/ambiguous
diagnostics remain retained, not silently relabeled as successes.

## SDK, build and use

The user-authorized .NET Framework4.8 SDK installation succeeded through normal
Windows UAC, installer exit0. Registry, `mscoree.h`, and x64 `mscoree.lib` were
verified. No installer remains running. The plugin built successfully with the
installed UE5.8/MSVC toolchain, UAT exit0; the shared project was not modified.

```powershell
python tools/build_ue_capture_plugin.py `
  --output artifacts.local/work/my-native-plugin

python tools/ue_native_capture.py capture --capture grounding `
  --spec artifacts.local/work/my-capture/spec.json `
  --output artifacts.local/work/my-capture/output `
  --plugin artifacts.local/work/my-native-plugin/package/BlindAssistCapture.uplugin `
  --timeout 1800
```

The already built plugin on this host is at
`artifacts.local/work/ue-capture-upgrade-20260907/native-build-v2/package/BlindAssistCapture.uplugin`.
It is loaded only by the owned capture process through `-PLUGIN` and
`-EnablePlugins`; no project or engine plugin installation is required.
`BA_UE_CAPTURE_PLUGIN` can supply the descriptor instead of `--plugin`.
Use `--depth-export native_probe` for a small same-target parity check.
The launcher snapshots code/spec and records the loaded DLL hash.

Native mode needs host `psutil`; the fallback additionally needs NumPy/OpenEXR
from the existing [requirements file](../nearfield/requirements-ue-capture.txt).
No binary, PDB, capture payload or SDK package is committed to Git.

## Evidence and limits

- `artifacts.local/work/ue-capture-cadence-20260907/`: cadence probe,
  `analysis.json`, native parity probe,48-frame production, RGB/geometry checks,
  and the retained `lowwarm/` diagnostic.
- `artifacts.local/work/ue-capture-upgrade-20260907/sdk-install-elevated/`:
  successful installation and SDK verification.
- `artifacts.local/work/ue-capture-upgrade-20260907/native-build-v2/`:
  successful compilation, binary hashes and process-release evidence.

Results apply to the tested fixed Willow geometry and procedural objects, not
arbitrary UE scenes. The validated fastest production run uses the grounding
capture adapter; the other two adapters share the opt-in cadence/export changes.
This still does not implement resumable dataset shards, a persistent cross-job
engine or asynchronous GPU readback. All successful capture maps were unchanged
and observed owned process trees were released.
