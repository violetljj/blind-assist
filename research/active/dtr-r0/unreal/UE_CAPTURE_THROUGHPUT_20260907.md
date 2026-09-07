# UE depth acquisition throughput upgrade

Engineering only: existing consumed G9 Development scenes, no training, model
scoring or new research terminal. Saved Willow map, RGB rendering, 640x360
resolution, settling counts, simulation timestamps and final dataset schema
remain unchanged.

## Measured result

On the same 48-frame specification, retaining eight settling frames:

| Measurement | Existing single-access conversion | Native EXR + host conversion |
| --- | ---: | ---: |
| Steady throughput (47 intervals, first frame excluded) | 2.772 frames/s | 3.984 frames/s |
| Steady elapsed | 16.953 s | 11.797 s |
| Median inter-frame interval | 359 ms | 250 ms |
| Median UE depth export | 139.224 ms | 31.642 ms |
| Script elapsed, including map load/warmup | 56.453 s | 50.985 s |
| Owned process elapsed, including startup/shutdown | 78.407 s | 74.078 s |

Steady throughput improves **43.7% (1.44x)** and steady acquisition time falls
**30.4%**. Full small-batch process time falls only **5.5%**, because fixed engine
startup and map settling remain. Each mode ran once, sequentially; this is a
small engineering measurement, not a thermal/storage endurance guarantee.

Linear extrapolation of these steady rates gives about 10.02 -> 6.97 hours for
100,000 frames under the same workload, excluding startup, later processing and
long-run drift. No 100,000-frame run was performed. Other settling schedules,
resolutions and scenes need their own timing; the factorial source currently
uses more settling frames.

Before timing production, 24 same-target frames (5,529,600 depth pixels) were
exported through legacy Python, single-access Python and native FLOAT EXR.
Host conversion matched both NPY references **byte for byte**, including header.
Median probe UE export costs: legacy196.084ms, single-access149.058ms,
EXR29.862ms; host conversion8.142ms. This probe performs three exports per frame
and must not be described as production throughput. The production host
conversion median was9.718ms, overlapping the next frame's rendering.

## Implementation and usage

[Launcher](../../../../tools/ue_native_capture.py) defaults to `exr` and accepts
`whisker`, `grounding`, or `factorial`. The existing launchers retain their
historical behavior. Use this new entrypoint to select the faster path:

```powershell
python tools/ue_native_capture.py capture --capture grounding `
  --spec artifacts.local/work/my-capture/spec.json `
  --output artifacts.local/work/my-capture/output --timeout 1800
```

Run with a Python environment containing the pinned
[dependencies](../nearfield/requirements-ue-capture.txt). On this host the added
OpenEXR/psutil packages are isolated at
`artifacts.local/work/ue-capture-upgrade-20260907/pydeps`; NumPy comes from the
existing bundled Python runtime. No global Python installation was modified.
For another environment, install that requirements file into a project-local
environment under `artifacts.local/work/`.

`--depth-export single_access` or `legacy` provides explicit comparators.
`--depth-export exr_probe` is for a **small** same-target parity probe only; it
retains reference files and does not implement the production queue bound.

The launcher snapshots the capture script, helper modules and specification
before starting an owned UE process. It loads no new plugin and modifies neither
the shared `.uproject` nor the saved map. All outputs use the canonical artifact
junction on F: on this machine.

The production flow is:

1. UE exports raw linear RGBA32F to lossless FLOAT EXR using its existing native
   exporter; a rename publishes the completed file.
2. One host worker reads the named `R` channel, checks FLOAT dtype and dimensions,
   applies the existing finite/0<cm<10000 filter, divides in float64, then stores
   float32 metres with the identical NPY header.
3. Atomic NPY publication acknowledges that frame. At eight outstanding frames
   the producer holds its current simulation state until conversion catches up.
4. After NPY commit, the worker removes only its intermediate EXR. Probe EXRs are
   retained. Per-frame hashes, dimensions, byte counts and timing remain in
   `transport.json`; RGB and NPY are durable outputs.
5. A conversion error fails the acquisition and releases the owned process tree.
   `completion.json` is PASS only after source capture and conversion complete;
   a host failure also changes `receipt.json` to FAIL, retaining the original
   engine receipt separately.

Generate a timing report with
`python tools/report_ue_capture_throughput.py <capture-output> --output <report.json>`.
The report separates probe, steady-state and startup/shutdown costs.

## Validation, evidence and limits

Evidence root: `artifacts.local/work/ue-capture-upgrade-20260907/`.
`summary.json` contains the measured numbers. `exr-probe-v2/` retains all three
same-target representations. `production-exr/` and `production-single-access/`
contain the paired production records; each completed48 frames with unchanged
map hash and released observed process trees. The final snapshot-based launcher
has an additional factorial integration smoke test.

Focused tests cover unusual float values, half-precision rejection, corrupt or
missing files, output preservation, atomic publication, worker completion and
host failure overriding an engine PASS. Depth parity is exact on the same render
target. RGB code is unchanged; separate UE runs are not claimed pixel-identical.
Model/evaluator separation and existing dataset consumers remain intact.

This upgrade does **not** implement restart/resume, packed dataset shards, a
cross-job persistent engine, asynchronous GPU readback, or a demonstrated
hundreds-of-thousands-frame service. The main remaining steady cost is settled
rendering plus RGB export. Preserve image quality when optimizing these next.

A direct C++ NPY prototype was attempted but could not build because the machine
lacks the .NET Framework SDK used by UE's build graph. The authorized installer
attempt requested only `Microsoft.Net.Component.4.8.SDK` and exited5007 because
the current process lacks administrative elevation. No SDK installation is
claimed. That unbuilt prototype and failure logs remain under the evidence root,
outside delivered source; the working EXR path requires no new SDK.
