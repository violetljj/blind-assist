# GPU readback pipeline and settled-history reuse

Follow-up to [background PNG encoding](UE_CAPTURE_ASYNC_RGB_20260907.md).
This is acquisition engineering on reused G9 Development geometry, with no model
training/scoring or claim of new independent evaluation samples.

## Result

480-frame repeated workload, 640x360 RGB plus native float32 depth:

| Pipeline | Acquisition including initial warmup and final drain | Rate | Full engine process |
| --- | ---: | ---: | ---: |
| Previous background PNG, full settling | 39.763 s | 12.072 fps | 99.906 s |
| GPU pair readback + settled-history reuse | **15.500 s** | **30.968 fps** | **76.937 s** |

Throughput is **2.565x**, and acquisition time falls 61.0%. Both runs consist of
ten repeats of the same 48 poses. The previous run is reused as a comparator;
these are single-run measurements, not confidence intervals. Startup/map loading
and post-exit host validation are excluded from acquisition but engine startup
is included in the last column. The new rate includes every frame, first-frame
warmup, and waiting for the last output; it is not a submission-only rate.

The 48-frame component comparison, using the same all-frame timing convention:

| Change | Acquisition fps |
| --- | ---: |
| Previous background PNG | 11.523 |
| Genuine GPU readback, full settling | 15.301 |
| Settled-history reuse, synchronous readback | 18.143 |
| Both | 25.600 |

Earlier reports used intervals after the first completed readback. That convention
is not directly comparable to asynchronous submission timestamps. The reporter
now also emits all-frame acquisition time, with explicit timestamp semantics;
historical all-frame starts are reconstructed from the first frame's profile.

## What changed

- RGB and depth copies are queued together using `FRHIGPUTextureReadback` after
  their capture render commands. Normal submission/polling does not flush the
  render thread or synchronously read render targets. Polling tests GPU fences;
  ready data is copied row-by-row with the actual row pitch and RGB channel order.
- Four bounded in-flight jobs overlap rendering, readback and background encoding.
  Staging buffers are recycled. Output files use no-replace partial writes and
  renames. A failed pair cannot publish successful capture completion, even if
  one file was already renamed. Final drain joins workers and releases staging
  resources before engine shutdown; the owned-process timeout remains the outer
  bound for device/I/O hangs.
- Within a clip, identical camera and object descriptions reuse the converged
  rendering history. Every observation still issues a real RGB render and a real
  depth render. The first frame and every changed clip/camera/object state retain
  their original warmup count. No frames are synthesized by copying old images.
  The 48-frame G9 RGB render count falls from 456 to 200.

In the 480-frame run all 480 pairs completed, zero failed, zero remained pending,
and the queue peaked at its bound of 4. GPU mapping/copy/conversion accumulated
0.448 s, PNG/NPY encoding 11.745 s, and writing 1.530 s. These overlapping counters
must not be added to infer wall time.

## Fidelity and scope

- A 12-frame same-target GPU probe produced pixel-identical RGBA images and
  byte-identical NPY files against synchronous exports (2,764,800 pixels in each
  modality). This checks transport, not temporal equivalence or throughput.
- With full settling, 48-frame maximum object/removed-object ROI RGB MAE versus
  the previous path was 0.178/255. Maximum global MAE was 0.917/255.
- With both changes, all 480 PNGs decoded and all depth arrays passed shape,
  float32, finite and range checks. Maximum global RGB MAE was 3.138/255; maximum
  object/removed-object ROI MAE was **2.220/255**, below the preceding diagnostic's
  5/255 ROI limit. Intended-object analytic depth coverage within 3 cm was 1.0;
  near-field ROI depth agreement within 3 cm was also 1.0, using the union of
  reference/candidate valid depths below 10 m to include unexpected near returns.
- History reuse is **not pixel-identical rendering**: temporal accumulation and
  background appearance differ. An additional full-range ROI comparison for the
  reuse-only run found background differences at 23-63 m behind a removed bar;
  neither run retained bar-depth pixels there. That diagnostic remains retained
  in `reuse48/quality.json`; no whole-scene exact-depth claim is made.

This addresses repeated **static settled poses**, not continuous motion, physics,
time-dependent scene equivalence, or arbitrary unseen assets. No resolution,
material or lighting settings were lowered. The earlier unconditional lower-warmup
diagnostic remains unadopted: this change retains full warmup after state changes.
The 480 repeated frames are a short soak, not multi-hour endurance.

## Defaults and use

`--pair-export auto` selects GPU pairs when the loaded plugin supports the API and
the capture is grounding/factorial with compatible depth settings. Old plugins
fall back to the previous path. Explicit RGB mode selection also keeps that path.
Whisker retains its existing cadence and export path.

`--settling-policy auto` enables reuse only for burst grounding/factorial specs
declaring `THREE_STATIC_SETTLED_POSES_SIMULATED_5HZ_NOT_MOTION_TEST`; other sources
retain full settling. Actual counts and the chosen policy are recorded. Use
`--settling-policy full` to retain original warmups, or additionally
`--pair-export off` to compare the previous readback implementation.

```powershell
python tools/build_ue_capture_plugin.py --output artifacts.local/work/my-gpu-plugin
python tools/ue_native_capture.py capture --capture grounding `
  --spec artifacts.local/work/my-capture/spec.json `
  --output artifacts.local/work/my-capture/output `
  --plugin artifacts.local/work/my-gpu-plugin/package/BlindAssistCapture.uplugin
```

Built host package:
`artifacts.local/work/ue-capture-upgrade-20260907/native-pair-build-v2/package/BlindAssistCapture.uplugin`.
The shared project/map is unchanged. New settings are integrated in grounding and
factorial; actual frame/quality benchmarks here cover grounding.

27 focused Python checks passed; the C++ plugin compiled and its build receipt
hashes match the delivered sources. Capture and build process-release receipts
confirm cleanup. Evidence, code snapshots, quality scripts and run summaries are
under `artifacts.local/work/ue-capture-gpu-20260907/`; no binaries/payloads are in Git.
