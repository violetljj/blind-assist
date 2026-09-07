# Background PNG encoding for UE capture

Follow-up to [native depth and burst capture](UE_CAPTURE_NATIVE_BURST_20260907.md).
The subsequent [GPU readback pipeline](UE_CAPTURE_GPU_PIPELINE_20260907.md) reaches
30.97 fps on the 480-frame repeated static workload; this report retains the
background-encoding-only intermediate result.
This is a data acquisition engineering check using reused G9 Development poses;
no model was trained or scored and no new independent evaluation samples are claimed.

## Measured result

The same 48-frame, 640x360 specification, saved Willow map, native depth exporter,
and settling render counts were used. This session reran the synchronous comparator.

| RGB export | Steady fps including final RGB drain | 47 intervals plus drain | Process elapsed |
| --- | ---: | ---: | ---: |
| Synchronous, fresh comparator | 9.087 | 5.172 s | 67.532 s |
| Background encoding | **12.129** | **3.875 s** | **64.062 s** |

This is a **33.5% throughput increase** against the fresh comparator, or 30% against
the prior 9.342 fps record. The previous and current rates exclude startup/map
loading; these still dominate a 48-frame run. Host image verification after engine
exit is also outside the steady acquisition number. Timings are one run per mode,
not confidence intervals or a guaranteed long-run rate.

The new production run's synchronous RGB submission median was 63.261 ms.
Cumulative PNG encoding was 1.320 s / 48 frames (27.503 ms/frame), now overlapping
subsequent work; writing was 0.101 s total. Readback accumulated 3.313 s including
the first frame's longer warmup. **Readback timing includes waiting for queued
scene rendering**, so it is not a measurement of PCIe transfer alone.

The queue peaked at 2 pending writes with a configured limit of 4. All 48 finished,
with zero failures or pending writes after drain.

A subsequent **480-frame bounded soak** (ten repetitions of the same 48 poses,
not 480 independent samples) achieved **12.127 fps including drain** over 39.500 s
after the first frame. The full process took 99.906 s. All 480 PNGs decoded and
all 480 float32 depth arrays passed dimensions/finite/range checks. Peak pending
writes remained 2; completed 480, failed 0, pending 0, with no partial files left.
The ten 48-frame blocks ranged from 11.394 to 13.795 fps without persistent queue
growth. This verifies a short repeated workload, not multi-hour endurance.

## Fidelity and completion checks

- A separate 12-frame same-target probe exported each render target through both
  paths without another render. All 2,764,800 RGBA pixels were identical after
  decoding. Probe throughput is deliberately excluded from production claims.
- Against the previous native/burst production images, 48-frame global RGB MAE
  was median 0.07875/255, maximum 0.94582/255; maximum intended-object ROI MAE
  was 0.13943/255. All tested projected object pixels had native surface depth
  within 3 cm of the analytic object geometry (minimum coverage 1.0).
- The 19 focused Python checks cover existing depth transport, bounded RGB
  backpressure, worker failure propagation, drain accounting, old-plugin fallback,
  overwrite rejection, source snapshots, and host decoded-pixel comparison.
- The plugin compiled successfully against the installed UE 5.8 toolchain. The
  saved map hash remained unchanged. Owned processes are released after each run.

## Implementation and use

`ExportRgbPng` uses the same `FImageUtils::GetRenderTargetImage` and lossless PNG
compressor defaults as the engine exporter. It retains a private CPU image and
encodes/writes it on a thread-pool worker. Pending jobs are bounded, and each file
is published through a new partial file and a no-replace rename. Polling surfaces
worker errors; capture completion and module shutdown drain remaining work.

This is **asynchronous CPU encoding with synchronous GPU readback**, not a GPU
readback ring. It preserves resolution, warmup counts and postprocessing. The
settled-pose/global-time limitations of the preceding burst implementation remain.

The launcher defaults to `--rgb-export auto`: it uses this path when the loaded
plugin exposes the new API and falls back to the original RGB exporter for older
plugins. `--rgb-export legacy` selects the comparator. `--rgb-export native_probe`
runs a same-target pixel check and needs Pillow on the host. The host also checks
all asynchronous production PNGs for successful decoding and expected dimensions
before publishing completion.

```powershell
python tools/build_ue_capture_plugin.py --output artifacts.local/work/my-rgb-plugin
python tools/ue_native_capture.py capture --capture grounding `
  --spec artifacts.local/work/my-capture/spec.json `
  --output artifacts.local/work/my-capture/output `
  --plugin artifacts.local/work/my-rgb-plugin/package/BlindAssistCapture.uplugin
```

The compiled host package is
`artifacts.local/work/ue-capture-upgrade-20260907/native-rgb-build-v1/package/BlindAssistCapture.uplugin`.
The shared UE project is not modified. All three capture adapters integrate the
helper; measured performance here is from grounding. Whisker retains tick cadence
under automatic cadence selection.

Evidence: `artifacts.local/work/ue-capture-async-20260907/` contains the source
specifications, probe, production, fresh comparator, quality checks and summaries.
The plugin build receipt and source/binary hashes are under `native-rgb-build-v1`.

Further gains require addressing rendering and synchronous readback; eliminating
the remaining native depth conversion alone cannot produce another large multiplier.
Hundreds-of-thousands-frame endurance, resumable shards, and physics/video temporal
equivalence remain untested.
