# City capture throughput engineering

Scope: paired engineering comparisons on the existing 50-group / 150-frame
pilot, one Street200V7 map and five real assets. No training, held-out evaluation
or newly independent worlds. One group contains three frames.

Subsequent implementation and actual larger-batch timing are recorded in
[500-group collection](CITY_COLLECTION500_20260908.md): 1,500 frames complete
with QA in 346.527 s after reducing routine settling and retaining first-use
protection. The capacity estimates below describe this earlier 32-tick stage.

## Full-chain findings and changes

The accepted cached worker baseline (`full-v2`) takes 184.904 s editor lifecycle
and 197.964 s complete job for 150 frames. Its script takes 162.640 s; geometry
verification 3.612 s and grouping 3.369 s. Keep startup separate from steady
acquisition and use one editor process for many cases. Persistent project DDC,
Zen and prepared assets stay on the worker; transfer only changed assets and
thin receipts/previews, not every raw dataset back to the primary.

The old `native_probe` exports synchronous RGB/depth reference copies for every
frame in addition to the canonical pair. The new production profile uses
`native_async`, retaining native completion, payload and geometry checks. Queue
polling occurs each tick with bounded backpressure. Reference probes remain
available for export-path acceptance; they are not needed for every training
sample after that path passes.

An optional 1280x720 appearance camera previously rendered every settling tick
and exported every frame. Production disables this camera entirely; the
640x360 model RGB remains available for visual QA. This changes preview work,
not model image resolution or native depth resolution. Default settings remain
compatible: `pair_export_mode=native_probe`, `export_appearance=true`.

The 32 settling ticks, native asset/shader/streaming readiness and extra
post-readiness settling for the first view/new mesh are retained. An observed
missing-RGB-asset defect justified the latter guard. Do not lower these merely
to obtain a better timing result.

The geometry verifier previously retained every depth frame in GPU memory.
It now streams frames and reloads only each analytic baseline/control pair.
On the primary CUDA backend, the 150-frame peak allocated memory falls from
152,782,848 to 15,464,448 bytes (89.9%); elapsed time 4.080 vs 4.027 s is not a
meaningful speed claim. A separate 15-frame set with two analytic controls also
passes. All 165 masks are byte-identical and all result rows exactly match.
Evidence: `artifacts.local/nearfield/city-pcg-20260908/streaming-verifier-regression-v1/results.json`.
GPU tensor retention is bounded by frame/pair processing; CPU report size and
disk usage still grow with dataset size.

## Measured export comparisons

All runs use the same worker, prepared project and persistent caches. These are
single-run engineering timings, not confidence intervals or sustained-load tests.

| Run | Frames | Editor lifecycle | Complete job | Output, decimal MB |
| --- | ---: | ---: | ---: | ---: |
| Standard small15-v2 | 15 | 42.988 s | 47.788 s | 85.46 |
| B: async, keep appearance | 15 | 40.462 s | 45.133 s | 62.50 |
| C: async, no appearance | 15 | 39.807 s | 44.484 s | 28.82 |
| Standard full-v2 | 150 | 184.904 s | 197.964 s | 916.63 |
| C: production full | 150 | 115.233 s | 127.286 s | 287.70 |
| A: standard repeat | 150 | 185.153 s | 198.414 s | 916.63 |

C150 completes 50 groups with native capture, geometry and grouping PASS, zero
failed exports and peak queue depth one. Script time falls from 162.640 to
93.218 s. Complete-job throughput improves 1.56x (35.7% less time); total output
falls 68.6%. The 15-frame comparisons are startup dominated; do not extrapolate
their rates as steady throughput or subtract them to assign the 150-frame gain
precisely between export modes.

B/C15 match every risk state, with maximum RGB MAE/255 below 0.00067 and maximum
fraction of known depth pixels changing over 3 cm below 0.000033. C150 also
matches all 150 BODY/HEAD support and distance states. However, whole-image
equivalence does **not** pass: maximum RGB MAE/255 is 0.04178 and maximum depth
change fraction 0.20920, concentrated in the plaza views. Primary inspection
finds intact props and normally rendered scenes. On diagnostic plaza frames
76/79, known BODY/HEAD support values and depth at support pixels are unchanged;
changes also occur within 8 m outside those support regions, not only far away.
The cause is not established merely by seeing foliage. The additional standard
A150 repeat also passes QA and all risk states. Against full-v2, its plaza depth
change fraction averages 1.04%, maximum 2.47%, versus C's 16.21% and 20.92%.
Its maximum plaza RGB MAE/255 is 0.00855 versus C's 0.04178. Thus standard
repeat variability does not explain the magnitude of C's changes, and C is
**not accepted as a pixel/depth-equivalent replacement** of the standard profile.

Delivery decision: keep the standard profile as the default and expose the
measured faster profile explicitly through `--production`. Current task-state
and visible-support checks pass for this pilot, but full-scene depth or strictly
matched-background comparisons should use the standard profile. Record the
profile with every batch and do not silently mix these render settings in
paired experiments. Further attribution of the background changes is pending;
this change makes no claim that all visual/geometry detail is preserved.

Evidence is under
`artifacts.local/nearfield/city-pcg-20260908/worker-profiles-v1/`:
`evidence/profile-disk-timing.json`, `compare-b15.json`, `compare-c15.json`,
`compare-c150.json`, and `plaza-diagnosis/near-far-support.json`.
The final standard repeat is recorded in `a150-evidence/`. Every task-owned
editor process and scheduled job is released; persistent caches and raw
acquisition evidence remain on the worker.

## Capacity estimate

Using the measured C150 complete-job rate, including repeated startup/QA at
the same batch size, gives the following linear estimates. These are not
completed large runs; new assets, cold caches, machine contention and thermal
throttling can change them. Decimal GB includes current outputs and metadata.

| Requested quantity | Frames | Estimated time | Estimated disk |
| --- | ---: | ---: | ---: |
| 1,000 frames | 1,000 | 14.1 min | 1.92 GB |
| 10,000 frames | 10,000 | 2.36 h | 19.18 GB |
| 1,000 three-frame groups | 3,000 | 42.4 min | 5.75 GB |
| 10,000 three-frame groups | 30,000 | 7.07 h | 57.54 GB |

In C150, canonical RGB is 78.33 MB, float32 depth 138.26 MB and support masks
69.14 MB. The standard run additionally writes 399.34 MB of appearance and
229.64 MB of reference copies. Masks/depth remain lossless and uncompressed;
packaging them is a possible later disk optimization, not required for current
capture throughput. Larger batches amortize startup further, but that rate
has not been measured. Persistent-editor resume, automated batch sharding and
reduced/adaptive settling are not implemented or validated by this change.

Native `gpu_ready_seconds` is accumulated submission-to-poll latency, overlapping
rendering and other work. It is not isolated GPU execution time and must not
be added to other timers. Baseline encoding is about 4.46 s, writing 0.39 s and
readback 0.16 s for 150 pairs, so compression/thread rewrites are not the first
priority. Validation costs about seven seconds in this baseline and remains
enabled.

## Running prepared production specs

Use the artifact-deployed `tools/run_city_worker_batch.py` with absolute paths:

```text
python -B run_city_worker_batch.py --spec <prepared-spec.json> --output <fresh-artifact-output> --production --timeout 1800
```

The timeout is for capture, in seconds; size it for the batch. Preserve whole
clear/center/lateral groups and their local baseline indices in each spec.
Use separate output directories for bounded batches, keep accepted batches,
and retry only failed batches. This entry does not resume midway through a
failed batch or generate new world diversity. Scaling requires genuinely new
scene/pose/geometry specs, not repeating the same 50 groups and counting them
as new information.
