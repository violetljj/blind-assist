# Three cold starts and sustained Development capture

Date: 2026-09-05 to 2026-09-06. Engineering and reused synthetic Development only.

## Startup result

The launch profile requesting `r.AsyncPipelineCompile 0` passed three consecutive
cold starts at 1280x720, with two scenes and 100 synchronized RGB/depth pairs per
start. Independent Pillow decoding verified all 600 PNGs against captured raw
RGBA hashes, with zero mismatches. Pair frame IDs, timestamps and camera poses
matched exactly; all three starts followed the same camera paths.

| Start | RPC readiness (s) | Capture (s) | Images | Ports released |
| --- | ---: | ---: | ---: | --- |
| 1 | 28.33 | 21.79 | 200 | yes |
| 2 | 16.41 | 21.00 | 200 | yes |
| 3 | 15.27 | 21.07 | 200 | yes |

The installed CARLA 0.9.16 UE4.26.2 executable contains the exact
`r.AsyncPipelineCompile` name and its asynchronous-PSO description. The profile
passes the setting through `-ExecCmds`; the shipping build offers no independent
CVar readback here. These results establish a working bounded launch profile,
not proof that this setting caused the improvement or permanently fixed the
native shader fault. GPU memory occupancy also differed from the earlier failed
session (about 1.2 GiB at this probe's preflight versus about 5.5 GiB previously).
There was no controlled concurrent-memory comparison.

The prior failed DX12 composite and DX11 warmup probe remain closed. Official
[CARLA rendering documentation](https://carla.readthedocs.io/en/0.9.16/adv_rendering_options/)
does not exclude DX11 from its offscreen description; one failed warmup does not
prove DX11 is unsupported. No machine-wide graphics or driver setting changed.

## Reproduction and admission

`carla/probe_carla_camera_startup.py` owns the three-start protocol and source
snapshots. `carla/validate_carla_camera_startup.py` independently checks all
payloads, recorded outcomes, pairing and planned paths, writing an exclusive
receipt. The actual 600-image check passed; a synthetic fixture also rejected
report overwrite and a rehashed PNG whose decoded pixels contradicted its raw
sensor digest.

The composite driver accepts this profile only with three complete startup
outcomes and the 600-image validation receipt bound to the protocol and result.
It still requires all source audits, missing-shard capture, independent fast-PNG
checks and native joins before detector preparation. Scene seeds, algorithm
arms, thresholds and scoring metrics remain unchanged.

Evidence is under `artifacts.local/runtime/carla-asset-library/experiments/`:
`camera-startup-sync-pso-20260905` owns the short probe; the separately identified
`dtr-fast-composite-sync-pso-20260905` owns the longer source continuation.
This reuses nine complete old shards and is explicitly
`DEVELOPMENT_COMPOSITE_REUSED_SOURCE_NOT_FRESH_CONFIRMATION`.

## Sustained capture result

All three missing shards completed on their first attempt in this new run,
without a new native crash. Each contains 1,092 frames across twelve scenarios.
Capture time includes native scene setup and cleanup, excludes server startup,
source audit, independent validation and downstream joining.

| Missing shard | Images | Capture elapsed (s) |
| --- | ---: | ---: |
| FINAL_A depth | 1,092 | 115.40 |
| FINAL_B wearable RGB | 1,092 | 136.67 |
| FINAL_B depth | 1,092 | 113.19 |

The 3,276 new images required 365.27 s of capture-client time in total. This is
completion evidence for the specified three shards, not a randomized throughput
comparison against the failed runs. Existing source protocols and all nine
reused shards were retained.

All 3,276 new PNGs passed independent decoding. All three groups then passed
native joins and produced `DEVELOPMENT_COMPOSITE_SOURCE_ADMITTED`. The original
failed executions were not reopened. No task-owned CARLA server remained and
the source-run capacity lease was released.

## Method preparation and visibility diagnosis

The first method preflight found concurrent X24/X25 interface changes, before
any detector or truth access. `stage_dtr_frozen_method_code.py` preserved a
separate 327-file method snapshot and restored only those two files from
`1560e3e8^`, each exactly matching its historical frozen hash. All 65 structural
dependencies and eight roster locks passed. The shared working checkout was
not reverted. Its manifest lives at
`artifacts.local/evidence/dtr-frozen-method-code-20260906/method-code-provenance.json`.

The new method preparation generated all 910 FIT_ONLY detector frames on CUDA,
then stopped after 118.00 s with
`ADMISSION_FAILED_MEASURED_RAW_COLLISION_CREDENTIAL_NO_INDEX_RESCUE`. No model
fit, final truth access or eleven-arm score ran. The failure is retained under
`artifacts.local/evidence/dtr-fast-composite-sync-pso-method-v2-20260906`.

The failure is S03's six-frame window `[23..28]`: sample 22 has twelve detector
candidates and a measured confirmed risk; sample 29 has zero candidates and no
measured confirmed risk. The RGB sequence shows the camera approaching the rear
of a van and becoming nearly black: at sample 29, RGB maximum is 13/255 and
standard deviation is 0.104. Earlier two- and three-frame windows meet their
adjacent observation conditions. Successful native geometry checks therefore
did not guarantee the planned post-dropout camera observation.

A separate **consumed FIT_ONLY diagnostic**, with no evaluator rows or fitting,
tested the post-hoc proposal `[7,8]`, `[11,12,13]`, `[16..21]`. All three windows
passed the same measured-risk and nonempty-candidate check on this one episode.
The result and original frame hashes are preserved in
`artifacts.local/evidence/dtr-s03-precontact-diagnostic-20260906/diagnostic.json`.
This is a candidate observable interval for a new Development design, not an
index rescue, a reopened R1 comparison, confirmation or evidence of method gain.
No FINAL_A/FINAL_B detector ledgers were opened to choose these windows.

The next algorithm comparison needs a separately identified pre-contact
intervention design whose camera recovery is actually supported. No more source
capture is needed merely to reuse the now complete raw data, but neither the
failed preparation nor its fixed windows may be relabeled as passed.

Experiment-index registration remains blocked by the existing line-252 input
fingerprint mismatch. These engineering and diagnostic records do not replace
any retained algorithm terminal or change X73/X94/X95 inheritance.
