# DTR-CARLA-N2 frozen-trace C2 replay result — 2026-08-31

## Decision

`DTR_CARLA_N2_FROZEN_TRACE_C2_REPLAY_COMPLETE`

The frozen N1 native-policy realization is now connected to a C2-compatible
four-modality source. One physics-off CARLA replay applies all 21 source actor
transforms atomically and captures instance segmentation, wearable RGB, metric
depth, and witness RGB on the same CARLA world frame. It then writes separate
C2 model and evaluator trees without rerunning Traffic Manager or AI walker
controllers.

This is a synthetic Development source/materialization result. It is not an
obstacle-algorithm result, a natural-traffic distribution, source-disjoint
confirmation, real-device evidence, product benefit, or safety evidence.

## Frozen run

- Evidence root: `E:\linnan\CARLA\experiments\dtr-carla-n2-frozen-trace-replay\evidence\n2-c2-replay-v1-20260830-2359`
- CARLA: `0.9.16`, `Carla/Maps/Town10HD_Opt`, DX12/Epic/off-screen GPU rendering
- Source trace: `481` frames, `21` actors, `0.05 s` fixed timestep
- Formal payloads: `1,924 = 481 x 4`, all `1280x720`
- Joined records: `481` model observations and `481` evaluator frames
- Evidence size at completion: `2,219,659,236` bytes (`2.067 GiB`)
- Result checks: `14/14` true
- Task cleanup: CARLA process count `0`; listeners on `26100--26102` count `0`

The observed maximum replay residual was `1.07896e-5 m` in position and
`9.99503e-5 degrees` in rotation. Wearable RGB, depth, and instance camera
transforms were identical on every sample. All four sensors shared the same
replay world frame for every sample.

## Source and alignment bindings

- Protocol SHA-256: `D1DCC9E4053A6895F5EF6AAEDA16AFCAF8AEAB290C77DF47C3AB39236B81AB16`
- Capture script SHA-256: `FC46817B5FEE807055D6F5D435E2EBDE278319D7004D48A2AF4CF24704D3FBD0`
- Helper module SHA-256: `5DF81D9071FA8925D2712840107F523652FE79F76F6D3E32B7CF012081593898`
- N1 behavior trace SHA-256: `794C7C909D77976C157A51A6D21A337C1EAB2520CDEAAC00BE59FAF19251E21A`
- N1 event receipts SHA-256: `7C9EB51EB221C9CFF462EE2339BE942A5F788319E136386D0F939450C8F02211`
- Frozen source-bundle receipt SHA-256: `93A80D4BB1BF911E47C1AC021505CBCABC32EEA0F19CC9055FAE3BA5F7422073`
- Four-modal alignment receipt SHA-256: `0079DAF1DA31D754116416AB047679B05840A514E3A24C9AA045CD4AE566C2E4`
- Alignment receipt file SHA-256: `517E27C13C772568EEB6C5CD4216DCED096F2A575CC393C285BC300DDA19D640`
- Payload inventory SHA-256: `A581D8BF495C9C7E67F0B0EE23CB7E0F43AD00E8B237B5AC6ECE4FA4FC2C43C9`
- Sealed model manifest SHA-256: `4B0E9BBB950444621BDB146CF1557F13474115B78364BF42F3D152238D24AE9D`
- Sealed evidence manifest SHA-256: `E37CF103139EFF6BE6D9F7BB38707EEA4A286FE558213D15F23D374C7A2F01CC`
- Contact sheet SHA-256: `B9EBB1885097D41462A35DFD1A29A3B3570CDCC4C6669AAAF858C75031D68F06`

## C2 separation

The model tree exposes only wearable RGB, metric depth, camera calibration,
the fixed synthetic observer pose, `NO_PLAN` navigation authority, and the
alignment/source hashes. A recursive scan found zero actor, instance, witness,
velocity, control, contact, or other evaluator-truth keys in that tree.

The evaluator tree retains instance segmentation, witness RGB, stable logical
actor identities, source control/state, replay transforms and residuals,
bounding boxes, visibility, contact diagnostics, and active tail-event IDs.
The `door_open` state missing from transform-only trace rows was restored once
from the frozen event receipt and closed once at its frozen end time.

## Exact boundary

N2 replays actor transforms, not the original native controller internals. The
observer is a newly frozen fixed synthetic observer because the N1 pilot did
not record a wearer trajectory. The source witness transform is retained from
the N1 actor manifest. Door state is explicitly receipt-replayed; vehicle-light
state, walker skeletal phase, and native-run pixels are not claimed to be
identical.

The result establishes that one native-policy realization can be turned into a
dense, trace-bound, truth-separated C2 source without cross-modality drift. It
does not alter or add evidence to X24, X31, C11, or the sealed C4 source.

## Reproduction

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_n2_frozen_trace_replay.ps1 `
  -RunId <new-unique-run-id>
```

The runner verifies the frozen N1 bundle before starting CARLA, copies the five
source files into the new evidence root, refuses overwrite/shared ports, and
verifies task-owned process and port release after completion.

## Next decision

The trace-to-source adapter is complete. A later consumer check may run the
unchanged C2 reader or frozen X24 arm on this model root once, but it must remain
a same-source synthetic Development compatibility check: no tuning and no
generalization or safety claim follow from N2.
