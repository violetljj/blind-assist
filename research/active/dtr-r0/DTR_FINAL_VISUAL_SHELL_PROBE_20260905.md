# DTR Final Reckoning R1: bounded visual-shell source probe

Date: 2026-09-05

Status: `CAPACITY_BLOCKED_BEFORE_SOURCE_CAPTURE_PROTOCOL_UNCONSUMED`

The user authorized one bounded non-collision visual-shell probe followed, only
if admissible, by the already frozen fair comparison. No X97 is authorized.
The earlier design-07 source remains closed and its pixels cannot be reused.

## Fixed intervention

The unchanged ten-cell probe receives visual-only advertisement panels for S09
and S10. S09 raises the panel bottom to preserve a partial target surface. S10
uses a lowered panel following the target/wearer midpoint during a fixed window.
Both panels retain `collision_relevant=false` and `collisions_enabled=false`;
no pedestrian or vehicle is reclassified to suppress a real hazard.

Two additional unobscured reference episodes preserve target, camera, scene,
and issued route while moving only the corresponding panel out of view. They
define synchronized native-raster visibility fractions, not method inputs.
There are twelve probe episodes; the final thirty-episode denominator and all
eleven arms remain unchanged.

Probe seed: `516938`, outside FIT_ONLY/FINAL_A/FINAL_B. Current pre-pixel v2
protocol SHA-256:
`1A41E1FD916DA9D76107B20D5EDD3DF8DF74B7D6C464E2CFD366BE55856121F1`.
The full window is 5.0 through 5.6 seconds. An analytic field-of-view review
before any pixels rejected the earlier 2.0-second window: target visibility and
eight preceding trackable samples were not established there.

The source gate requires six contiguous partial-visibility frames in [0.05,0.45]
with no zero-pixel frame in the frozen partial window, and six contiguous fully
hidden frames plus eight visible reappearance frames for S10. Source-native
responsible assets and independent witness replay must agree. An invisible
reference cannot establish an occlusion. The instance gate precedes witness
capture; failure stops the probe without model capture or inference.

## Execution boundary

The roster validator passes all eight direct locks; a read-only audit also
verified all 33 nested C44 component hashes. An eleven-arm execution adapter and
shared final event matcher are still missing. The historical five-arm runner
cannot be used for final scoring, particularly for S10's two contact intervals.
Source success alone does not authorize skipping remaining source semantics,
runtime closure, intervention-index, FIT_ONLY, or final evaluator freezes.

Protocol and freeze receipts live under
`artifacts.local/evidence/dtr-final-reckoning-roster-r1`. The first launch was
refused before server start or run-directory creation because that raw output
location was outside the CARLA storage guard's experiment root. The next launch
uses the existing governed root
`artifacts.local/runtime/carla-asset-library/experiments/dtr-final-reckoning-roster-r1`.
Both resolve through the same canonical F: artifact junction. The guard and its
capacity ceiling were not weakened. No joined model/evaluator root, method
prediction, or fit/final access is permitted during this source probe.

The first evaluator freeze overlapped an implementation-only context-boundary
repair. Because the first launch had no server, run directory, or pixels, the
original receipt was preserved and a new pre-pixel v2 receipt bound the final
evaluator bytes before the next launch. No output was evaluated under the first
receipt. Current evaluator SHA-256:
`C09440DC47B0EFD0036EC92CA5F82433083F3713501F3E02B020AD8C9487F1EC`.

The materializer's three focused tests and the source evaluator's fourteen
synthetic tests pass. The PowerShell runner parses successfully. Its new
`-VisualShellSourceProbeOnly` mode captures instance first, gates it, then captures
and checks witness only if eligible; ordinary four-shard capture is unchanged.

The separate `carla/dtr_final_reckoning_event_metrics.py` candidate now implements
maximum one-to-one alert/event matching, explicit UNKNOWN coverage, fragmentation,
lead, CLEAR censoring, pooled counts and paired episode-cluster bootstrap. Its
sixteen synthetic checks pass. A continuous alert spanning multiple truth events
can match only one; abstention does not erase a known truth event from recall.
Empty metric denominators remain `None`, and unevaluable bootstrap replicates
are reported separately. This closes a metric-implementation gap, not final
execution integration: its detailed candidate semantics must enter the complete
execution freeze before any fit or final truth is opened.

Central registration was attempted via `tools/knowledge.py register-experiment`
but its prerequisite validation rejected the pre-existing input-fingerprint
mismatch at `experiments/index.jsonl:252`. No registry bytes were changed by this
task. This is a local registration gap, not a source or algorithm verdict.

## Observed execution outcome

The governed launch waited the existing 300-second capacity window and exited
with `CARLA capacity did not recover: processes=0; free_physical_gb=2.44`.
An unrelated user-owned `UnrealEditor.exe` remained active; it was not stopped.
The 4 GiB startup floor was preserved. No raw run directory, source pixel,
CARLA process, detector ledger, model prediction, or fit/final access exists for
this run. Cleanup inspection found zero task-owned processes and no matching
storage lease; RPC ports 2000/2001/2002 had no listeners.

The bounded visual-shell experiment has therefore **not executed** and its
protocol is unconsumed. This is a resource blocker, not a failed occlusion
source and not an algorithm result. Resume the same unconsumed design when the
concurrent job releases sufficient memory; do not recast this as a scored probe.

## Resume after capacity is available

The protocol remains unconsumed until the first durable source pixel. Verify
the pre-pixel v2 freeze before launching the already prepared command:

```powershell
pwsh -NoProfile -File tools/run_dtr_carla_c2_rich_scene.ps1 `
  -RunId visual-shell-probe-20260905-01 `
  -RawEvidenceRoot artifacts.local/runtime/carla-asset-library/experiments/dtr-final-reckoning-roster-r1 `
  -Protocol artifacts.local/evidence/dtr-final-reckoning-roster-r1/visual-shell-probe-20260905-01-protocol-prepixel-v2.json `
  -VisualShellSourceProbeOnly -StorageReservationBytes 4294967296 `
  -CaptureTimeoutSeconds 1800
```

Source failure with durable pixels is terminal for this bounded probe. It must
not trigger an automatic scene revision or a fresh run ID.
