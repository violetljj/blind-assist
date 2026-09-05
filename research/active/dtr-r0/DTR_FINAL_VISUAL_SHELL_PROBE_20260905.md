# DTR Final Reckoning R1: bounded visual-shell source probe

Date: 2026-09-05

Status: `SOURCE_SHELL_GATE_MET_PENDING_FULL_ROSTER_GATES`

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

## Initial capacity-blocked launch (historical)

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

## Resumed execution (2026-09-05)

The same v2 freeze was verified unchanged when free physical memory recovered
to 7.85 GiB and no CARLA or Unreal editor was running. The original run ID was
then launched successfully. The protocol is now consumed by source pixels;
the earlier capacity-blocked receipt remains historical evidence only.

The instance shard captured all twelve episodes (1,092 frames). Every expected
outcome and responsible set matched. The frozen paired-raster gate passed:

| Pair | Frozen window | Observed result |
| --- | --- | --- |
| S09 / ep_09 vs ep_11 | 1.3–2.4 s | 12 contiguous qualifying frames; native/reference ratio 0.0738955823–0.3012805588; no zero-pixel frame |
| S10 / ep_10 vs ep_12 | 5.0–5.6 s | 7 contiguous zero-pixel frames; required visible context before and after disappearance passed |

Independent witness capture also completed all twelve episodes (1,092 frames),
and the frozen witness replay check passed. The runner exited zero after both
gates. This completes the bounded shell probe, not formal roster admission.

Raw evidence is retained under
`artifacts.local/runtime/carla-asset-library/experiments/dtr-final-reckoning-roster-r1/visual-shell-probe-20260905-01`.
SHA-256 receipts:

- `source-gate-instance.json`: `BFCB6D2E2131687BA2F0C434246D63AB68018C3A7C2D623A91C4598D70457D02`
- `source-gate-witness.json`: `839E99DCA0F38D6BB47AA1F410F510C7E7FCEC968FB4B64FBEE2597BC2C0DF88`
- `shards/instance/result.json`: `8035AADC30F0A0BCA652C14ADD65CC671C17D0662C65D5D0B6415504191C4D6B`
- `shards/witness/result.json`: `2C417CEA5A8CF891E4AF75DEA08DFAE359BC38046D9299BF063D9A10F0616D30`
- `postprobe-s10-temporal-audit.json`: `9CCD89BF3177ACEE602293438BD815E2D79FE0CABD9150225B13FF7BE32CDDB0`

Final cleanup inspection found zero matching task-owned processes, zero RPC
listeners on 2000/2001/2002, and zero matching storage leases. Durable source
payloads, manifests, logs and receipts remain retained. No model capture,
prediction, fit or final truth was opened.

### Remaining formal-source gap

A separately labelled post-probe diagnostic of native S10 truth finds positive
future-contact samples at 0.0–1.1 s and 2.8–9.0 s, separated by 16 known-negative
samples at 1.2–2.7 s (1.6 s under the 0.1 s sample-cell convention). Proven
reference-visible disappearance instead occupies 5.0–5.7 s. **There are zero
known-negative samples inside that disappearance.** This establishes the shell
mechanism, but does not establish R1's stronger requirement that disappearance
contain a known-negative interval before renewed risk. Two contact windows and
an unrelated occlusion window cannot substitute for their temporal binding.

The diagnostic is not a changed frozen probe gate or a method score. Its inputs
and hashes are retained in `postprobe-s10-temporal-audit.json` under the raw run.
The successful shell result remains intact. No additional source run, changed
scene, detector inference, FIT_ONLY or FINAL access was started to repair this.

A read-only source audit also found that the complete ten-stratum evaluator and
three-group formal materializer are absent. Existing analytic center-contact
checks do not establish all captured-source semantics. The next formal-source
implementation needs a versioned execution annex with joint S10 temporal
binding, the remaining per-stratum source checks, frozen observation-removal
indices, and explicit auxiliary-reference identities outside the final metric
denominator. Existing eleven-arm mechanisms and the final roster stay frozen.

Central registration was retried after capture resumed and still rejected the
pre-existing fingerprint mismatch at `experiments/index.jsonl:252`; the registry
was not rewritten to bypass it. A local structured component disposition is
retained in `DTR_FINAL_VISUAL_SHELL_PROBE_20260905.inheritance.json`; central
registration remains explicitly pending, rather than claiming it succeeded.

## Launch command (historical; do not repeat this consumed run)

The following command produced this source run after the memory blocker cleared:

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
