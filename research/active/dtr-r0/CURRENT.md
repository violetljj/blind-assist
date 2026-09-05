# DTR-R2 current

Updated: 2026-09-05

Status: `DTR_R2_DYNAMIC_RETAINED`

## Capability question

Can future obstacle occupancy intersecting the wearer's route produce stable
`ONSET / HOLD / ESCALATE / CLEAR` events while missing evidence stays `UNKNOWN`?
The unresolved contribution question is what collision-state information X94
adds beyond a raw motion baseline and the same temporal event smoothing.

## Retained evidence and baseline

- **Public/JRDB X21:** same-live-track transport of an already authorized X13
  component; six consumed sequences reached `5/6` CONTACT, 11 false segments,
  45.45% Event F1, `3.061 s` median lead, and `8/18` dropout recovery. This is
  same-source Development; genuinely source-disjoint confirmation is pending.
- **CARLA X73:** the latest complete source-disjoint synthetic Development
  confirmation. On C35, parent-hull reconstruction improved X72 by
  `+6 TP / +0 FP / +2.24 pp` frame F1, reaching `132/18/40` TP/FP/FN and
  `88.00/76.74/81.99%` precision/recall/F1. The same-map, scripted-motion,
  detector and evaluator boundaries remain; successors lack its confirmation.
- **CARLA X94:** consumed post-hoc Development reference, `1478/84/417`
  TP/FP/FN and `94.62/77.99/85.51%` frame precision/recall/F1 across eleven
  cohorts. Its one-observation full-dropout bridge added six TP and no FP over
  X93; same-parent evidence and an unchanged valid issued plan remain required.
- **Credible simple comparator:** in the consumed C35 raw-input pilot, Kalman
  CV + route tube + `0.60 s` hold tied X94 at 66.67% Event F1 and four false
  segments, with two versus ten fragmentation runs. X94 retained better frame
  F1 (`84.62%` versus `81.52%`), median lead (`2.80` versus `2.65 s`) and CLEAR
  (`0.00` versus `0.20 s`). This tradeoff has no fresh eleven-arm adjudication.
- **X95 remains a challenger:**
  `DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_GATE_NOT_MET`. Its consumed replay
  gained `3.74 pp` Event F1 and removed five false segments versus X94, but lost
  `7.02 pp` frame F1 and added 30 fragment gaps. Simple `0.60 s` hysteresis
  reached 89.07% frame F1 on that replay; complexity has not established a win.

## Current bottleneck

Frozen avoidance-only R1 reached `30/30` instance/witness source-stratum passes
and a complete FIT_ONLY RGB/depth join. FINAL_A's depth server then exited
before its first depth frame with `Shader compilation failures are Fatal`;
FINAL_B RGB/depth never started. Status is
`NOT_EVALUABLE_SOURCE_CAPTURE_INTERRUPTED`: nine complete sensor shards remain,
no detector, fit, prediction or final method score was opened, and task-owned
processes, ports and leases were released. The eleven-arm adapters have focused
implementation checks only. This source failure changes no algorithm inheritance.

The completed two-scene 720p probe passed 100 synchronized RGB/depth pairs.
Synchronization alone did not improve capture time (`57.23 -> 59.13 s`). With
both arms synchronized, fast lossless PNG reduced `63.28 -> 22.00 s` (2.88x);
400 independently decoded images exactly matched their raw pixels, with 8.96%
more encoded bytes. This was one ordered short comparison excluding warmup,
not statistical throughput or long-run stability evidence. It preserves depth
bytes but does not validate metric-depth decoding or establish a shader-crash fix.

A subsequent Development composite reused nine intact shards, but FINAL_A depth
again hit a shader fatal before any payload. A separate DX11 probe reached RPC
but failed camera warmup. Both runs ended without detector, fit or scores; the
client now detects server death promptly. Neither failed run is reopened here.

A separately identified launch profile requesting synchronous PSO compilation
passed three cold starts (600 independently checked images), then all three
missing shards (3,276 images) and all native joins. The composite source is now
admitted as reused Development. This is bounded completion, not a permanent
shader-fix claim. See [startup and method diagnosis](CARLA_CAMERA_STARTUP_20260905.md).

Method preparation generated 910 FIT_ONLY detector frames but failed S03's
six-frame dropout recovery condition: frame 29 is nearly black, has zero
candidates and no measured collision credential. No fitting or final scoring
ran. A consumed FIT_ONLY proposal with earlier 2/3/6-frame windows passed on
that episode only; it neither rescues the failed run nor confirms method gain.
Registration remains blocked by the existing input-fingerprint mismatch.
The older eleven-cohort raw comparison cannot be reconstructed fairly: only
C35 retains the required dense model/evaluator payloads. Derived tracks or
recaptured pixels do not restore the original raw-input comparison.

## Next decision

1. **Development:** use the complete source for a separately identified
   pre-contact dropout design with observable recovery. The consumed FIT_ONLY
   candidate windows are `[7,8]`, `[11,12,13]`, `[16..21]`; their selection is
   post-hoc and changes no original verdict. Validate their source role before
   comparing methods. Preserve the isolated hash-matching X24/X25 method snapshot
   rather than reverting concurrent interface changes. No recapture is needed
   merely to reuse the now complete source.
2. **Confirmation:** a new admitted source authority is required before the
   pending comparison can proceed. Preserve the frozen eleven-arm decomposition,
   including raw Kalman + emitter and X94 + the same emitter, shared event
   primaries and secondary frame diagnostics. If collision-state quality adds
   useful effect under that common emitter, retain that contribution; if the
   simple baseline ties or wins, simplify the proposed architecture. Missing
   source support leaves the algorithm question unresolved.
3. **Exploration:** this update starts no X97 or new learner. A separately named
   Development hypothesis may reuse disclosed consumed inputs; it does not revise
   frozen scores or restore confirmation authority. JRDB X21 promotion still
   needs its own unchanged, genuinely source-disjoint confirmation; CARLA cannot
   substitute for it. No experiment is started by this documentation update.

## Boundaries

- Frozen R1 is not retried or reclassified as fresh confirmation. Preserve its
  seals, nine shards and crash evidence; no protected fit/final access follows
  from capture-engineering success.
- C35-C41 cannot be rerun as confirmation. C8-C11 admitted no evaluable X31
  occlusion source; N4 v1 cannot resume or retry. It is consumed incomplete,
  not a three-town result. Historical outcomes and source gates remain unchanged.
- Consumed diagnosis is Development, not fresh authority. Keep frozen thresholds,
  source gates, lifecycle, association, seeds and denominators with their original
  results; changes belong to a separately identified Development version.
- `UNKNOWN` and `NOT_EVALUABLE` are not `CLEAR`, negative evidence or safety.
  Wearer-global route conflict owns event correctness; component identity is
  diagnostic. Public replay and CARLA do not establish Android readiness,
  natural-distribution performance, user benefit, deployment or safety.
- Uncommitted candidates and outputs remain WIP. Existing structured inheritance
  roles and historical verdicts remain authoritative; this compaction changes none.

## Evidence links

- [X21 result](X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md), [X73 confirmation](carla/DTR_CARLA_C35_X73_FRESH_CONFIRMATION_20260901.md), [X94 result](carla/DTR_CARLA_X94_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md).
- [C35 raw-input pilot](DTR_BASELINE_RECKONING_C35_RAW_PILOT_20260905.md), [X95 result and simple controls](carla/DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_20260901.md).
- [Frozen comparison design](DTR_FINAL_RECKONING_ROSTER_R1_20260905.md), [latest source execution and crash](DTR_FINAL_SOURCE_EXECUTION_20260905.md); the later execution record owns the design's current execution status.
- [Completed RGB/depth and lossless-PNG probe](CARLA_RGBD_THROUGHPUT_20260905.md).
- [Failed Development composite and DX11 probe](CARLA_FAST_COMPOSITE_SOURCE_20260905.md).
- [Detailed ledger and reproduction](README.md), [formal research governance](../../../docs/formal/RESEARCH_GOVERNANCE.md).

The full superseded current is preserved exactly in Git at
`daf5720064d98a93b75336469d18e9a2fe0023e5:research/active/dtr-r0/CURRENT.md`.
Use that history anchor and the existing result files for the X24-X94 trajectory;
this page owns the present decision, not a second history ledger.
