# Fast-PNG Development source continuation

Date: 2026-09-05. Working engineering result: `NOT_EVALUABLE_NEW_COMPOSITE`.
The avoidance comparison remains unresolved; no detector, model fitting,
prediction or evaluator scoring ran in this continuation.

## Question and fixed scope

Can the tested lossless encoder finish the three missing camera shards while
reusing nine complete shards from the failed R1 capture? This is explicitly
`DEVELOPMENT_COMPOSITE_REUSED_SOURCE_NOT_FRESH_CONFIRMATION`. The original R1
failure remains intact. Scene protocols, eleven algorithm arms and metrics were
not changed. Each missing shard had one attempt; the failed composite is closed.

## Observed result

- All three source audits passed using the retained instance/witness data.
- FINAL_A depth produced **zero** payloads. The DX12 engine crashed after 29 s
  with `Shader compilation failures are Fatal` (native process 16044).
- The client spent 361.12 s in capture and cleanup timeouts. This identified a
  separate orchestration defect: the parent did not monitor server exit while
  waiting for its client. FINAL_B RGB/depth were never attempted.
- A separate DX11 engineering probe reached RPC readiness in 23.70 s, but its
  camera warmup received no usable frames and ended with `Empty()`. No new native
  crash record was observed for this probe. This is a failed bounded probe, not
  proof that DX11 can never work. It did not authorize another full capture.

The earlier 2.88x short-probe PNG improvement remains an encoding measurement;
it did not fix cold-start shader stability. Do not attribute this native failure
to the encoder or GPU memory pressure without further evidence.

## Implemented continuation and fixes

The wrapper preserves original capture metadata and writes a separate raw-RGBA
hash journal and implementation receipt. The composite driver reuses intact
shards with hard links, then requires independent PNG decoding and all joins
before admission. The source/preparation/scoring chain carries the Development
claim and binds each admitted source root and joined result.

After this failure, the driver now terminates its own capture child promptly
when the server exits. A subprocess test verifies the child is gone. Independent
review also found and fixed the validator's `root/GROUP` versus `root/raw/GROUP`
layout mismatch; a three-shard fixture exercises the actual layout and corrupt
pixel rejection. These fixes apply to future separately identified runs only.
The failed run's original nine implementation files were preserved byte-for-byte
before edits; future runs snapshot them automatically.

## Retained evidence and disposition

Under `artifacts.local/runtime/carla-asset-library/experiments/`:

- `dtr-fast-composite-20260905`: authority, three audit outputs, nine linked
  complete shards, failed encoder receipt, source-failure record, original code
  snapshots and native crash context.
- `rgbd-dx11-stability-20260905`: independent engineering protocol and failed
  warmup result. No algorithm data or score was admitted.

Both task-owned server/client runs ended. Ports 2000–2002 are free and both
storage leases were released and are absent. Durable evidence remains retained.
The existing experiment-index input fingerprint error prevented registration;
this record does not create or replace a structured research terminal.

Next decision-changing work is an isolated shader-startup diagnosis with a
camera-frame acceptance check, followed by a separately identified Development
capture only if that check succeeds. Repeating the same full DX12 launch or
assuming DX11 stability from successful RPC connection is not supported.
