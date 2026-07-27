# RCLE real positive-approach role admission R2 CID-SIMS

Status: preclaim implementation frozen; candidate payload access remains forbidden
until the guarded formal runner exclusively creates the hash-bound claim.

This module performs only geometry-based data-role admission for the single
frozen ScienceDB CID-SIMS V6 `floor3_1` archive. It does not decode RGB pixels, import or
run the RCLE RGB algorithm, tune thresholds, qualify performance, or provide
confirmation authority.

Entry points:

- `formal_runner.py`: sole claim creator and one-shot formal orchestrator.
- `acquire.py`: one exact ScienceDB GET after claim, with official byte/MD5
  gates and no redirect, retry, resume, or replacement.
- `producer.py`: frozen first complete 10-second geometry window.
- `validator.py`: independent geometry and access-contract replay.
- `pilot.py`: host-only throughput pilot on already-burned data; refuses
  any `floor3_1` path.

All downloaded payloads and formal receipts must remain below
`artifacts.local/`. After any `floor3_1` payload or geometry access, the entire
CID-SIMS V6 Floor3 three-run family and shared pointcloud are unavailable for
confirmation. A source failure or
geometry-gate miss is a no-retry HOLD. Unsafe members, forbidden access,
binding drift, implementation failure, or validation disagreement are INVALID.

Even an admitted and independently valid R2 result only permits creation of a
separate performance-qualification task. R2 never authorizes that work itself.
