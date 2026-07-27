# RCLE real positive-approach role admission R1

Status: preclaim implementation frozen; candidate payload access remains forbidden
until the guarded formal runner exclusively creates the hash-bound claim.

This module performs only geometry-based data-role admission for the single
frozen ETH3D `sofa_3` RGB-D archive. It does not decode RGB pixels, import or
run the RCLE RGB algorithm, tune thresholds, qualify performance, or provide
confirmation authority.

Entry points:

- `formal_runner.py`: sole claim creator and one-shot formal orchestrator.
- `acquire.py`: one exact official GET after claim, no retry or replacement.
- `producer.py`: frozen first complete 10-second geometry window.
- `validator.py`: independent geometry and access-contract replay.
- `pilot.py`: host-only throughput pilot on already-burned data; refuses
  `sofa_3`.

All downloaded payloads and formal receipts must remain below
`artifacts.local/`. After any `sofa_3` payload or geometry access, the entire
ETH3D sofa scene family is unavailable for confirmation. A source failure or
geometry-gate miss is a no-retry HOLD. Unsafe members, forbidden access,
binding drift, implementation failure, or validation disagreement are INVALID.

Even an admitted and independently valid R1 result only permits creation of a
separate performance-qualification task. R1 never authorizes that work itself.
