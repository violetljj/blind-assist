# RCLE degradation / flow-quality diagnostic R0

This Development diagnostic keeps the existing R3 response, strict
`> 0.01/s` threshold and three-consecutive-pair confirmation unchanged.

`extract.py` is the stage-1 firewall. It reads only the already-open ADVIO
RGB, timestamps and pose, and writes label-blind proxy ledgers. It must not
read RCLE response ledgers. `analyze.py` is stage 2: it joins frozen proxy
ledgers to the existing R3 pair ledgers and reports per-session descriptive
attribution plus the effect of a response-blind abstention gate.

ADVIO sequence16 remains `SEALED_UNSEEN` and is rejected by both entrypoints.
Pair rows are longitudinal measurements, not independent samples.
