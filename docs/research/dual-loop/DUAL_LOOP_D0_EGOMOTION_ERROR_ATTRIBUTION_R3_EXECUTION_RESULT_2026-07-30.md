# D0 ego-motion error attribution R3 execution result

## Terminal

`EXECUTION_INVALID / CONSUMED / NO_RERUN / NO_R4 / NO_SCIENTIFIC_EXIT`

The only R3 formal producer created its immutable `formal_start.json`, opened the
frozen scientific inputs, and stopped before producing any event row. The
failure receipt records:

- error: `BBOX log-area closure mismatch`;
- completed events: `0 / 469`;
- no `event_table.jsonl`, `analysis.json`, `producer_receipt.json`,
  `execution_validation.json`, or `execution_receipt.json`;
- `rerun_authorized=false`.

The D0 three-way operational priority exit is therefore not evaluable.
`NO_PRIORITY_IDENTIFIED` is not a substitute for an invalid execution.

## Static root cause

This is a cross-stage semantic-lineage defect, not an environment, input-integrity,
or floating-point-tolerance failure:

1. The frozen LITE BBOX arm defines its signed scale rate as
   `0.5 * (log(current_area) - log(previous_area)) / dt`.
2. The D0 R1/R2/R3 protocol and byte-identical producer instead recompute
   `(log(current_area) - log(previous_area)) / dt`.
3. D0 then requires that full log-area rate to match the frozen half log-area
   rate within `1e-12 / s`.

Every finite non-zero BBOX row is therefore systematically separated by a
factor of two. The producer correctly failed closed relative to its frozen,
but semantically incorrect, D0 contract. The independent validator duplicated
the same full-rate assumption, while synthetic tests checked only D0-internal
self-consistency and missed the upstream formula lineage.

R1 and R2 failed earlier for unrelated missing runtime dependencies, so they
never exposed this latent contract defect.

## Execution-envelope incident

The reviewed guarded launch passed its live capacity gate and started the exact
frozen `python -I -B ... produce` command. The caller's foreground tool timed
out after the formal child had started, terminating the outer wait/monitor
process but not the producer. The existing producer process was observed
without restart until it emitted the immutable failure receipt.

This monitoring interruption is preserved as an execution-envelope deviation.
It does not rescue or alter the stronger scientific terminal: the producer
itself consumed the one-shot and failed its frozen closure before any event
output.

## Disposition

- Do not edit or rerun R3.
- Do not create D0 R4 under the same burned input route.
- Do not infer ego-motion or temporal-trend priority.
- Any future diagnostic would require a new protocol identity that explicitly
  distinguishes radius-equivalent half-log scale rate from full log-area rate.
- Mainline implementation may proceed only through the separate, non-scientific
  shadow-wiring contract, with no active geometry source or effect claim.
