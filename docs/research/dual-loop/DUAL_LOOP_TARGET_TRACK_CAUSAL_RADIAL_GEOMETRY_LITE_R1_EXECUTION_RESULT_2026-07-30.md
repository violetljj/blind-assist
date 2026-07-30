# Dual-loop causal radial geometry LITE R1 execution result

## Terminal

`EXECUTION_INVALID_STOP_NO_RERUN / NOT_EVALUABLE`

Execution date: 2026-07-30  
Terminal scope: R1 evidence version only  
Development truth joined: `false`  
Evaluator invoked: `false`  
R1 rerun authorized: `false`

The producer process exited zero and created a complete pre-truth ledger, but the
guarded-host launcher rejected the formal attempt because its progress timestamp
freshness check misinterpreted the UTC `Z` timestamp after JSON deserialization.
The frozen R1 activation requires guard exit zero. Therefore the output is retained
as failure evidence but is not admitted to the evaluator.

## Bound execution evidence

- Activation SHA-256:
  `2247d3d165207b9cbf6d4d8dfb48b1053f760689fc8e217ae44de0d41ca744dd`.
- Activation review SHA-256:
  `a3befd18d5e479870b78637400a1e84740652051839b7b7f55a4f9e69299003b`,
  terminal `ACTIVATION_REVIEW_PASS`.
- Implementation lock SHA-256:
  `20faa22021fc144b07883190d3034e8e020a1729648350861f3a93a9e985c80e`.
- Guarded-run summary SHA-256:
  `abf5096471c8a1617c9e7c4547836b9cf889a04661720b7474e64f1369e2d6d5`.
- Guard terminal: `PROGRESS_CONTRACT_VIOLATION`; Python exit code `0`;
  success path present; failure path absent.
- Producer output SHA-256:
  `2577a1b42e96b6dcede656bf08c641a5e3e765951008d5c0dd073e06fbf464ba`.
- Producer receipt SHA-256:
  `1ae1403bab718687b849d43bc471616b554b33ad252022b006cb930e4111c811`.
- Producer progress SHA-256:
  `f913338f693a42a31eca9cde22333620cfa780a55083a5dc5634c52688392125`.

The retained producer receipt reports 13,014 input rows, 26,028 output rows,
32 shape-change opportunities, 64 common arm abstentions, `mode=formal`,
`truth_joined=false`, and elapsed time 132.703 seconds. These observations localize
the execution-envelope failure; they do not authorize evaluation or constitute an
arm comparison result.

## Root cause

The producer wrote:

`2026-07-30T05:22:14.559650Z`

PowerShell 7 `ConvertFrom-Json` materialized that value as a UTC `System.DateTime`.
The guarded launcher then cast the value to a culture-formatted string before
`DateTimeOffset.TryParse`. The cast removed the UTC designator, so the second parse
treated `05:22:14` as Hong Kong local time and shifted it to the preceding UTC day.
The guard consequently emitted `last_progress_at predates this runner invocation`
even though the progress file write time and producer timestamps were fresh.

This is an execution-envelope timestamp-type bug, not evidence about either radial
geometry arm.

## Stop enforcement

- The evaluator was not invoked.
- Development truth and natural-event ledgers were not opened after the attempt.
- The complete producer output is not rescued, reclassified or reused as an R1
  scientific result.
- Old F-1B decision output remained sealed.
- No Confirmation, Android, product, runtime or safety authority was created.

## Forward repair

The shared guarded launcher now preserves deserialized `DateTime` /
`DateTimeOffset` values and converts them directly to UTC before comparison. Its
integration fixture was changed to emit the same trailing-`Z` representation and
passes. This repair does not reopen R1. A new R2 evidence version may independently
freeze the repaired execution envelope and a new output namespace before any new
producer attempt.
