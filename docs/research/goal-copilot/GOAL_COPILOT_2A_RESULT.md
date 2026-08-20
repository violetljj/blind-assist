# GOAL-COPILOT-2A result

Terminal status: `GOAL_COPILOT_2A_COMPLETE`.

Decision: `GOAL_COPILOT_2B_NOISE_ROBUST_SKY_SEARCH_PROTOCOL_DESIGN_ADMITTED`.

GC2-A made zero model calls. GOAL-COPILOT-1 remains permanently closed, and
neither GC1 nor GC2-B model search is authorized by this result.

## Frozen identity and replay

- execution commit: `2fe3acd156be630d91ffd47c19726e067d209d47`;
- winner SHA-256:
  `24d4e57374dd99363700ae881d18db536e48ec5f79f39e95c5b873e96edbc3a1`;
- protocol seal digest:
  `149a605be0d29c828022f0d452ec6669874884c4991f6d594c16c2defccf692f`;
- condition cells per policy: `22`;
- model calls: `0`;
- deterministic replay: `PASS`.

## Primary result

The frozen winner retained `12/12` completion, `0` unsafe guidance, and `0`
premature completion in CLEAN. Under the preregistered primary
`COMBINED_MODERATE` condition it had:

- completion: `0/12` (`0.0`);
- eligible true reacquisition: `0/3` (`0.0`);
- unsafe guidance: `0`;
- premature completion: `1`;
- wrong-way actions: `68`;
- timeouts: `12`.

This trips three frozen admission clauses: completion below `0.8`, premature
completion above `0`, and eligible true reacquisition below `0.8`. The unsafe
guidance clause did not trip.

The single premature completion occurred in `dev_track_fine_alignment` after
the corrupted observation stream drove the trace
`FORWARD, FORWARD, ALIGN_RIGHT, SCAN_RIGHT, SCAN_RIGHT, COMPLETE` before the
hidden completion state. This is a mechanism observation in the deterministic
harness, not a rate estimate.

## Robustness structure

Winner completion by isolated corruption (`MILD / MODERATE / STRESS`) was:

| Corruption | Completion | First enumerated failing severity |
|---|---:|---|
| Target dropout | `12/12 / 12/12 / 2/12` | `STRESS` |
| Bearing jitter | `11/12 / 9/12 / 0/12` | `MILD` |
| False target | `10/12 / 11/12 / 8/12` | `MILD` |
| Nearness error | `12/12 / 8/12 / 7/12` | `MODERATE` |
| Tracking collapse | `12/12 / 12/12 / 2/12` | `STRESS` |
| Delayed evidence | `12/12 / 12/12 / 0/12` | `STRESS` |

The isolated deterministic schedules are diagnostic cells, not samples from a
calibrated distribution. In particular, false-target completion is non-monotonic
between MILD and MODERATE, so the table must not be read as an estimated severity
response curve. The interpretable failure mechanisms are directional instability
under bearing jitter, false-target diversion during search/reacquisition, belief
overstatement and premature completion under nearness error, and timeout/recovery
collapse under sustained dropout, tracking collapse, or delayed evidence.

Combined completion was `10/12` at MILD, `0/12` at MODERATE, and `0/12` at
STRESS. No condition produced observed unsafe motion guidance, but the completion
and premature-completion failures prevent a safety or readiness claim.

## Decision and claim ceiling

The only next authorized route is GC2-B noise-robust Sky search **protocol
design**. Before any search, it must separately freeze its search surface,
budgets, provider identity, evaluator ownership, and acceptance gates. This
closeout does not authorize Sky/EvoX/Codex calls or a multi-arm experiment.

The claim ceiling is
`consumed_dev_deterministic_perception_corruption_characterization_only`.
These results are not fresh evidence, real-RGB evidence, device evidence,
statistical robustness estimates, proof of safer guidance, or product readiness.

Authoritative local evidence is stored under
`artifacts.local/evidence/goal-copilot/GOAL-COPILOT-2A/`.
