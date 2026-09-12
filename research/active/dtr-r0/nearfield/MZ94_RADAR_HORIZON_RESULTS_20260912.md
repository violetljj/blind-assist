# MZ94: horizon admission changes decisions, but fails the proxy gate

Decision: `HORIZON_GATE_NOT_MET`. One frozen consumed-Development run,
code `4b8bda023862ce3736a564ae0e1fee4a5ce077fb`; no parameter sweep.
Protocol: [MZ94](MZ94_RADAR_HORIZON_20260912.md).

| Regime / method | TP | FP | FN | F1 | False segments | Fragments |
|---|---:|---:|---:|---:|---:|---:|
| ideal matched_hold | 457 | 41 | 34 | .9242 | 8 | 27 |
| ideal range-only | 463 | 41 | 28 | .9307 | 8 | 27 |
| ideal horizon A | 470 | 11 | 21 | .9671 | 4 | 28 |
| proxy matched_hold | 333 | 151 | 158 | .6831 | 29 | 49 |
| proxy range-only | 400 | 169 | 91 | .7547 | 34 | 54 |
| proxy horizon A | 388 | 164 | 103 | .7440 | 38 | 57 |

Same 1,920 frames and 491 positives per regime. Proxy A adds72 TP but loses17,
removes24 FP but adds37. It rescues44/69 frozen range-blocked FN; range-only
rescues47. Of108 prior current-Radar FP, A retains95; of168 accompanying TP,
it retains159. Range-only retains all108 FP and168 TP. Future-only TP rises
43 to95 for A,96 for range-only. No additional baseline event is missed;
maximum added shared-event delay is0.2s. Extra FP/segments/fragments fail the gate.

The mechanism now reaches final decisions. However, higher F1 than matched_hold
does not establish a trajectory-estimation benefit: simple range expansion scores
better on this proxy panel. Ideal gains cannot override proxy losses. Current
in-corridor occupancy includes t=0, even for a receding target; the task is not
finite-body collision. Four focused tests passed, including causal/permutation
behavior and admission integration. Scalar causal fitting/scoring runs on CPU.

Retain the frozen policy as a challenger/comparator only; its full replacement
claim failed here. The separately authorized fresh MZ96 comparison reuses it
unchanged and supplies additional scope-specific evidence. Do not tune these
consumed outcomes or attribute all gains to CV prediction.

Evidence: `artifacts.local/work/mz94-radar-horizon-20260912/run-v1/` contains
sealed predictions, result, paired/cohort outcomes and hashed receipt.
