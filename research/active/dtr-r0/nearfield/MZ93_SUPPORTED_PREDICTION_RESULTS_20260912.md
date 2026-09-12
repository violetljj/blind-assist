# MZ93: extra return exclusions do not reach final decisions

Decision: `SUPPORTED_PREDICTION_GATE_NOT_MET`, NEGATIVE_CONTROL. Keep matched_hold.
Frozen protocol/code57d149fb; [protocol](MZ93_SUPPORTED_PREDICTION_20260912.md).
One fixed run per consumed MZ90 regime, no training, tuning or source expansion.

| Sensor proxy arm | TP | FP | FN | F1 | UNKNOWN | Positive UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| matched_hold | 333 | 151 | 158 | .6831 | 1330 | 121 |
| Frozen MZ92 control | 333 | 151 | 158 | .6831 | 1330 | 121 |
| MZ93 supported prediction | 333 | 151 | 158 | .6831 | 1330 | 121 |

All have29 false segments,49 within-event fragments,1 missed event,43 future-only
TP. Zero added/lost TP/FP, zero lost baseline events and zero added delay. Strict
FP gain fails; other gates pass by equality. Ideal remains457TP/41FP/34FN,
F1.9242,8 false segments,27 fragments,0 missed events. All predictions are equal,
not merely aggregate metrics.85 real-only FP and45 persistent-only FP remain.

## Where the intervention disappears

Read-only reconstruction from sealed scores and unchanged branch logic:

| Stage | Ideal changed frames | Sensor proxy changed frames |
|---|---:|---:|
| Any accepted raw Radar return | 1 | 2 |
| Radar hysteresis output | 1 | 2 |
| After ToF/Radar branch selection | 0 | 0 |
| After one-frame hold | 0 | 0 |

In sensor_proxy, vetoed raw returns rise3 to11, an additional8. Multiple-return
support reduces this to2 changed raw-support frames (807,1182); hysteresis output
differs at1183,1184. After ToF branch selection, no difference remains. Thus the
one-frame hold is not responsible for erasing the last differences in this run.
The internal readout operates, but it does not intervene on final errors.
Ideal has2 to3 vetoed returns, a changed raw frame1893 and hysteresis frame1894,
also no difference after branch selection. Frame numbers index the sealed regime.

This narrows the failure beyond an unchanged F1: the fixed front-end exclusion
has little relevant decision coverage. It does not prove that more aggressive
exclusion, removing ToF authority or disabling hysteresis is beneficial. No such
counterfactual policies were run. Repeating additional upstream filters without
showing their final decision coverage is not supported by these results.

## Verification and limits

Five focused tests pass: exact MZ92 score decomposition; causal prefixes/slot
permutation and subset relation; current versus future geometry; missing evidence
remains UNKNOWN; inward direction requires history and rate evidence. Independent
review found no added implementation defect and recorded that suppressed future
mass includes possible radial boundary effects; the1.96sigma directional rule
does not imply calibrated95-percent collision confidence.

All22 MZ90 and12 MZ92 input receipt hashes verify; control predictions reproduce
MZ92 exactly in both regimes. Both new prediction payloads sealed before evaluator
access. Label-only source replay matches raw/evaluator arrays. All12 output hashes
verify. `validation.json` holds later read-only stage diagnostics; the parent
`pre-run-diagnostic.json` discloses the consumed score analysis preceding this run.
Original negative predictions were not changed.

Artifact: artifacts.local/work/mz93-supported-prediction-20260912/run-v1.
Preserve code, predictions, reports and receipts as task-owned decision evidence.
CPU scalar scoring for the two readouts sharing one track pass took0.573s ideal,
0.500s proxy; not device latency or a controlled speed comparison to earlier runs.
No persistent process, worker or paid allocation remains. This is horizontal
synthetic Development evidence, not hardware, finite-body collision or safety.
The experiment ends here; no automatic successor or parameter sweep.
