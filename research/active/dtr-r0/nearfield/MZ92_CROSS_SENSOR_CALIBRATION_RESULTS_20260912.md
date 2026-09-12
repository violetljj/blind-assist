# MZ92: relative bias estimation improves, alert effect does not

Decision: `CALIBRATION_GATE_NOT_MET` / NEGATIVE_CONTROL for the full policy.
Retain matched_hold as the task baseline. Online relative-angle calibration shows
a narrow diagnostic benefit; it does not establish better obstacle alerts or
justify integrating the extra computation. No post-result threshold changes.

Frozen protocol/code: `460625fd`.
[Protocol](MZ92_CROSS_SENSOR_CALIBRATION_20260912.md).
Reused consumed MZ90 source:48 episodes/1920 frames and491 truth-positive frames
per regime. This is Development, not new independent confirmation or hardware.

| Sensor proxy arm | TP | FP | FN | F1 | UNKNOWN | Positive UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| matched_hold | 333 | 151 | 158 | .6831 | 1330 | 121 |
| Frozen MZ91 control | 333 | 151 | 158 | .6831 | 1330 | 121 |
| MZ92 calibrated | 333 | 151 | 158 | .6831 | 1330 | 121 |

All have29 false segments,49 within-event fragments,1 missed truth event and43
future-only TP. Final predictions are identical, not just equal aggregate counts:
zero added/lost TP/FP, zero lost baseline events and zero added delay. Strict FP
improvement fails; other gates pass by equality. Ideal also stays identical at
457TP/41FP/34FN, F1.9242,8 false segments,27 fragments,0 missed events.

## The calibration did acquire some useful information

Sensor_proxy supplies155 accepted co-observation measurement frames. At least
three recent paired frames are available at424/1920 timestamps, including283
without valid ToF. These include69 baseline FP (55 with no ToF) and168 baseline
TP.219of469 gate-qualified Radar returns fall within active calibration;
218 qualifying support scores differ from MZ91. Thus lack of any calibration
opportunity does not explain the unchanged alerts.

In a posthoc evaluator-only check on those424 active timestamps, mean absolute
error relative to the simulator's fixed Radar angle ambiguity falls from7.123deg
(zero correction) to2.128deg (estimated correction). This compares the same active
timestamps, includes repeated correlated frames, and only measures the declared
relative offset. It is not an estimate of total bearing error, calibrated posterior
coverage, independent samples, or true target localization accuracy. Simulator
offsets were never predictor inputs. Ideal true offsets and estimates are zero.

MZ91 vetoes1 raw qualifying return; MZ92 vetoes3. None changes a final alert after
the existing combination of other returns, ToF, hysteresis and hold. Reduced
relative-offset error has therefore not translated into enough decision-relevant
exclusion under this fixed readout. This result cannot isolate whether other
uncertainty, trajectory prediction or the readout is the remaining cause.

## Error origins remain unchanged

Using the same available-support attribution as MZ91 (not causal source removal):

| Baseline support | TP | FP | MZ92 removed TP / FP |
|---|---:|---:|---:|
| Valid current ToF | 107 | 16 | 0 / 0 |
| Real Radar only | 204 | 85 | 0 / 0 |
| Persistent phantom only | 7 | 45 | 0 / 0 |
| Transient Radar only | 0 | 1 | 0 / 0 |
| Mixed Radar | 6 | 3 | 0 / 0 |
| Hold without recent qualifying Radar | 9 | 1 | 0 / 0 |

The85 real-only FP and45 persistent-only FP persist. Fixed-ambiguity calibration
does not prove rejection of persistent phantoms. Original Radar candidate gates
also mean this attempt cannot recover dangerous returns already rejected there.

## Verification and retained evidence

Five focused tests pass: causal prefixes/slot permutation; known offset with
variance floor and expiry/reset/missing packets; ambiguous-pair abstention;
prior equivalence and no artificial angular-rate change; shared sensor rotation
cancellation. Independent pre-run design review checked coordinate convention,
association ambiguity and source-specific limits. No evaluator was used for fit.

Verified22 MZ90 input hashes and12 MZ91 receipt hashes before execution. MZ91
control predictions and matched_hold reproduce exactly in both regimes; both new
prediction files sealed before evaluator/provenance access. Provenance replay
matches all raw and evaluator arrays. All12 MZ92 output hashes verify. Later
`validation.json` contains the disclosed read-only calibration coverage/error
diagnostic and did not alter predictions.

Artifacts: `artifacts.local/work/mz92-cross-sensor-calibration-20260912/run-v1`.
Keep frozen code/protocol, predictions, result, provenance and receipts as
task-owned decision evidence. CPU scalar scoring of both arms together takes
0.983s ideal and0.930s sensor_proxy; these are host timings, not device latency.
No worker, paid allocation, training or persistent process remains.

This one attempt is closed. The relative-bias diagnostic may inform a separately
authorized hypothesis; the full recipe remains a negative control. Do not turn
the lower calibration error into an alert-quality claim, or sweep thresholds on
these consumed outcomes. No successor run was launched.
