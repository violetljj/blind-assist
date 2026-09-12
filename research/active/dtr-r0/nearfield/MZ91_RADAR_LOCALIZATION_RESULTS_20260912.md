# MZ91 result: no alert benefit from fixed localization veto

Decision: `LOCALIZATION_GATE_NOT_MET`, NEGATIVE_CONTROL. The fixed multiframe
localizer changes estimated support but does not change a single final alert in
either regime. Retain matched_hold as this comparison's baseline. Do not promote
the added computation, tune this run or infer that all multiframe localization
is ineffective. This is consumed Development evidence, not independent confirmation.

Frozen implementation/protocol: `fb592864`.
[Protocol](MZ91_RADAR_LOCALIZATION_20260912.md).
Source: sealed MZ90,48 episodes/1920 frames per regime,491 positive frames and26
truth events. Same raw input, ToF logic, hysteresis and one-frame hold for all arms.

| Sensor proxy method | TP | FP | FN | F1 | UNKNOWN | Positive UNKNOWN |
|---|---:|---:|---:|---:|---:|---:|
| Frozen matched_hold | 333 | 151 | 158 | 0.6831 | 1330 | 121 |
| Single-frame uncertainty veto | 333 | 151 | 158 | 0.6831 | 1330 | 121 |
| Five-frame localization veto | 333 | 151 | 158 | 0.6831 | 1330 | 121 |

All three have29 false segments,49 within-event fragments,1 missed event and43
future-only positive detections. Added delay, lost baseline events, added/lost
TP and added/removed FP are all zero. Six of seven gates pass by equality;
strictly fewer FP fails. Ideal predictions also remain identical457TP/41FP/34FN,
F1.9242,8 false segments,27 fragments and0 missed events.

## Why this is an active but ineffective intervention

Sensor_proxy has469 raw gate-qualified Radar returns;403 have at least three
observations in the causal track.402 qualified scores differ from the single-frame
ablation. Five-number support-score summaries are:

| Model | Min | Q1 | Median | Q3 | Max |
|---|---:|---:|---:|---:|---:|
| Single-frame | .0996 | .4297 | .5254 | .6523 | .7305 |
| Multiframe | .0078 | .5430 | .6563 | .7695 | .9648 |

Each model vetoes only one qualified return in sensor_proxy; neither veto changes
an alert after unchanged other-return support, hysteresis, ToF and hold logic.
Ideal has1 single-frame and2 multiframe vetoed returns, also no alert change.
The localization code therefore ran and altered internal estimates, but the fixed
exclusion rule has almost no effective intervention surface here. These scores
are uncalibrated model support fractions; larger values are not evidence of more
accurate localization. No posterior calibration or spatial-error gain is claimed.

## Separate persistent clutter and real-object support

The following counts describe available baseline support, not causal ablations.
For no-ToF alerts use the most recent qualified Radar return within current and
previous two frames (the existing Radar plus hold lifetime). They are not object
identities exposed to the predictor. Mixed frames remain mixed.

| Baseline support category | TP | FP | Multiframe removed TP/FP |
|---|---:|---:|---:|
| Current valid ToF | 107 | 16 | 0 / 0 |
| Real Radar only | 204 | 85 | 0 / 0 |
| Persistent phantom only | 7 | 45 | 0 / 0 |
| Transient Radar only | 0 | 1 | 0 / 0 |
| Mixed Radar origins | 6 | 3 | 0 / 0 |
| Hold without recent qualifying Radar | 9 | 1 | 0 / 0 |

Thus none of the85 real-only FP or45 persistent-only FP is removed. This baseline
is matched_hold; earlier joint_spatial's126FP/202TP fallback audit is a different
denominator. Phantom/real-static kinematic overlap remains a limitation; source
periodicity and detection-probability differences were not used as features.

## Integrity, limits and delivery

Five focused tests pass: prefix and slot permutation invariance; unrestricted
acceptance reproduces matched_hold; common angle bias floor survives added
history; missing evidence remains UNKNOWN; deterministic continuous geometry
matches the benchmark over100 checks. Pre-run independent code review identified
and corrected the strict-depth contact edge. The uncertainty model explicitly
does not propagate the full correlated pose posterior into angle/rate covariance.

All22 MZ90 sealed input hashes verify; reconstructed baseline predictions match
in both regimes. Both candidate prediction files were sealed before evaluator
access. Label-only source replay exactly reproduces all raw/evaluator arrays.
All12 output receipt hashes verify; later `validation.json` records read-only
score diagnostics without modifying predictions. No training or parameter search.

Artifacts: `artifacts.local/work/mz91-radar-localization-20260912/run-v1`, retained
as task-owned decision evidence with result/receipts, per-frame predictions,
provenance and frozen code. CPU scalar backend recorded0.880s ideal and0.814s
sensor_proxy for both candidate arms together, not device latency or a baseline
runtime speedup. No worker, paid allocation or persistent process remains.

This run ends here. A future proposal must change an evidenced source of spatial
information or the justified uncertainty/readout model, rather than claim that
more persistence alone is the missing ingredient. No successor was launched.
