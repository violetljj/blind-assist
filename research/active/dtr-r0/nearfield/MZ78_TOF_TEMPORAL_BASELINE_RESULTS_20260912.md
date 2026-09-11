# MZ78 strong pure-ToF geometry and temporal baseline

Decision: `MZ78_TOF_GEOMETRY_TEMPORAL_DOMINATES_ON_CONSUMED_SIMULATION`.

On the existing MZ77 controlled approach replay, a label-free 8x8 two-return
geometry expert substantially outperforms the frozen learned DIVERSE system.
Causal three-sample gap extrapolation removes all 30 query-bit losses caused by
the fixed central outage without adding a false query bit. Retain this expert as
the controlled-simulation baseline and reject the fixed OR dual-expert rule: OR
adds only one true bit but inherits 28--31 additional false bits from DIVERSE.

## Question and fixed method

The question was whether the learned RGB+ToF system still shows incremental
value against a credible pure-ToF opponent with the same packet and temporal
history. This is `EXPLORE` on already consumed Development evidence, not fresh
confirmation.

MZ78 reads only MZ77's ordered 64-zone, two-return ranges and validity flags,
clip boundaries, nominal timestamps, and the fixed camera/body calibration. It
projects every usable return through its zone-center ray into the unchanged
BODY/HEAD and near/far boxes. The temporal arm uses the previous five samples,
requires two direct observations, takes the median consecutive closing slope,
and extrapolates at most three missing samples with a fixed 3 m/s speed cap.
Static/receding returns are never filled. It neither predicts beyond the current
3 m corridors nor converts an invalid return to clearance.

The code, configuration and predictions were hash-sealed before the frozen MZ77
evaluator was opened. There were zero fits, learned parameters, new cutoffs or
threshold searches. `TEMPORAL_OR_DIVERSE` is a fixed logical OR, included only
to test the proposed dual-expert architecture.

## Paired result

Each condition contains 130 positive and 510 negative native query bits across
the four 40-frame clips. Counts are TP / FP / FN / TN over the original four
BODY/HEAD near/far queries.

| Method | CLEAN | CENTER_GAP |
| --- | --- | --- |
| Geometry, current frame | 124 / 2 / 6 / 508 | 94 / 2 / 36 / 508 |
| Geometry + causal temporal fill | **124 / 2 / 6 / 508** | **124 / 2 / 6 / 508** |
| Frozen DIVERSE | 79 / 33 / 51 / 477 | 60 / 30 / 70 / 480 |
| Temporal geometry OR DIVERSE | 125 / 33 / 5 / 477 | 125 / 30 / 5 / 480 |

For head bar, thin pole and wall, temporal geometry first hits every eligible
far query at 3.1 m and every detected near query at 1.6 m in both conditions.
It preserves all bar HEAD samples and all pole/wall HEAD_FAR samples through the
two three-frame gap windows. The remaining six misses are three BODY_NEAR samples
for the thin pole and three for the wall, caused by zone-center projection at
the closest poses. Its two false bits are one thin-pole BODY_FAR bit and one wall
BODY_NEAR bit. The OR rule recovers one of those six BODY_NEAR bits but keeps
DIVERSE's false alerts, so it is not retained.

The CLEAN packet has 58 all-invalid frames; CENTER_GAP has 64. These include all
40 control frames, where every model-visible ToF slot is invalid. No method alerts
on the control, but that is `UNKNOWN` sensor coverage rather than measured clear
space. Every remaining frame also has only partial zone/slot coverage. Temporal
fill synthesizes 206 slots on 28 CLEAN frames and 584 slots on 34 CENTER_GAP
frames; those are model estimates, not observed returns.

## Interpretation and limits

This result changes the immediate algorithm decision: the current learned fusion
does not demonstrate added value on this source once a strong physical ToF
baseline is included. The controlled packet is generated from simulator native
depth and is much cleaner than measured hardware; the result does not establish
that an actual 8x8 sensor will preserve a thin pole, head bar, wall or two ordered
surfaces this reliably. Conversely, DIVERSE was not trained as a temporal model.

Retain the MZ78 geometry/temporal operator as `COMPONENT_OR_CHALLENGER / CHALLENGER`
for comparable simulated packets. The next decision-changing evidence, if later
authorized, is the unchanged expert on calibrated real packet recordings with
range status and target-independent truth. Do not tune it on MZ77, claim hardware
dominance, or treat missing observations as negative evidence. The MZ77 corpus
also lacks the requested independent ordinary BODY-obstacle clip, so four-class
scenario coverage remains incomplete.

## Verification and reproduction

Fourteen focused tests pass: six expert tests plus the existing eight MZ77
sequence-metric tests. The final complete CPU replay and scoring took 0.189 s
(`TASK_NOT_GPU_SUITABLE`); no process, GPU allocation or worker remained.

```powershell
Set-Location E:\linnan\linnan\research\active\dtr-r0\nearfield
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe -m unittest test_tof_geometry_temporal_expert.py test_mz77_sequence_metrics.py
E:\codex-tools\tools\venvs\blindassist-torch-gpu\Scripts\python.exe mz78_tof_temporal_baseline.py --root E:\linnan\linnan --output E:\linnan\linnan\artifacts.local\work\mz78-tof-temporal-baseline-20260912\run-v2
```

Canonical evidence: `artifacts.local/work/mz78-tof-temporal-baseline-20260912/run-v2/`
contains sealed predictions, predictor receipt, full per-clip metrics, aggregate
summary, source snapshots and the final hash receipt.
