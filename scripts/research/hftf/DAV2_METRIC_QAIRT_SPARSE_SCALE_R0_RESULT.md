# DA V2 Metric QAIRT + sparse scale R0

Date: 2026-08-03

Terminals:

- `DAV2_METRIC_392X518_HTP_DEPLOYMENT_PARITY_SUPPORTED`
- `DAV2_METRIC_392X518_UNCALIBRATED_TASK_QUALITY_FAIL`
- `PER_SEGMENT_SPARSE_SCALE_ANCHOR_DEVELOPMENT_SIGNAL_5_OF_5`
- `GLOBAL_ONE_SHOT_CAMERA_SCALE_FALSE_CLEAR_FAIL`
- `PERIODIC_METRIC_SCALE_ANCHOR_REQUIRED`

## Decision

Keep one conditional mobile candidate:

```text
external calibrated RGB
  -> DA V2 Small Metric Hypersim 392x518 on HTP
  -> sparse three-band metric scale anchor (for example multi-zone ToF)
  -> stale or missing anchor => UNKNOWN
  -> calibrated left / center / right clearance
```

Do not keep either the uncalibrated model or a one-time global scale as a
standalone metric-clearance source. This is the first tested arm in this branch
that combines a metre-trained checkpoint, genuine SM8650 HTP execution near
eight pure executions per second, and a conditional 5/5 task-gate signal. The
signal depends on recurring metric anchors and is not a pure-RGB result.

## Deployment result

The exact GPU-quality geometry (`518x686`) and one preselected reduced arm
(`392x518`) were exported from the same official Hypersim ViT-S checkpoint.
PyTorch-to-ONNX maximum differences on the deterministic input were about
`0.0013-0.0014 m`. Both graphs converted to HTP FP16 and executed with four HVX
threads and no observed CPU fallback.

| Arm | MACs | Cached HTP mean | Accelerator mean | ORT/HTP relative difference |
|---|---:|---:|---:|---:|
| `518x686` | 15.113 G | 277.92 ms | 274.56 ms | 0.631% |
| `392x518` | 8.636 G | 123.19 ms | 119.99 ms | 0.786% |

`burst` did not materially change execution. Cached runs retained the existing
QAIRT warning that the compatible `HTP_V75_SM8650_4MB` record's VTCM/DSP
metadata did not exactly match the target. Cross-platform GPU/phone timing is
diagnostic only.

## Task quality

The reduced arm was selected before its 120-frame consumed TUM task result was
opened. Without any scale anchor it failed badly: clearance MAE `0.6942 m`,
collision agreement `64.87%`, false-clear `35.01%`, and only 2/5 gates passed.
Resolution reduction alone is therefore not a Pareto solution.

A separate fixed calibration diagnostic used a shared scale across all three
bands. It took the median of `metric clearance / candidate clearance` from each
segment's first ten frames and evaluated only the remaining twenty frames.
There was no threshold, model, band, or calibration-length search.

| Scale role | Eval frames | MAE | Agreement | False-clear | Temporal delta MAE | Gates |
|---|---:|---:|---:|---:|---:|---:|
| no scale anchor | 120 | 0.6942 m | 64.87% | 35.01% | 0.1270 m | 2/5 |
| recurring segment-prefix anchor | 80 | 0.0981 m | 93.77% | 4.95% | 0.0858 m | 5/5 |
| one global first-prefix scale | 110 | 0.1300 m | 91.13% | 6.99% | 0.1039 m | 4/5 |

The strict global check is decisive: camera intrinsics plus a one-time scale do
not establish stable enough false-clear behavior. The positive result means a
small metric source can repair the fast observer when refreshed, not that the
RGB model became metrically reliable by itself.

## Implemented boundary

`metric_scale_anchor.py` provides the causal runtime primitive: robust shared
three-band scale estimation, strictly ordered anchor updates, and fail-closed
`UNKNOWN` for absent, future, stale, or empty anchors. The expiry interval is a
caller-owned protocol value; this result does not choose it from consumed
outcomes.

The next experiment is hardware registration, not another depth-model search:
bind a low-resolution multi-zone ToF field to the RGB left/center/right bands,
measure anchor availability and age, and rerun the same task gates on the final
camera. Until that succeeds, this candidate remains host/device Development
evidence and cannot drive alerts or change the mainline/default App.

Machine-readable result:
`DAV2_METRIC_QAIRT_SPARSE_SCALE_R0_RESULT.json`.
