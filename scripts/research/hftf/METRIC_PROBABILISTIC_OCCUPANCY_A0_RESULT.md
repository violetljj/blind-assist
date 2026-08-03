# Metric probabilistic occupancy A0 result

Date: 2026-08-03

Terminals:

- `METRIC3D_EMPIRICAL_RESIDUAL_OCCUPANCY_NOT_SUPPORTED`
- `UNIDEPTH_DETERMINISTIC_CLEARANCE_NOT_SUPPORTED`
- `UNIDEPTH_CONFIDENCE_STRATIFIED_OCCUPANCY_NOT_SUPPORTED`
- `MOTION_CONDITIONED_LEARNED_OCCUPANCY_REMAINS_OPEN`
- `RESEARCH_MAINLINE_UNCHANGED / DEFAULT_APP_UNCHANGED`

## Metric3D empirical residual probability

Calibration used 119 fixed-world-floor Development residuals per band.
Evaluation used 1,440 known band×horizon opportunities from consumed
`walking_rpy`.

| Measure | Result | Gate | Pass |
|---|---:|---:|:---:|
| Brier reduction vs deterministic | 12.28% | >=15% | no |
| Log-loss reduction | 51.21% | >=20% | yes |
| Expected calibration error | 0.10279 | <=0.10 | no |
| High-confidence-clear false-clear | 9.80% | <=5% | no |
| High-confidence-clear coverage | 24.10% | >=10% | yes |
| Occupied recall at P>=0.50 | 84.10% | >=85% | no |

The large log-loss improvement is a positive ranking/soft-probability signal,
but only two of six continuation gates passed. The residual distribution did
not transfer well enough from static/translation-dominant calibration to RPY.

## UniDepth source and confidence conditioning

The deterministic UniDepthV2-S field on the same fixed reference achieved
0.19662 m clearance MAE and 87.80% collision agreement at 32.66 ms/frame, but
false-clear was 7.46% and temporal-delta MAE 0.18877 m. It did not pass A0.

UniDepth confidence was then frozen as the median `log1p(confidence)` near each
band's clearance support and used to create 12 band×confidence residual strata.
On 1,434 known opportunities:

| Measure | Result | Gate | Pass |
|---|---:|---:|:---:|
| Brier reduction vs deterministic | -27.65% | >=15% | no |
| Log-loss reduction | 38.43% | >=20% | yes |
| Expected calibration error | 0.14527 | <=0.10 | no |
| High-confidence-clear false-clear | 7.44% | <=5% | no |
| High-confidence-clear coverage | 23.43% | >=10% | yes |
| Occupied recall at P>=0.50 | 91.56% | >=85% | yes |

Native confidence increased occupied recall but worsened overall probability
accuracy. Three of six gates passed; no fresh sequence was opened.

## Decision

Stop unconditioned empirical residual CDFs and fixed confidence-quartile
stratification. Do not tune probability thresholds, confidence bins, smoothing,
or clearance geometry on `walking_rpy`.

The remaining coherent successor is a direct low-capacity occupancy model in
which clearance margin, depth confidence, ground-fit quality, obstacle support,
and causal global camera-motion/HFTF features jointly predict band×horizon
occupancy. It must be evaluated by held-out whole windows before any new fresh
source is opened. This is a different estimand and representation, not another
post-hoc threshold rescue.
