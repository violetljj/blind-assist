# MZ90 common observable-contract results

Decision: `OBSERVABLE_CONTRACT_COVARIANCE_TRANSFER_NOT_ESTABLISHED`.

The exact joint covariance recipe keeps an isolated benefit under ideal
observations but does not transfer its no-added-FP benefit to the declared sensor
proxy. No complete policy replaces the simple matched geometry/hold controls.
This does not disprove covariance mathematics or all probabilistic fusion; it
limits what MZ89's idealized mechanism result establishes.

## Literature-led implementation

[Literature and frozen protocol](MZ90_LITERATURE_AND_PROTOCOL_20260912.md) summarize
four inspected papers and the current official ST UM3109 Rev12. Their useful
implications were multi-hypothesis Radar ambiguity, limited observability,
separate missing/valid states, and caution about temporal accumulation. They did
not establish that soft spatial compatibility would solve the MZ89 tradeoff.

Code/protocol/tests were frozen at `5bcf8d75` before the sole run. One common world
generator created48 scenes of40 frames, with491 positive frames and26 continuous
hazard events per regime. Ideal and sensor_proxy have identical truth and differ
only in declared sensor generation. Both remove truth motion/height hints and
provide raw rotating-frame Radar; both use the same causal estimated yaw and
nearest-two valid-return budget. Current and matched hard geometry are separated
so the new temporal baseline is not weakened by MZ88's legacy first-slot handling.

## Paired regime results

Each row has1,920 frames. Regimes are paired observations of the same48 scenes,
not96 independent scenes. Numbers are not comparable directly with MZ89's panel.

| Arm | Ideal TP/FP/FN | Ideal F1 | Sensor proxy TP/FP/FN | Sensor proxy F1 |
| --- | --- | ---: | --- | ---: |
| Current geometry + Radar fallback | 336/37/155 | .7778 | 292/132/199 | .6383 |
| Causally matched hard geometry | 455/37/36 | .9257 | 309/142/182 | .6561 |
| Matched + one-frame missing-ToF hold | 457/41/34 | .9242 | 333/151/158 | .6831 |
| MZ88 full negative control | 347/106/144 | .7352 | 309/144/182 | .6547 |
| Independent marginals + spatial Radar | 221/36/270 | .5909 | 262/128/229 | .5948 |
| Joint covariance + coarse Radar | 400/87/91 | .8180 | 311/142/180 | .6589 |
| Joint covariance + spatial Radar | 299/36/192 | .7240 | 276/134/215 | .6127 |

The isolated covariance change adds78TP/0FP in ideal, with no TP removed. Under
sensor_proxy it adds14TP/6FP, also with no TP removed. False segments increase23
to28 versus independent marginals, though missed events improve5 to2. Thus the
predeclared covariance-transfer gate fails; the reduced uncertainty is not a
free recall improvement in this source. The inherited yaw-only model omits
measurement angular/range errors and full shared Radar/ToF pose covariance.
This combined stress does not isolate which omission causes the six false frames.

Full joint+spatial loses33TP versus matched hard, and57TP versus matched hold.
Its134FP are below142/151, but that tradeoff does not give better F1. Relative to
matched hard, one previously detected event is lost and maximum added first
correct support delay among shared detections is0.3s. It fails full qualification.

| Sensor-proxy arm | UNKNOWN | Positive UNKNOWN | False segments | Missed events | Within-event fragments | Future-only TP/146 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current geometry | 1363 | 145 | 22 | 5 | 46 | 24 |
| Matched hard | 1363 | 145 | 29 | 1 | 55 | 35 |
| Matched hold | 1330 | 121 | 29 | 1 | 49 | 43 |
| MZ88 full | 1426 | 170 | 21 | 5 | 37 | 31 |
| Independent + spatial | 1468 | 206 | 23 | 5 | 58 | 24 |
| Joint + coarse | 1385 | 156 | 25 | 2 | 46 | 34 |
| Joint + spatial | 1428 | 191 | 28 | 2 | 67 | 32 |

Future-only means no direct current hazard but a positive future geometric
intersection; it is not a true object-motion label. A current-only system may
coincide with this truth through broad Radar or erroneous geometry, so such TP
do not establish a correct prediction of the particular future object.

## Observation availability and responsibility

| Observable quantity | Ideal | Sensor proxy |
| --- | ---: | ---: |
| ToF packet gaps | 0 | 219 |
| Received ToF packets with no valid return | 931 | 1472 |
| Frames with any valid ToF | 989 | 229 |
| Raw valid ToF zones | 1348 | 239 |
| Selected valid ToF returns | 1306 | 239 |
| Positive frames with any valid ToF | 440/491 | 144/491 |
| Frames with any valid Radar | 1590 | 1678 |
| Missing IMU increments | 0 | 48 |

These availability rates result from our declared finite FoV, random geometry,
range cutoffs, reflectivity draws and detection probabilities. They are NOT
estimated frequencies for VL53L8CX or any actual Radar. Radar has no packet-drop
model this round; its detection misses and clutter remain distinct.

Read-only decomposition of sealed predictions finds that126 of134 joint+spatial
FP in sensor_proxy occur in the inherited **no-valid-ToF Radar fallback**. That
same branch provides202 of276TP. The remaining valid-ToF branch has74TP/8FP;
all six covariance-added FP are in this branch. In ideal, fallback supplies32TP
and35 of36FP. Consequently, simply changing the spatial gate used when ToF is
present leaves the dominant proxy FP branch untouched. With the fallback and
other decisions held fixed, even removing all eight valid-ToF FP leaves126FP.
This is a conditional structural observation, not proof that Radar fallback can
be replaced without losing its202TP.

The shared any-valid-ToF flag also does not establish observability of every
hazard: a valid background or outside return can suppress Radar fallback while
another object is not observed. The nearest-two budget can discard a third
object. These common inherited limits are now explicit; this run does not test
a local-observability replacement or prove its benefit.

## Decision and evidence

Preserve MZ89's original isolated46TP result in its original domain. Do not claim
its no-added-FP benefit transfers to the sensor proxy. Retain MZ90's joint+spatial
transfer result as NEGATIVE_CONTROL and the shared raw observation/evaluator
harness as diagnostic infrastructure. Simple matched/hold controls remain
comparators, not a promoted robust product. This study provides no justification
for choosing a soft Radar head as the sole next change. Stop without a parameter
search, new source panel, training, hardware work or automatic successor.

The common-scene source and all inputs/outputs are under
`artifacts.local/work/mz90-observable-contract-20260912/run-v1/`.
Both prediction files were sealed before either evaluator was opened:

- ideal SHA-256: `b336edd87fdf938fcaa44def57dde9a1618f80bd0bb36247f1147a9639afbdfa`
- sensor_proxy SHA-256: `9bd42940d585b4838048f647d36e8e7d3f5b5d8248cd9d62375f29e6e9bad28d`

All22 sealed hashes verify. `validation.json` records the later read-only
responsibility decomposition and observation-schema check; it does not change
predictions. Eleven focused tests pass, covering coherent continuous geometry,
paired truth, source reproducibility, raw coordinate/schema boundaries, invalid
hidden returns, missing versus no-return, nearest-two order invariance, causal
prefixes, non-reseeding hold and per-event misses. Independent pre-run code review
found no scoring blocker. CPU predictor batches took0.2224s ideal and0.0656s proxy;
the complete paired run took0.6215s. The smaller proxy runtime reflects sparse
returns, not greater model efficiency. No worker, paid allocation or persistent
process was started; MZ90 owns the retained evidence directory.

This is an uncalibrated horizontal point-scene surrogate with arbitrary error and
detection assumptions, not a full8x8 optical/electromagnetic simulator. Range sigma
is retained in raw outputs but not consumed by the frozen MZ89 policy. No actual
height, finite body swept volume, thin-pole detectability, obstacle identity,
natural-data reliability, device latency, deployment or safety claim is made.
Reproduction requires the checkout at the frozen Git revision and its Python
dependencies; copied scripts alone are not a standalone reproducible package.
