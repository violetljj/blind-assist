# MZ79 datasheet range-cap sensitivity

Decision: `MZ79_DATASHEET_RANGE_CAP_BREAKS_MZ78_CURRENT_RGB_RESCUE_INSUFFICIENT`.

MZ78's 124/130 true query bits do not survive a bounded range-availability
stress derived from the VL53L8CX datasheet. Under the 8x8 continuous 15 Hz
typical endpoints, pure ToF temporal geometry falls to 87/130 for a 17%
reflectance target in darkness and to 42, 32 and 20/130 at 5 klux for 88%, 54%
and 17% reflectance. The current frozen RGB branch recovers some missing bits,
but a fixed OR introduces 64 additional false bits in every condition. Retain
MZ78 as the ideal-packet comparator, not a sensor-realistic baseline; current
RGB is not an acceptable rescue.

## Question, source and fixed intervention

The falsifying question is whether MZ78 remains strong after model-visible ToF
availability is limited by published ambient-light, reflectance and zone-position
range endpoints. This is `EXPLORE` on the consumed MZ77 Development sequence.

[ST DS14161 Rev 12 Table 20](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)
reports these typical maximum distances for continuous 8x8 ranging at 15 Hz:

| Reflectance | Dark inner / corner | 5 klux inner / corner |
| --- | ---: | ---: |
| White, 88% | 4.00 / 3.95 m | 1.55 / 1.40 m |
| Light gray, 54% | 3.30 / 3.10 m | 1.40 / 1.25 m |
| Gray, 17% | 2.45 / 1.95 m | 1.15 / 0.95 m |

The gray 5 klux minimum endpoints, 0.90 / 0.70 m, are an additional conservative
stress. The code applies each endpoint pair as a hard cap to MZ77 CLEAN packet
returns and radially interpolates from the center four zones to the corners.
Every removed return becomes invalid/`UNKNOWN`. The original MZ78 five-frame,
three-gap causal expert then runs unchanged. The frozen MZ77 RGB margins are
reused because RGB input is identical; `TEMPORAL_OR_RGB` is a fixed logical OR.

The datasheet supplies maximum-range endpoints, not a per-distance detection
curve. No probability was invented. Radial interpolation, uniform scene-wide
reflectance and hard cutoff are explicit sensitivity assumptions. Code,
configuration and predictions were sealed before the evaluator was opened;
there were zero fits, threshold searches or outcome-driven retries.

## Result

Each condition has 130 positive and 510 negative native BODY/HEAD near/far query
bits over 160 frames. Counts are TP / FP / FN / TN.

| Condition | ToF temporal geometry | RGB | ToF OR RGB |
| --- | --- | --- | --- |
| Ideal 4 m | 124 / 2 / 6 / 508 | 22 / 64 / 108 / 446 | 125 / 66 / 5 / 444 |
| Dark, 88% typical | 124 / 2 / 6 / 508 | same | 125 / 66 / 5 / 444 |
| Dark, 54% typical | 124 / 1 / 6 / 509 | same | 125 / 65 / 5 / 445 |
| Dark, 17% typical | **87 / 1 / 43 / 509** | same | 89 / 65 / 41 / 445 |
| 5 klux, 88% typical | **42 / 0 / 88 / 510** | same | 50 / 64 / 80 / 446 |
| 5 klux, 54% typical | **32 / 0 / 98 / 510** | same | 41 / 64 / 89 / 446 |
| 5 klux, 17% typical | **20 / 0 / 110 / 510** | same | 33 / 64 / 97 / 446 |
| 5 klux, 17% minimum | **10 / 0 / 120 / 510** | same | 27 / 64 / 103 / 446 |

ToF recall is therefore 95.4% ideal, 66.9% for dark gray, 32.3% for 5-klux
white, 24.6% for 5-klux light gray, 15.4% for 5-klux gray typical and 7.7% at
the gray minimum endpoint. The first BODY/HEAD union warnings contract from the
ideal 1.6/3.1 m query onsets to about 1.4/1.5 m for 5-klux white, 1.2/1.3 m for
light gray and 0.9/1.1 m for gray. At the gray minimum, BODY union is never
detected in the pole or wall clips.

The 5-klux conditions leave only 30, 24, 18 and 10 of 160 frames with any direct
valid return. All-invalid frames rise to 130, 136, 142 and 150. These counts
include the 40 control frames and the early out-of-range approach samples; they
are coverage failure, not negative obstacle evidence.

RGB contributes real but poor-quality complementary information: relative to
stressed geometry, OR adds 8, 9, 13 and 17 true bits across the four 5-klux
conditions, but also 64 false bits every time. This rejects the current fixed OR
as a deployment candidate. It does not reject better learned/selective fusion;
it establishes the concrete target: recover unavailable-range positives without
inheriting RGB's present false-alert burden.

## Interpretation and limits

This result falsifies the robust-4-m reading of MZ78 and restores a legitimate
multimodal research problem, but it is not hardware evidence. Datasheet maximum
range is converted to a deterministic cutoff; no signal rate, ambient rate,
target status, integration-time dynamics, range error, cover glass, sub-zone
occupancy, material map or device motion is simulated. Table 20 is measured at
15 Hz while MZ77 poses are sampled at nominal 10 Hz. Uniform reflectance also
changes all visible surfaces at once rather than only the intended target.

Retain the range-cap operator as `COMPONENT_OR_CHALLENGER / COMPONENT` for
sensitivity analysis, MZ78 as the ideal-packet baseline, and RGB as a weak
complementary diagnostic. Do not tune caps on this consumed cohort or proceed
automatically to dual-surface and ego-motion experiments. The next stronger
evidence is calibrated real packet survival/status across distance, ambient
light, reflectance and zone, followed by the unchanged MZ78 replay.

## Verification and evidence

Eighteen focused tests pass: four range-cap tests, six MZ78 expert tests and the
existing eight sequence-metric tests. CPU execution took 0.365 s
(`TASK_NOT_GPU_SUITABLE`). No worker, GPU allocation or persistent process was
created.

Canonical evidence is under
`artifacts.local/work/mz79-tof-range-cap-20260912/run-v1/`, including the sealed
predictor receipt, full per-clip metrics, aggregate summary, source snapshots
and final hash receipt.
