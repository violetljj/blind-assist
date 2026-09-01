# DTR CARLA C40 X81 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C40_X81_GENERALIZATION_GATE_NOT_MET`

## Frozen question

C40 admitted one new scripted CARLA source at seed `401081`, with four changed
weather/render assignments and fresh pixels. Unchanged X80 and X81 predictions
were sealed before the evaluator was opened. The single formal score required
X81 to exercise at least one zero-shift shape release, lose no true positives,
remove at least one false positive, meet the full precision/recall/F1 floors,
and preserve every authority, contact-recall, and safe-segment constraint.

## Result

X81 exercised its release mechanism on three fresh frames and removed three
false positives with no true-positive loss:

| Arm | TP / FP / FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| X24 | 79 / 28 / 93 | 73.83% | 45.93% | 56.63% |
| X80 | 129 / 28 / 43 | 82.17% | 75.00% | 78.42% |
| X81 | 129 / 25 / 43 | 83.77% | 75.00% | 79.14% |

The X81 delta versus X80 was `0 TP / -3 FP / +0.72 pp F1`. The exact
zero-shift cross-route shape mechanism therefore reproduced its precision
direction on genuinely new pixels. All four contact episodes remained above
55% recall, every safe episode stayed at or below two false-alert segments,
the total was five segments, and every required authority invariant remained
zero.

The full gate nevertheless did not pass because X81 precision was `83.77%`,
below the preregistered `85%` floor. Recall, F1, incremental effect, mechanism,
contact, safe, and authority requirements all passed. This is fresh positive
incremental evidence for the X81 mechanism, but not full-arm promotion.

## Evidence identity

- Protocol SHA-256:
  `130658DF02FE31CBFA0C6662870149222F6F9F5C9700DA3ECD968F8FE87DF108`
- Source result SHA-256:
  `E3F8A34BD23D341CC9771A64EB23E820CA54B5332ABE3327601038CDBE618E10`
- Model manifest SHA-256:
  `AAFF677A396FED54ADF86836514864F0928164C0F6DF6FB05CDD6E60D35D6896`
- X24 freeze SHA-256:
  `839B67A7CEFFB05D8DC17861CBDE99CD62F619D61E4FDC8DC626EE1B72ADB1CD`
- X24 prediction SHA-256:
  `C678925DF7DEC58020CEC15000D01670A627FBB122405DE5D493EE4D27B221BE`
- X80 prediction SHA-256:
  `9F7BD4C6F95845F2558253862E1C979E7FD14BED17E618A8FEA86891A50BC2B4`
- X81 prediction SHA-256:
  `0790E4CE2C03FCFD40D6B50AFB2C2F9914374BD87B97DA5349ECA52A9ABA30F4`
- X81 predictor SHA-256:
  `7612B5AE997ACD7F1109924D9F7C37ECDE578B35020BB1B8E048A61DE72CE232`
- Confirmation runner SHA-256:
  `BC09DA609AFF8C885445EDF22EEB8BA43BF7D39862DDBA4396460641FB15C3FA`
- Summary SHA-256:
  `09394D752A3C26151772C32EB241B537AC07E4AAA147454B9893057EA5E6EEC0`

## Claim boundary and next decision

X81 now has fresh source-disjoint evidence for its incremental precision
direction, but it did not meet the full-arm precision gate. X73 remains the
latest arm with a complete positive fresh generalization gate.

C40 is consumed diagnosis material. Do not rerun or resample it as
confirmation. A successor may use its remaining false-positive structure to
design a different observable mechanism, but any promotion claim requires a
later preregistered fresh source. This remains same-map, same-route-layout,
scripted synthetic CARLA Development evidence, not natural-distribution,
real-world, product-default, deployment, user-benefit, reliability, or safety
evidence.
