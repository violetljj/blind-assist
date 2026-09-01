# DTR CARLA C39 X79 fresh confirmation

Date: 2026-09-01

Decision: `DTR_CARLA_C39_X79_MECHANISM_NOT_EXERCISED`

## Frozen question

C39 admitted one new scripted CARLA source at seed `391079`, with four changed
weather/render assignments and fresh pixels. Unchanged X78 and X79 predictions
were sealed before the evaluator was opened. The single formal score required
X79 to exercise at least one lateral-only release, lose no true positives,
remove at least one false positive, meet the full precision/recall/F1 floors,
and preserve every authority, contact-recall, and safe-segment constraint.

## Result

X79 did not exercise its release mechanism on the fresh cohort. It was
classification-identical to X78:

| Arm | TP / FP / FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| X24 | 84 / 34 / 88 | 71.19% | 48.84% | 57.93% |
| X78 | 137 / 22 / 35 | 86.16% | 79.65% | 82.78% |
| X79 | 137 / 22 / 35 | 86.16% | 79.65% | 82.78% |

The X79 delta versus X78 was `0 TP / 0 FP / 0.00 pp F1`. Both mechanism
requirements failed: `0` lateral-only release frames and no false-positive
reduction. The frozen full-arm floors passed, all four contact episodes stayed
above 55% recall, each safe episode had at most two false-alert segments, and
every required authority invariant remained zero.

C39 therefore supplies no fresh incremental evidence for X79. This is
mechanism-not-exercised rather than a demonstrated negative effect: the
fresh source never reached the state on which X79 differs from X78.

## Evidence identity

- Protocol SHA-256:
  `EC62FF07F2E1FBF2A43046083D4792D6A8A6ADF1CFAB65102505BCBE965637F3`
- Source result SHA-256:
  `79DD2907DE3A44BF3B24EA160C057B825191A3C636873C736E5F60E040749F46`
- Model manifest SHA-256:
  `CB7B7061BC826D9259457227834027DCAA96E45A3A7F053C60E4ABD89F66AF3A`
- X24 freeze SHA-256:
  `EEBED6DD8BE893F075DE257E0BFA6A8348CBC5C2A686A080B0E31CA3382A4250`
- X24 prediction SHA-256:
  `9798535B36A6D4FD4BE64168BCB214DCD27B603302347D3F298BEAB838F97B5A`
- X78 prediction SHA-256:
  `35A720163AA71047DCFE35D10ED3F877EB30756173823F91683F85297AC0944B`
- X79 prediction SHA-256:
  `4D97C89F839ADFFB2D6321FA31E0A473206C95DD73C7A6F156F6767E54C834EF`
- X79 predictor SHA-256:
  `537C7BC5ECB842548583282EEFD1FB28F6A630CD457075795AF62332D7C88F93`
- Confirmation runner SHA-256:
  `344578216D28D5AD364A7F7D204A188CB98BD3D9D156FEEB05D871DBBC5F6632`
- Summary SHA-256:
  `7BAB6C74BA64F772941C6128ED17A50B06214BF0607CA3933681EFFA38A71C72`

## Claim boundary and next decision

X79 keeps its eight-consumed-cohort Development result of `0 TP / -15 FP`
versus X78, but C39 does not promote that effect. X73 remains the latest arm
with positive fresh source-disjoint confirmation authority.

C39 is now consumed diagnosis material. Do not rerun or resample it as
confirmation. A successor may use its opened false-positive structure to
design a different observable mechanism, but any promotion claim requires a
later preregistered fresh source. This remains same-map, same-route-layout,
scripted synthetic CARLA Development evidence, not natural-distribution,
real-world, product-default, deployment, user-benefit, reliability, or safety
evidence.
