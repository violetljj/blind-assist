# DTR CARLA X83 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X83_CONSUMED_ELEVEN_COHORT_AUTHORITY_EFFECT_POSITIVE`

## Change

X83 separates a correct route-risk classification from the set of tracks
allowed to own that classification. When a confirmed set contains both at
least one eligible `RIGID_DYNAMIC` carrier and a non-rigid carrier, X83 keeps
the route-risk decision and minimum entry unchanged, moves the non-rigid
reference back to the candidate set, and derives confirmed parent identity
from the remaining rigid carriers. Track identity, geometry, motion, lifecycle,
and candidate evidence are retained. Frames with no mixed authority are
unchanged. The rule adds no detector, association, distance, route, time,
weather, class, or fitted metric threshold.

## Result

The replay applied X83 to the sealed X82 predictions from
C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41 and opened only their already
consumed evaluators for Development scoring.

| Cohort | Classification delta vs X82 | Projection frames | Authority-defect delta | Result |
|---|---:|---:|---:|---|
| C26 | 0 TP / 0 FP / 0 FN | 0 | 0 | not exercised |
| C27 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C28 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C32 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C34 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C35 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C36 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C37 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C39 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C40 | 0 / 0 / 0 | 0 | 0 | not exercised |
| C41 | 0 / 0 / 0 | 1 | -2 | positive authority effect |

X83 demoted one C41 `STATIC_SCENE` confirmed reference while retaining the
co-occurring valid `RIGID_DYNAMIC` owner and true-positive route-risk frame.
This removed the one confirmed non-rigid reference and the one confirmed
parent-identity mismatch. All five required authority invariants are zero in
all eleven cohorts, and every cohort is classification-identical to X82.

Pooled X83 remains `1,472 TP / 142 FP / 423 FN` at
`91.20/77.68/83.90%` precision/recall/F1. All contact-recall, safe-segment, and
full-arm reference checks pass.

## Evidence

- Output directory suffix in each cohort's existing evidence run:
  `x83-consumed-development-20260901-175700`
- C41 summary SHA-256:
  `F19AA85F65DB7BE2C6B60F32E3998B523971FA628D9AD10BF870992C355C699D`
- X83 predictor SHA-256:
  `BEC2E8251CF019AD1AED3C7CC2C4BBE4865B0621D42049DE17718C61DEB2289D`
- X83 runner SHA-256:
  `4E5D07B6FCE8C298506CED3A3448428C9D6CCA7CA10C6FD978F0E62A626B1ADF`

## Claim boundary

X83 was designed after C41 truth and invariant results opened. All eleven
cohorts are therefore consumed, post-hoc synthetic Development evidence for
X83. This result establishes an authority-normalization mechanism without a
classification regression; it is not fresh confirmation. X73 retains the
latest complete source-disjoint confirmation authority. Any promotion of X83
requires a new preregistered source that exercises the exact projection while
preserving classification and zero required invariants. This is not real-world,
deployment, reliability, user-benefit, or safety evidence.
