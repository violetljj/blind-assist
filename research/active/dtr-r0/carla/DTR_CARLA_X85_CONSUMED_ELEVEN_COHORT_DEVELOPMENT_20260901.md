# DTR CARLA X85 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X85_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X85 corrects a stage-ordering defect between X68 and X72. X68 may use current
object-local metric velocity to falsify inherited surface collision geometry,
but X72 could reopen risk in the same frame from historical surface credential
completion. This double-counts object-local rigid evidence after the
route-level geometric falsifier has already cleared the surface decision. X85
gives that current X68 release precedence only when the reopened confirmed set
consists entirely of X72 completion proxies. Direct carriers, frames without a
same-frame X68 release, and later independent evidence remain unchanged. The
rule adds no fitted metric, detector, route, weather, lighting, or class
threshold.

## Result

The replay applied X85 to sealed X84 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X84 TP/FP/FN | X85 TP/FP/FN | Delta vs X84 | Release frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/10/37 | 136/10/37 | 0/0/0 | 0/0 |
| C27 | 134/5/39 | 134/5/39 | 0/0/0 | 0/0 |
| C28 | 128/10/45 | 128/10/45 | 0/0/0 | 0/0 |
| C32 | 130/3/42 | 130/3/42 | 0/0/0 | 0/0 |
| C34 | 135/8/37 | 135/8/37 | 0/0/0 | 0/0 |
| C35 | 132/11/40 | 132/11/40 | 0/0/0 | 0/0 |
| C36 | 143/23/29 | 143/21/29 | 0/-2/0 | 2/3 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0/0 |
| C39 | 137/16/35 | 137/16/35 | 0/0/0 | 0/0 |
| C40 | 129/22/43 | 129/22/43 | 0/0/0 | 0/0 |
| C41 | 135/12/37 | 135/12/37 | 0/0/0 | 0/0 |

Pooled X84 is `1,472 TP / 139 FP / 423 FN` at
`91.37/77.68/83.97%` precision/recall/F1. Pooled X85 is
`1,472 TP / 137 FP / 423 FN` at `91.49/77.68/84.02%`, or
`0 TP / -2 FP / +0.05 pp F1`. C36 alone improves from
`86.14/83.14/84.62%` to `87.20/83.14/85.12%`. The two released frames are
exactly the same-frame X68-release/X72-reopening partition and remove three
completion proxies. Every contact-recall, safe-segment, full-arm, and required
authority-invariant check passes; the other ten cohorts are classification
identical to X84.

## Evidence

- Output directory suffix beside each cohort's existing X84 run:
  `x85-consumed-development-20260901-183100`
- C36 summary SHA-256:
  `B50DCAB07716D685D9771D57972042FC09EC7AFABC4C65DA8346CE0080E3262C`
- X85 predictor SHA-256:
  `05DD9B81E940295A16A7D5445C7E5B17F9DC2DEC7D67CE3C3E4FC8CE3431B323`
- X85 runner SHA-256:
  `FF6475FC4C8A6996B143E8B88DAB2433DF963DB79488A416AFA5FE3111D6ECFB`

## Claim boundary

X85 was designed after all eleven evaluator truths and inherited mechanism
flags were opened. The result is consumed, post-hoc synthetic Development
evidence for a stage-precedence mechanism, not fresh confirmation or an
estimate of natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X85 requires a new
preregistered source that exercises the same-frame precedence rule while
preserving frozen recall, contact, safe-segment, and authority constraints.
This is not real-world, deployment, reliability, user-benefit, or safety
evidence.
