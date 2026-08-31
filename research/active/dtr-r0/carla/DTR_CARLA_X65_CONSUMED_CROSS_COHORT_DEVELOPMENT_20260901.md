# DTR CARLA X65 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X65_CONSUMED_CROSS_COHORT_REFERENCE_TARGET_MET`

## Structural change

X65 retains X64 and adds a pre-conflict cross-representation credential. When
the X64 surface route-risk state and the X24 metric route-risk state agree
before an X44 edge-direction conflict, X65 remembers the exact X24 identity.
The first conflict handback still requires current measured support from the
X44-suppressed surface lineage; after that join, only the same motion-supported
X24 identity may continue on HOLD. Stable parent ancestry permits a measured
sibling surface track to satisfy the join. An X59 evidence-supported receding
release clears the active handback and still wins.

No detector, route, duration, distance, speed, weather label, or numeric
threshold was added. X62 motion support and all X64 crossing-release behavior
remain unchanged.

## Cross-cohort result

All four cohorts were already consumed before X65 was designed. These results
are Development/mechanism evidence only.

| Cohort | X64 TP/FP | X64 F1 | X65 TP/FP | X65 P/R/F1 | Delta vs X64 | Handback frames |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C26 | 126/20 | 79.00% | 128/20 | 86.49/73.99/79.75% | +2 TP, +0 FP, +0.75 pp F1 | 2 |
| C27 | 126/15 | 80.25% | 129/15 | 89.58/74.57/81.39% | +3 TP, +0 FP, +1.13 pp F1 | 3 |
| C28 | 125/14 | 80.13% | 125/14 | 89.93/72.25/80.13% | identical | 9 |
| C32 | 117/7 | 79.05% | 127/7 | 94.78/73.84/83.01% | +10 TP, +0 FP, +3.95 pp F1 | 10 |

Pooled over the four equal-size cohorts, X64 was `494 TP / 56 FP` with
`89.82/71.39/79.55%` precision/recall/F1. X65 was `509 TP / 56 FP` with
`90.09/73.55/80.99%`: `+15 TP / +0 FP / +1.44 pp F1`. X65 improved three
cohorts, tied one, and regressed none.

Every cohort met the Development reference floors: precision at least 85%,
recall at least 70%, F1 at least 78%, every contact episode recall at least
55%, each safe episode at most four false-alert segments, total safe segments
at most ten, and all required authority invariants zero. The pre-conflict
credentialed handback was exercised on every cohort. C32 specifically moved
ep_05/ep_07 contact recall from `50/50%` under frozen X64 to
`56.52/64.58%` under X65 without adding a false positive.

## Evidence identity

Frozen task outputs:

| Cohort | Output directory | Predictions SHA-256 | Summary SHA-256 |
| --- | --- | --- | --- |
| C26 | `x65-consumed-transfer-v3-20260901-044500` | `65BD42E7C949BE4EB2777E8D72618E601AC4BFB0104262E32798B7048B7ABFE4` | `5605EECDF386AAD4F158235C341B217584857AAC0C1EEFA3490BFC6C8655D594` |
| C27 | `x65-consumed-transfer-v3-20260901-044500` | `991B681E078884D40ECA90241B15E163530B46A753C14B4A936B9F1BC2023B23` | `1B184D2FD21AB567B42F2FEDFDA01A0DC23B392391E11898C559D4E82A22117E` |
| C28 | `x65-consumed-transfer-v3-20260901-044500` | `72C28540668D3A6BAB4B0386C89442D30EB2A88E5B7EC5767EAA2B5E8D18192C` | `69AA66CCFF61B232C4DA6CCF7863E8BF65AC84BB6E721D44CEDCAE2E2F74849A` |
| C32 | `x65-consumed-development-v5-20260901-044500` | `86999A546713831C1C4F19A94A6DDAF3E886B64BBB6C65F0861A7376DDF51711` | `55F235AB7F80FD9E9F6CB3DE0136DB089EFF995E8752B7D8E03EC8D960BE792A` |

The shared X65 predictor SHA-256 is
`B87E444384CF6BE4A2B69A4B8536F9EA4CD10FE8A46DD9B5D0499A60AB94E4F1`;
the consumed runner SHA-256 is
`96186BA2DC48D3FE13A543A0CCDA43EADA3181F6776C519E6EF34E0C498C166D`.

## Claim boundary and next decision

X65 is the strongest current CARLA Development arm across C26-C28 and C32.
The cross-cohort non-regression and the concentrated C32 gain support the
pre-conflict credential mechanism, but they do not create fresh confirmation:
the algorithm was designed after observing C32 and all four cohorts are now
consumed. This is not unseen-map, open-world traffic, natural-distribution,
real-sensor, Android runtime, user-benefit, deployment, reliability, or safety
evidence.

Freeze X65 before the next scored model invocation. A promotion claim requires
one new source-disjoint cohort that admits the physical source gate and
actually exercises the pre-conflict credentialed handback. Do not rescore
C26-C28 or C32 as fresh authority.
