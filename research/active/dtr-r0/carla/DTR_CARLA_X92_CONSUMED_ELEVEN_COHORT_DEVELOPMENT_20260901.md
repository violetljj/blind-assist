# DTR CARLA X92 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X92_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X92 makes X91's held-only alert-birth suppression temporally complete. Without
a latch, the same inherited alert segment can reappear one frame after X91
clears its first frame even though no measurement, parent identity, or evidence
horizon state changed. X92 carries the X91 release across the same parent
lineage while every confirmed carrier remains a held lineage envelope and its
minimum route entry remains later than X24's existing `0.60 s` evidence hold
window. The latch ends immediately when current or mixed support appears,
parent identity changes, entry reaches the authorized horizon, or the inherited
route decision becomes clear for another reason. The rule reuses inherited
identity and time authority and adds no fitted numeric, detector, route,
weather, lighting, or class threshold.

## Result

The replay applied X92 to sealed X91 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X91 TP/FP/FN | X92 TP/FP/FN | Delta vs X91 | Latch frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/7/37 | 136/5/37 | 0/-2/0 | 2/2 |
| C27 | 134/3/39 | 134/3/39 | 0/0/0 | 0/0 |
| C28 | 128/5/45 | 128/1/45 | 0/-4/0 | 4/4 |
| C32 | 130/2/42 | 130/2/42 | 0/0/0 | 0/0 |
| C34 | 135/7/37 | 135/7/37 | 0/0/0 | 0/0 |
| C35 | 132/8/40 | 132/8/40 | 0/0/0 | 0/0 |
| C36 | 143/18/29 | 143/16/29 | 0/-2/0 | 2/2 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0/0 |
| C39 | 137/11/35 | 137/10/35 | 0/-1/0 | 1/1 |
| C40 | 129/17/43 | 129/15/43 | 0/-2/0 | 2/2 |
| C41 | 135/11/37 | 135/11/37 | 0/0/0 | 0/0 |

Pooled X91 is `1,472 TP / 108 FP / 423 FN` at
`93.16/77.68/84.72%` precision/recall/F1. Pooled X92 is
`1,472 TP / 97 FP / 423 FN` at `93.82/77.68/84.99%`, or
`0 TP / -11 FP / +0.27 pp F1`. All eleven latch releases are false
positives, distributed across five cohorts; the other six cohorts are
classification identical to X91. Every contact-recall, safe-segment, full-arm,
and required authority-invariant check passes.

## Evidence

- Replay output suffix: `x92-consumed-development-20260901-220000`
- C28 summary SHA-256:
  `DF2DF084F556CC73405442A71456F1959D3CBCB7E8A338AAED95A2579ACE5044`
- C40 summary SHA-256:
  `3F533A7CC8A8C34921E54A53C524283C382315126608E604262E128738419A6B`
- X92 predictor SHA-256:
  `D81811C32DF29BA485FFC34D682439A1A6FD9BA012FCB3187F9855ADEFF51B92`
- X92 runner SHA-256:
  `6EFD60736AA95E228FA7D002B671028220729BE7DAD8E74A519CD884672CC7FC`

## Claim boundary

X92 was designed after all eleven evaluator truths and transport histories
were opened. The result is consumed, post-hoc synthetic Development evidence
for a temporal alert-lifecycle mechanism, not fresh confirmation or an
estimate of natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X92 requires a new
preregistered source that exercises the same held-only risk-birth and latch
partition while preserving frozen recall, contact, safe-segment, and authority
constraints. This is not real-world, deployment, reliability, user-benefit,
or safety evidence.
