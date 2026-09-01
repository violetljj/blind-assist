# DTR CARLA X87 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X87_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X87 bounds an isolated X72 surface-boundary completion by the lifetime of the
metric evidence that originally authorized its parent. Completion proxies may
preserve a still-live parent through a short measurement gap, but a proxy-only
decision cannot independently forecast a later route entry after the inherited
X24 `0.60 s` hold window has expired. X87 clears risk only when X72 completion
is the sole active mechanism, every confirmed carrier is an X72 completion
proxy, and every predicted route entry lies beyond that existing evidence
horizon. Current or immediate entries, direct measured carriers, mixed
mechanisms, and mixed evidence remain unchanged. The rule reuses the frozen X24
hold window and adds no fitted metric, detector, route, weather, lighting, or
class threshold.

## Result

The replay applied X87 to sealed X86 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X86 TP/FP/FN | X87 TP/FP/FN | Delta vs X86 | Release frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/10/37 | 136/10/37 | 0/0/0 | 0/0 |
| C27 | 134/5/39 | 134/5/39 | 0/0/0 | 0/0 |
| C28 | 128/10/45 | 128/10/45 | 0/0/0 | 0/0 |
| C32 | 130/2/42 | 130/2/42 | 0/0/0 | 0/0 |
| C34 | 135/8/37 | 135/8/37 | 0/0/0 | 0/0 |
| C35 | 132/11/40 | 132/9/40 | 0/-2/0 | 2/2 |
| C36 | 143/21/29 | 143/21/29 | 0/0/0 | 0/0 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0/0 |
| C39 | 137/14/35 | 137/14/35 | 0/0/0 | 0/0 |
| C40 | 129/22/43 | 129/21/43 | 0/-1/0 | 1/2 |
| C41 | 135/11/37 | 135/11/37 | 0/0/0 | 0/0 |

Pooled X86 is `1,472 TP / 133 FP / 423 FN` at
`91.71/77.68/84.11%` precision/recall/F1. Pooled X87 is
`1,472 TP / 130 FP / 423 FN` at `91.89/77.68/84.19%`, or
`0 TP / -3 FP / +0.07 pp F1`. The releases span two cohorts and all are false
positives. Every contact-recall, safe-segment, full-arm, and required
authority-invariant check passes; the other nine cohorts are classification
identical to X86.

## Evidence

- Replay output suffix: `x87-consumed-development-20260901-185000`
- C35 summary SHA-256:
  `DA316AF48E4317D07800057B9A5B33F3ECF4412952689FBD3205B38972A1030D`
- C40 summary SHA-256:
  `4E27E3EE686B674C7024528D8D218DCA14B31771AF3F2DE68AAD02E2C3FC0F00`
- X87 predictor SHA-256:
  `8B741E1B58EDC2BE7319A206FFCDBD217218CA7BC8613FA289BDF360D85408F9`
- X87 replay runner SHA-256:
  `2CAFE960B0E7A63BA40FB2C5AA07E18FDF1C3F4D463E8CA267912191D3F0A85E`
- Committed X87 runner SHA-256 after removing one trailing blank line:
  `849A44DD4F13CCF9DCBC6EC42F9EA7413216F88F65E7E692DDA76692565328B8`
  (execution semantics are unchanged)

## Claim boundary

X87 was designed after all eleven evaluator truths and completion-proxy timing
were opened. The result is consumed, post-hoc synthetic Development evidence
for an evidence-horizon mechanism, not fresh confirmation or an estimate of
natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X87 requires a new
preregistered source that exercises the same proxy-only timing partition while
preserving frozen recall, contact, safe-segment, and authority constraints.
This is not real-world, deployment, reliability, user-benefit, or safety
evidence.
