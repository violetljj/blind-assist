# DTR CARLA X86 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X86_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X86 bounds route-forecast authority by the lifetime of its supporting metric
evidence. A forward-receding X57 handback can remain correct when it is already
on the route or enters immediately, but it must not authorize a later lateral
entry after the inherited X24 measurement hold window has expired. X86 clears
risk only when every confirmed carrier is a forward-receding X57 metric
handback and its minimum predicted route-entry time exceeds that existing
`0.60 s` authority horizon. Closing handbacks, current or immediate entries,
direct surface carriers, and mixed evidence remain unchanged. The rule reuses
the frozen X24 hold window and adds no fitted metric, detector, route, weather,
lighting, or class threshold.

## Result

The replay applied X86 to sealed X85 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X85 TP/FP/FN | X86 TP/FP/FN | Delta vs X85 | Release frames |
|---|---:|---:|---:|---:|
| C26 | 136/10/37 | 136/10/37 | 0/0/0 | 0 |
| C27 | 134/5/39 | 134/5/39 | 0/0/0 | 0 |
| C28 | 128/10/45 | 128/10/45 | 0/0/0 | 0 |
| C32 | 130/3/42 | 130/2/42 | 0/-1/0 | 1 |
| C34 | 135/8/37 | 135/8/37 | 0/0/0 | 0 |
| C35 | 132/11/40 | 132/11/40 | 0/0/0 | 0 |
| C36 | 143/21/29 | 143/21/29 | 0/0/0 | 0 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0 |
| C39 | 137/16/35 | 137/14/35 | 0/-2/0 | 2 |
| C40 | 129/22/43 | 129/22/43 | 0/0/0 | 0 |
| C41 | 135/12/37 | 135/11/37 | 0/-1/0 | 1 |

Pooled X85 is `1,472 TP / 137 FP / 423 FN` at
`91.49/77.68/84.02%` precision/recall/F1. Pooled X86 is
`1,472 TP / 133 FP / 423 FN` at `91.71/77.68/84.11%`, or
`0 TP / -4 FP / +0.10 pp F1`. The four releases occur across three cohorts and
all are false positives. The 30 receding X57 handback true positives remain
protected because their route entry is current or within the inherited hold
window; all five closing handback true positives also remain unchanged. Every
contact-recall, safe-segment, full-arm, and required authority-invariant check
passes.

## Evidence

- Output directory suffix beside each cohort's existing X85 run:
  `x86-consumed-development-20260901-184000`
- C32 summary SHA-256:
  `B867BFBEBD15372BC27D066539D9FCEFC5A7D6D71C070FA63AA0DCA8D3A4868D`
- C39 summary SHA-256:
  `A75C12161DA67A1B59F1A0A6628B1900FF924496091BB23BF745C0900FCBC05C`
- C41 summary SHA-256:
  `AC40764D8DE45727F694D496E9BBA27A4428610DD34DFD55D028AF85B015681E`
- X86 predictor SHA-256:
  `12E4BC2F6EA196703A2EF6F6CEB61A3F4CE31111843BF52525AE1F2B7B3DF1FB`
- X86 runner SHA-256:
  `061038B58839AF74C722E8D43CB63686A34B72FB8FA219A101A76F43A7D493A7`

## Claim boundary

X86 was designed after all eleven evaluator truths and X57 handback timing
were opened. The result is consumed, post-hoc synthetic Development evidence
for an evidence-horizon mechanism, not fresh confirmation or an estimate of
natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X86 requires a new
preregistered source that exercises the same timing partition while preserving
frozen recall, contact, safe-segment, and authority constraints. This is not
real-world, deployment, reliability, user-benefit, or safety evidence.
