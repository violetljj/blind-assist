# DTR CARLA X91 consumed eleven-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X91_CONSUMED_ELEVEN_COHORT_PRECISION_EFFECT_POSITIVE`

## Change

X91 unifies the evidence-horizon principle introduced by X86 and X87 at the
alert lifecycle boundary. When the inherited X90 decision was clear in the
previous frame, a new alert cannot be born solely from held lineage envelopes
if their predicted route entry occurs after X24's existing `0.60 s` evidence
hold window. Such a frame contains no current footprint measurement capable of
renewing the forecast authority. X91 uses the inherited X90 risk sequence, so a
release cannot create a second artificial risk birth in the following frame.
Ongoing alerts, currently measured or mixed-support carriers, and current or
within-horizon entries retain the X90 decision. The rule reuses an inherited
time horizon and adds no fitted numeric, detector, route, weather, lighting, or
class threshold.

The design is consistent with two external primary-source observations:
coasted tracks accumulate uncertainty and become stale without new association
evidence, while dynamic occupancy-grid collision prediction is fundamentally a
short-horizon operation unless additional behavior priors are introduced.

## Result

The replay applied X91 to sealed X90 predictions and opened only the already
consumed evaluators from C26/C27/C28/C32/C34/C35/C36/C37/C39/C40/C41.

| Cohort | X90 TP/FP/FN | X91 TP/FP/FN | Delta vs X90 | Release frames/tracks |
|---|---:|---:|---:|---:|
| C26 | 136/8/37 | 136/7/37 | 0/-1/0 | 1/1 |
| C27 | 134/4/39 | 134/3/39 | 0/-1/0 | 1/1 |
| C28 | 128/6/45 | 128/5/45 | 0/-1/0 | 1/1 |
| C32 | 130/2/42 | 130/2/42 | 0/0/0 | 0/0 |
| C34 | 135/8/37 | 135/7/37 | 0/-1/0 | 1/2 |
| C35 | 132/9/40 | 132/8/40 | 0/-1/0 | 1/1 |
| C36 | 143/21/29 | 143/18/29 | 0/-3/0 | 3/3 |
| C37 | 133/19/39 | 133/19/39 | 0/0/0 | 0/0 |
| C39 | 137/14/35 | 137/11/35 | 0/-3/0 | 3/3 |
| C40 | 129/20/43 | 129/17/43 | 0/-3/0 | 3/3 |
| C41 | 135/11/37 | 135/11/37 | 0/0/0 | 0/0 |

Pooled X90 is `1,472 TP / 122 FP / 423 FN` at
`92.35/77.68/84.38%` precision/recall/F1. Pooled X91 is
`1,472 TP / 108 FP / 423 FN` at `93.16/77.68/84.72%`, or
`0 TP / -14 FP / +0.34 pp F1`. All fourteen releases are false positives,
distributed across eight cohorts; C32/C37/C41 are classification identical to
X90. Every contact-recall, safe-segment, full-arm, and required
authority-invariant check passes.

## Evidence

- Replay output suffix: `x91-consumed-development-20260901-211500`
- C26 summary SHA-256:
  `DB8E1D3731524000D47B756D27D12145D4CF42FD3C966554CC58F04A53F29E31`
- C40 summary SHA-256:
  `F8ADE1AB6FD2F8768729995705A288E74B688D82E6651CC7102888445E0FB874`
- X91 predictor SHA-256:
  `9AB89C111762E0C8DA45E22E100D456D08354E0BFAC721C01A470CD891ECB430`
- X91 runner SHA-256:
  `50AD52D36AEA3A308666C18357F1D72D8E012F75374971A17D86339B045C4D3E`
- Track-staleness design context:
  <http://www.cs.columbia.edu/~areiter/CS_Webpage/Publications_files/Track_Stitching_SPIE_2007.pdf>
- Short-horizon occupancy-risk design context:
  <https://inria.hal.science/hal-01011808v1/file/ISER2014.pdf>

## Claim boundary

X91 was designed after all eleven evaluator truths and transport histories
were opened. The result is consumed, post-hoc synthetic Development evidence
for a temporal evidence-authority mechanism, not fresh confirmation or an
estimate of natural-distribution performance. X73 retains the latest complete
source-disjoint confirmation authority. Promotion of X91 requires a new
preregistered source that exercises the same held-only risk-birth partition
while preserving frozen recall, contact, safe-segment, and authority
constraints. This is not real-world, deployment, reliability, user-benefit,
or safety evidence.
