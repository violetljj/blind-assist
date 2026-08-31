# DTR CARLA X71 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X71_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X71 adds an object-local occupancy birth for frames where X70 has no current
surface carrier. Both X24 metric-point and X25 rigid-footprint arms must already
confirm route risk for the same semantic class. The metric point must lie inside
the current rigid footprint, their route-forward velocity signs cannot oppose,
and their separately transported positions must remain within the inherited
X24 `1.5 m` association distance at the later of the two current predicted
route-entry times.

The later entry time tests whether the representations continue to describe one
object through the decision point instead of merely overlapping now. An
explicit X69 mature rigid-contradiction release blocks X71 before birth. X71
reuses X24/X25 route-entry times, footprint geometry, and association distance;
it adds no numeric threshold and no class-specific prior.

Current same-class point-in-footprint containment alone recovered the three
retained true positives but also introduced eight false positives. Entry-time
co-transport removed six of those false positives; route-forward direction
agreement removed the remaining two. The full structural rule retained all
three true positives and zero false positives.

## Research context

Two Exa search angles reviewed 16 results. Three primary papers most directly
supported the design choice: [DSC-Track](https://arxiv.org/abs/2508.11323)
emphasizes stable spatiotemporal cue consistency over indiscriminate geometry;
[MCTrack](https://arxiv.org/abs/2409.16149) reports that distance or overlap
alone is inadequate and explicitly evaluates downstream motion quality; and
[PD-SORT](https://arxiv.org/abs/2501.11288) combines depth-volume overlap with
velocity-direction consistency for occlusion-robust association. X71 is a
small deterministic realization of those converging principles, not a claim
that their benchmark results transfer to BlindAssist.

## Five-cohort result

All five sources had been scored before X71 was designed. These are consumed
Development results only.

| Cohort | X70 TP/FP | X71 TP/FP | X71 P/R/F1 | Delta vs X70 | Birth frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 130/16 | 130/16 | 89.04/75.14/81.50% | neutral | 0 |
| C27 | 129/7 | 129/7 | 94.85/74.57/83.50% | neutral | 0 |
| C28 | 125/12 | 125/12 | 91.24/72.25/80.65% | neutral | 0 |
| C32 | 128/3 | 128/3 | 97.71/74.42/84.49% | neutral | 0 |
| C34 | 128/13 | 131/13 | 90.97/76.16/82.91% | +3 TP, 0 FP, +1.12 pp F1 | 3 |

Pooled across the five equal-size cohorts, X70 was `640 TP / 51 FP / 223 FN`
at `92.62/74.16/82.37%` precision/recall/F1. X71 is
`643 TP / 51 FP / 220 FN` at `92.65/74.51/82.59%`: `+3 TP / +0 FP / +0.23 pp
F1`. The mechanism was positive in one cohort and classification-neutral in
four. All required authority invariants remained zero, and every
contact-recall and safe-segment constraint remained satisfied.

Relative to frozen X65 on C34, X71 removes 13 false positives and recovers four
true positives. C34 rises from `83.01/73.84/78.15%` to
`90.97/76.16/82.91%` precision/recall/F1.

## Evidence identity

- X71 predictor SHA-256:
  `D67A9A4722A2AD212C2137A0E8C666FF750DFA3556F486D788B25ED9D15E1FFA`
- Consumed runner SHA-256:
  `6AC38B3BCC2B62FB864C62B5FB8ACE8158C1760E2ECF9DC419308D8C14E49248`
- C26 summary / X71 prediction / X25 rigid prediction SHA-256:
  `87ABAC056BA6CCE41F06A2D34BEF83AC3A2982AA343CA8035791CF237962B852` /
  `B1F77127AFCDBD03D0C111F08EDFF03E11CC79F1078077B29D701D9439CBD2FF` /
  `0B1D5904057C8E6E0751F00F97546163A51FED33C04A4CAD47F9659571183868`
- C27 summary / X71 prediction / X25 rigid prediction SHA-256:
  `BB56036E816E3999705F58D03D7D81CDB2BCF33DC78F308A9161297F4EB23AE7` /
  `084C1C050545F15B2230CEA9328D20E6D4D0D9BA9CDE9A06C1BD85D9242D2359` /
  `A2C9FFA93F6219B89DADEBE9A00CAE6B3285870F77931063F4488EE78BA94E0B`
- C28 summary / X71 prediction / X25 rigid prediction SHA-256:
  `2A9688BD5367C20964D55B9D475AFF45D7D04C81CFC8DE9AC67C8CEBD919B340` /
  `8822A5F9993E9EA986AC5B97F6BAA6106943477C976DB33CDAA876BB68884C1C` /
  `A9BECAC8047AB3AC61CDA2064F941F9E4CAE5F5047A28E1212BFBF5621B5E7C3`
- C32 summary / X71 prediction / X25 rigid prediction SHA-256:
  `1F89CFF258ADD80A889102E4FD6C2E67FA068BFC915852ECE003CA5052379004` /
  `77DD78108AEA6773DA38B9A375C95FB420A0F62978175518BC58C50C8965CBF6` /
  `844DF9E132E5EF90F063D4DA45C01D62EF4638A190363AF1FBD10E1314057ED3`
- C34 summary / X71 prediction / X25 rigid prediction SHA-256:
  `6419A9A3D53668A9A9F88C4A99E6DEE8EC6BB03968B04148AF1D95F58CDF7847` /
  `F368D092468E32218D3A7498C29C0161D1598992F701A2E8E6D1E1BE7C38C0B7` /
  `15C792B658406F5919616FA1A45968DEF7AFD8B56FBABF070282599E3EFDB69C`

## Claim boundary and next decision

X71 is the strongest current cross-cohort CARLA Development arm. It shows that
two already-authorized geometry representations can birth a narrow occupancy
decision during surface absence when their current overlap and predicted
co-transport agree. The effect is localized to one cohort, while the other four
provide non-regression evidence. All sources were consumed before X71 was
designed, so this is not fresh confirmation or promotion authority.

The remaining pooled error is `220 FN / 51 FP`; C34 remains `41 FN / 13 FP`.
Most remaining misses still lack jointly confirmed point/footprint route risk.
The next recall mechanism should improve observable support or earlier
object-local occupancy formation rather than weakening X71 association. A new
confirmation source should be frozen only after another visible structural gain
or to adjudicate unchanged X71.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
