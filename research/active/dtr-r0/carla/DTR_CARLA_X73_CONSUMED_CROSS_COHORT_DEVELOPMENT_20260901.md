# DTR CARLA X73 consumed cross-cohort Development

Date: 2026-09-01

Decision: `DTR_CARLA_X73_CONSUMED_CROSS_COHORT_NONREGRESSION_EFFECT_POSITIVE`

## Structural change

X73 reconstructs current object geometry across all measured fragments of one
still-live, collision-credentialed surface parent. It takes the convex hull of
the current fragments, transports that hull with their area-weighted current
velocity, and reruns the inherited X25 footprint-to-route entry test. This
covers the complementary failure mode to X72: the object remains visible, but
no individual surface fragment retains enough extent to predict collision.

The reconstruction is rejected whenever the center of any current measured
X25 footprint lies inside any current fragment of that parent. In that state,
the rigid representation already explains the surface support and parent-wide
filling is ambiguous. X69 explicit contradiction release retains precedence.
No detector, route, class, distance, duration, weather, score, or other numeric
threshold was added.

The unrestricted credentialed-parent hull recovered 14 true-positive frames
but added three false-positive frames. The current-rigid containment veto
removed all three false positives and deliberately sacrificed two ambiguous
true-positive frames. The retained rule therefore had a pooled formal effect
of `+12 TP / +0 FP`.

## Five-cohort result

All five sources had been scored before X73 was designed. These are consumed
Development results only.

| Cohort | X72 TP/FP | X73 TP/FP | X73 P/R/F1 | Delta vs X72 | Reconstruction frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| C26 | 133/16 | 136/16 | 89.47/78.61/83.69% | +3 TP, 0 FP, +1.08 pp F1 | 3 |
| C27 | 129/7 | 134/7 | 95.04/77.46/85.35% | +5 TP, 0 FP, +1.86 pp F1 | 5 |
| C28 | 126/12 | 128/12 | 91.43/73.99/81.79% | +2 TP, 0 FP, +0.76 pp F1 | 2 |
| C32 | 128/3 | 130/3 | 97.74/75.58/85.25% | +2 TP, 0 FP, +0.76 pp F1 | 2 |
| C34 | 135/13 | 135/13 | 91.22/78.49/84.38% | neutral | 0 |

Pooled across the five equal-size cohorts, X72 was `651 TP / 51 FP / 212 FN`
at `92.74/75.43/83.19%` precision/recall/F1. X73 is
`663 TP / 51 FP / 200 FN` at `92.86/76.83/84.08%`: `+12 TP / +0 FP / +0.89 pp
F1`. The mechanism was positive in four cohorts and classification-neutral in
C34. All required authority invariants remained zero, and every contact-recall
and safe-segment constraint remained satisfied.

C34 rejected five credentialed parent groups through current rigid-center
containment and added no frame. This is the intended conservative behavior,
not a hidden regression. The same veto sacrifices one otherwise recoverable
C32 contact frame and one C26 contact frame rather than selectively admitting
them after labels were opened.

## Evidence identity

- X73 predictor SHA-256:
  `8722FAB54E441459EDE6E1EBE61CE1BE0FD7E8956BB2C9B139BF67E3BF51BBD2`
- Consumed runner SHA-256:
  `7F9F2B4D173C795E8C0098DBF4D5FF058FA170E391F3A8D2ECBC15016F11A198`
- C26 summary / X73 prediction / X25 rigid prediction SHA-256:
  `E7363A36570D3C4764E59189D8B22101FFD38397EA71A5727ACA2C915E461CF6` /
  `6FF3FF4D90E233D4FB4E0FBED40785BDBF78A696D187DF8D90259AB5E9243E83` /
  `13C6BB18ABADFAE37E6D08F30371B1889E13E3A47F2BEB683B8694CFD262C14D`
- C27 summary / X73 prediction / X25 rigid prediction SHA-256:
  `C282515A0E5874FE4A4DFB99DDBDCE3B3ECC4550AE53168C0BEEEBB94D4E0AA1` /
  `E30EAB3D94FF444BCBCCBD488F0E99FC9661CC5F89C4AAE08F1C375E8842CDCE` /
  `81FBFEA5B6BBA25E5251C1F2AFF305CD65389B9D0A130A29FACACB2FCEAE73CF`
- C28 summary / X73 prediction / X25 rigid prediction SHA-256:
  `62DDF0BD852F5F85E3463C46170B9B35ADC77FE3750C62D46BA6D95DD36FC722` /
  `51222C58ADA3574A0ACA41E84F7C90B89CE58CA5F82454306A9DDAF32E729994` /
  `D8B211266FEE6AD4D30461016E8402BD9D3E630221D15B4FE243E1D893B90E1F`
- C32 summary / X73 prediction / X25 rigid prediction SHA-256:
  `5246BBE6293E656329E32185907AC126D3817EF57082D64DF690B804BEEB726A` /
  `C8617BE8CFC2B7335BCD8EE86974984D103984E5C9C18290A549FDBEEEDFFF5A` /
  `144018801E9FE6673574F1593D813E4FB48FACA522651B3D36F4F0E66ED1EBCB`
- C34 summary / X73 prediction / X25 rigid prediction SHA-256:
  `CC46743DF4450D9202D683A7ED0DDDF40C43BF86B1625D25B793447175204548` /
  `7B7D41240B5A28FD1CE0EB1C682CA1C384AA7ADAD9BB43D7AD42C4A08091E866` /
  `4837E53172CC0D31BF6F1DBDF9099DD33A4C53F885A2E18F6EEFDBBDBECEA116`

## Claim boundary and next decision

X73 is the strongest current cross-cohort CARLA Development arm. Its positive
effect in four cohorts shows that parent-level reconstruction can recover
extent lost to component fragmentation, while a current rigid-center veto
prevents the known contained-object near-miss mode. All sources were consumed
before X73 was designed, so this is not fresh confirmation or promotion
authority.

The remaining pooled error is `200 FN / 51 FP`; C34 remains `37 FN / 13 FP`.
The visible cross-cohort effect authorizes one genuinely new frozen source to
confirm unchanged X73. Do not tune X73 on that source after pixels or outcomes
are opened.

This is synthetic Development evidence, not real-world, natural-distribution,
product-default, deployment, user-benefit, reliability, or safety evidence.
