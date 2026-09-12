# MZ88 uncertainty-aware geometric association results

Decision: `SCALAR_UNCERTAINTY_ASSOCIATION_NOT_RETAINED`.

The fixed scalar one-sigma tri-state association does not rescue MZ85 under the
fresh analytic motion/calibration source. It correctly abstains near uncertain
geometry and its credentialed one-frame hold obeys the declared authority
rules, but the full policy loses too much early crossing/weak-Radar recall and
lets coarse Radar evidence recreate more outside-boundary false positives than
the hard baseline. All predeclared full and attribution-arm performance checks
fail. Do not tune `k`, the uncertainty envelope, Radar thresholds or hold length
on this consumed source.

## Frozen comparison

[Protocol](MZ88_UNCERTAINTY_AWARE_ASSOCIATION_20260912.md) was committed before
source materialization. The source contains 36 new episodes and 1,080 frames:
six families, six distinct yaw/pitch trajectories, both medium error polarities,
variant-specific one/two-frame IMU gaps, new corridor offsets, crossing speeds,
gap times and ranges. It does not reuse MZ84/MZ87 frames or failure instants.

All arms see identical ToF, Radar and corrupted IMU packets. `hard_mz85` uses
the deterministic stabilized 12-degree corridor. The other arms use the fixed
`k=1` scalar uncertainty envelope; only their `UNCERTAIN` responsibility differs.

| Arm | TP | FP | FN | F1 |
| --- | ---: | ---: | ---: | ---: |
| hard MZ85 association | 398 | 124 | 17 | 0.850 |
| tri-state, uncertain to UNKNOWN | 295 | 48 | 120 | 0.778 |
| plus Radar confirmation | 326 | 151 | 89 | 0.731 |
| plus one-frame credentialed hold | 338 | 155 | 77 | 0.744 |

The full arm needed F1 above 0.850, FP at most 62 and TP at least 396. It instead
has 155 FP and loses 60 TP. The nominal/no-gap variants also fail invariance:
hard association is 69 TP / 0 FP / 2 FN, while full association is
62 / 31 / 9. The timing gate fails because lateral-crossing first alerts are
delayed by 0.5--0.7 s, above the declared 0.2 s ceiling.

## Responsibility attribution

The three association states contain 329 `CERTAIN_IN`, 300 `UNCERTAIN`, 433
`CERTAIN_OUT` and 18 missing frames. The uncertain band contains 120 true and
180 false frames, so uncertainty alone is not a hazard likelihood.

1. Conservative tri-state geometry removes 76 of the hard baseline's 124
   stressed boundary/head-motion FP, but loses 103 TP. Lateral crossing falls
   from 77 TP to 25, and weak-Radar obstacles fall from 111 to 87.
2. Radar confirmation recovers 31 TP over the conservative arm but adds 103 FP.
   On outside-boundary clutter it changes 48 FP to 148 FP; the frozen coarse
   Radar gate (`abs(azimuth)<=20 deg`) is not precise enough to confirm a
   12-degree corridor merely because ToF association is uncertain. It also adds
   three head-motion FP.
3. The one-frame credentialed hold adds 12 TP and four FP over Radar confirmation.
   It creates zero uncertain-ToF-only hazards, has zero uncredentialed holds and
   fabricates zero height labels. The rule is internally correct but cannot
   repair delayed first association and slightly transports already-wrong Radar
   confirmation.

The large crossing loss also exposes a representation problem: a single
orientation sigma, then a conservative one-second angular forecast, treats much
of the correlated temporal pose error as if it were independent pointwise
uncertainty. The resulting band expands faster than useful crossing evidence.
Changing that propagation after observing these results would be a new method,
not a repair of MZ88.

## Decision

Retain MZ88 as a negative control for the next geometry-state falsifier, not as
an operating component. The failure does not show that uncertainty-aware
geometry is unnecessary. It shows that this exact scalar margin plus
`UNCERTAIN + coarse Radar` authority is insufficient:

- scalar abstention trades too much early crossing evidence for FP reduction;
- Radar needs spatially compatible likelihood/track evidence, not merely higher
  authority inside a broad uncertainty band;
- a one-frame hold can transport a credential but cannot establish one.

Per the frozen stop rule, MZ88 ends here. A successor must change the
representation to a correlated occupancy/collision tube or equivalent track
distribution, propagate pose/calibration uncertainty jointly across time, and
test Radar compatibility against that same spatial object. Do not fit that
successor on MZ88 outcomes, and do not resume the separate MZ86 continuity patch
as a substitute.

## Evidence and implementation note

Canonical evidence is
`artifacts.local/work/mz88-uncertainty-aware-association-20260912/run-v2/`.
Its prediction SHA-256 is
`ab4392b2e84db93195437d3c8d35f5513cf0192e5c61870c554dcd01076d1756`.
The first complete predictor/evaluator attempt (`run-v1`) failed only while
serializing an infinite missed-event delay to strict JSON. Its sealed prediction
hash is identical to run-v2; the repair represents a missed post-baseline event
as JSON `null` and makes the timing gate fail explicitly. The earlier partial
fixed-20-frame materializer attempt was removed before any predictor output.

Six focused unit tests and Python compilation pass. This is fresh constructed
analytic evidence only. It does not establish a real covariance model,
calibration quality, hardware performance, alert benefit, deployment readiness,
user benefit or safety improvement. `UNKNOWN` remains distinct from `CLEAR`.
