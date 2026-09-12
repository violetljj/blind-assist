# MZ87 IMU rotation realism falsifier results

Decision: `ROTATION_COMPENSATION_TOLERANCE_NOT_ESTABLISHED`.

MZ85 is not an oracle that fails under the smallest analytic perturbation, but
the preregistered small-to-medium tolerance region does not hold. Every smallest
single-factor stress preserves `160 TP / 0 FP / 11 FN`; however, selected
2-level single factors and both signed medium combinations cross frozen geometry
boundaries. The ideal `0 FP` result therefore has a non-zero but narrower and
strongly coupled synthetic margin than MZ85 alone suggested.

## Frozen comparison

[Protocol](MZ87_IMU_REALISM_FALSIFIER_20260912.md) runs 36 zero-fit replays:
33 single-factor polarity/seed cases and three frozen combinations. MZ84 sensor
observations, MZ85 radar and fusion policy, ToF association, thresholds and
hysteresis are unchanged. The ideal replay is bitwise identical to sealed MZ85.

Stress values are pressure levels only. They are not distributions or
specifications of a real IMU.

## Single-factor boundaries

| Stress | Fully stable region observed | First declared failure |
| --- | --- | --- |
| Gyro bias | both signs at 0.5 deg/s | +2 deg/s adds 2 lateral-crossing FP; -2 deg/s remains within the permissive TP gate but loses 2 lateral TP |
| Timestamp offset | both signs through 50 ms | none on this trajectory; 50 ms produces at most 1.5 deg angular mismatch |
| Extrinsic yaw error | both signs at 0.5 deg | +2 deg adds 2 lateral-crossing FP; -2 deg loses 2 lateral TP but remains inside the declared gate |
| White gyro noise | all four seeds at 0.5 deg/s RMS | 1 of 4 seeds at 2 deg/s adds 1 and loses 1 lateral-crossing frame |
| IMU dropout | 1 frame | 2 and 3 frames restore all 8 head-motion FP |

All smallest stresses preserve 160 TP, zero FP, zero added delay and zero added
fragmentation. This rejects `FRAGILE_AT_SMALLEST_PRESSURE`.

High stress gives the expected breakdown: signed 5 deg/s bias and 5 deg
extrinsic errors produce four head-motion FP in the adverse polarity; two-frame
dropout restores all eight. Ten deg/s noise is seed-dependent and can add FP,
lose TP and add one fragment.

## Combination falsifier

Neither signed medium combination is stable:

| Combination | TP / FP / FN | Head-motion FP | Other change |
| --- | ---: | ---: | --- |
| medium positive | 160 / 8 / 11 | 4 | 4 new lateral-crossing FP |
| medium negative | 157 / 4 / 14 | 4 | 3 lost lateral-crossing TP |
| high positive | 121 / 24 / 50 | 4 | broad lateral, clutter, weak-reflector and multi-target degradation |

Thus independent-looking acceptable errors cannot be assumed to compose. The
medium combinations reach about 7.4--7.9 degrees maximum ray mismatch and move
both the original head-motion cases and lateral projection across hard corridor
boundaries.

## Mechanism interpretation

The first failures are not limited to resurrection of the original eight FP.
Accumulated bias, fixed extrinsic offset and noise first perturb lateral-crossing
projection, creating polarity-dependent FP or TP loss while head-motion FP can
still remain zero. The dominant robustness issue is therefore the hard angular
boundary operating on a miscalibrated stabilized frame, not just insufficient
head-motion cancellation.

The apparent tolerance to +/-50 ms is source-specific: the constructed head
motion is locally flat at the two association frames. It does not establish a
50 ms synchronization allowance for rapid natural motion.

MZ85 remains the ideal mechanism baseline, but `160/0/11` must not be presented
as a robust operating point. A successor should change the evidence source and
include calibration/timing uncertainty explicitly. Soft uncertainty-aware
association may become justified only after that evidence; threshold relaxation
or learned fusion on this consumed grid is forbidden.

## Evidence boundary and verification

Canonical evidence is
`artifacts.local/work/mz87-imu-realism-falsifier-20260912/run-v2/`. It contains
the frozen grid, all 36 scenario metrics, family-specific gained/lost decisions,
angular mismatch, and artifact hashes. Three focused tests, Python compilation,
knowledge validation and `git diff --check` pass. No measured gyro, ToF, radar,
clock, extrinsic or natural motion is present; no hardware or safety claim is
supported.
