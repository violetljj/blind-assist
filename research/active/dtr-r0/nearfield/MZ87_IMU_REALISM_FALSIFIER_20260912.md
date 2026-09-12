# MZ87: IMU rotation realism falsifier

Mode: `EXPLORE`, zero training, consumed MZ84/MZ85 analytic source.

## Question

Does the fixed MZ85 rotation compensation retain a non-zero tolerance region
when its IMU-to-rotation transform is perturbed, or does the zero-FP result
require oracle orientation?

MZ86 remains reserved for the two bounded authority-continuity defects. MZ87
does not change ToF, radar, fusion, association, corridor, hysteresis or alert
thresholds. It changes only the rotation supplied to the frozen MZ85 ToF expert.

## Frozen stress grid

No full factorial search is allowed. Single-factor stresses are:

- yaw gyro bias: `+/-0.5`, `+/-2`, `+/-5 deg/s`;
- timestamp offset: `+/-5`, `+/-20`, `+/-50 ms`;
- fixed yaw extrinsic error: `+/-0.5`, `+/-2`, `+/-5 deg`;
- white gyro noise: `0.5`, `2`, `10 deg/s RMS`, four fixed seeds each;
- causal IMU dropout: `1`, `2`, `3` consecutive frames ending at the
  constructed head-motion association instant; missing increments are held.

Three frozen combinations are added after the single-factor sweep: positive and
negative signed medium combinations (`2 deg/s`, `20 ms`, `2 deg`, `2 deg/s RMS`,
one dropped frame), and one high positive combination (`5`, `50`, `5`, `10`,
two frames). These are pressure probes, not claimed device error distributions.

## Metrics and decision

Every replay records head-motion FP, generic TP/FP/FN, new non-head-motion FP,
first-correct-alert delay, event fragmentation, and stabilized-ray angular
mismatch relative to ideal MZ85.

A scenario is stable only if:

- head-motion FP is at most 2;
- TP is at least 158;
- new non-head-motion FP is zero;
- maximum added first-alert delay is at most 0.2 s;
- no event gains a fragment.

`BOUNDED_SYNTHETIC_TOLERANCE` requires every small/medium single-factor replay
(bias <=2 deg/s, absolute sync <=20 ms, extrinsic <=2 deg, noise <=2 deg/s RMS,
dropout <=1 frame) and both signed medium combinations to be stable. High
stresses may fail and define the breakdown boundary. If any smallest stress
fails, the decision is `FRAGILE_AT_SMALLEST_PRESSURE`; otherwise failure of the
bounded gate is `TOLERANCE_NOT_ESTABLISHED`.

## Evidence boundary

All stresses are analytic and aligned to the consumed MZ85 trajectory. They omit
real spectra, temperature drift, axis coupling, scale-factor nonlinearity,
timestamp jitter distributions, sensor clocks and measured extrinsics. Passing
shows a synthetic tolerance interval, not compatibility with a particular IMU,
hardware performance, deployment readiness, user benefit or safety.
