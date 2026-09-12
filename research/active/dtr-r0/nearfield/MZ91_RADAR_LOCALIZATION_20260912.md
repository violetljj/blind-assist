# MZ91 causal Radar localization falsifier

EXPLORE on consumed MZ90 Development observations. Question: can causal spatial
localization reject nonhazardous real returns without losing useful alerts?
Baseline is sealed MZ90 matched_hold, not joint_spatial (whose126FP audit is not
the baseline denominator). No source expansion, training or threshold sweep.

Change only the raw no-ToF Radar acceptance before the existing two-of-three
hysteresis and one-frame non-reseeding hold. ToF matched geometry is identical.
Radar returns link by mutual unique nearest normalized range/angle/Doppler cost,
gates0.5m/15deg/0.5m/s, max gap0.3s. No slot identity or simulator IDs. Use at most
five observations over0.5s; fit only from three observations. Fit range and radial
speed by weighted least squares, angle and angle rate by regularized least
squares (rate prior0 +/-30deg/s). Single-frame ablation uses current range,
Doppler and angle with zero angle rate, and the same uncertainty and veto rule.

Independent measurement sigma: range0.06m, Doppler0.08m/s,
angle sqrt(2^2+10^2/12)deg. These are declared proxy-scale assumptions, not fitted
error calibration. Common bearing sigma10deg is added after estimation and never
divided by history length. Shared causal MZ90 pose sigma is retained, plus a
non-decaying5deg per missing-IMU-frame quadrature floor; tracks reset on gaps.
Angular-rate uncertainty has an additional2deg/s floor. This approximation does
not model the complete correlated pose posterior or prove interval coverage.

Use512 fixed antithetic normal samples, seed91012, to estimate support for the
benchmark current wedge or continuous one-second constant-velocity wedge
intersection. The benchmark is horizontal, not physical body collision. Veto only
support probability<0.1, a fixed90-percent exclusion convention. Ambiguity retains
the original coarse Radar alert authority; it is NOT confirmed localization or
clearance. All unalerted no-ToF frames remain UNKNOWN. Probability is a model
support score, not a calibrated hazard probability. No new raw Radar triggers
outside the baseline range/closing/angle gate are admitted.

Primary consumed sensor_proxy pass: TP at least baseline-2, FP strictly lower,
F1 at least baseline; no additional false segments or within-event fragments;
no missed baseline event and maximum added shared-event delay<=0.2s. These are
explicit development tolerances, not safety requirements. Separately report
single-frame comparison to avoid attributing its effect to temporal localization.
Ideal is a regression diagnostic, not a second tuning set. Save all predictions
before evaluator and provenance access, verify sealed baseline parity, and record
paired TP/FP changes, UNKNOWN, events and origin slices. Use baseline raw
qualifying support origins for removed-frame attribution, not causal source
ablation. Persistent phantom and true-static kinematics overlap; no categorical
phantom rejection claim is allowed.

One fixed run on each of the two sealed regimes, stop after evaluation regardless
of outcome. Mechanical implementation failures can be repaired with logs and
identity retained, never tuned to outcome. Gain retains a Development challenger;
failure records a negative control; source/parity failure is NOT_EVALUABLE.
CPU scalar scoring through research_backend; no worker or paid allocation.
Artifact: artifacts.local/work/mz91-radar-localization-20260912/run-v1.
