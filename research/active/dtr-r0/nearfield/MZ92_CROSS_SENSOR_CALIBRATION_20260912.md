# MZ92 causal cross-sensor bias calibration

EXPLORE: one fixed attempt on consumed MZ90 ideal/sensor_proxy observations.
MZ91 changed scores but no alerts. Hypothesis: occasional unambiguous co-observed
ToF/Radar returns can constrain a shared relative angular bias, adding information
that Radar-only averaging cannot provide. Not a new threshold search.

Current-frame valid raw ToF/Radar returns are compatible if range differs<=0.15m
and wrapped bearing differs<=25deg. Require mutual unique compatibility and
exactly one surviving pair in a frame; otherwise no measurement. The offset is
Radar angle minus ToF angle in the same sensor frame; common ego rotation cancels.
Use all valid raw ToF bins for calibration; retain baseline nearest-two ToF risk
logic unchanged. Invalid/missing returns are not clearance or calibration evidence.

Maintain eligible offsets over the current and preceding20 frames (2s at fixed
MZ90 cadence). Minimum3 paired frames, Gaussian prior0deg/sigma10deg,
measurement variance 2^2+10^2/12+5.625^2/12 from declared proxy noise/quantization.
Posterior variance additionally retains a2.5deg sigma common floor. No simulator
bias, identity, reflectivity, clutter flags, truth or detection probability enters
the estimator. Each episode resets it; expired/insufficient evidence returns to
the original0deg/sigma10deg prior. This model is approximate and uncalibrated;
repeated quantization and false unique matches can still overstate confidence.

At each frame reproject the entire causal raw-stabilized track using the current
bias estimate; never mix previously corrected angles. Keep MZ91 association,
five-sample temporal fit, pose/gap uncertainty, fixed512 samples, support veto0.1,
raw candidate gate, hysteresis and one-frame hold unchanged. No raw candidate
outside baseline Radar acceptance may be introduced. MZ91 without calibration
is the mechanism control; matched_hold is the task baseline. Original negative
results remain frozen. New-arm scores/predictions are sealed for both regimes
before evaluator/provenance access.

Primary sensor_proxy gates inherited from MZ91: FP strictly decreases; TP at least
baseline-2; F1 not lower; no added false segments/fragments or lost baseline event;
max added first-alert delay<=0.2s. Report real-only and persistent-phantom errors
separately, calibration coverage, paired changes, UNKNOWN and future-only recall.
Ideal is a regression diagnostic. Any gain remains consumed Development evidence.
One fixed run per regime; no outcome tuning or automatic successor. Pass retains
a development challenger, failure retains a negative control, parity/source
failure is NOT_EVALUABLE. CPU scalar scoring via research_backend, no workers.

Artifact: artifacts.local/work/mz92-cross-sensor-calibration-20260912/run-v1.
