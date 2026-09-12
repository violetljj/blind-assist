# MZ85 residual generic-FN audit

Mode: `EXPLORE`, read-only post-result diagnostic on sealed MZ84/MZ85 run-v2.

## Question

What observable mechanism accounts for each of the 11 remaining generic hazard
false-negative frames after MZ85 rotation compensation?

No prediction, threshold, state, source or label is changed. The audit first
checks every consumed artifact against its sealed receipt, then opens existing
observations, predictions and evaluator arrays.

Primary categories are mutually exclusive:

1. `LATERAL_HISTORY_COLD_START`: ToF is valid, but the first frame has no prior
   ray for the fixed angular projection; radar has no active evidence;
2. `AVAILABLE_RADAR_EVIDENCE_NOT_ACTIVATED`: ToF is UNKNOWN and a current raw
   radar candidate passes geometry/Doppler thresholds, but fixed hysteresis has
   not activated;
3. `DUAL_INSTANTANEOUS_OBSERVATION_ABSENCE`: ToF is UNKNOWN and no current raw
   radar return passes the fixed expert thresholds;
4. `OTHER`: none of the above, retained explicitly rather than forced.

Gap membership, gap onset, preceding hazard state and HEIGHT_UNKNOWN attribution
are orthogonal tags. HEIGHT_UNKNOWN generic true positives are reported as
attribution debt and never counted among generic FN.

This is a diagnosis of a consumed constructed source, not a promotion test. It
may bound what a later authority-aware hold can target, but does not implement
or score that hold.
