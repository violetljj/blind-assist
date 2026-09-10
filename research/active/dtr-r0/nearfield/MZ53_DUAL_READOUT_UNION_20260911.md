# MZ53: fixed dual-readout union after negative-coverage learning

2026-09-11 EXPLORE, explicitly post hoc Development. MZ51's OLD_NEG GATED
reduces old noncalibration added false events from19 to5, but its442 nonfit
true events miss the453 target. The saved paired comparison shows that OPEN
recovers different true events. Test whether the union of the two existing
readouts retains this complementarity at a useful total false-alert cost.

Use only the sealed MZ51 run/score arrays and its exact original MZ50 arrays.
For each of fixed MZ50, NEW_NEG and OLD_NEG, OR the already-thresholded
OPEN and GATED decisions, identically for every query/frame/profile. No new
cutoff, selective query rule, model inference, fitting or native-depth access.
Keep each individual readout, MZ37, and both other unions as comparators.
This is a new composition test; it does not retroactively pass MZ51's gate.

Report TP/FP/FN, exact frames, paired gains/losses and UNKNOWN for every
existing cohort and IDEAL/MERGE_CLOSE/DROP_CLOSE. Preserve the original
MZ48 fit1280, rich calibration256, held-out-site640 and nonfit-family384
groups. Aggregate old noncalibration separately from new nonfit1024; fit
outcomes remain explicitly labeled. Keep all400 MZ36 attempts/80UNKNOWN,
original MZ37 positives, full frame identity and fixed source hashes.

Primary practical comparison under DROP_CLOSE: OLD_NEG union should reach
at least453 new nonfit TP with at most21 FP, and add at most19 old noncal
FP over MZ37. Also report paired costs against fixed MZ50 union and NEW_NEG
union, rather than attributing a generic OR gain to negative-source coverage.
An observed gain retains a challenger for this consumed source; a remaining
false-alert cost precludes a no-cost/default claim. Failure retains the
individual tradeoffs without searching thresholds or selectively dropping queries.

Verify each union using a separate scalar truth table, reproduce individual
counts against the sealed MZ51 result, and record exclusive OPEN/GATED correct
additions with saved native winning-cell evidence on MZ48. Native winning
flags are source-bound evaluator outputs, not newly independently derived
depth labels. Do not infer that every correct union event has local support.

Budget: one CPU saved-output pass plus these checks, a concise result,
structured disposition and scoped delivery. Stop afterward. This is small
Boolean bookkeeping (TASK_NOT_GPU_SUITABLE), not a hardware or fresh
confirmation result. Keep all original outputs immutable and release owned
processes after the pass.
