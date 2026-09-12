# MZ95 experiment B: horizontal pointwise ToF authority

One fixed EXPLORE independent of A. Radar raw gate and hysteresis are unchanged.
Only negative ToF branch authority changes. A positive ToF decision always acts.
When Radar alerts and ToF does not, veto Radar only if every currently qualifying
Radar return (or latest preceding frame when hysteresis sustains it) has a valid
ToF support with range difference<=0.15m and bearing within reported ToF bearing
+/-2.8125deg. Use causal yaw for a shared horizontal frame, and Doppler for a
one-frame range transport. Use all valid raw ToF bins for this coverage diagnostic.
Missing/invalid returns never certify coverage or empty space.

This is deliberately pointwise range/bearing compatibility, not proof the whole
uncertain Radar cone is observed or the same object was identified. It does not
provide height, true empty-space evidence, occlusion reasoning or posterior
coverage. A compatible nonalerting ToF observation is a heuristic local veto,
not certified clearance. Fixed range/bearing tolerances are not outcome fitted.
No MZ92 calibration or A admission is used in this separate B experiment.

Keep original outer one-frame hold. No-op coverage=all true must exactly reproduce
matched_hold. Freeze prior conflict cohorts: proxy20FN/8TN; ideal9FN/82TN. Report
recovery/loss on the same frames and total paired TP/FP changes. B only releases
vetoes, so it cannot reduce baseline FP; require F1 higher, TP not lower and no
extra FP for full gain, plus unchanged event/delay/fragment gates. Partial benefit
is not a promotion. Predict both sealed regimes before evaluator access.

One fixed run per regime, no parameter sweep, no learned head here. The baseline
and candidate remain consumed Development evidence. New UE comparison is separate.
Artifacts: artifacts.local/work/mz95-coverage-authority-20260912/run-v1.
