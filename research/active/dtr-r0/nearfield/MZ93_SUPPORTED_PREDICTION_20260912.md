# MZ93 evidence-supported future occupancy

EXPLORE, one fixed consumed MZ90 comparison. MZ92 reduces relative bias error
without changing alerts. A pre-run score-only decomposition exactly reproduces
MZ92 scores: of469 qualified sensor_proxy returns,8 have current occupancy score
below0.1 but current-or-future score at least0.1; all8 lack supported inward angular
motion under the criterion below,6 are in no-ToF frames. This is a small possible
intervention, not evidence of final alert gain. Diagnostic is disclosed consumed
analysis, not an independent protocol design source.

Hypothesis: broad angular-rate uncertainty allows hypothetical future crossings
to preserve currently non-threatening returns. Separate current evidence from
predictive support. Retain the MZ92 total score only when history has at least3
points, mean_angle*mean_angular_rate<0, and abs(mean_angular_rate)>1.96 times its
model standard deviation. Otherwise use current-occupancy score only. This fixed
approximate95-percent directional criterion is not calibrated coverage. Keep
current score even when motion is unsupported; no future observation is used.
It treats the mean bearing sign as known and suppresses all future-only sample
mass when unsupported, including possible radial range-boundary crossings.

No changes to pairing, calibration, track fitting, uncertainty,512 samples,
veto threshold0.1, raw candidate gate, ToF or hysteresis/hold. Current-only score
uses the same posterior samples. No new triggers outside the baseline raw gate.
Unalerted no-ToF states remain UNKNOWN, not confirmed clearance. The readout
changes how speculative motion contributes, not the sensor model. This may lose
real crossings with poor motion evidence; measure that failure rather than hide it.

Baselines: sealed matched_hold and unchanged MZ92. One run per existing ideal and
sensor_proxy regime; predictions for both sealed before evaluator/provenance.
Primary gates unchanged: FP strictly lower, TP>=baseline-2, F1>=baseline;
no added false segments/fragments, no lost baseline event, added delay<=0.2s.
Report future-only TP, positive UNKNOWN, paired TP/FP changes, origin slices.
Ideal is a regression diagnostic. Pass retains Development challenger only;
failure is NEGATIVE_CONTROL; invalid parity is NOT_EVALUABLE. Stop after one run,
no threshold sweep, new source or automatic successor. CPU scalar workload.

Artifacts: artifacts.local/work/mz93-supported-prediction-20260912/run-v1,
with pre-run-diagnostic.json in its parent. No real-device or body-collision claim.
