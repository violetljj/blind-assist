# Fixed-origin rotational ToF jitter contrast

2026-09-10 EXPLORE, authorized by the user's walking-jitter suggestion.
This is rotational sensitivity on consumed static scenes, not a walking sequence.
It excludes translation, changing occlusion, changing RGB, pose estimation error,
and moving body-query coordinates. Frozen RGB and original B alerts stay fixed.

Use the same 600 source frames and 530 admitted endpoints as the static simulation.
Compare stationary and rotating sensor axes at the same origin, 25 samples over
2 seconds. Pitch is 2 degrees at 2 Hz, yaw 1 degree at 1 Hz; independent uniform
axis perturbations +/-0.25 degrees use seed23. These are assumed amplitudes, not
measured gait. Fixed15degree diagonal footprint and all three frozen return laws.
Never choose amplitude, phase, return law or threshold from results.

Pair both arms with identical per-frame/per-time range noise and dropout using
the existing disturbance generator with seed29. Measurement reads native scene
depth only. Model input receives scalar range, validity and known ideal pose;
no isolated target mask, truth, identity or correct association. Recompute the
body-frame possible-return bounds for each pose. Invalid or non-contained ranges
add no event, preserve UNKNOWN and every RGB event; no temporal accumulation.

Report each law on the common admitted denominator including invalid packets:
valid fraction, near-support fraction and any-time near support, added false
events, instantaneous fused HEAD near hit, and consecutive spatial-state flips.
Flips compare time-adjacent outputs of this assumed sequence, not measured walking
jitter or original alert instability. Original alerts remain unchanged by design.

Classify gain and cost separately: increased any-time correct near support is
coverage gain; reduced mean support or increased output flips is a stability
cost. Mixed evidence remains mixed. No automatic promotion, tuning, smoothing,
new network or fresh source capture. Stop after one paired pass and verification.
