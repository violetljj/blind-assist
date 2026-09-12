# MZ106: causal temporal geometric contradiction

2026-09-12. EXPLORE, consumed rendered Development. User authorized one bounded
short-history geometric check after MZ105. Keep SGBM+ToF, spatial boxes and final
two-on/two-off state fixed. No training, TAO replay, new source or threshold sweep.

Question: does an additional translated view disambiguate near support that
survived single-pair SGBM validation? MZ105 fixed photometric score gates failed
lossless task benefit. This intervention adds actual past-frame correspondence
and camera displacement; repeated predictions are not additional geometry.

## Inputs and opportunity

Use all576 MZ101/MZ102 frames, only previous frame in the same episode (0.25s).
First frames are unavailable. A previous-to-current baseline below0.10m does
not authorize temporal rejection. Camera position/orientation metadata may be
read to audit source opportunity and for an explicitly ideal-pose diagnostic;
they are not observable RGB pose. Actor identity/geometry and native depth do
not enter the verifier. Preserve the original576-frame task denominator.

One fixed RGB-derived metric pose probe: ORB2000, Hamming ratio0.75, rectified
stereo |dy|<=1.5px and disparity>=1px, known0.10m baseline; temporal PnP RANSAC
2px, at least12 correspondences/inliers, >=50% inliers, >=3 cells of a3x3 image
grid. Seal estimates before comparing to simulator camera pose. Never use that
comparison to choose per-frame poses. The observable arm uses only its own
validity flags; ideal poses are a separately reported control, never a fallback
inserted into the observable arm. If pose availability collapses, classify that
arm as limited/not evaluable and retain the ideal diagnostic's restricted scope.

## One fixed verifier

For every current SGBM pixel contributing to BODY or HEAD, track its left-image
location into the previous left image with OpenCV pyramidal Lucas-Kanade:
21x21 window,3 pyramid levels,30 iterations/0.01 epsilon. Accept tracking only
with successful forward/backward status, <=1px cycle error, <=20 mean grayscale
absolute residual, and complete10px image margin in both frames. Unavailable
tracking supplies no negative evidence.

Project the current SGBM near point into the past using the supplied relative
pose. Independently project the current pixel ray's depths4m..infinity; under
positive projected depth these form an image line segment. Compare the tracked
location with near projection (e_near) and closest point on that far segment
(e_far). Require valid projection and >=1px separation between the predicted
near and4m projections. Reject a stereo point only when e_near>2px, e_far<=1px,
and e_near-e_far>=1px. Rejected stereo evidence becomes absent/UNKNOWN. Keep
independent valid ToF support. Apply no final-alert gate, size criterion or hold.

This is a **contradiction-only** intervention: first frames, inadequate baseline,
occlusion/track failure or inconclusive geometry retain original stereo support
with verification UNKNOWN. They are not certified correct. Requiring every new
obstacle to already have history would build an unavoidable recall loss into the
test. Report unchanged/unavailable coverage explicitly. Dynamic surfaces are
outside the rigid-scene assumption; report moving-family outcomes separately,
without runtime actor-label gating. Existing moving-family camera translation
is zero; erroneous RGB camera motion may still expose a failure.

## Evidence and decision

Seal all pixel masks/support/predictions before task truth/family evaluation.
Verify RGB hashes, MZ105 cached depth source hashes and baseline support parity.
Evaluate RGB-pose and ideal-pose arms with the identical fixed verifier; this is
one mechanism plus the pose attribution control, not a tuned method comparison.

Both panels must lower raw FP without increasing raw FN, preserve >=95% of
each critical family baseline raw TP, and preserve all detected events. Report
exact raw/final TP/FP transitions, smallHEAD/poles, unavailable coverage, false
alert segments, UNKNOWN, elapsed processing and paired first-correct delays.
No final-FP reduction or >0.25s added delay prevents an end-to-end benefit claim.
If there is no useful benefit, stop without changing window, pose, history,
threshold, event state or source. A positive ideal-only result is diagnostic
headroom, not an achieved observable method. Any fresh confirmation is a
separate next decision, not automatically launched by this experiment.

Budget: one RGB pose pass and one fixed temporal verifier pass, plus evaluator
and narrow numerical checks. Mechanical repairs preserve failed logs. OpenCV
sparse feature/flow backend is recorded honestly; no CUDA claim from CPU work.
Artifacts: artifacts.local/work/mz106-temporal-geometry-20260912. Preserve source,
estimates, sealed masks and receipts; release owned processes on completion.
Registry failure, if still present, is reported without editing old entries.

Reference: [multi-view geometric consistency](https://demuc.de/papers/schoenberger2016mvs.pdf).
That literature does not validate this causal optical-flow gate or obstacle task.
