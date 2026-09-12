# MZ107: single-camera association canary

EXPLORE, fresh controlled simulation, 2026-09-13. One bounded 12-scene x 8-frame
UE source, one RGB camera and synchronized hypothetical ToF/Radar/IMU. No stereo,
training, threshold search, hardware validation or automatic successor.

Question: can observed image extent refine uncertain Radar angular support while
preserving independent ToF support? Baseline uses all valid 8x8 ToF returns and
Radar range/bearing, stabilized with the same IMU increments as the candidate.
Current corridor: forward 0.2--3.6m, lateral +/-0.3m, height 0.4--2.05m;
orientation is initial walking heading, not inferred future intention.

The candidate adds fixed grayscale local-contrast/contour proposals from actual
RGB pixels: grayscale, 3x3 median, Otsu threshold, 5x5 closing, connected components
of at least12 pixels, excluding regions spanning80% image width or95% height.
This assumes the controlled source contrast and does not establish a general
obstacle detector. Both arms are current spatial-support readouts without final
alert confirmation/holding; this is not a comparison to all historical temporal
policies or a full alert-system upgrade. Radar velocity is recorded but not needed
by this current-occupancy readout; no closing-speed benefit is claimed.
No object identity, native depth, truth masks or rendered semantic
colors enter prediction. A unique proposal within the Radar bearing uncertainty
(12 degrees) can refine angular extent; an observed ToF return within that image
region must agree in range within 0.35m to establish cross-sensor association.
Multiple admissible regions or conflicting ranges leave the association UNKNOWN
and retain the baseline support. Missing image proposals also retain the baseline.
Independent ToF positives are never removed. Refined Radar geometry uses the same
measured Radar range; it does not estimate monocular metric depth.

Freeze all predictions before opening evaluator geometry/identity. Compare the
same frames with RGB enabled and RGB disabled (which must exactly reproduce the
baseline). Report TP/FP/FN, old TP lost/new TP gained, critical strata, episode
errors, UNKNOWN/no support, accepted associations and their evaluator-only identity
audit. Report actual algorithm latency separately from offline source rendering.

Retain as a component only if FP decrease, no baseline TP is lost (including pole
and HEAD), and observed associations have positive identity evidence. Otherwise
stop and describe the failure/opportunity limitation. Even a passing canary is
not promotion or independent scene confirmation. Simple shapes and controllable
contrast limit any visual localization claim.
