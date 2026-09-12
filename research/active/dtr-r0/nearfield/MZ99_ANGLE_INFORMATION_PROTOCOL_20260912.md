# MZ99: paired Radar angle-information diagnostic

EXPLORE, one fresh128scene x40frame UE capture, seed99013. No training,
parameter selection or successor launch. All5120frames are a new synthetic
diagnostic panel; legacy train/validation/test metadata are retained but unused
for fitting. Shared generator families are not unseen-family confirmation.
Source SHA256 1fd71e296d6cb3716286463c4ab2a64698b10b315bde7ded063e1d88e97e1153.

## One-page task contract

Coordinate x is lateral, z forward, metres, stabilized ground plane. Wearer
translation and obstacle trajectories remain scripted straight/CV motions.
Body is an evaluation-only axis-aligned0.6m wide x0.4m deep rectangle at the
wearer's position: half extents(.30,.20). These are explicit simulation design
parameters, not validated human safety dimensions. Object footprint uses native
UE bounds including actual width, not only its center. No BODY/HEAD height claim.

Two separate new targets (never pooled):
1. CURRENT_ROUTE: any real object footprint intersects the fixed-width rectangle
   x in[-.30,.30], z in[.20,3.60] at the current time. Spatial relevance only.
2. BODY_1S: any real footprint intersects the simulated body within[0,1]s under
   current relative CV. Compute exact continuous slab intersection of the object
   center with Minkowski-expanded body, including t=0. Also report current contact
   and strict future contact (BODY_1S AND NOT current contact) separately.
Legacy point-center wedge current/future labels remain untouched and scored as
a third, separate compatibility result. These are new task definitions, so their
scores cannot be presented as changes to previous MZ96/MZ97 accuracy.

All four existing methods still output a single boolean alert; evaluate that
same output separately under each target. This diagnoses semantic suitability,
not a new learned two-head architecture or a collision warning guarantee.

Shared causal presentation rule: start a warning session on first positive,
retain session through at most3 consecutive negative frames(.3s), close on the
fourth; one prompt per session start and no repeat while active. Reset each
episode. No feedback into algorithm state. Report algorithm frame F1 and fragments
separately from session starts/minute, false starts/minute (target false at start),
fully false sessions (never overlap target) and repeated starts within target
events. These are simulated prompts, not measured user annoyance or device UI.

Contact event timing uses per-object first contact from continuous native-footprint
CV geometry within[0,3.9]s. Warning window starts1s before contact; timely means
session active at some sample in[contact-1,contact-.5], leaving>=.5s model lead.
Count fresh session starts versus carried active warnings separately. Contact
before1s is left-censored; outside recorded3.9s is right-censored and excluded
from the timely denominator. Report censored counts. Later warning before contact
is late; no warning by contact is missed. One general warning can cover multiple
objects: report this shared-warning interpretation, not source-specific detection.
For current-route truth intervals, separately report fully missed intervals.

## Paired input intervention

Clone frozen MZ96 collector into new file, preserving RNG calls/order, measurement
laws, visibility, ToF aggregation, noise, detection, ghosts, noisy tuple sorting
and truncation. Add evaluator-only provenance AFTER original ordering: every
selected slot identifies real actor / persistent ghost / transient, pre-noise
range/velocity, exact sensor-relative bearing (real/ghost), observed triple.
No provenance or true fields enter the original16-key raw predictor contract.

Measured arm receives raw packets. Angle-corrected diagnostic arm substitutes
ONLY valid real/ghost radar_angle with its exact pre-noise sensor-relative bearing.
No re-sorting/re-gating/regeneration. Keep measured range/Doppler, all valid/packet
flags, IMU/yaw errors, ToF and temporal sampling identical. Persistent ghosts are
corrected to their own simulated direction, not removed; transients have no true
geometric bearing and remain unchanged. This removes Radar angular bias/noise/
quantization but is not an all-sensor or flawless-pose oracle and not achievable
calibration evidence. Predictor receives no truth range, motion or identity.

Run frozen baseline, R, R+F, A identically on each arm (MZ98 instrumentation
and unchanged downstream hysteresis/ToF/hold). Seal both inputs and all8 sets of
predictions, dependency hashes and source receipt before scoring labels.
Input assertions prove that only permitted Radar angle entries change. Include
exact per-slot provenance and final sensor/current/history source attribution;
mixed supports remain mixed, never uniquely assign a causal physical source from
co-occurrence. Provide paired removed/lost/added TP/FP and future-contact recall.

## Decision and limits

Question: does angular information materially limit current-route decisions, and
does correcting it also repair strict future-contact warnings? Primary diagnostic
readout is R on CURRENT_ROUTE. Predeclare angular benefit when corrected R has
>=20% fewer FP, >=98% paired TP retention and no additional missed current-route
truth interval. Otherwise report any tradeoff rather than calling the angle
information useless. Examine BODY_1S and A versus R+F separately; no mixing targets
or selecting a best method from corrected-input scores.1000seed99013 paired episode
bootstrap F1 differences are descriptive uncertainty, not a selection mechanism.

Do not interpret corrected-input gains as an algorithm improvement or product
readiness. No new hardware/RF/real-user evidence. Source distributions are unchanged
apart from seed/IDs; no extra source sweep. Unit-check slab/contact timing and
causal session boundaries before outcomes; independent review of intervention
and evaluator. Freeze before full capture/comparison; preserve mechanical failures.
Capture timeout1800s and release owned UE/editor/Zen processes. Scalar geometry/
scoring on CPU, no training or paid worker. Artifacts only under
artifacts.local/work/mz99-angle-information-20260912/. Finish result report,
inheritance and normal scoped commit/push; no automatic new experiment.
