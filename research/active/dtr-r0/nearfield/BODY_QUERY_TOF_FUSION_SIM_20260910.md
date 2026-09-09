# Three-way spatial evidence contrast under hypothetical single-zone returns

2026-09-10 EXPLORE. User explicitly confirmed simulation-only hardware and asked
for RGB-only, ToF simple threshold, and visual/body-region fusion. This first
contrast is single-frame; it cannot measure continuous trigger-distance jitter,
dynamic approach or HEAD false-alarm rate without the corresponding denominators.

Use the unchanged600 captured frames and previously admitted530 endpoints:
53 original-fixture pairs,106 simplified ordinary pairs,106 size-matched pairs.
Preserve failed primary source gates and consumed status. No training, new capture,
checkpoint selection, threshold fitting or source replacement.

Model-visible input is RGB plus one sparse, explicitly HYPOTHETICAL range packet.
Fixed centered15deg and27deg diagonal ideal square footprints; sensor and camera
axes/origins coincide. Native radial-depth histogram bins are0.1m across0.1-4m,
weighted by solid angle. A mode needs2% of the full footprint's geometric weight.
Three predeclared response hypotheses select nearest supported mode, greatest-area
mode, or farthest supported mode. These are NOT calibrated VL53L1X algorithms.
Measurement computation may not read isolated target depth, target identity,
BODY/HEAD truth, labels, or predictions. Unknown native pixels do not vote.

Use seed17,20% simulated dropout and uniform +/-0.05m perturbation, shared across
response laws and footprints for each frame. Report the resulting effective valid
fraction separately; this is assumed simulator availability, not hardware quality.
Packet carries range/validity, FoV, hypothetical0.1m error bound (bin halfwidth
plus perturbation), zero same-frame age and an opaque ID. No histogram, mask or
correct association goes to the fusion method. Zero age is a synchronous static
simulation assumption; the corpus is not a timed approach sequence.

Methods:
- RGB-only: saved frozen JOINT range events, all original B alerts retained.
- ToF threshold: naively assign a valid in-range scalar to HEAD_NEAR at<=1.63m
  or HEAD_FAR above it. This deliberately exposes the simple baseline's unverified
  within-ROI association. Invalid packets assert no event (UNKNOWN, never CLEAR).
- RGB plus body association: the positive-only cone-support rule adds only events
  supported by full possible-return-region containment in a body query. Preserve
  every original visual event, including FAR when a near return is added. Invalid,
  stale(>100ms) or spatially ambiguous ToF cannot erase visual evidence. Original B
  alert decisions are unchanged in all versions; ToF/spatial results stay separate.

Evaluate common admitted denominators INCLUDING dropped readings for each law
and FoV. Report HEAD-near event hits, wrong-far on near endpoints, strict pairs,
BODY/HEAD range errors and valid observation fraction. Do not confuse event hits
with the historical count-query hit metric. Also disclose added false events and
120/480/all600 original alert parity. No ToF validity-filtered accuracy as headline.

Decision: retain as a candidate only if primary15deg fusion increases HEAD-near
event recall by>=10percentage points over RGB under EACH response law with zero
added false HEAD-near events. Separately compare with ToF threshold; if fusion
only repeats its performance, do not claim a fusion contribution or expand a
network.27deg is a fixed sensitivity control, not a selectable winner. Failing
any condition is partial/negative evidence, not a reason to tune this run.
Stop after one packet-generation pass, frozen-output fusion scoring, checks and
report. No real-sensor, fresh-generalization, continuous-time or safety claim.
