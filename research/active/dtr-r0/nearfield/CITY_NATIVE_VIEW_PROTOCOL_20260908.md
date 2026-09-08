# Same-instance native bollard view probe

Pre-outcome EXPLORE protocol, 2026-09-08. Previous replay/thin weights alert on
4/4 exposed TRAIN bollard positives but0/2 consumed route positives. Route
distance/size is inside TRAIN ranges, while viewing direction and context differ.
Question: does that trained original instance retain its BODY alert when the
camera approaches across rather than along the bollard row?

One eight-view source-preserving capture in original Small_City_LVL. Controls
0..3 reproduce TRAIN6,8,11,13 exactly. Treatments4..7 use the same original TRAIN
bollard, distances1.5/2.4m and camera-local lateral offsets-0.08/+0.08m, with
camera yaw90 instead of0. Height2.43m, pitch-5, roll0, target/scene unmodified.
Reuse surrounding full-detail/HLOD loading and settled RGB-D capture. Check
map integrity, target visibility, floor, corrected independent collision checks
and release task-owned processes. Missing or disagreeing geometry stays UNKNOWN.

Compare original, completed native-only and completed replay/thin seed17 weights.
No optimizer steps, augmentation, threshold tuning, extra instances or further
views selected from outcomes. Save predictions before evaluator labels are used.
Historical and locked DEV cutoffs remain unchanged; replay BODY is primary.
Measure each paired BODY score delta, alert flips, reliable target overlap and
joint hits. Compare recaptured controls with existing TRAIN scores to expose
render/load drift; do not assume byte-identical RGB. Report target support sizes,
whole-query labels and any newly visible geometry that changes comparability.

If control alerts reproduce and reliable transverse views lose alerts, retain
evidence that changing the same-instance view/context affects response. This
does not isolate camera yaw, target appearance, occlusion or background causally.
If transverse alerts persist, that change alone does not reproduce the original
route miss. If controls drift or target visibility is unreliable, report the
comparison as not evaluable for that claim rather than model failure.
Eight views only; no rescue fit or source expansion. These selected same-map
Development observations cannot promote a model or establish fresh confirmation.
