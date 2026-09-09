# Original crossbar, translated background diagnostic

2026-09-09 EXPLORE. One bounded attribution experiment after the failed combined
region/fixture shift. No training, model selection, threshold changes or ordinal head.

Question: does the unchanged JOINT spatial branch retain near/far attribution
when the exact original fixture and camera-relative geometry move to other backgrounds?
Use all 60 previously reviewed positions (30 in each region). Select the first
unused original EVAL crossbar BASE pair with matching absolute camera yaw in
historical metadata order. Copy both endpoints' complete 13-part assembly,
dimensions, rotations, materials, relative camera pose and lighting settings.
Translate camera, wearer and every component together; change only location and
streaming rectangle. Never reconstruct positions from nominal distance fields.
No prediction-based selection. These backgrounds and original samples are consumed
Development; this is a mechanism diagnostic, not fresh confirmation.

Budget: 60 near/far pairs, 120 new RGB/native frames, one frozen B/JOINT inference
pass. Original checkpoint, normalization and thresholds are hash-bound by
`body_query_fresh_size_eval.FROZEN`. Retain original B alerts. No size-matched arm.

Admission precedes predictions: native HEAD-only scene and intended exclusive
range, actual capture pose, floor support, component identity and visible target
extent, using the previous native checks with target `adjustable_cross_member`.
Require at least 24/30 admitted pairs in EACH region; otherwise stop the model
experiment as NOT_EVALUABLE. Preserve every exclusion and do not replace samples.
The 80% coverage floor is specified for this small diagnostic, not a revision of
the previous experiment's failed 90% gate.

Report four metrics on admitted new pairs and their EXACT historical partners:
near query hit, near-to-far confusion, strict both-endpoint pair-correct, original
alert parity. New pair-correct >=80% with exact parity supports background
retention in this cohort. A drop >=20 percentage points against the matched old
cohort supports substantial background sensitivity; smaller effects remain
limited evidence. If old matched pair-correct is below80%, it cannot serve as a
strong retained-capability control. No outcome grants broader promotion.

Geometry/material invariants must pass before capture; model receives RGB only.
Native arrays enter scoring, not the model. A location change can also alter
occlusion, illumination and reflections: this isolates the native background
intervention from fixture redesign, not every photometric pathway separately.
Stop after source admission, frozen inference if admitted, metric audit, report
and scoped delivery. Preserve evidence and release task-owned capture resources.
