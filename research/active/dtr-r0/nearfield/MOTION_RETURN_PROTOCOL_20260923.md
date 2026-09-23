# Public motion consistency as return attribution evidence

2026-09-23 EXPLORE. User authorized trying public attribution after the matched
range diagnostic. Reuse the complete consumed stability and rescue cohorts;
no capture, training, parameter sweep or alert-policy modification.

Question: does current/previous RGB motion distinguish actual contributors to
the current observed ToF return within a zone, beyond a static-image and wrong
zone control? Pure forward/backward translation with fixed rotation is the
explicit controlled scope. Public extraction never receives camera translation,
object identity, target masks, native depth, time direction or labels.

Fixed public extractor: sample every fourth point in each axis of the existing
192x256 native sampling lattice, projected to public 320x180 pixel coordinates.
Retain all 3072 grid points; mark public FoV membership rather than selecting
points using truth. OpenCV pyramidal LK current-to-previous and reverse uses
21x21 windows, three pyramid levels, 30 iterations and epsilon .01. Quality
requires both statuses, finite in-image endpoints and <=1 pixel forward/backward
error. This is a fixed tracking-quality test, not a selected ownership cutoff.

For a tracked current pixel q and previous pixel p, use the current and tracked
previous public zones' nominal valid distances. The prediction is
q_hat = c + (p-c)*z_previous/z_current, c=(159.5,89.5). Score is minus Euclidean
pixel residual. Invalid returns or tracking yield missing, never distant range
or clear space. Static control substitutes p=q using the same selected ranges.
Wrong-zone control replaces the previous zone by column+4 modulo8 in its row.
Do not use future RGB, a known approach direction or actual camera displacement.

Seal public arrays before opening evaluator geometry/native depth. Reproduce
all saved ToF from original simulation identities. At sampled points, derive
current observed winning-bin contributor membership and separately native
target-in-corridor contributor membership. Target masks are evaluation only.
Contributor identity is weaker than target ownership and current membership
does not prove the same surface generated the previous return.

Primary comparison: macro ROC AUC within each frame/zone containing both
contributor and noncontributor grid points on a COMMON available subset for
all three scores. Report each cohort and geometry group, exact evaluable zones,
all-point and contributor tracking/matched coverage, missing reasons, first
frames, phase, shape, relation and same-pose dwell. Nonmixed zones are explicitly
not evaluable for AUC. Repeated frames are not independent examples. Auxiliary
target-in-corridor AUC cannot replace failed primary results.

Retain as a public attribution COMPONENT only if both cohorts have primary
macro AUC>=.70, >=.05 paired macro improvement over BOTH controls, >=50%
matched coverage of all current contributor grid points (including first-frame
missing history), and at least four geometry groups with evaluable mixed zones.
Otherwise retain measured partial signal and this exact diagnostic as a
NEGATIVE_CONTROL. No cutoff rescue or automatic student follows this test.
Even a pass needs a separate end-to-end benefit experiment; no safety claim.

Budget: one extraction/evaluation over 1152 saved frames, no resimulated noise
or extra scene generation. CPU TASK_NOT_GPU_SUITABLE: installed OpenCV CPU LK
and NumPy bookkeeping; log backend/runtime. Execute through research-ue with
source hashes and receipts, preserve prior frozen evidence and UNKNOWN.
