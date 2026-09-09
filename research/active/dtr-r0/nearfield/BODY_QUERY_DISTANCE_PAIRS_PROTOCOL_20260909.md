# Frozen B forward-distance pairs

2026-09-09 EXPLORE. Preserve expanded B and stop R0 loss tuning. Audit of5000
existing frames found2000 native exclusive HEAD-near/far frames but zero strict
same-camera/map/assembly near-far groups after removing only forward translation.
The source audit has no model inference. Capture32 new pairs /64 frames instead.

Use first two accepted sites in each of big01,big03,big05,big06, four existing
HEAD_ONLY fixture families per site. Translate the complete fixture assembly,
including supports, rigidly along camera forward axis. Within each pair keep
camera, map, dimensions, material, rotation, lateral coordinates and height
fixed. Endpoint distances are1.0/2.1m at the first site and1.3/2.6m at the second.
Fixed source specs and analytic intended extents precede capture; native visible
geometry is authoritative. Preserve failed frames/pairs and count any shortfall;
do not select replacements using predictions. No automatic capture expansion.

Both endpoints must have native BODY=0, HEAD=1 and HEAD capped count sum>=3
in exactly the intended distance half. Preserve native UNKNOWN; it is not FREE.
Verify paired spec equality, map/project identity, native capture health and
process release. Inspect all64 source frames before inference for capture defects.

Only expanded B checkpoint c7aef143bcc7ac464097e95219dc5239523df7ce5716fc95926f37592f94e776
is evaluated, with the original expanded-run DEV thresholds. RGB144x256 and
original inference are unchanged; native data is evaluation-only. No training,
calibration, R0 deployment, new depth model or conditional probability1-S_far.

Primary descriptions: both endpoint HEAD alerts per pair; delta=S_far(far)-
S_far(near), where S_far is the original three-cell capped-count event; positive
direction count, median/mean delta, and direction by family, size and site/region.
Record raw signs and tolerance1e-6 ties. Report BODY false alerts separately;
this all-HEAD-positive source cannot estimate HEAD FPR. Also keep near scores,
raw count distributions and per-frame predictions for interpretation.

Correct response supports within-object distance sensitivity in these controlled
assemblies, not metric depth, reliable near-cell attribution or arbitrary-object
transfer. Translation naturally changes angular size, occlusion and shadows;
the test does not separate those cues from explicit3D reasoning. Existing sites,
assets and geometry templates remain consumed shared-world Development; pairs
are correlated across locations/specifications, not32 independent natural trials.
Report unstable or family-specific response directly. Stop after this fixed
capture/inference comparison and deliver results without automatically opening
another model or data experiment.
