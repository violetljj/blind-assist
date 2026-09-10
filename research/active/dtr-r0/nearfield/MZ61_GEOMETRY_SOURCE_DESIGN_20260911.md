# MZ61: geometry-disjoint source candidate

DESIGN ONLY: no registration, UE launch, capture, label result or model fit is
authorized by this document. Root must review the exact candidate before a
separate source registration and capture decision.

The MZ55 audit found all 160 held shallow-awning label tensors already present
among actually fitted frames: changing sites repeated the same camera geometry.
This candidate changes metric dimensions, aspect ratios, front/lateral positions
and three-axis tilt, rather than increasing site replicas alone. It uses the
same native mesh assets and a variable primitive rod; it does not claim new mesh
identity, through-hole grille geometry or natural installation diversity.

The 4096-frame matrix is four families x four BODY/HEAD/BOTH/nonintruding
relations x two distance bands x 32 geometry recipes x two site replicas x two
support contexts. The same eight consumed sites are used. Each family has four
angular-width targets, two depth/thickness settings and four tilt settings.
Their joint combinations partition into 16 TRAIN, eight CALIBRATION and eight
HELDOUT_GEOMETRY recipes: 2048/1024/1024 frames. Every split covers every marginal
factor value; the joint combinations are disjoint. Both sites, both support
conditions and all relation/range versions of a recipe retain its split.
Site names are not the holdout unit, and geometry IDs never enter a predictor.
Geometry hashes use only actual camera-relative target asset/material,
dimension/scale, origin and rotation fields; names, recipe IDs and roles are
excluded. Even this separation does not establish different native masks.

Supported capture fields are `scale=[sx,sy,sz]` for explicit native meshes,
`size_m=[depth,width,height]` for the engine-cube rod, `center_m`,
`rotation_deg={pitch,yaw,roll}` and `placement='actor_origin'`. Nonuniform scale
changes physical aspect ratios. The preparer composes camera-axis tilts with
the existing mesh orientation and converts back to supported Euler angles;
actual native actor bounds must later agree with the configured bounds.

Near front anchors are 1.04/1.08/1.12 m; far anchors 1.93/2.04/2.15/2.26 m.
Physical width is front times one of 0.35/0.45/0.55/0.65, making near and far
angular-size ranges overlap instead of giving distance a fixed-shape size cue.
Depth spans are 0.055/0.14 m before tilt; vertical spans vary by family. Tilts
combine pitch -4/2/-2/4, yaw -5/-1/2/5 and roll -8/-3/3/8 degrees. Lateral
offsets and BODY/HEAD heights also vary. The preparer verifies full RGB projected
bounding boxes remain inside the 640x360 image, configured targets stay within
their intended near/far and height band, and at least 75% of each family/split's
projected widths fall in the shared near/far interval. This is bounding-box
design evidence, not proof of rendered visible size or native event truth.

The 64-frame canary is counted inside 4096: one paired case for every family x
relation x range, across the existing sites. Only worker capture is proposed
while primary GPU fits run. Expand the remaining 4032 only after root admission:
all 64 source health/identity checks, 32 exact target-mask/depth pairs, at least
56 collapsed relation-intent matches, every family/relation represented
correctly in at least one range, and both near and far intended event coverage
within every family. Actually view all 64 RGB/native panels for absent
faces, unexpected occlusion and geometry/material mismatch. Independently hash
the nonempty native event-cell tensors, collapse deliberate support/site
replicas, and report held-to-TRAIN collisions. A repeated positive tensor is
evidence against useful diversity even when metadata fingerprints differ; do
not claim native uniqueness before capture. On complete source admission, report
this collision table over all configurations, with all-zero nonintruding tensors
separate. If held positive coverage still repeats TRAIN broadly, stop the claim
of new geometry evidence and report the source limitation rather than relabeling.

Native truth remains the original full-image, valid native point contract with
at least three pixels per query. Preserve every actual bit, including unexpected
near+far positives, source failures, intent mismatches and UNKNOWN. Configured
AABB intention never overrides truth or permits deleting positive bits. Paired
target event masks/depth must remain identical; background valid counts may
change. ToF remains the same weak 45-degree range/valid packet. Native arrays,
geometry, roles and labels are training/evaluator-only. No cutoffs, calibration
outputs or model predictions inform this candidate.

Reuse the compact MZ55 path: original RGB bytes, small packets/metadata,
full-frame uint8 counts[45,80,4]+valid counts[45,80], angular auxiliary labels,
and only 80 predeclared raw native audits (canary64 plus eight extra pairs).
All raw remains on the capture-owner host; optional transparent compression is
a separate authorized storage operation. At the MZ55 measured mix, 4096 frames
would be about 1.76 GB compact and 7.58 GB raw logical capture bytes. Worker-only
pipeline work is roughly 83 minutes; ideal balanced two-host work roughly
44 minutes before canary gating/startup/availability overhead. These are
extrapolations, not measured MZ61 performance. Primary may join only after its
GPU task finishes and an unstarted-shard allocation is explicit.

Preparation command:

```powershell
python research/active/dtr-r0/nearfield/mz61_geometry_prepare.py --root E:/linnan/linnan --output E:/linnan/linnan/artifacts.local/work/mz61-geometry-source-20260911/spec-v2
```

A future authorized worker canary uses the existing frozen runtime command:

```powershell
python <runtime>/tools/run_city_pcg_capture.py --project <existing-project> --plugin <frozen-plugin> --engine <existing-engine> --spec <one-canary-spec> --output <fresh-capture> --timeout 2400 --ddc-path <existing-warm-DDC>
```

No backend change is needed. A separate tiny dataset adapter will be needed
before capture to replace MZ55's hardcoded `(32,312)` shard-size assertion,
admit `HELDOUT_GEOMETRY` and retain `geometry_id/recipe_id` in evaluator metadata.
Do not edit the frozen MZ55 adapter. Keep its native truth, source health,
target-pair checks and packet reconstruction unchanged. Full-frame derivation
can reuse the frozen MZ52 recipe with the new exact source index. An eventual
model experiment must declare its own fit/calibration use; source role names
alone do not authorize tuning on held geometry.
