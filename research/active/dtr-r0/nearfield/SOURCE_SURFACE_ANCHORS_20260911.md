# Source geometry repair: actual surface candidates

The worker exported actual LOD0 vertex/triangle data for wood, a landing frame,
a duct and a whole birch mesh. A subsequent CPU computation produced concrete
placement candidates for the two failed MZ72 HEAD_FAR families. It took 0.604 s;
no UE capture, model inference or source admission occurred in this computation.

Coordinates below are actor origins in camera-forward, camera-right and height
above the site-floor plane, in metres. Original rotations and 1x scales remain.
The actual lowest transformed vertex is placed on that plane. Horizontal
translation moves a surface cross-section at height 1.62 m onto the query centre.

| Asset | Proposed origin | Horizontal displacement | HF intersecting triangles, before / after | Fixed target-only rays |
| --- | --- | --- | ---: | ---: |
| Landing frame | [3.078248627, 0.420761599, 0.012788000] | [-0.255196898, -0.699314579] | 0 / 96 | 9 / 9 |
| Whole birch mesh | [2.060817560, 0.110537401, 0.060051756] | [+0.010817560, -0.929462599] | 0 / 750 | 3 / 9 |

These intersections use actual triangle surfaces, with the original HF query
box x=[1.63,3.13], |y|<=0.18 and z=[1.4,1.85]. Target-only rays do not include
scene occlusion, native rendering, material visibility or collision identity.
The birch witness is thin and belongs to a mesh component whose lowest point
is 0.985 m above the floor; this does not establish a connected branch or leaf
structure. No new semantic diversity count follows from these mesh names.

The wood is a single connected mesh under both index-edge and exact-coordinate
edge connectivity; the earlier suggestion of multiple loose pieces is not
supported by this topology. Its two rack gaps are 3.579 and 1.162 mm. The duct's
two support gaps are 502.197 and 2.875 mm, so the first rail especially requires
actual-surface repositioning. Bounding-box contact is insufficient.

The next small source capture should use these explicit poses, preserve geometry
across context partners, and check actual terrain contact plus isolated target
depth and original component identity separately from whole-scene event labels.
MZ72's failed source and stopped expansion remain unchanged. The prepared poses
are geometry candidates, not newly validated data or a started capture job.

[CPU report](../../../../artifacts.local/work/mz74-return-survival-20260911/surface-anchor-v1/REPORT.md),
[exact transforms](../../../../artifacts.local/work/mz74-return-survival-20260911/surface-anchor-v1/candidate-transforms.json),
[script](../../../../artifacts.local/work/mz74-return-survival-20260911/surface-anchor-v1/compute_surface_anchors.py),
[receipt](../../../../artifacts.local/work/mz74-return-survival-20260911/surface-anchor-v1/receipt.json).
Receipt SHA-256: 911fc76187051be08164b15d1044e155a6aa9ed4bac65962d2ed341420d6a9c1.
