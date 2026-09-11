# Corrected surface placement: four actual diagnostic frames

The secondary worker completed exactly four frames using the original collector
and settling settings: corrected landing-frame and whole-birch poses, each in
its two original context partners. The target meshes remain at1x scale. Original
MZ72 sources and asset bytes are unchanged; its stopped expansion was not resumed.
This engineering check produced rendered placement evidence, not admitted training
data or a new model result.

| Target | Contexts | BODY_FAR pixels per frame | HEAD_FAR pixels per frame |
| --- | ---: | ---: | ---: |
| Landing frame | 2 | 2,014 | 231 |
| Whole birch mesh | 2 | 2,085 | 1,287 |

All four frames have zero BODY_NEAR and HEAD_NEAR pixels under the inherited
query geometry. The extra BODY_FAR positives are retained: these are not pure
head-only samples. Counts are native query pixels, not detected obstacles.
Full-scene and isolated-target query masks/counts agree, and corresponding depths
are within the inherited3cm diagnostic tolerance. Both context pairs have exact
query masks, query-union depths and isolated-target depth arrays. This supports
rendered target contribution but does not certify collision geometry.

Original-component ray checks remain UNKNOWN in all four frames. The two landing
frames each have2 matches out of32 sampled rays; the remaining results are30
unknown, then27 unknown and3 occluded. Both birch frames have32 unknown. This
sample is not restricted to the newly recovered head pixels. A correct component
path with a depth discrepancy is still not a matching rendered-surface witness.
This UNKNOWN belongs to the evaluator's collision comparison, not to ToF return
validity. It must not be reported as an all-invalid ToF frame.

A subsequent read-only diagnosis identified125 correct-component hits and3
other-component hits across the128 rays, with no empty hits. Of the125 correct
hits,121 still fail the depth agreement check. Near the optical axis, birch
render depth is2.397076m versus collision depth1.503031m; the radial/axial factor
is only1.000641 there. Pixel sampling, depth decoding and the inspected projection
algebra are consistent, so evidence favors a collision/render representation
discrepancy. This is not an independent camera calibration or proof of a specific
collision-LOD/Nanite setting. [Read-only diagnosis](../../../../artifacts.local/work/source-surface-smoke-20260911/ray-diagnosis-v1/REPORT.md).

The first-frame330-ray ground grid includes181 sidewalk hits at world z=0.73m
and149 hits on the controlled landing component. It therefore does not establish
unobstructed terrain beneath each contact point. Four camera-floor probes agree
at0.73m, but stable support and a full load path remain unverified. Context names
`supported` and `unsupported` are inherited recipe labels, not mechanical findings.

Root and worker actually viewed all four RGB images. The shifted assembly and
bare whole tree are visible; the whole tree does not establish leaf coverage or
isolate individual branch semantics. See [four-frame preview](../../../../artifacts.local/work/source-surface-smoke-20260911/preview-all4.jpg).

Capture and world verification both exited0; combined execution took91.820s.
The return archive is4,999,589 bytes;56 returned files were hash-verified. Owned
UE, Python and Zen processes were released. No model training, additional capture
or source admission occurred. Remaining work is specific: verify terrain while
ignoring controlled actors, and resolve collision/render-depth mismatch before
using original-component rays as surface identity evidence. Keep rendered depth,
collision and physical-support claims separate.
The existing floor trace API already accepts `ActorsToIgnore`; passing this
frame's controlled actors is the concrete next fix, without asset changes.

[Execution and pixel accounting](../../../../artifacts.local/work/source-surface-smoke-20260911/result.json),
[receipt](../../../../artifacts.local/work/source-surface-smoke-20260911/receipt.json),
[worker report](../../../../artifacts.local/work/source-surface-smoke-20260911/REPORT.md),
[root visual review](../../../../artifacts.local/work/source-surface-smoke-20260911/root-visual-review.json).
Result SHA-256: d491655dadfe5d7e492af51f2aa6eec6179adb27ae52a01eab4990ad6fcaddd4.
Receipt SHA-256: ec02613c2af19815abcb54362a08522c7da604bb288146107fcc4492f6f66fa1.
