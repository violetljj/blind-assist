# Fixed original Large stereo with ideal-ray VPP guidance

2026-09-23 EXPLORE. User authorizes one matching-stage guidance experiment on
the same 576 consumed MZ101/MZ102 stereo pairs. This follows the original Large
geometry challenge: correct-query Z5cm improves 48.67% to 87.62%, but final
SGBM+ToF 402/40/44 becomes 377/22/69 and small_head_flat is missed. Keep those
failures and the MZ104/MZ105 negatives. No checkpoint or downstream change.

Question: can sparse measured distances change correspondence estimation enough
to improve correct near geometry and contact/events, without losing small HEAD
and thin obstacles? Not a real zonal-ToF, fresh-scene, current-A or safety claim.

One intervention: upstream bartn8/vppstereo revision
ecf12d40a816286d541c96c46746f43401836cfa, vpp_standalone.py, rnd method,
3x3 patches, blending0.4, non-uniform random colours, subpixel interpolation,
left-to-right traversal. These are its defaults, fixed before candidate scores.
No distance/adaptive patch, confidence filter, native-depth occlusion mask or
learned fitting. Numba RNG is seeded per panel/frame by SHA256; no seed search.
Use exact upstream projection function; preprocessing on CPU is small IO/ETL,
the unchanged Large model runs on observed CUDA. Verify source/config hashes.

ToF is ideal 64 centre rays, radial range0.5..4m, 45x45 degrees, same origin as
LEFT camera; right camera offset0.1m. Convert radial range to camera-forward Z
using normalized [1,tan(az),tan(el)], project into original640x360 left image,
round to pixel, then compute fB/Z. Do not use radial range as axial depth.
Preserve exact ray3D and subpixel coordinates. Rasterization shifts at most half
a pixel per axis; a 3x3 patch assumes local constant disparity and can cross an
occlusion/boundary. No native truth may validate, prune or fill input hints.
Out-of-image hints are excluded by calibration only. Invalid returns add nothing.
Raw RGB/ranges/valid hashes must match historical capture receipts.

Use the previous original23-51-11 weights/config, original resolution, official
padding,32iterations,max_disp416,mixed precision,low_memory0 and unchanged depth
eligibility. First the same3engineering canary IDs, without task-score selection;
then one complete576pair run. Store raw disparity, raw axial depth, historical
filtered depth, applied hint metadata, image-change counts and output hashes.
No failure rescue by parameter sweep or output filtering.

Primary contrast: guided Large+ToF versus unguided Large+ToF under exact common
FOV, poses, voxel support and two-on/two-off state. Also report both pure-depth
arms, with historical SGBM+ToF as retained reference. Thus four arms need only
one new inference; do not confound guidance with removing the final ToF union.

All candidate supports sealed before evaluator native geometry and task labels.
Keep fixed native-in-query pixel denominators, missing failures, conditional
MAE separately,5/10/20cm correct-query accuracy, far>4m and near-outside origins,
false-provenance TP, thin_left/right/occluded_thin/small_head, and final events/
delays. Count unguided-to-guided corrected AND regressed pixels/query frames.

Stratify whole-frame zero hints separately from no in-query hints. In-query and
nearby hints use measured3D transformed by declared pose; nearby means outside
the box but within0.25m Euclidean distance to its AABB. These are query-spatial
categories, not proof of same-object ownership. Pixel-change diagnostics use
distance to actual stamped hint centres: Chebyshev<=1px,2..16px,>16px, or no-frame
hint. The latter categories do not prove causal propagation or true association.
Whole-frame zero hints must leave BOTH input images and deterministic model
outputs identical to unguided. Synthetic empty-hint identity is also required.

Retain only demonstrated component/whole-task gains. Strong task regression or
failure to restore correct geometry closes this fixed recipe; no confidence or
alpha/patch sweep. Failure is not an information upper bound on ToF. Partial
spatial gains may inform separate structure work, never conceal small-obstacle
loss. Real regional/multitarget/CNH adaptation is outside this run.

Run materialization/inference/evaluation/audit via explicit research-ue RunSpec;
preserve raw observations, failures, seals and receipts under artifacts.local.
No truth enters producer; no GT-derived hint substitution. Release owned workers
when finished. Publish scoped report/terminal/inheritance via supported ledger.
