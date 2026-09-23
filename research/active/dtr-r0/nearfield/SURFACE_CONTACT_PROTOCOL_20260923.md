# Frozen depth, bounded local surface contact boundary

2026-09-23 EXPLORE, consumed MZ101/MZ102576pairs. User authorizes one fixed
local-surface recipe on saved unguided and VPP Large depths, no model retraining.
Hypothesis: continuous locally fitted visible support can recover more accurate
camera-forward contact boundaries, without inventing gap occupancy or deleting
thin obstacles. VPP's94.75%positive-pixel accuracy does not establish adequacy on
negative queries; retain its false near backgrounds and zero-support small HEAD.

## Geometry and contrasts

Two frozen inputs: ba-foundation-geometry-20260923-frontend-v1 and
ba-vpp-geometry-20260923-frontend-v1, exact576public IDs. Historical eligible
depth.5..4m,640x360,hfov70 and commonFOV az22.5/elevation20degrees remain.
No RGB model, ToF refitting, scene labels, pose inference or temporal fusion.

One conservative recipe: inverse-depth planar least squares in3x3pixel windows
only when all9original samples are valid, originaldepth span<=.05m and metric
fit residual<=.01m. Otherwise original vertex remains. No missing-depth fill,
component-area removal, smoothing across invalid pixels or unbounded planes.
Every valid vertex retained, including narrow/isolated evidence. Adjacent edges
require originaldepth differences<=.05m. Two bounded triangles per2x2cell need
all4original corners valid and allowed triangle edges. No enclosing occupancy
box, cross-gap patch, infinite plane, closed mesh or solid interior assumption.

For each input compare raw points, fitted points, and bounded surfaces including
the same fitted vertices and permitted edges. Fitted-point control separates
vertex movement from interpolation; all arms share eligibility/FOV conditions.
Thin fallback is explicit retained point/edge support, not proof of reliability.
Coherent false surfaces can survive this entire construction.

## Frozen camera-aligned task

Coordinate x/y/z=forward/right/up in current left-camera axes. Fixed height
bands BODY[-1.05,-.30]m and HEAD[-.30,.15]m, corresponding to the historical
dimensions at nominal camera height1.7m. They rotate with the image, not gravity;
no claim of recovering physical body direction or complete walking clearance.

Contact distance d(w)=minimum camera-forward x on supported geometry inside
the height band and lateral fullwidth w, with .5<=x<=4m and commonFOV.
Widths BODY[.36,.56,.76]m, HEAD[.24,.36,.48]m. Defaultwidths .56/.36m.
Critical fullwidth w*=2min|y| inside the heightband, .5<=x<=3m, w*<=1m.
These are metric boundary outputs, not distances inferred from a probability.
No contact output means UNKNOWN/no supported contact, never known clear.

Direct and fitted-point minima use the same AABB queries as analytic clipping
of finite surface triangles/segments. Report distance and width<=.05m success
on a fixed ground-truth-contact denominator, counting missing output as failure;
conditional MAE separately. Each frame/query equally weighted, wall separately,
thin_left/right,occluded_thin,small_head separately. Do not pixel-weight totals.
Critical width zero (central intrusion) reported separately from positive width
to avoid a large trivial-zero group hiding lateral-boundary error.

For pointwise alert use d(defaultwidth)<=3m; report pure geometry and union with
unchanged public ToF point support. Freeze two-on/two-off episode reset. Raw and
final results separate; these camera-aligned labels differ from old body/world
labels, so do not connect these counts to historical377/22/69 as one curve.

## Independent truth and error attribution

Seal all depth-derived predictions before evaluating actual scene geometry.
Truth uses capture manifest actual native cuboid bounds transformed exactly to
camera coordinates, intersected with query/FOV constraints by independent LP.
Do not use camera AABB enclosing a rotated object. Include declared background
and floor when they intersect this scope; retain hidden foreground obstacles.
Native visible depth provides a secondary sampled-visible point reference, not
complete geometry or the same fitted surface as the method being tested.
Record textured render/collision surface offsets and scene-oracle limitations.

Report negative-query false contact separately, particularly old VPP far-near
cases. Report missing/UNKNOWN and thin/HEAD coverage, not only errors where a
prediction exists. small_head_flat absent input remains included as failure;
no requirement or permission to recover it by unsupported surface expansion.

## Fixed execution and decision

Synthetic plane/gap/isolated-point/rotated-cube tests precede cohort scoring.
No threshold/window/horizon/width/seed sweep after outcomes. Interface/timing
canary may reuse fixedfirstframe without labels, then one576pair replay.
Use governed research-ue RunSpecs, preserve sources, predictions, receipts and
failures under artifacts.local. Compare supported CPU/GPU actual workloads when
selecting backend; record observed device and timings honestly.

Keep only demonstrated boundary benefit beyond matched fitted-point control.
If only walls/averages improve, boundaries do not, or gains rely on thin support
loss, close this recipe. Do not rescue by threshold tuning. Separate new layouts
would be needed to confirm a useful Development result. This run is neither
real ToF nor current-A/App/safety validation, and is no originality claim.
