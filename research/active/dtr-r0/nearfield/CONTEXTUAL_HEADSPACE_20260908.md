# Attached fixtures: geometry, intrusion and observation controls

The user rejected floating cabinet/window/branch proxies in the middle of a
sidewalk. The replacement starts with a complete supported assembly, then moves
the wearer along its route. The earlier 64-frame primitive suite is diagnostic
history only and must not be promoted to training or realistic-scene evidence.

## Delivered design

The requested first phase is 4 fixture families x 4 requested region relations
x 3 lateral positions x 2 distances = 96 core conditions. Another 32 conditions
cover BODY/HEAD edge (8), HEAD/CLEAR edge (6), thin members (4), oblique members
(4), paired partial occlusion (4), paired lighting (2), pitch (2) and path offset
(2). This is 128 deterministic single frames, not 288–480 stochastic variants,
video, independent worlds or a held-out model benchmark.

- Crossbar: a height-adjustable maintenance access frame has grounded feet,
  uprights, visible clamps and side rails.
- Side cabinet: the back is mounted to the existing building facade. The shell,
  shelf, hinge, door and handle remain present at every tested mounting height.
- Oblique rod: the diagonal member is clamped to grounded uprights. Tilt changes
  also move its attachment sleeves; an unsupported tilted stick is not accepted.
- Hanging sign: a shop bracket meets the wall, with two side suspension rods
  outside the walking corridor. Changing panel height also changes rod length.

A separate two-view casement preview has a shallow frontage bay with a real
opening, sill, jambs, lintel and hinged open sash. It is not one of the four core
families and does not inflate the 128-condition count. These are assembled
synthetic fixtures using existing materials, not scanned complete furniture or
certified photorealism. The rod is a construction member, not a claimed tree.

The fixture site is Street200V7 near X=-76 m and the facade plane around Y=15 m.
All core near/far and lateral observations of one assembly keep its world
geometry unchanged. A height counterfactual moves the attached cabinet assembly
along the wall or adjusts the supported member; physical supports may therefore
change with the intervention. This is an explicit assembly intervention, not a
claim that only one actor changes. The UE adapter recreates identical actors
between views; persistent actor reuse and random world/NavMesh generation remain
future work. Camera pitch changes do not change the wearer sweep direction.

## Label and sampling contract

Keep the existing versioned BODY/HEAD envelopes: X forward, Y lateral, BODY
Z=[0.65,1.40], Y=[-0.28,0.28], HEAD Z=[1.40,1.85], Y=[-0.18,0.18] metres above
floor. Forward extents are 0.18 and 0.13 m. The user's alternative Y-forward,
1.50/1.90 m example is not silently applied to old G10 data. Touching counts as
contact. `BOUNDARY` is an orthogonal edge-perturbation flag, not a fifth label
that erases actual BODY_ONLY/HEAD_ONLY/BOTH/CLEAR geometry.

`contextual_geometry.py` computes continuous +X sweeps against the union of all
supplied solid cuboids, including supports, over 3 m. Exact YZ separating-axis
tests support UE roll for diagonal members; empty regions between parts remain
empty. This reference covers the attached assembly, not arbitrary mesh collision
or all background map geometry. The separate native-depth verifier observes all
visible scene surfaces. Occluded/unseen geometry may yield a visibility gap;
missing pixels never prove all-scene CLEAR. Native agreement is simulator
engineering evidence, not model accuracy.

Each condition retains sampling intent, effective height, actual geometry,
first-contact distances, all contacting component witnesses, and projected
lateral intrusion width/coverage per region. The last metric unions clipped
component projections somewhere inside the horizon; it is **not** volume
occupancy, contact probability or danger severity. Thin members keep their
positive collision labels. These evaluator-side fields are not model inputs.

Requested class is advisory until geometry checks pass. The core diagonal HEAD
height is 1.70 m so the left-offset BODY envelope also clears the inclined rod;
the cabinet uses 1.78 m to account for panel thickness. All 96 core conditions
have matching intended/computed relations, with 24 of each. Hard conditions keep
their actual outcomes even when they differ from the intended sampling bucket.
Thick-panel boundary perturbations use panel edges rather than nominal centres.

Core distances are 1.0 and 2.5 m BODY-front gap, lateral offsets -0.1/0/+0.1 m.
Hard pitch is -15/+10 degrees, route offset -0.35/+0.35 m, thin member width
0.01/0.02 m and tilt 15/30/45/60 degrees. A grounded edge notice stand supplies
partial occlusion. The darker-light member scales actual directional light to
0.15 and skylight to 0.35; both reset to base values on subsequent cases. Fixed
EV100=12 improves detail in the shaded facade. No post-capture image editing is
used for truth or model input.

World/site, parent and factor identifiers support paired analysis. The entire
source site must stay together in any prospective source-disjoint split; family
IDs do not turn one facade into independent TRAIN/VAL/TEST worlds. This suite
does not fit a model or establish appearance invariance, monotonic model scores,
distance robustness, model IoU or real-device transfer. Those need an explicitly
run model comparison on accepted data, preserving old G10/G13 decisions.

## Reproduction and evidence

Use `tools/make_contextual_headspace.py --map-spec <pinned-spec> --output
<fresh-spec> --collection-128`. Omitting the last flag emits the 10-view placement
preview. Then run `tools/run_city_pcg_capture.py`,
`tools/verify_city_pcg_world.py` and `tools/check_contextual_headspace.py` against
the fresh artifact capture. Source/map/material hashes accompany the generated
spec; original capture, UNKNOWN accounting and mismatches remain immutable.

All evidence is under `artifacts.local/nearfield/city-pcg-20260908/` on F:.
`contextual-capture-v1` contains ten supported-placement views with passing native
geometry. Its darker initial exposure was inspected, then a four-frame check
at `contextual-smoke-capture-v3` verified the brighter setting and actual light
scaling: cabinet HEAD/BOTH, inclined HEAD and darker hanging-sign HEAD all agree
with native support. Both task-owned editor sessions were released.

The final specification is `contextual-128-v3.json`; payload root is
`contextual-128-capture-v3`. All 128 captures and native geometry checks pass.
`contextual-check.json` records core agreement 96/96 and hard agreement 32/32,
with core geometry CLEAR/BODY_ONLY/HEAD_ONLY/BOTH each 24. No mismatches or
samples were removed. This is geometry-versus-native-support agreement, not
learned-model performance. The editor lifecycle was 168.844 s; release receipt
reports `released=true` and no surviving task processes.

The four inspected final mid-distance HEAD views retain their visible supports
and wall connections (`supported-fixtures-overview.png`, sample indices
15/39/63/87). The partial-occlusion pairs retain target-visible pixel counts
30332 -> 23826 for the cabinet and 18362 -> 16406 for the sign; these counts
use native points inside known target solids with a 3 mm surface tolerance.
The daylight/darker pair's RGB means are 141.327/103.833 on the 0–255 scale.
`hard-observation-check.json` records these diagnostics. They demonstrate the
requested observation changes occurred, not a model response to them.

Final spec SHA256:
`593decc5cfa55705d075fe8901172e65214fa249e4048e707e916f4fcae30032`.
Native report SHA256:
`985ceb3f9a4a321523e0843789450a3ee999068fdfff7046652c0e73c22a1b51`.

Sixteen focused sampling/geometry/scene tests pass, including tilted geometry
whose enclosing AABB would falsely predict BODY contact and overlapping
intrusion intervals counted only once. Prior detached controls remain rejected
for scene use; no old training data, threshold or model has been changed.
