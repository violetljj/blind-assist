# Eight-family head-space geometry controls

**Rejected by the user as a scene/training design.** These detached cuboid
controls are retained only as geometry debugging history. The active delivery
is [attached contextual fixtures](CONTEXTUAL_HEADSPACE_20260908.md). Passing
geometry checks cannot waive implausible placement or missing supports.

User-requested extension of the HEAD/BODY data design. The frozen G10/G13 data
and models retain their original contracts. This is a new G14 engineering suite,
not a retrospective change to G10 or a training run.

HEAD-only means the obstacle intersects the future HEAD sweep but not the BODY
sweep. BODY-only, BOTH and CLEAR follow the two independent intersections.
The existing floor-relative boxes are BODY Z=[0.65,1.40] m, half-width 0.28 m,
and HEAD Z=[1.40,1.85] m, half-width 0.18 m; forward extents are 0.18/0.13 m.
Boundaries are closed. A very thin obstacle at 1.60 m is therefore HEAD-only,
not automatically BOTH. Body geometry, thickness, route and horizon determine
the label. Camera direction alone must not redefine an eventual walking route.

`headspace_spec.py` supplies eight explicitly synthetic cuboid compound proxies:
horizontal bar, branching low branch, side sign, scaffold tube, cabinet door,
stepped stair underside, open window frame and segmented hanging rope. These
are geometry controls, not botanical meshes, round pipes, deformable ropes,
finished supported furniture or accepted realistic assets. The current City
capture adapter renders the specified solid cubes. Do not replace an arbitrary
mesh with its whole-object AABB and call the result collision truth.

Each family has four rigid vertical translations calculated from the actual
body boxes and component height extent. Shape, part identity, scale, XY and
camera remain identical within each quartet. CLEAR raises the target above the
head; a later lateral-clear intervention can preserve a lower visible target.
Each world is fixed before two straight-route views are sampled, at nominal
BODY-front gaps 0.65 and 2.25 m. This yields 32 target configurations, 16 paired
quartets and 64 frames in one existing map. Near/far observations share exactly
the same world geometry. This is not 32 independent source worlds. The existing
adapter recreates identical target actors between cases; persistent actor reuse,
NavMesh routes, curved sweeps and balanced world-first random sampling are still
pending. This controlled extension does not complete the architecture in the
user's scene-first proposal.

The target-only reference uses continuous constant-X sweep against the union
of individual solid cuboids. Empty space between parts stays empty. It records
first-contact distances separately for BODY and HEAD over a 3 m horizon.
The existing native depth verifier independently measures all-scene **visible**
support and its configured 1/4/8 m distance states. These have different scopes:
occluded or missing pixels cannot establish CLEAR, while background geometry
can add visible support. `check_city_headspace.py` retains every mismatch and
reports it as `VISIBILITY_OR_SCENE_GAP`; it never copies an expected label into
observed truth, removes difficult samples, or claims model accuracy.

All geometry, expected contacts, family names and group metadata remain
evaluator-side. Training input is RGB/calibration only. All siblings, views and
trajectories from a world must stay in the same source split in any future
training set. Test outcomes must not tune acceptance quotas or choose assets.

Reproduce with `tools/make_city_obstacle_suite.py --headspace-controls --build
<verified-build> --output <fresh-artifact-spec>`, then the existing
`tools/run_city_pcg_capture.py`, `tools/verify_city_pcg_world.py` and
`tools/check_city_headspace.py --capture <fresh-capture>`. Preserve the build,
spec, code hashes and original native verification. This suite performs no fit,
inference, threshold tuning or model promotion.

The next natural-asset step is to replace each named proxy with a supported
asset/assembly while retaining geometry-derived interventions, and to verify
native collision versus visible surface truth separately. The larger world-first
generator must add plausible asset zones, route ownership independent of gaze,
view rejection for invalid placement, visibility/UNKNOWN accounting, and
source-world grouped splitting before collecting a balanced training cohort.
