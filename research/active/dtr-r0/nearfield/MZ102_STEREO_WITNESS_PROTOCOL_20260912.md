# MZ102: query-local stereo surface witnesses

Date2026-09-12. EXPLORE, one fixed primary mechanism, one diagnostic contrast.

## Capability and fixed decision

MZ101 union adds54TP but19FP over full64-zone ToF. Fourteen final FP queries still
have current stereo support; many small far-background mismatches pass the
one-occupied-voxel readout. Investigate **query-local connected image-plane span**
as the primary change. Do not alter the ToF branch, matcher, camera calibration,
common FOV, task boxes, or 2on/2off state. No detector or decision-head training.

S: accept a stereo component only if its estimated horizontal OR vertical
image-plane span is at least0.10m. Components use4-neighbor pixel adjacency and
neighboring disparity difference<=1px. First restrict to each individual BODY or
HEAD query; an outside wall or another body part cannot sponsor a small fragment.
Span=max(pixel horizontal extent,pixel vertical extent)*minimum axial depth/f.
Depth-axis variation cannot supply qualifying length. No minimum pole width.
The0.10m task prior is explicitly not valid for every conceivable small obstacle.

S+I: fixed secondary diagnostic, start with S-qualified points and require each
to remain in the query after disparity+/-1px, then recheck S on surviving components. The endpoints
fB/(d+1),fB/(d-1) require d>1. This is an uncalibrated perturbation-stability check,
not a probabilistic confidence interval. It can delete real boundary evidence.
It is not the primary candidate and cannot replace S after seeing outcomes.

The bounded literature check found Hu/Mordohai's primary abstract evaluating17
stereo confidence measures: [TPAMI2012](https://doi.org/10.1109/TPAMI.2012.46).
Only the abstract was retrieved; full-text fetch failed. It motivates separating
depth estimates from confidence evaluation, not this specific span threshold or
an implementation/novelty claim. Large correlated wrong disparities can pass S
and S+I; neither proves object identity or solves all stereo ambiguity.

## Sources and runs

One disclosed consumed MZ101 Development replay evaluates this fixed configuration
and exact old prediction parity. No threshold scan or outcome-driven parameter
change. Then one new seed102013 rendered panel:24 appearance-paired episodes,
12 geometry families,288frames,4Hz posed sampling. New positions, speeds,
thicknesses, texture seeds and background pattern; overlapping generator families
are disclosed. Small HEAD protrusion and partially occluded thin rod are explicit
counterexamples to indiscriminately deleting small/short visible support.

Use the existing MZ101 collector and its warmed task-owned DDC; no hardware.
Shared640x360/70deg stereo/0.10m baseline, complete64 ToF centre rays, known pose.
Primary ideal-ToF and same common-FOV task as MZ101. No extra proxy sweep in this
run. Ground truth stays evaluator-only; incomplete depth stays UNKNOWN/no support.
Scene geometry/ToF point sampling and posed unlit textures are not physical sensor
or real-world validation. Fresh seed is not unseen-family external confirmation.

Freeze code before Development and fresh outcomes. Recompute fresh baseline
ToF, stereo, old union, ToF+S and ToF+S+I on identical packets. Seal predictions
and support counts before GT scoring; native depth may be used only for later
diagnostic/geometry checks. Primary candidate is fixed as ToF+S throughout.

## Primary useful-gain conditions

Relative to old union and ToF on the same fresh frames:

1. Preserve all ToF TP and detected events (an implementation invariant for union).
2. Preserve>=90% of **old-union-only TP**, and>=90% of its thin_left/right-only TP;
   no meaningful denominator means insufficient opportunity, not perfect retention.
3. Reduce>=50% of **extra FP**, defined per frame/query as candidate AND NOT ToF
   AND NOT GT compared with old union AND NOT ToF AND NOT GT. No extra FP in the
   old baseline means no falsifiable reduction opportunity.
4. No new entirely false sessions relative to ToF; no lost old-union detected
   events; no more than one sample(0.25s) additional first correct delay per
   retained event, and F1 not below old union.

Report allTP/FP/FN, BODY/HEAD and family slices, entire false sessions, false
segments, fragments, directly unsupported queries, and computation time. Separate
new warning onset from carried state and exclude left-censored/carried events
when discussing paired entry timing. Scores describe current forward corridor
awareness, not future collision prediction. FP duration sums both task queries.

## Stop and delivery

One configuration, one consumed replay, one fresh panel; S+I only as prescribed
diagnostic. If S fails, do not select S+I or lower thresholds retrospectively.
Keep positive component effects with explicit failed gates. If the source fails,
repair mechanical acquisition with logs and unchanged design; do not call it a
negative algorithm result. End after report, structured inheritance, scoped push,
video/overlays and owned process release; no automatic successor.
