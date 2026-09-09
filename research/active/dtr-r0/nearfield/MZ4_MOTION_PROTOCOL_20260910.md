# MZ4: does motion add angular information beyond one 8x8 observation?

2026-09-10 EXPLORE. Authorized following the MZ0/MZ1/MZ2 and single-zone
motion review. This is a new finite-scene information experiment, not continued
MZ1 training, a hardware emulator, real walking, or a new RGB algorithm.

Hypothesis: a short moving sequence can disambiguate BODY/HEAD membership that
remains ambiguous after repeated or individually decoded 8x8 observations.
Keep the previous zero/one/two nonoverlapping 2-degree patch family, centers on
the integer [-9,9]^2 angular grid, foreground axial X=2.5m, eye height=1.7m.
Use the same 500 preselected scene indices as the previous motion result and
verify their catalog identity. The background plane is X=3.25m (previously3.5m)
so the enlarged rotated45x45degree footprint stays within4m radial range.
This explicit change preserves angular geometry and query labels.

Generate25 exposures from the existing seed23 two-second trajectory. This is
generic simultaneous8x8 sampling at12Hz, not the exact VL53L8CX acquisition
schedule. Equal angular bins tile the45x45degree sensor FoV. Native640x360
camera rays/HFoV100degree and solid-angle weights define zone coverage. Verify
full footprint visibility and both distance-band bounds. No ray is a zone-center
point. Scene patches jointly obscure the background; consecutive returns need
not share a source object.

The primary hypothetical law reports a foreground return when total foreground
solid-angle coverage of a zone is>=2%; otherwise it reports background. Observation
is a quantized distance band: foreground radial[2.5,2.6)m, background[3.25,4)m.
Decode at2.925m; bounded +/-0.05m noise cannot cross bands. Discard within-band
range information to isolate angular attribution.20% independent whole-zone
dropout uses seed43, common case/time/zone masks across compared methods.
This is deliberately simpler than sensor target selection; no raw histogram or
ground-truth return identity enters inference. A background-only law is a separate
negative control for measurements that carry no foreground evidence.

Before scoring, predict codes for every allowed empty/one/two-patch scene.
Retain all scenes compatible with every valid observation. Query BODY and HEAD
from their physical rectangle intersections separately. Certainty requires
agreement across the nonempty feasible set; empty sets are model-mismatch UNKNOWN.
Known absence is only conditional within this finite family, never CLEAR.

Fixed comparisons (all500 cases, all25 steps, no selection):

1. Initial8x8 frame.
2.25 stationary repeats, jointly decoded (same first orientation).
3.25 moving frames decoded separately; accumulate only each frame's BODY/HEAD
   conclusions, without combining scene constraints. This separates seeing a
   better individual view from information requiring joint inference.
4.25 moving frames, joint exact-pose constraints.
5. Same moving observations, joint inference incorrectly using the first pose
   for every frame (unaligned control).
6. Same moving observations, joint inference using a fixed+0.5degree yaw bias
   (single pose-error stress, no retraining or amplitude sweep).

Run comparisons1-6 under the primary law; repeat1,2,4 under background-only.
Record both-query exact-known coverage, BODY/HEAD true-positive resolution,
false-positive and false-negative assertions, UNKNOWN, empty sets, feasible counts,
and new resolved cases relative to stationary/framewise. Assert true-scene
inclusion for matched exact-pose arms. Under that assumption zero wrong assertions
is a consistency property, not independently measured safety/generalization.

Success for the ideal information mechanism means moving joint constraints
resolve additional cases over BOTH stationary and moving-framewise controls
without wrong assertions in matched arms. Saturation/no gain retains the result
without parameter rescue. Biased/unaligned errors qualify deployability and the
need to represent pose uncertainty; they do not revoke the exact-pose comparison.
Stop after one fixed pass, CPU bitset recomputation of inference, independent
geometry samples, report and scoped delivery. No model fitting, new UE capture,
asset search, motion tuning, sensor integration, or automatic successor.

Artifacts: `artifacts.local/work/mz4-multizone-motion-20260910/run-v1/`.
Hash the protocol, source scene receipt/observations, code and results. Reuse
`tools/research_backend.py` for actual tensor-work placement and record timing.
Any mechanical retry retains its failed evidence separately and changes no
scientific settings. Release task-owned processes after completion.
