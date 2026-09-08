# Secondary-worker 1000-frame attached-fixture collection

User authorization: collect 1000 data samples on the secondary Windows worker,
using the accepted contextual experimental scene. This means 1000 RGB/native
depth pairs, not 1000 independent worlds or 1000 quartets. No model training.

Design: 250 matched four-state groups. CLEAR/BODY_ONLY/HEAD_ONLY/BOTH each have
250 exact assembly-geometry labels. Crossbar and cabinet have 63 groups each;
inclined member and suspended sign have 62 each. The fixed seed is 20260908.
Every group preserves camera, geometry shape, material, tilt and thickness
while changing fixture height and its physically attached supports.

Group-level changes: BODY-front distance 0.8–2.8 m, lateral offset ±0.1 m,
eye height 1.6–1.8 m, pitch -8..6 degrees, inclined-member tilt 15..35 degrees,
member thickness 0.03–0.08 m and common height jitter ±0.005 m. All use the
same Street200V7 frontage; keep this entire site together in future source
splits. Exact cuboid sweeps include supports. Visible native support is measured
separately and visibility gaps stay in the full denominator.

Use `tools/make_contextual_headspace.py --collection-1000` with a pinned map
spec and a fresh output. Production capture uses native asynchronous RGB/depth,
640x360, fixed EV100=12, 32 settling ticks and existing native readiness gates.
No duplicate reference payloads or 720p appearance exports are requested.

Deployment is an isolated hashed runtime under the worker artifact root,
`work/city-pcg-20260908/contextual1000-runtime-v1`. It uses the existing
CitySampleSliceV2 project, map and plugin. Source checkout and existing assets
must remain unchanged; add only missing verified material instances. Transfer
the on-disk material dependency closure, not just unchecked material names.

Run a four-frame first-use smoke spanning the four fixture families, then one
1000-frame job. A launch acknowledgement is not completion. Full acceptance
requires native capture integrity, exact 1000/250 coverage, recomputed geometry,
reported visibility agreement/gaps, selected model RGB inspection, hashes,
terminal job result and process release. Stop at 1000 or a failure; preserve
partial evidence without automatically redispatching or increasing the budget.

Raw owner/location: BlindAssist contextual acquisition, worker artifact tree
`work/city-pcg-20260908/contextual1000-full-v1`. Controller thin evidence belongs
under `artifacts.local/nearfield/contextual-worker1000-20260908/`. Do not pull all
raw depth/masks back to the primary or remove the worker's retained dataset.
