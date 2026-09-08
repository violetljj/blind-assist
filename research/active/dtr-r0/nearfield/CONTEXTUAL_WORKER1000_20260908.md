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

Completed 2026-09-08 on the secondary RTX 3060 worker. The production job ran
16:07:14–16:19:40 +08:00 (12 minutes 26 seconds), exited successfully, and
retained exactly 1000 RGB PNGs and 1000 native depth arrays. Native capture
integrity passed. All 250 quartets and four geometry classes are complete;
999/1000 visible native labels agree with assembly geometry. The report status
is `VISIBILITY_GAP`, not an all-labels-pass result.

The sole gap is index 849, `contextual1000_g212_body_only` (crossbar).
Its assembly is BODY_ONLY while visible native support is [0, 0]. The
hash-verified RGB shows the near low cross member outside the bottom of the
image, consistent with its 0.840 m BODY-front distance, 1.785 m eye height and
3.791 degree upward pitch. Preserve the geometric positive and visibility-gap
flag; do not interpret the absent support as physical CLEAR or silently drop
the sample. No replacement capture was made.

The primary verified all 63 thin-package files, exact case/settings equality
after worker path adaptation, runtime manifest equality, and independently
reprojected eight hash-matched native frames. All eight agreed with the worker.
32 selected RGB hashes were verified (plus the gap RGB); the all-frame pixel
hash report contains zero duplicates. The four-family smoke RGBs were visually
inspected before dispatch. No model training or accuracy evaluation was run.

Pinned runtime source: `f17805ee`; original 1000-frame spec SHA-256:
`fd72fc798e64f87343f847fcd867d448f293e5b55d36171f3da5e27e8e2476b5`.
Evidence in the controller directory above includes `full-primary-acceptance.json`,
`primary-independent-audit.json`, `full-evidence/capture/contextual-check.json`,
and `final-release.json`. Final live worker inspection found zero task-owned
processes and zero scheduled tasks; the complete raw dataset remains at the
declared worker location. This is one-site controlled development data.
