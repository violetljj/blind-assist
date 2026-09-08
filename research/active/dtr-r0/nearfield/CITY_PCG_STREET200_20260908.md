# City PCG 200 m street: engineering delivery

2026-09-08. Native capture and geometry PASS; full visual acceptance pending.
No model training, threshold promotion or safety claim.

`/Game/BAResearchSlice/Street200V5` extends the plaza into an approximately
200 m street with three building rows, a crossing street, sidewalks and a
tree-lined plaza connection. Official building and paving graphs are copied
into task-owned graphs and parameterized; original assets remain unchanged.
Map SHA256: `d7d5452bfee59f678f6ae02fdd24829fc46ed89a43564c9e477ae505dc9cfc49`.

Evidence root: `artifacts.local/nearfield/city-pcg-20260908/` on F:.
`street200-layout-v3.json` and `street200-build-3/` retain the geometry recipe,
seeds and source hashes. `street200-finish-layout.json` and
`street200-asphalt-layout.json` record subsequent material edits. Warm V3
generation took 11.687 s, following initial asset compilation.

`street200-capture-3/` contains 15 native RGB/depth pairs, 1280x720 previews,
source snapshots, dependency manifest and receipts. Capture-script time was
151.172 s including dependency export, excluding host startup/shutdown.
`world-verification.json` passes on CUDA RTX5060 Laptop in 1.06 s: payloads,
source identity, process release, declared floor patches and two analytic cube
controls. The real railing has BODY support; a real sign pole and a composite
frame have both BODY and HEAD support. Mesh bounds are not mask truth.

The verifier retains 3 m visible-surface BODY/HEAD masks. Separate continuous
`distance_evidence` uses an 8 m review range, with experimental 1 m DANGER and
4 m WARNING divisions and a three-visible-pixel requirement. The same frontal
wall at approximately 0.8/3/7 m yields DANGER/WARNING/OBSERVE for both regions;
a near lateral wall yields NO_VISIBLE_SUPPORT. A zero in the 3 m mask does
not mean no obstacle or no warning evidence. This is not a 4 m warning mask.

These are gaps to observed surfaces along a fixed forward corridor, not
complete swept-mesh first-contact distances or safety thresholds. Occluded
geometry, collision-proxy agreement, per-instance rendered IDs and motion TTC
are not established. The proposed scene-group/ordinal-model schema remains
under review; no ordinal model or large dataset was started.

Earlier attempts remain retained: paving axes and join spacing were fixed;
the sample road material repeated white bands, which persisted after a parent
material swap. V5 samples an existing asphalt texture through a new material.
The road remains too bright and lacks markings; far boundaries remain open.
A foliage-intersecting plaza camera failed the first floor check; a later
overview exceeded the 100 m depth range. Final validation uses first-person
views. Curbs and route collision walkability are not certified.

Worker prepositioning reuses the prior plaza assets. New props added 10 files /
84,011,700 bytes; used rendering dependencies added 1,659 files /
5,766,087,253 bytes, individually SHA256 verified in 437.98 s. See
`worker-transfer-street-render-v1/worker-transfer-receipt.json`. Unused full
generator dependencies were not copied; worker UE has not been run.

Preparation was slowed by repeated map/material iterations and label-scope
expansion after cold compilation. The user requested no further expansion:
deliver this engineering scene with its limits and review the data proposal.
