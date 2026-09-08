# City source collection and worker handoff, 2026-09-08

The primary host captured 39 successful source reconnaissance frames on the
original Big City map: 5 initial views, 33 views across seven candidates, and
one native HLOD-export probe. All successful sessions retained unchanged source
hashes and released their owned editor processes. A failed zero-frame helper
attempt remains in the evidence tree. These are source views, not the 336-frame
research cohort, and no model was run on them.

Evidence root: `artifacts.local/nearfield/city-crossregion-v1-20260908/`.
`source-delivery-summary.json` joins seven v4 capture receipts, release receipts,
and floor grids. Each grid has 961/961 native vertical first hits; these are
candidate surfaces, not semantic sidewalk or walkability labels. Contact sheets
are in `scouting/big_candidate_01-review-v4/` through `07-review-v4/`.

Visual review found that several farthest-distance candidates favor waterfront
plazas and shared skyline backgrounds. Seven regions remain `NOT_ADMITTED`.
Three additional TRAIN-only dense-city candidates were prepared using local
building density and geographic distance, without model outputs. The worker
packet `worker-dense-specs-v1/manifest.json` requests five source views per
candidate; image review must still identify actual sidewalk, intersection, and
narrow-passage routes. Density of actor descriptors alone does not establish them.

## Reusable collection commands

```powershell
python tools/city_region_scout.py prepare --candidates <candidate-json> --project <CitySample.uproject> --output <fresh-spec-directory>
python tools/city_region_scout.py run --manifest <spec-directory/manifest.json> --project <CitySample.uproject> --plugin <BlindAssistCapture.uplugin> --engine <UE58-directory> --output <fresh-capture-directory>
```

The runner verifies the frozen specification and host-local source-map hashes,
refuses to share a host with an existing Unreal process, preserves failed
receipts, and writes contact sheets plus batch progress/completion. Outputs must
resolve under that checkout's canonical `artifacts.local`. The launcher retains
RGB, depth, camera parameters, native probes and optional HLOD membership.

## Secondary worker

The worker uses an isolated City Sample project and explicit hashed source
snapshot, leaving its older project and checkout unchanged. The asset dependency
closure contains 128,070 files / 62,645,882,810 bytes, including all 101,981 Big
City external actor files. A separate 370-file engine-content hash comparison
passed. Transfer uses 124 resumable chunks with per-file verification; only
verified transport archives are removed.

The queued sequence requires the complete transfer receipt and source identity
checks, then runs one native-source smoke frame followed by the three TRAIN
scouts (15 frames). Smoke failure stops the chain. Raw data stay on the worker;
thin receipts and contact sheets return for review. At documentation time the
transfer is still running, so neither worker capture nor formal cohort completion
is claimed. Live receipts are under
`artifacts.local/work/city-crossregion-worker-20260908/`.

## Background identity limit

The native plugin built successfully and the rendered one-frame probe exported
925 loaded HLOD proxies with 3,342 source references and zero unresolved native
containers. These references point to another HLOD level; this is not a complete
original-actor identity graph. Full-map expansion remains incomplete. The
[background audit](BACKGROUND_FRUSTUM_AUDIT_20260908.md) records conservative
frustum overlap, shared visible global surfaces, and the remaining limits.

Target for the next collection remains useful complete routes with natural
obstacles and matched controls. Formal cross-region isolation is an additional
requirement for the research split; successful file transport or capture does
not certify it.
