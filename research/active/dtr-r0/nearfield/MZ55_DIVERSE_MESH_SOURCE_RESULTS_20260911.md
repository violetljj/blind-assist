# MZ55: diverse rigid-mesh source admission

The [frozen source protocol](MZ55_DIVERSE_MESH_SOURCE_20260911.md) completed its
original 2560-frame budget. All frames passed native source health and collapsed
BODY/HEAD relation intent. This admits a source component; no model was trained
or evaluated by this collection.

The matrix contains 1920 new-mesh frames (640 each shallow awning, sign panel and
square grille) plus 640 retained rod controls, across eight consumed sites, four
relations, two range anchors, two support contexts and five settings. Roles are
1600 TRAIN_CANDIDATE, 320 CALIBRATION and 640 HELDOUT_SITE. These role names do not
turn previously consumed sites into fresh blind evidence.

| Acceptance | Result |
|---|---:|
| Canary, counted within total budget | 64/64 source and intent; 32 invariant pairs |
| New family by relation canary cells | All 12 represented, four correct frames each |
| Unique frames / source-valid / intent-matching | 2560 / 2560 / 2560 |
| Paired target event-cell arrays unchanged | 1280/1280 |
| Pairs with changed background valid counts | 1280/1280; preserved |
| Independent native block-loop audit | 80 frames; 1,152,000 event and 288,000 valid counts exact |
| Independent raw mask/depth pairs | 40/40 |
| Original RGB hashes verified | 2560/2560 |
| Predeclared RGB/native panels actually viewed | 80/80 |

Actual BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR positives are 640/784/640/800.
The extra 144 BODY_FAR and 160 HEAD_FAR bits are retained: range anchors do not
promise exclusive event bands. Query UNKNOWN bits are zero for the admitted
frames; 7,417,741 of 9,216,000 full-frame cells remain UNKNOWN.

Full-frame supervision uses native-only uint8 `fullframe_event_counts[N,45,80,4]`
and `valid_counts[N,45,80]`, with maximum count 64 per 8x8 pixel block.
Presence is count greater than zero, knownness is valid count greater than zero,
and global event truth requires at least three native pixels. Angular auxiliary
labels remain `cell_event_presence[N,64,49,4]` and `cell_known[N,64,49]`.
These arrays are training/evaluator supervision, never predictor inputs. The
observable predictor surface remains original 640x360 RGB and 45-degree ToF
range/valid packets; source admission does not expand ToF coverage.

Square grille is admitted as the unchanged native **opaque patterned rigid
panel**. Reviewed dark center pixels had foreground depth; this does not establish
through-hole geometry. Root explicitly accepted that bounded interpretation
before expansion. There was no material or geometry adjustment, perforation
claim, optical ToF material validation or natural-installation claim.

| Owner | Completed frames | Raw capture bytes | Compact source ZIP bytes | Full-frame NumPy CPU seconds |
|---|---:|---:|---:|---:|
| Primary | Three main shards, 936 | 1,743,981,939 | 408,931,628 | 9.535 |
| Worker | Five main shards, 1560, plus 64 canary | 2,995,040,037 | 690,844,649 | 29.068 |
| Total | 2560 | 4,739,021,976 | 1,099,776,277 | 38.604 |

All original RGB bytes total 1,067,808,186 bytes and are retained unchanged in
the compact packages. The merged full-frame supervision is 568,029 bytes; it is
not a lossless replacement for raw depth. Raw capture trees remain on their
original owner hosts. Storage numbers are logical file bytes, not reclaimed or
allocated physical disk space.

Worker main throughput was 0.8195 frames per measured pipeline-second; primary
was 0.7192. Worker timing excludes separately measured, overlapped full-frame CPU
derivation; primary includes it. These are per-shard work sums, not a controlled
hardware comparison or two-host elapsed time. Final independent aggregation
took 7.337 seconds on NumPy CPU.

Root's independent `efficiency-analysis.json`, derived from the same saved
stage/storage receipts, attributes 91.8755% of primary pipeline time to capture
and only 0.2812% to packing plus roundtrip verification. RGB accounts for
97.0932% of package bytes. Packages are 76.7932% smaller than raw capture trees
in logical bytes, with raw still retained. Further speed work should therefore
target capture; compressing labels or optimizing packing has little remaining
measured leverage here. This is not a two-host wall-clock benchmark.

The worker initially owned four main shards. After they completed, root explicitly
transferred the still-unstarted `06_site_003` shard from primary to worker. Only
that shard was dispatched; the original specs and runtime stayed fixed. Primary
finished its other three shards, then its controller exited 1 at the ownership
guard before attempting the transferred fourth. Original
`primary-completion.json` remains marked FAIL and is preserved; the separate
`primary-release-final.json` records this expected authorized stop and verifies
that no fourth source attempt existed. Both hosts' final release receipts pass,
with no task-owned process or Zen listener remaining. No raw source was deleted.

Evidence is sealed under `artifacts.local/work/mz55-diverse-mesh-source-20260911`:
`source-index.json`, `source-v1/{result,receipt,native-audit}.json`, `parts.json`,
both final release receipts, and `storage-summary{,-receipt}.json`. The source
index SHA256 is `84848bc92ea077b9bbc651f6d080eedb9cdc97205c5ddc9f1033a1961032e176`;
the aggregate receipt SHA256 is
`8ef314c51bf96569d1a9e20f45ed5441d4f1d2de3404cc2738efb32e925a17fc`.

Retain this as a COMPONENT for fixed-checkpoint transfer and future explicitly
scoped supervision. Algorithm benefit, deployment cost, hardware performance and
safety remain separate questions. Do not tune a frozen transfer model or its
cutoffs from these source-admission outcomes.
