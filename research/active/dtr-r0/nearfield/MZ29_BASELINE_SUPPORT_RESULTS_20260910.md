# MZ29: most baseline false positives are outside the geometric repair surface

2026-09-10 EXPLORE, consumed Development. **80 of 81 placement baseline false
bits have no original geometric candidate.** Correct baseline positives also
occur without candidates: 48 relation BODY_NEAR bits and 28 oldDEV BODY_NEAR
bits. Missing local support cannot be interpreted as absence of an obstacle.
MZ22's supported-only residual branch did not test this correction responsibility;
MZ28's add-only composition preserves these baseline positives by construction.

[Protocol](MZ29_BASELINE_SUPPORT_PROTOCOL_20260910.md),
[diagnostic](mz29_baseline_support.py), [independent audit](mz29_audit.py).
No model, cutoff or prediction changed. There was no fit or neural inference.

| Normal cohort | Frames | Baseline TP bits | Baseline FP bits | FP without geometry | TP without geometry |
|---|---:|---:|---:|---:|---:|
| oldDEV |1000|715|34|33|28|
| clean sequence |200|17|27|24|0|
| stress sequence |200|17|27|24|0|
| relationDEV |2000|1502|47|46|48|
| distanceDEV |1000|935|34|34|0|

The five normal views contain 4400 frame-view entries and 17,600 query positions;
clean/stress reuse the same 200 sequence frames. Their 3355 baseline-positive
records are 3186 true and 169 false bits, not independent obstacle events.
Wrong-visual controls are outside this diagnostic's scope. Query order below is
BODY_NEAR, BODY_FAR, HEAD_NEAR, HEAD_FAR.

Placement baseline FP counts are relation `[2,5,26,14]` and distance `[0,0,25,9]`.
The sole geometrically supported placement FP is relation frame1049/global12249,
HEAD_NEAR, cabinet at `big05_site_168`. Its MZ5 margin is1.815525; frozen MZ20
local margin is-2.752551. It has no actual query contributor, and MZ26/MZ28 both
remove its local support. Its original baseline-positive decision remains.

All81 placement FP bits have zero actual selected-return query contributors.
However,54 placement baseline TP bits also lack those contributors: the48
unsupported BODY_NEAR bits plus6 BODY_NEAR bits with geometric candidates.
The latter distinguish a geometric hypothesis from an actual source witness.
All28 oldDEV baseline TP geometry gaps likewise lack selected-return query
contributors. These are source/coverage gaps despite positive native event truth.

Among the2437 placement baseline TP bits, original geometric support is missing
for48, MZ26 learned availability support for75, and MZ28 packet availability
support for59. Both availability mechanisms also leave all81 baseline FP bits
without local support. These overlapping positive/negative partitions require
independent observable evidence for correction; this result proposes no veto.

Frozen local-score agreement is a separate property. At the unchanged MZ20 task
cutoffs, none of the81 placement baseline FP bits receives a positive local score
from MZ20, MZ26 or MZ28. Yet443 baseline TP bits also lack a positive MZ20 local
score, and488 lack one under each of MZ26 and MZ28. Unsupported raw-score sentinel
values are not measured negative evidence or calibrated absence probabilities.

Here, actual-query contributor counts are occupied query-labeled subcells of
selected echo sources, summed over zones/echoes/cells. They are neither all native
query pixels nor whole-object segmentation. Global task truth remains the frozen
native event definition; known/source masks are evaluator-only. Missing source
membership, UNKNOWN, and geometric candidate absence must not become CLEAR.

Artifacts are under `artifacts.local/work/mz29-baseline-support-20260910/run-v1/`:
`partitions.npz`, `result.json` with all3355 rows, receipts and `audit.json`.
Explicit placement family/site/group identities are retained; old/sequence row
identities remain empty in this export and are not guessed.

Independent audit PASS: every exported row and partition matches frozen
predictions and authoritative event/source labels; a separate NumPy float32
implementation verifies110,387,200 candidate-geometry bits, including near/far
boundary conventions. Stress echo labels are aligned. Summary counts are
recounted from exported records, and input/code/protocol hashes verify. The
audit performs no neural work; its process exited0. No new source, protected
EVAL, App/device change, deployment or safety conclusion follows.
