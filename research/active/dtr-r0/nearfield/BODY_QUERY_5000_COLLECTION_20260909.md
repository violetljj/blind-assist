# BODY-QUERY 5000-frame expanded source collection

User authorized expansion after the 120-frame pilot passed 120/120 native labels
and 24/24 complete groups. Target: 5000 NEW accepted frames, not counting the old
pilot or engineering reconnaissance. No model fitting or scoring is authorized
by this collection operation. Preserve all historical/frozen source contracts.

Proposed distribution: 10 BigCity regions, 25 distinct XY sites each, 20 frames
per site. Every site contains four fixture-family quartets (CLEAR/BODY_ONLY/
HEAD_ONLY/BOTH), and the crossbar adds LOW/ABOVE/LATERAL_OUT/FAR_OUT. Thus 250
sites, 1000 groups, 5000 frames. Existing supported fixture variants are reused
with new distance/placement combinations; this is controlled synthetic data,
not 5000 independent natural street scenes. Near/far and family distributions
are balanced within regions; headings span cardinal directions. Preserve fixed
optical height 1.70m, zero pitch/roll, 640x360, HFOV 100 degrees.

Proposed Development roles preserve prior source assignments: big01..04 and
dense01 TRAIN; big05 and dense07 DEV; big06/07 and dense02 EVAL. This yields
2500/1000/1500 frames if all regions admit 25 sites. The reconnaissance-exposed
regions have shared CitySample assets and may share backgrounds. Region/site
IDs enforce grouping, not renderer-proven isolation or protected TEST status.
No model score participates in camera, fixture or group selection.

First capture source-only empty camera probes from existing native floor grids.
Require exact camera floor agreement, native BODY/HEAD no-witness, successful
capture/resource checks and visual route review. Candidate positions are at
least 6m apart within each region. Include preordered reserves; choose accepted
sites by this source-only order, retaining reasons for every rejection. This
cannot establish continuous walkability or hidden free space. Engineering
probes do not count toward 5000 final frames.

Final capture runs in region chunks on available primary/secondary hosts with
owned process trees, isolated cache-service ports, immutable specs and source
hashes. Keep RGB, native depth, raw/capped 12-query counts, support and UNKNOWN,
ray/local ownership, role/site/group/fixture metadata, render health, source
integrity and release receipts. Complete groups are the admission unit.
Mismatched/failed frames are retained and never silently relabeled. Necessary
source repairs or preordered replacement sites get fresh versioned specs before
any model access. Count only final admitted complete groups toward the target.

Acceptance reports actual frames, groups, unique XY sites, regions/headings,
geometry variants, native near/far positives and difficult negatives separately.
Raw-count/query aggregation must reconstruct original native near bits exactly.
Deliver 5000 accepted frames with documented exclusions, or report the precise
remaining shortfall; never substitute repeated frames or call launch completion.

Execution notes: source admission required 520 empty-view probes. Of these,
100 had existing native BODY/HEAD witnesses; a further 103 native-eligible
views showed invalid building interiors/shells. The final 250 sites were
selected in recorded source order from 317 eligible views. Two difficult
regions used fresh directed candidates near already reviewed street corridors,
with the same native/visual checks and 6m spacing; no model selected these.
Primary and worker specifications were frozen separately as their source
reviews completed, then bound in `collection-plan-frozen-v1.json`.

The worker's first full attempt stopped before frame 0 because its trimmed
asset copy lacked an explicitly requested fixture material. Preserve that
failure and supply the original material/dependency files with hash receipts
before a fresh attempt. Do not substitute materials or alter fixture geometry.
