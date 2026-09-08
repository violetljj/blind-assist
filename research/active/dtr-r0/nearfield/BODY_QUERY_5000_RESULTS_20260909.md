# BODY-QUERY 5000 expanded collection

Status: PASS — 5000 new frames, 1000 accepted complete groups, zero native
intent mismatches. Final source, label, image and array hash audit passed.

[Source protocol](BODY_QUERY_5000_COLLECTION_20260909.md). The previous
120-frame pilot passed 120/120 native labels and 24/24 complete groups before
this separately versioned 5000-frame expansion. No model fitting or scoring
is part of this operation.

## Source admission completed

520 empty-view probes were rendered and visually reviewed. 100 contained
existing native BODY/HEAD witnesses. Visual reviews were performed by Codex
agents; native geometry supplies the labels. An additional 103 otherwise eligible
views showed building-shell/interior problems. From the remaining 317,
250 sites were selected in recorded source order; 67 eligible reserves remain
unused. No camera-floor mismatch or insufficient-known-depth rejection occurred.
Two regions needed directed proposals along already reviewed street corridors;
these proposals still underwent the same native and visual checks.

The completed collection has 10 BigCity regions, 25 distinct XY sites per region,
and 20 frames per site: four fixture-family quartets plus four additional
crossbar negatives. This gives 1000 complete groups, with 500 distinct local
geometry-state designs repeated across the 10 backgrounds/regions. It is not
5000 independent natural street scenes. The four families are crossbar,
cabinet, oblique rod and hanging sign. Every site includes CLEAR, BODY_ONLY,
HEAD_ONLY and BOTH; crossbar adds LOW, ABOVE, LATERAL_OUT and FAR_OUT.

Actual roles are TRAIN 2500, DEV 1000 and EVAL 1500 frames. Regions share
CitySample assets and may share backgrounds; these are controlled Development
roles, not a renderer-proven background-isolated or protected TEST cohort.
Previously frozen/consumed cohorts remain unchanged. All cameras use 1.70m
optical height, zero pitch/roll, 640x360 and 100-degree horizontal FOV.

## Capture and data

Primary and worker each own five sequential 500-frame region chunks. Each
chunk must finish native verification, resource-health checks, source/hash
binding and process release. Full per-frame labels additionally preserve
raw/capped 12-query counts, native support and UNKNOWN, exact-ray ownership,
local membership/known mass and masks. Raw-count aggregation must reconstruct
the original native BODY/HEAD bits. Intent mismatches are retained and cannot
be silently relabeled or counted as accepted complete groups.

A source-only asset defect stopped worker full-v1 before any frame: the trimmed
worker copy lacked MI_Bldg_PaintedMetal_Black.uasset. The recursive closure
contained 74 packages / 58 project files; 57 existing files already matched the
primary. Only the missing 6299-byte original file was added, with no overwrite.
All 58 project files then matched, and a fresh full-v2 batch passed. Geometry,
materials and roles were not substituted. The zero-frame failure is retained.

Artifact root: `artifacts.local/work/body-query-5000-20260909`.

- `collection-plan-frozen-v1.json`: immutable per-host source specifications.
- `accepted-sites-{primary,worker}-v1.json`: source selection and hash bindings.
- `source-admission-summary-v1.json`: all source rejection counts.
- `visual-review-v1.json`: all 520 reviewed source views and reasons.
- `primary-full-v1/<region>/labels`: primary complete ownership labels.
- `worker/full-<region>-v2/returned/capture`: worker raw data and receipts.
- `worker-labels-v1`: all 2500 worker-frame ownership labels, PASS.
- `full-visual-review-v1.json`: selected first/last site inspection only;
  40 frames per region, 400 total, with no identified visual defect. This is
  not a claim that all 5000 were visually reviewed.
- `full-visual-v1/ten-regions-head-only.jpg`: ten-region comparison with
  original-image hashes in its adjacent manifest.
- `dataset-v1/index.json`: final 5000-row RGB/native/ownership-array index.
- `dataset-v1/summary.json` and `manifest.json`: final acceptance and bindings.

## Final acceptance

All ten regions have 500 accepted frames, 100 complete groups and 25 sites.
There are 250 globally distinct XY sites, with at least 6m separation within
each region. All 5000 RGB hashes are distinct; no repeated RGB frames were
counted. Four cardinal headings contribute 1160 (-90), 1320 (0), 1220 (180)
and 1300 (90) frames. The 500 local geometry-state hashes are reused across
regions, so image uniqueness does not imply independent geometry or assets.

CLEAR, BODY_ONLY, HEAD_ONLY and BOTH each contribute 1000 frames. LOW, ABOVE,
LATERAL_OUT and FAR_OUT each contribute 250 additional crossbar controls.
Native BODY and HEAD each have 2000 positive frames. Every region has 50
groups with actual near HEAD positives and 50 with actual far HEAD positives.
All 1000 groups passed, with zero failed frames or rejected final groups.
These are geometry/intent agreement counts, not trained-model accuracy.

Three focused generator tests passed. The full finalizer verified all 5000
cases against their frozen source specs, all bound input/label-array hashes,
consistent labeling code/calibration across six label chunks, complete-group
acceptance, site spacing, role overlap by RGB hash, and all 400 selected visual
review image bindings. Completed source admission also exercises directed
proposals. No model inference or training steps were performed.

Final SHA-256 bindings:

| Artifact | SHA-256 |
| --- | --- |
| dataset-v1/index.json | `88dd9687fbefe2432ac05ac57d54510ec1f0d0e65faac646f324365ae380166e` |
| dataset-v1/summary.json | `6b04622d8fef97204871487e4edd7dd2045244dd4b960055bd1c92162fc6ac96` |
| dataset-v1/manifest.json | `a2717528ec08fdf98c278be9a56868d51f8d65d93e9c52f1b562be69334a9fda` |

Primary and worker capture processes were released; final primary inspection
found no collection Python, UnrealEditor or Zen process and no Zen listener
on port 28656. Worker release receipts report zero running jobs, UE or Zen.
Task-owned capture temporary directories were removed with verified receipts
(`temp-cleanup-receipt-v1.json`, `temp-cleanup-receipt-v2.json`, and
`worker/full-v2-temp-cleanup.json`). Durable raw data, failed-attempt evidence,
archives, labels and shared caches remain retained.
