# MZ36: frozen methods on newly captured XY locations

2026-09-10 EXPLORE. Question: do MZ28's additional detections and MZ30/MZ35's
false-alert corrections survive new camera/fixture placements without fitting?
MZ5 remains baseline; MZ28, MZ30 and MZ35 are separately frozen comparators.
All model, cutoff, normalization and bank hashes are sealed before new capture.
No fitting, bank updates, cutoff calibration or outcome-based checkpoint choice.

Use only dense_candidate_05 and dense_candidate_06 source-only floor inventories.
These CitySample regions were already used in a prior size experiment: this is
new XY/fixture capture, not unseen assets, independent regions or natural data.
Exclude XY within6m of every prior proposed/accepted site in both historical
proposal versions. Select15 sites per region using the retained floor selector;
require explicit road/sidewalk mesh and local floor agreement. Review all30 empty
views before predictions, and take the first10 source-valid ranked sites per
region. No extra candidate expansion. Insufficient sites means NOT_EVALUABLE.

Source validity requires successful capture/render/world/process receipts,
correct native floor, a camera above the floor and an interpretable forward
view without a camera inside opaque geometry or missing-scene render failure.
Record floating/background defects and exclusions; source review is not clearance.
After empty admission, generate20 fixtures per site from the retained TRAIN-only
catalog: four families, body/head relation states and existing crossbar controls.
Use new names/rank offset1000, fixed640x360 HFOV100, eye1.7m, level camera,
32/64 settling ticks and native_async export. Freeze full specs before capture.
Budget:30 empty views, then400 full frames; two regions, no outcome retries.
Native per-frame and complete-group admission precedes model prediction. Retain
all exclusions/UNKNOWN and attempted denominators; insufficient valid groups
limits the result rather than permitting replacement based on model behavior.

The predictor reads only RGB and ideal64-zone two-echo observations synthesized
from native depth. Native event labels are evaluator-only. This readout omits
sensor physics and is not the current single-zone ToF hardware interface.
The raw-frame adapter must first pass same-input historical numerical and task
parity, preserving failed attempts. Mechanical adapter repair is allowed before
new inference; no tolerance widening to relabel a numerical failure as success.

Report per-query TP/FP/FN, UNKNOWN, all-four-known exact frames, complete groups,
each region/family and paired changes. Test MZ28 against MZ5 for added TP/FP and
lost TP. Test MZ30/MZ35 independently against MZ28 for removed FP, lost TP and
preserved additions. A correction transfers usefully if it removes FP, loses
zero MZ28 TP and preserves additions on admitted new frames. Report failure or
zero opportunity explicitly; this finite result does not prove universal benefit.

Prefer the existing worker's hash-verified capture runtime and pinned plugin;
only rewrite the spec's map_file to its verified worker path. Each stage has a
fresh job/output, bounded3600s per region and verified owned process/Zen release.
Raw outputs stay on the worker; return the400 RGB/depth pairs specifically needed
for primary frozen inference plus native labels/receipts. Preserve durable inputs,
outputs, failures and hashes; release task-owned temporary capture material.

Stop after this fixed paired evaluation and audit. Retain components/tradeoffs
according to measured changes; no protected EVAL, App default or safety promotion.
