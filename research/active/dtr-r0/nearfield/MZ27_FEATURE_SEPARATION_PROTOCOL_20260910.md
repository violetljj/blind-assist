# MZ27: frozen feature separation at unsupported angular locations

2026-09-10 EXPLORE, consumed Development. One no-training diagnostic; no model,
availability threshold, task cutoff, normalization, or capture changes. Register
before execution. Output: artifacts.local/work/mz27-feature-separation-20260910/run-v1.

Freeze anchors from MZ26 step4800: all three placement added false positives;
eight largest negative-availability bags each from exact TRAIN, oldDEV,
relationDEV and distanceDEV; actual contributor witnesses for all 75 original
MZ20 placement far additions. Negative bags mean eligible & known0, not task
false alarms. Task-error winner maximizes availability among eligible frozen
MZ20 cutoff-passing candidates. Positive witness is the frozen MZ20 argmax among
eligible & actual query contributors, as saved by MZ26 extrema. Preserve missing
witnesses. Ties use flattened zone/echo/cell order, bag ties global ID/query.
Collapse identical frame/zone/cell anchors, retaining all selection memberships.

At each anchor's exact zone/cell, references are exact MZ26 TRAIN IDs excluding
the anchor and any identical explicit site or group. Map old5000 through
observations.old_index into the source index's site_id/group_id, validating RGB
hash identity; map placement cache rows by global ID = 3700 + cache_index to
their explicit site/group. Never infer identity from filenames. Missing site or
group excludes a reference; an anchor missing either is UNKNOWN_IDENTITY and
unscored. MZ6 sequence has no declared site/group and is excluded unless an
explicit authoritative mapping is present; no mapping search or guessed ID.

Choose exactly the nearest 128 eligible reference frames by Euclidean distance
on the four-vector [clean two ranges/4, two good bits] in the same zone. good =
valid & finite & 0 < range <= 4; clean = range if good else 0. Stable tie break
global ID. Selection is label-blind. Only afterward inspect native known labels
at this same angular cell. Require at least five known1 and five known0; otherwise
INSUFFICIENT_CLASS_SUPPORT, without expanding or replacing references.

Compare fixed RMS Euclidean neighbors for four descriptors: packet-only4;
raw64 (bilinear normalized MZ16 HIGH_DETAIL feature sample); raw3x3_576 (nine
raw64 samples at offsets dx,dy in {-1,0,1} feature pixels, row-major, normalized
grid offsets 2/28, align_corners=False, zero extension outside the 28x28 map);
learned40 (MZ26 step4800 local16, zone-average16, absolute2, relative2, packet4).
Use the identical 128 references for every descriptor, no learned projection or
metric. Top five pooled neighbors vote known1 by majority (three or more).
Also save the closest-five distances separately by known class. No candidate
operating threshold is fitted or adopted.

Report every reference ID/identity/label/packet distance, descriptor distance,
top-five prediction and agreement; missing identities/classes/witnesses; anchor
duplication; per-selection-cohort label-specific agreement and balanced mean
when both labels exist; paired descriptor agreement changes on identical anchors.
Native known labels define retrieval groups only; query truth defines diagnostic
anchor selection only. Keep TRAIN tail and consumed DEV slices separate. Cells,
queries and repeated frames are correlated observations, not independent trials.

A raw-descriptor advantage over both packet-only and learned40 under the fixed
identity/packet controls would support retained visual information with a possible
compression/readout limitation. Absence of that advantage falsifies this particular
local-distance mechanism, not representational sufficiency in general. No claim
that arbitrary nonzero distances demonstrate semantic separability. No threshold
search, learned probe, alternative neighborhood, extra anchor sweep or fitting.

Bind caches, source identity index, normalization, MZ20 checkpoint/cutoffs,
MZ26 checkpoint/predictions/batches/extrema and their receipts. Frozen model
sampling runs on CUDA; scalar retrieval uses NumPy CPU. Verify selected positions,
saved extrema and descriptor-to-head output parity. Preserve failed outputs and
repair only mechanical implementation errors in a fresh output. Release process
resources at terminal. No protected EVAL or device/product/safety claim.
