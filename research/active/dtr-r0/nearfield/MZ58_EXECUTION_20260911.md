# MZ58 execution and current efficiency bottlenecks

The single frozen inference pass completed on all 2560 admitted MZ55 frames
in 64.391 seconds on the primary CUDA GPU. Ten fixed methods share three
observed-packet profiles and three RGB views. The global 256x144 BOX view,
original 224x224 crop and full 640x360 view were each encoded once per frame;
their features were shared across readouts and profiles. RGB was loaded 2560
times, totaling 1,067,808,186 bytes. View encoding took 21.147 seconds and fixed
readouts 13.605 seconds; the total also includes source binding, loading and
output work. This is a desktop batch measurement, not Android latency.

The run used original checkpoints and four original cutoff vectors. It performed
zero training steps, zero new calibration, zero old-cohort replay and zero
native-depth reads. The predictor did not read new evaluator labels. No dense
feature cache was created. The process exited with code 0; both recorded process
IDs were subsequently absent, and compact archive handles were closed.

For MZ55 collection, the three primary shards spent 1195.638 of 1301.368
pipeline seconds in capture: 91.875%. Packing plus roundtrip verification took
3.660 seconds, or 0.281%. The frozen capture already used asynchronous paired
export, zero deliberate settling interval, 32 settling ticks and 64 first-use
ticks, with native resource-readiness checks. These measurements identify the
capture stage as the main throughput target; they do not identify how much
time is rendering, settling or target export. Reducing settling requires a
separate image/label consistency check, not an assumed free speedup.

The two hosts completed the original 2560-frame budget. The secondary host took
the primary's final unstarted shard through an explicit exclusive handoff. This
avoided duplicate collection and allowed primary GPU inference to resume.
Per-host stage totals are not a controlled measurement of dual-host speedup.

Compact source packages total 1,099,776,277 bytes, about 0.430 GB per 1000 frames.
Original RGB accounts for 97.093% of that payload. The packages are 76.793%
smaller than the original capture trees by logical bytes, but the originals are
retained on their owner hosts: this percentage is not freed disk space. The
merged full-frame labels occupy only 568,029 bytes. Further label compression
has little remaining payoff; any smaller RGB representation would need a
measured effect comparison before replacing the preserved original images.

Earlier MZ54/MZ56 feature reuse used a hardlink, followed by verified removal of
the last 4,408,012,928-byte temporary cache. That cache removal is completed;
the original sources, checkpoints and evidence remain available.

Execution evidence is under
`artifacts.local/work/mz58-diverse-transfer-20260911/`:
`execution-receipt-v1.json`, `process-exit-verification-v1.json`, and
`run-v1/receipt.json`. Source measurements are under the corresponding
`mz55-diverse-mesh-source-20260911/` task in `storage-summary.json`,
`efficiency-analysis.json`, and the primary per-shard receipts. These execution
measurements are separate from the independent transfer score.

## What the transfer result changes

The independent [transfer score](MZ58_DIVERSE_TRANSFER_RESULTS_20260911.md)
rejects all three predeclared no-added-FP checks. DROP gives the old union
1832 TP / 317 FP / 1032 FN; adding LOCAL, GLOBAL or SUPPRESSED produces
1836/319/1028, 1835/319/1029 and 1834/318/1030. The old union itself adds
122 TP and 51 FP over MZ37, so its legacy operating point also needs scrutiny
on this source. No threshold was changed from these results.

All additions from the three scale unions occur on retained rods. Across the
1920 frames of the three new families, they add zero TP and zero FP. GLOBAL's
three added true events are HEAD_FAR, with only one native-positive winner;
the other two winners are locally known but not native-positive. There are no
new native-positive outside-sensor winners. Its two additional false events
are HEAD_NEAR on BODY_ONLY rods. Active GLOBAL versus the same weights with
its vector suppressed adds one TP and one FP under DROP.

The remaining problem is more specific than a small aggregate improvement:
for shallow awnings the old union detects only 7 of 160 BODY_NEAR positives
under DROP. The fixed learned full-image paths do not repair that gap. This
points to a concrete shape/observation failure to investigate; it does not by
itself distinguish insufficient training coverage from representation or
metric-observability limits. The new data have not yet been used for fitting.

For the old union, 338 of 1280 pairs change some decisions despite identical
target-event arrays. RGB background and observable ToF returns can both change
between contexts, so this is context sensitivity, not proof of an RGB-only
shortcut. UNKNOWN local regions remain explicit. Merely preserving an overall
query label does not establish localization or measured clearance.

Retain MZ58 as a negative control for fixed no-cost transfer to these shapes.
Preserve the earlier MZ57 consumed-source challenger and its original evidence;
the new negative result does not erase that scoped benefit. Any later learning
comparison must distinguish added training coverage from a changed mechanism,
retain the original comparator, and declare its evaluation use. Neither source
expansion nor this check demonstrates calibrated VL53L8CX hardware behavior.
