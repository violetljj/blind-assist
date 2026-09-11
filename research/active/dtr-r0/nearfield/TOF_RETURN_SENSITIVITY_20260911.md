# Preserve one reported return: paired sensitivity inputs

The user corrected the emphasis on deliberately all-invalid ToF. Ordinary and
partially usable measurements are the main research target; all-invalid remains
a separate failure test. None of the existing profile frequencies estimates a
real VL53L8CX failure probability.

## Existing ordinary and partial-input results

A fresh extraction of the already sealed MZ70 score confirms that ordinary and
partial profiles were measured. The table reports the existing final OR decision
on consumed Development HELD geometry, not new predictions or real-device tests.
Each source has 1,024 frames and 1,024 positive query events; the four queries
together have 3,072 known negative events and zero frame-query UNKNOWN entries.
Event counts are not counts of uniquely detected obstacles.

| Source / profile | CONTROL TP / FP | DIVERSE TP / FP | DIVERSE recall |
| --- | ---: | ---: | ---: |
| New MZ67 / IDEAL | 769 / 48 | 861 / 46 | 84.1% |
| New MZ67 / MERGE_CLOSE | 804 / 57 | 911 / 57 | 89.0% |
| New MZ67 / DROP_CLOSE | 664 / 71 | 753 / 71 | 73.5% |
| Old MZ61 / IDEAL | 978 / 86 | 971 / 78 | 94.8% |
| Old MZ61 / MERGE_CLOSE | 987 / 33 | 987 / 31 | 96.4% |
| Old MZ61 / DROP_CLOSE | 858 / 38 | 832 / 38 | 81.3% |

New-source IDEAL improvements concentrate in HEAD_FAR (+76 TP net) and BODY_NEAR
(+21), while HEAD_NEAR loses 5 TP and 2 FP. Old-source DROP loses 26 TP net at
unchanged FP. More diverse training therefore helps the new shapes, but the
benefit is uneven and retains an old-source cost. The source allocation halved
old-source exposure at fixed updates, so that cost does not isolate forgetting.
The MERGE gain is not evidence that realistic sensor degradation improves
performance: its synthetic midpoint changes the geometry supplied to the model.

The [descriptive extraction](../../../../artifacts.local/work/vl53l8cx-packet-contract-20260911/ordinary-partial-audit-v1/report.md)
retains exact source hashes, all four queries, candidate versus final OR,
known-negative denominators and UNKNOWN for the other source groups. No model,
cutoff or current challenger disposition changed.

## Implemented input scenarios

The existing [MZ40 proxy](mz40_packets.py) replaces unresolved close dual returns
with their midpoint in MERGE_CLOSE, or deletes the whole zone in DROP_CLOSE.
Neither is a measured device distribution. Keeping only the near return would
also assume that a small near obstacle survives, which is not established.

[tof_return_sensitivity.py](tof_return_sensitivity.py) therefore implements two
explicit, paired input scenarios:

| Scenario | In a valid dual-return zone with separation below 0.600 m |
| --- | --- |
| CLOSEST_REPORTED_PROXY | Preserve the original near slot; mask the far slot |
| FARTHEST_REPORTED_PROXY | Preserve the original far slot; mask the near slot |

All other bytes remain unchanged. The retained distance stays in its original
slot; no midpoint, new distance, valid zone or quality field is created. A
previously missing zone remains missing. The threshold uses the actual stored
float values and a strict comparison; no epsilon is silently added. These are
two sensitivity scenarios, not guaranteed upper/lower bounds on sensor accuracy,
not strongest-return selection, and not a model of weak-target detection, signal,
ambient light, noise or multipath. Both could differ from the real device.

The function accepts only the existing float32/bool [batch,64,2] geometric
first/last arrays. It rejects misordered usable pairs and does not truncate or
reorder a real four-target device packet. `as_measurement_packet` wraps one case
as SIMULATION_PROXY with status, target count, quality and timestamps unavailable.
The mode, changed-zone mask and provenance are audit metadata, not model inputs.
No evaluator labels, native depth or intended HEAD/BODY class are read.

## Engineering verification

Six new tests and the eight packet-contract tests pass. A replay of all 8,192
existing MZ61/MZ67 observable packets verified both scenarios without RGB,
native-depth or evaluator-label reads. This is input transformation verification,
with zero model inference, zero training and no new operating threshold.

| Source | Frames | Changed zones per scenario | Valid slots before / after | Missing frames before / after |
| --- | ---: | ---: | ---: | ---: |
| MZ61 | 4,096 | 46,869 | 134,277 / 87,408 | 15 / 15 |
| MZ67 | 4,096 | 36,850 | 102,798 / 65,948 | 53 / 53 |

Each 4,096-frame transformation took 6.1–7.0 ms on CPU in this run, excluding
file reads and checks. Unaffected bytes and surviving slot values were exact;
source hashes were unchanged. These missing-frame counts describe the saved
geometric proxies, not actual sensor reliability. [Replay receipt](../../../../artifacts.local/work/vl53l8cx-packet-contract-20260911/return-sensitivity-v1/receipt.json).

The next model comparison should replay both scenarios with fixed checkpoints
and original cutoffs, separately reporting near/far body/head TP, FP and UNKNOWN
on old and new sources. Do not choose the surviving endpoint using labels or
select only the better endpoint after scoring. No model benefit from these new
inputs has yet been measured. [Quality and timing contract](VL53L8CX_MEASUREMENT_PACKET_20260911.md).
