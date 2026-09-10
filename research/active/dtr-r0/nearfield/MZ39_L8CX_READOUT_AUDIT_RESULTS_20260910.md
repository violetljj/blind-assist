# MZ39: ideal dual returns frequently violate the documented separation limit

2026-09-10 EXPLORE. On380 saved MZ36 frames,4234of4893dual-return zones
(86.53%) have a stored separation below600mm. These occur in346frames;
every frame with a dual-return zone contains at least one such pair. Retain
this as a diagnostic of exercised ideal-readout mismatch, not model accuracy
or a measured physical-sensor detection rate.

[Protocol](MZ39_L8CX_READOUT_AUDIT_PROTOCOL_20260910.md),
[runner](mz39_l8cx_readout_audit.py),
[successor observation contract](VL53L8CX_OBSERVATION_CONTRACT_20260910.md).

| Saved packet property | Count |
|---|---:|
| Frames / zones per frame |380 /64|
| All zones |24320|
| Zero valid returns |16592|
| One valid return |2835|
| Two valid returns |4893|
| Dual-return frames |346|
| Dual pairs below600mm |4234|
| Frames containing a sub600mm pair |346|
| Dual pairs below100mm |3225|
| Pairs exactly600mm in stored float values |0|

Dual-pair separation min/25th/median/75th/max is
0.002680/0.029362/0.051243/0.218004/2.231162m. These are separation statistics
of the synthetic outputs, not a measured sensor resolution distribution.

The original generator partitions true radial depths into10cm bins and emits
first/last supported bins, each needing only3pixels. Neighboring bins can yield
two close means; a continuous slanted surface may therefore be represented as
multiple returns. The audit identifies close pairs, without reconstructing or
asserting their actual surface identity.

[ST UM3109 Rev12 section4.10](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)
states a600mm minimum for detecting two targets. The same document describes
default single-target output and strongest ordering; an explicitly configured
two-target mode does not guarantee valid detections of arbitrary geometric
surfaces. The present ideal generator must not be reported as equivalent
VL53L8CX emulation. Surviving659pairs are not certified hardware-valid either.

Independent standard-library loops recount all24320zones, validity bins,
strict threshold predicates and exact violating identities against vectorized
NumPy. Input receipt/packet/source hashes remain unchanged; audit and execution
PASS. No labels, model inference, training, new frames or observation alteration.
CPU saved-array work is TASK_NOT_GPU_SUITABLE. Compact result, violations,
audit and receipt are in artifacts.local/work/mz39-l8cx-readout-audit-20260910/run-v1/.

Retain as diagnostic COMPONENT. Next implement a separately defined constrained
readout and compare unchanged-method sensitivity before matched retraining or
new-object capability claims. Preserve the old ideal comparator, MZ37's measured
17TP/2FP tradeoff, and UNKNOWN. This result does not establish that the existing
algorithm fails on real hardware or quantify the eventual effect of correction.
The broad improvement goal remains active; no physical or safety promotion.
