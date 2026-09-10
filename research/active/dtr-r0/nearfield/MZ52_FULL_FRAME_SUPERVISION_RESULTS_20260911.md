# MZ52: full-frame supervision from existing native data in538KB

The [fixed derivation](MZ52_FULL_FRAME_SUPERVISION_20260911.md) completes all
2,560 existing MZ48 frames without recapture, inference or training. The merged
compressed labels occupy538,415 bytes. All2,432 original positive events have
full-frame spatial witnesses, including the352 BODY_NEAR events without a
45-degree witness. This closes a training-label coverage gap; model benefit
from the additional RGB region remains untested.

The output uses nonoverlapping8×8 blocks of the original640×360 frame:
`fullframe_event_counts[N,45,80,4]` and`valid_counts[N,45,80]`, both uint8
with maximum64. Event presence and cell-known masks are derived with`>0`;
they need no duplicated stored arrays. Summing counts exactly recovers the
original full-frame query counts and the≥3-pixel event rule. Original finite,
axial/radial validity and BODY/HEAD near/far geometry stay unchanged.
All7,562,077 unknown cells remain explicit out of9,216,000 cells. No data
are converted from unknown to clearance by this representation.

| Host | Existing frames | Native bytes read locally | Compressed host labels | CPU derivation |
|---|---:|---:|---:|---:|
| Primary | 936 | 862,737,408 | 211,368 bytes | 15.925s |
| Secondary | 1,624 | 1,496,886,272 | 328,055 bytes | 24.192s |

Each machine reads its original native files. Returning compressed labels
and provenance uses2,828,759 bytes of payload, with zero native-depth bytes
returned, avoiding transfer of2.36GB of full native depth. The combined
uncompressed count arrays are46.08MB; merged NPZ is538KB including IDs. This
is a compact task-specific supervision representation, not a lossless copy
of all original depth or an assertion of physical disk reclamation. Original
raw evidence and immutable MZ48 packages remain on their owning hosts.

All2,560 native SHA256 bindings match. Every one of10,240 full-event counts
and labels matches the original source. All1,280 target/context pairs have
identical event-cell counts. Valid-point counts differ in all1,280 pairs,
as expected when backgrounds/supports change; the check deliberately does
not require those background counts to match. The80 originally prescribed
native audit files are independently reaggregated using a separate block-loop
path. All1,152,000 event-cell values and288,000 valid-cell values match the
serialized arrays. Independent verification and merging took10.832 seconds
on NumPy CPU. No neural/GPU workload is required for this bookkeeping.

The source keeps512/640/640/640 positive BODY_NEAR/BODY_FAR/HEAD_NEAR/HEAD_FAR
events. There are no global query counts of1–2 pixels. Frame identities and
the MZ50/MZ51 partitions remain1280 fit,256 calibration,640 held-out-site,
384 withheld-family. These are the same consumed controlled sources.

The secondary's initial hidden launch ended when its SSH session closed:
no native output directory or data processing had begun. A synchronous
remote execution then completed the unchanged request and code. That
scheduling failure and its empty logs are preserved separately; it is not
a native-data failure or an extra dataset. Both hosts' owned processes have
exited, and final release receipts bind their observed absence.

Retain this as a COMPONENT source for a subsequent full-RGB local model.
Only original RGB and the original45-degree ToF packet/status may enter a
predictor. Full native depth and these cell labels are training/evaluator
truth; they cannot supply ToF ranges outside the sensor field. The result
does not establish RGB metric-ranging accuracy, unseen-space clearance,
physical VL53L8CX calibration or natural-scene performance.

Evidence is under`artifacts.local/work/mz52-full-frame-supervision-20260911/`:
`primary-v1/`, `worker-v1/`, `verified-v1/`, the bound manifest, source index
and final receipts. Original source hashes remain unchanged.
