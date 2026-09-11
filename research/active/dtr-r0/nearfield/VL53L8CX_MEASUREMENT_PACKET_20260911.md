# VL53L8CX measurement packets: observed quality and explicit time

The next research priority is RGB plus usable ToF under ordinary and partial
degradation. Deliberately all-invalid packets remain a secondary failure test,
not a measured device operating distribution or the primary optimization target.
The current MZ70 predictor receives RGB plus range_m/valid[64,2]; it does not
yet consume signal, ambient, status, sigma or timestamps. Its geometric first/last
slots are not equivalent to the device's default strongest-target ordering.

[vl53l8cx_measurement_packet.py](vl53l8cx_measurement_packet.py) adds an opt-in
input contract and causal alignment helper. Frozen collectors, archives,
checkpoints, calibration and model feature dimensions remain unchanged. This is
an implemented interface with focused integrity tests, not a new sensor simulator,
device integration or demonstrated algorithm improvement.

## Verified device interface

UM3109 Rev12 section4.4/Table2 specifies4x4 at1..60Hz and8x8 at1..15Hz.
RGB30FPS plus ToF15Hz is a reasonable declared starting configuration, not a
guarantee of synchronous alternating frames. Autonomous8x8 has four integration
periods; a single timestamp does not imply simultaneous acquisition of all zones.
Section4.10 describes1..4 configured targets per zone, default1, with strongest
ordering by default. Its600mm minimum separation does not guarantee detection
of a small rod against a wall. [Official ULD manual](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf).

| Field | Shape in an8x8 packet with T configured targets | Host unit |
| --- | --- | --- |
| distance | [64,T] | mm at ULD boundary; metre copy for existing code |
| target_status | [64,T] | original status code |
| signal_per_spad | [64,T] | kcps/SPAD |
| range_sigma_mm | [64,T] | mm |
| reflectance | [64,T] | estimated percent |
| nb_target_detected | [64] | reported count |
| ambient_per_spad | [64] | kcps/SPAD |
| nb_spads_enabled | [64] | enabled count |

The adapter takes decoded ULD host units; raw firmware fixed-point values require
the appropriate driver conversion first. Disabled outputs remain unavailable.
Reported count is distinct from the number of status-accepted targets. The
default usability policy accepts status5 only; explicitly passing(5,6,9) includes
the manual's lower-confidence first/merged returns. Raw codes remain available
either way. The manual's confidence guidance is not a calibrated probability
of scene correctness, and status9 must not be treated as a precise isolated
surface merely because a distance is present.

The pasted motion[64] simplification needs correction. In the inspected official
ST driver API2.0.0, motion_indicator is140bytes with motion[32]. Its default8x8
plugin mapping creates16 aggregate regions, each combining2x2 ranging zones.
It is not an independent64-cell velocity or flow map. This first packet version
therefore leaves motion out; a later consumer must carry the actual motion-region
mapping. [Official API header](https://github.com/STMicroelectronics/fp-ind-datalogmc/blob/74414a1edcf4701a0ed8a048c7d78fcbedb5cbd4/Drivers/BSP/Components/vl53l8cx/modules/vl53l8cx_api.h)
and [motion plugin](https://github.com/STMicroelectronics/fp-ind-datalogmc/blob/74414a1edcf4701a0ed8a048c7d78fcbedb5cbd4/Drivers/BSP/Components/vl53l8cx/modules/vl53l8cx_plugin_motion_indicator.c).

DS14161 Rev12 Table19 confirms the quoted8x8/15Hz/5klux typical ranges:
88% white inner1550mm,17% gray inner1150mm and gray corner950mm. Those measurements
use full-FoV targets and stated detection, calibration, temperature and lighting
conditions. They do not specify thin-branch range, a universal0..2m operating
guarantee or a dropout probability. [Official datasheet, sections8.2.1/8.2.3](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf).

## Behavior and boundaries

`from_legacy` copies old float range and bool validity arrays byte-for-byte,
including invalid zero/NaN representations. It does not invent status, target
counts, quality or time. `observed_field` returns values plus an availability
mask; an observed zero signal and an unavailable signal are different inputs.
Quality is not synthesized from native pixel counts, desired HEAD/BODY labels,
RGB brightness or the chosen degradation profile. Provenance and measurement
identifiers are audit metadata, not predictor features.

`from_uld` preserves slot order, target count, raw status and provided quality.
Its accepted-status policy is recorded. It neither sorts targets into the old
first/last semantics nor silently truncates4 targets to the old2-slot model.
Invalid or absent returns do not generate clearance labels. Using this new
packet with a neural model still needs an explicit consumer and comparison;
quality gains have not been measured.

`align_to_rgb` requires an explicit common clock and continuous sequence.
It chooses the latest acquisition completed no later than the RGB reference
time and received no later than the decision time. It returns age, sample reuse
and AVAILABLE/STALE/NO_ALIGNED_TOF explicitly. A reused15Hz measurement is not a
new observation for every30FPS RGB frame. Maximum age is caller-declared, not
a fitted safety threshold. Clock calibration, subframe motion correction and
per-zone acquisition timing are not implemented here. Existing independent
synthetic case IDs have no such timeline and cannot be joined across cases.

The next simulator change should declare which quality quantities are measured,
which are controlled proxies and which are unavailable. Test ordinary and partial
degradation first; leave all-invalid as a secondary diagnostic. A calibrated
signal/ambient/sigma model needs device evidence; a named sensitivity proxy can
be evaluated earlier without calling it validated VL53L8CX physics. Richer
physical obstacle collection and compact storage remain parallel workstreams.

[Official-source verification](../../../../artifacts.local/work/vl53l8cx-packet-contract-20260911/official-verification-v1/notes.md)
and [local implementation gap](../../../../artifacts.local/work/vl53l8cx-packet-contract-20260911/local-gap-v1/notes.md)
record inspected revisions, exact hashes and limits.
