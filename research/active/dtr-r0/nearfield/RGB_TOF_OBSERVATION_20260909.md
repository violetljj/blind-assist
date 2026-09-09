# RGB plus a constrained VL53L1X-like observation

2026-09-09 input-route decision following the user's linked task
`01a086bf-8805-71c0-a95d-be15bc9eb28a` (评估当前推进思路).
Existing JOINT and LOCAL transfer diagnostics motivate checking observation quality
before another decoder fit. No ToF simulation, fusion measurement or hardware
validation is claimed by this document.

## Authoritative capability and existing interface

ST's [VL53L1X datasheet](https://www.st.com/resource/en/datasheet/vl53l1x.pdf)
describes programmable ROI, range in mm, range status, return signal rate and
ambient rate. The receiving ROI is4x4 to16x16 SPADs; this does NOT provide a dense
16x16 depth image. Nominal FoV is DIAGONAL:27degrees at16x16,20at8x8 and15at4x4
for centered ROI in the documented conditions. Do not treat27degrees as both
horizontal and vertical angular width. A host can move the ROI sequentially;
sequential measurements have different timestamps, not simultaneous multizone data.

The [ST driver manual](https://www.st.com/resource/en/user_manual/um2510-a-guide-to-using-the-vl53l1x-ultra-lite-driver-stmicroelectronics.pdf)
documents distance/status functions and signal/ambient outputs. Simulation from
ordinary UE RGB or geometry cannot provide measured infrared reflectance, signal
rate, ambient rate, cover-window crosstalk or real mixed-target response.

Current Android `AtomS3rMjpegFrameSource` reads frame-bound ToF timestamp, validity,
range mm and age-at-JPEG-ready. `RangingSample` preserves the device clock domain.
The inspected transport does not carry ROI or camera/ToF extrinsics. Those require
a separate verified configuration/calibration record before geometric fusion;
timestamp presence alone does not prove cross-device synchronization.

## Model-visible packet

Keep RGB plus one distance value (nullable), validity, measurement timestamp,
frame association/age, ROI configuration and fixed sensor-to-camera calibration.
Attach provenance identifying REAL_MEASUREMENT versus IDEALIZED_SIMULATION.
ROI is configuration, not target identity. Missing/invalid distance cannot clear
a visual HEAD warning. No UE world pose, target ID, full depth raster, BODY/HEAD
truth or oracle statement that the range belongs to the bar enters the model.

## Next bounded experiment

First measure native surface coverage within a FIXED, declared sensor footprint
on existing captured RGB/native pairs. Explicitly assume a coincident, aligned
virtual sensor until actual hardware calibration is available; this is an ideal
observation study, not a calibrated VL53L1X simulator. Preserve mixed surfaces,
missing native support and target-outside-ROI as separate evaluator observations.
Do not substitute center-pixel depth, nearest surface or arbitrary mean for a real
sensor return. Do not select ROI using native obstacle identity.

Only after that coverage check, define an ideal homogeneous-surface distance
packet and compare RGB-only, ToF-only and RGB+ToF on identical RGB and common
denominators. Keep mixed-surface response unmodeled until a real sensor fixture
measurement constrains it; do not inject oracle target association as fusion.
Record error recovery AND wrong associations/invalid returns. A useful large-wall
result cannot stand in for thin-HEAD-bar performance. Start with deterministic
geometry/association rules before considering a fusion network.
