# VL53L8CX-oriented simulation: constrained observations before task claims

2026-09-10. The user explicitly sets VL53L8CX as the intended ToF capability
target and asks not to assume strong sensing. The existing physical single-zone
adapter does not override this research target. Diverse replacement objects,
including nonvegetation shapes, remain the data direction; sensor assumptions
must be corrected before describing their results as VL53L8CX performance.

## What is known and what is assumed

| Surface | Inspected source or current fact | Required treatment |
|---|---|---|
| Spatial resolution | ST supports4x4/8x8 zones | Explicit mode and angular calibration; a zone is not an RGB pixel |
| Target count | UM3109 describes1..4configurable outputs, default1 | A second slot requires declared configuration; never guarantee two valid detections |
| Target separation | UM3109 section4.10 states600mm minimum | Do not emit sub600mm geometric peaks as independently resolved device targets |
| Target order | Closest/strongest; default strongest | Nearest/farthest surface extraction is not a strongest-target signal model |
| Validity | Target count/status, signal and range sigma exist in the documented interface | Invalid, mixed and unavailable measurements remain distinguishable from absence of an obstacle |
| Range/detection | Datasheet specifies measurement conditions and environmental dependence | Up-to4m is not guaranteed detection of every small/dark target at4m |
| Sensor physics | Current generator has depth but no calibrated photons, NIR reflectance or ambient response | Do not fabricate measured signal/status or infer NIR reflectance from RGB color |
| Timing | Current evidence is static frame capture | Frame rate, integration time, sensor/camera offset and motion require explicit configuration/evidence |

Sources inspected: [DS14161 Rev12](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf),
[UM3109 Rev12 sections4.9/4.10/5.1/5.5](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf).
These are documented constraints, not a calibrated physical simulator or
statistical guarantee for branches, foliage, wire, cloth or mesh.

## Existing ideal comparator

`multizone64_observation.observe(readout='multi_surface')` bins native radial
depth into40 ten-centimeter bins and returns the first/last bin with at least
three pixels. It has no target-separation limit or photon/quality model. Native
depth and3-pixel support remain valid definitions for its disclosed geometric
comparison, but not a sensor detection law. Preserve its frozen artifacts and
comparators; do not silently change old generated inputs.

[MZ39](MZ39_L8CX_READOUT_AUDIT_RESULTS_20260910.md) audits one concrete exercised
mismatch. Even pairs surviving this single constraint are not certified real
returns. The audit does not measure a corrected model's task accuracy.

## Successor observation and evaluation rules

1. Keep native surface/event truth evaluator-only; the model receives only the
   declared constrained observation. No obstacle identity, contributor label,
   native depth map or desired HEAD/BODY relation enters the predictor.
2. Declare1-target and explicitly configured2-target assumptions separately.
   Missing second returns are masked; mixed/unresolved pairs are flagged rather
   than represented as two precise observations. Select/merge behavior without
   measured signal evidence must be named a simulation proxy or sensitivity arm.
3. Treat weak-target loss, measurement noise, invalid zones and temporal mismatch
   as named uncertainty dimensions. Choose sensitivity levels before task outcomes;
   tag unsupported numerical levels as assumptions. Random noise alone does not
   establish hardware fidelity, and deliberate excessive degradation is not a
   substitute for a credible restricted model.
4. Report correct/missed/false events and UNKNOWN/coverage under the restricted
   readout, alongside RGB-only and the frozen ideal comparator. Separate the
   immediate mismatch of a model trained on ideal packets from an eventual
   matched-training comparison under the new observation contract.
5. Replacement assets include pipes, open frames, mesh, static cloth-like forms,
   bags, bulky irregular objects and vegetation where available. Preserve paired
   old fixtures and visible nonintruding controls. Re-render RGB/native depth,
   verify material holes and label actual surfaces; box intent is not mesh truth.

## Storage for future acquisition

Preserve a canonical recoverable RGB/depth source and compact observations,
labels, calibration, manifests and frozen model references. Derived dense
features are explicitly regenerable caches, not a mandatory permanent copy at
every experiment stage. Avoid new raw-depth copies and retained transfer ZIPs
after a separately authorized, verified lifecycle step; existing evidence is
not deleted by this contract. Lossless compression must roundtrip exactly;
float16/quantization is a separate numerical change, not transparent compression.

No new capture, fitting, device interaction or baseline promotion is performed
by this contract. MZ38 asset scout establishes file availability only. The next
executable sensor-limited comparator needs its own defined assumptions and scope.

