# Multi-zone ToF to RGB adapter R0

Date: 2026-08-03

This is the hardware boundary for the retained quality/cost candidate. It does
not assume that a phone camera exposes ARCore depth and does not require the ToF
sensor to share the RGB optical center.

## Geometry

For each ToF zone, the sensor geometry supplies a unit ray `r_tof`. A valid
range `d` becomes a point in the ToF frame:

```text
p_tof = d * r_tof
p_rgb = T_rgb_from_tof * [p_tof, 1]
u = fx * p_rgb.x / p_rgb.z + cx
v = fy * p_rgb.y / p_rgb.z + cy
```

The adapter samples the candidate RGB depth near `(u, v)` and estimates one
robust global scale from `p_rgb.z / candidate_z`. It does not divide raw radial
ToF range by a forward-clearance value. This distinction is required whenever
the sensors have different optical centers or nonzero viewing angles.

## Required inputs

ToF geometry uses schema `hftf_multizone_tof_geometry_r0`:

```json
{
  "schema": "hftf_multizone_tof_geometry_r0",
  "tof_sensor_id": "exact-device-id",
  "zones": [
    {"zone_id": "0", "ray_tof_unit": [-0.1, -0.1, 0.989949]}
  ]
}
```

Calibration correspondences use JSONL schema
`hftf_tof_rgb_correspondence_r0`. Each row binds one physical target observed
by one ToF zone to its point in the already-rectified RGB frame:

```json
{
  "schema": "hftf_tof_rgb_correspondence_r0",
  "tof_sensor_id": "exact-device-id",
  "rgb_calibration_id": "json:SHA256",
  "zone_id": "0",
  "range_m": 1.5,
  "rgb_uv_px": [320.0, 240.0]
}
```

`calibrate_multizone_tof_rgb.py` solves a rigid PnP transform and emits schema
`hftf_tof_rgb_registration_r0`. Observation count, zone coverage, range span,
inlier fraction, reprojection RMSE, and the caller-selected gates are retained
in the output. A failed result is written with `admitted=false`; the runtime
loader rejects it.

Runtime ToF frames use JSONL schema `hftf_multizone_tof_frame_r0`:

```json
{
  "schema": "hftf_multizone_tof_frame_r0",
  "sequence_id": "capture-001",
  "timestamp_ns": 123456789,
  "clock_domain": "host_monotonic",
  "tof_sensor_id": "exact-device-id",
  "registration_id": "rig-r0",
  "zones": [
    {"zone_id": "0", "range_m": 1.48, "sigma_m": 0.012, "status": "VALID"}
  ]
}
```

The driver must convert device time into the declared RGB clock domain before
writing the row. A clock-domain mismatch is not repaired by arrival order.

## Runtime gates

`run_external_rgb_clearance_sidecar.py --tof-jsonl` requires explicit values
for maximum RGB/ToF skew, maximum ToF sigma, minimum valid zones, minimum
covered RGB bands, maximum scale MAD, and anchor expiry. These values belong to
the hardware protocol and must be selected from sensor specifications or an
independent bench fixture before task results are inspected.

The adapter fails closed for:

- mismatched sensor, registration, or RGB-calibration identity;
- different RGB and ToF clock domains;
- future or over-skew ToF frames;
- invalid range, sigma, zone, projection, or RGB depth;
- insufficient zone or band coverage;
- inconsistent per-zone scales;
- absent or expired last valid scale anchor.

No alert is emitted by this sidecar.

## Acquisition sequence

1. Calibrate the exact external RGB camera, focus, resolution, and distortion.
2. Export exact ToF zone rays from the selected device/driver.
3. Collect correspondences across the field of view and at several distances.
4. Run the rigid registration and retain only an admitted output.
5. Emit both streams in one host monotonic clock domain.
6. Run the sidecar and measure anchor availability, skew, expiry, task quality,
   camera-to-clearance latency, memory, power, and thermals on the target device.

Synthetic geometry tests prove the transform and failure mechanics only. They
do not establish the accuracy of a physical ToF module or mounting rig.
