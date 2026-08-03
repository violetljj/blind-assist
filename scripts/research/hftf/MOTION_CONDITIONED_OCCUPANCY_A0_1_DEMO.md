# Motion-conditioned occupancy A0.1 public-data demo

Date: 2026-08-03

This renderer turns the ignored A1 machine report into a deterministic contact
sheet. It selects the middle complete frame from each of the seven frozen
`walking_halfsphere` windows and displays the frozen A0.1 probability plus
sensor-depth truth for left, centre, and right metric bands at 1.5 m.

The colored strip is a schematic summary, not an image-space segmentation.
Green means probability below 0.20, amber 0.20 through 0.49, and red at least
0.50. `T=1` means the registered sensor-depth reference says that metric band
is occupied within 1.5 m.

The output is current-frame collision occupancy only. It must not be presented
as the unsupported A2 0.5-second future field, an alert recommendation, or
external-camera evidence.

The verified seven-window output is ignored at
`artifacts.local/evidence/hftf/metric3d-clearance-field-a0/motion-occupancy-a0-1-demo.png`.
Its SHA-256 is
`7936654A5ABA9934A037D615870E0CB3B36113F9126C503CCD89C930E89D0BFF`.

Run:

```powershell
python scripts/research/hftf/render_motion_occupancy_a0_demo.py `
  --report artifacts.local/evidence/hftf/metric3d-clearance-field-a0/collision-risk-field-a1-walking-halfsphere-report.json `
  --output artifacts.local/evidence/hftf/metric3d-clearance-field-a0/motion-occupancy-a0-1-demo.png
```
