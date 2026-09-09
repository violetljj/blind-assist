# VL53L8CX as the assumed simulation hardware

Subsequent user steering: prioritize generic MultiZone-ToF-64 clean information
experiments [MZ0](MZ0_RESULTS_20260910.md), then learned fusion and resolution
curves. The hardware details below are reference constraints, not immediate
high-fidelity simulation work or a prerequisite to algorithm experiments.

2026-09-10: user asks to assume VL53L8CX. Retain all prior single-zone results as
their own controls. No physical hardware availability or multi-zone experimental
gain is asserted by this specification note.

## Verified capability, not simulated performance

[ST datasheet DS14161](https://www.st.com/content/st_com/en/technical-documents/DS14161.html)
describes4x4/8x8 ranging,64zones in8x8 mode, multitarget distance per zone and a
nominal45degree horizontal/vertical detection volume (65degree diagonal). Its
maximum range claims depend on reflectance, illumination, configuration and
coverage; do not use4m as universal valid range.

[ST ULD manual UM3109 Rev12](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)
specifies8x8 at1-15Hz and4x4 at1-60Hz. Adopt8x8/15Hz as the first simulation
configuration. The60Hz headline does not describe8x8. Up to four targets per zone
can be configured; default is one, with Strongest ordering. The manual gives600mm
minimum target separation for multiple-target detection. Do not interpret every
geometric surface as a reported target or assume a thin bar is always returned.

Useful outputs include zone identity, target count, per-target distance/status,
range sigma and signal, plus ambient and motion indicator. Motion indicator is
scene-change intensity, not calibrated sensor pose/IMU. Lens orientation flips
horizontal and vertical zone mapping; derive camera/body mapping explicitly.
In autonomous8x8 mode, a frame has four integration periods. A first ideal
simultaneous-frame model must disclose that simplification; later temporal
alignment must account for exposure timing rather than blindly using one pose.

## How this changes the research question

Single-zone motion supplied angular constraints that were absent in one scalar
reading. A multi-zone sensor already supplies coarse angular support per frame.
Motion may now supply additional boundary observations between these coarse zones,
improving body-relative attribution or separating mixed returns across time.
This is a hypothesis of additional spatial information, not guaranteed depth
super-resolution. A zone is an angular area, not an exact 3D point at its center.

The next comparison should hold8x8 hardware observations constant and distinguish:
single-frame spatial inference; temporal evidence without pose alignment; and
pose-aware joint inference. Stationary repetitions remain a control for averaging
and missing-data recovery. Preserve multi-source explanations and invalid returns.
Score attribution resolution and false assertions, not just a smoother picture.

Start with known synthetic pose to test information gain, then challenge pose
error, timing and mixed-response assumptions. No immediate neural training or
hardware purchase is required. The single-zone271/300 ideal result is not an
8x8 result and must never be relabeled as one.
