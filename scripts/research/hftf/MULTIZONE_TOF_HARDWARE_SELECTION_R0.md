# Multi-zone ToF hardware selection R0

Date: 2026-08-03

Terminal:

`VL53L8CX_DEFAULT_CANARY_VL53L5CX_AVAILABILITY_FALLBACK`

## Decision

Use the **ST VL53L8CX** as the default sparse metric-anchor sensor.

- First bench configuration: `X-NUCLEO-53L8A1 + STM32 Nucleo USB bridge`.
- Compact A568 prototype: `SATEL-VL53L8 + direct host I2C/SPI`, if the board
  exposes a verified bus and voltage-compatible carrier.
- Compact phone/USB prototype: `SATEL-VL53L8 + small MCU USB bridge`.
- Availability fallback only: `VL53L5CX-SATEL`; keep the same JSONL and spatial
  registration contract.

No hardware was purchased by this work. Exact regional stock, total delivered
price, carrier voltage levels, and connector availability must be checked at
procurement time.

## Why VL53L8CX is the balance choice

The official product information gives the properties needed by this branch:

| Property | VL53L8CX | VL53L5CX | Relevance |
|---|---|---|---|
| native zones | 4x4 or 8x8 | 4x4 or 8x8 | enough spatial support for left/center/right scale coverage |
| maximum advertised range | 4 m | 4 m | matches the current clearance field's 0.2-4.0 m working range |
| maximum frame rate | 60 Hz | 60 Hz | well above a sparse-anchor refresh requirement |
| diagonal FoV | 65 degrees | 65 degrees | materially better than a single-point range source |
| host interface | I2C up to 1 MHz or SPI up to 3 MHz | I2C up to 1 MHz | L8 offers a lower bus-risk deployment option |
| continuous-mode example | about 215 mW | 216-313 mW depending supplies | sensor cost is bounded but still requires full-rig measurement |
| generation emphasis | enhanced ambient performance and reduced power | earlier wide-FoV generation | L8 is the safer outdoor/bright-scene canary |

The VL53L8CX does not replace RGB geometry. Its 64 measurements are too coarse
to define a complete body clearance field, but they can supply absolute scale,
per-zone sigma/status, multi-target information, and freshness to the faster RGB
observer. This is exactly the missing variable demonstrated by the consumed
replay.

## Why the development kit and final carrier differ

`X-NUCLEO-53L8A1` is the default first purchase because ST describes it as a
complete evaluation kit compatible with STM32 Nucleo Arduino R3 boards and
supplies examples and a GUI. It reduces initial electrical and firmware risk,
but it is too large for the final wearable form.

`SATEL-VL53L8` contains compact breakout boards and is the correct second-stage
mechanical candidate. It still needs a verified power/interface carrier. On
A568, direct host I2C is simplest for an initial low-rate canary; SPI is
preferred when supported and measured because the sensor supports 3 MHz SPI.
For a phone, an MCU bridge must translate the sensor bus to USB and timestamp
frames into the same host monotonic clock used by RGB.

The official `STSW-IMG040` VL53L8CX ULD is C source with an isolated platform
layer for low-level bus access. It is the preferred driver basis; the repository
adapter remains independent of the vendor API by consuming normalized JSONL.

## Initial canary configuration

Start with:

```text
8x8 zones
15 Hz continuous ranging
all reported targets retained by the raw capture
per-zone range + sigma/status
host-monotonic capture timestamp
rigid shared mount with the exact external RGB camera
```

`15 Hz` is an engineering starting point, not a selected optimum or deployment
policy. The bench must retain raw frames so 5/10/15 Hz scheduling can later be
evaluated without changing the spatial registration. Anchor expiry, sigma,
scale-MAD, and skew gates remain prospectively chosen hardware parameters.

## Minimal acquisition list

1. One `X-NUCLEO-53L8A1` and one compatible STM32 Nucleo board for USB bench
   capture, or one `SATEL-VL53L8` if a known-good host carrier already exists.
2. The final candidate RGB camera and a rigid two-sensor mount.
3. A flat high-contrast correspondence target usable from approximately
   0.5-4 m and across left/center/right overlap.
4. A stable power source and, for deployment measurement, an external power
   meter.

Do not buy a single-zone ToF or ultrasonic module for this branch: neither can
establish multi-band correspondence in a multi-obstacle scene. Do not treat the
phone proximity sensor as a substitute; the bounded device probe exposed no
public multi-zone range stream.

## Admission sequence after hardware arrives

1. Export the exact sensor's zone rays and raw quality fields.
2. Calibrate the final RGB camera.
3. Collect multi-distance, multi-zone RGB/ToF correspondences.
4. Run `calibrate_multizone_tof_rgb.py`; reject a non-admitted registration.
5. Capture both streams in one host clock and run the existing sidecar.
6. Report anchor availability/skew/staleness, clearance gates, end-to-end P50/
   P95, memory, sensor-plus-bridge power, and sustained thermals.

The route remains a candidate until these physical measurements exist.

## Primary sources

- [ST VL53L8CX product page](https://www.st.com/en/imaging-and-photonics-solutions/vl53l8cx.html)
- [ST VL53L8CX datasheet](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf)
- [ST VL53L5CX product page](https://www.st.com/en/imaging-and-photonics-solutions/vl53l5cx.html)
- [ST VL53L5CX datasheet](https://www.st.com/resource/en/datasheet/vl53l5cx.pdf)
- [ST X-NUCLEO-53L8A1 evaluation board](https://www.st.com/en/evaluation-tools/x-nucleo-53l8a1.html)
- [ST SATEL-VL53L8 breakout](https://www.st.com/en/evaluation-tools/satel-vl53l8.html)
- [STSW-IMG040 VL53L8CX ULD](https://www.st.com/en/embedded-software/stsw-img040.html)
