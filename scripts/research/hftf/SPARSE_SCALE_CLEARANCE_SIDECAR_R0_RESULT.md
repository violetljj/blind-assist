# Sparse-scale clearance sidecar R0

Date: 2026-08-03

Terminals:

- `SPARSE_SCALE_CLEARANCE_SIDECAR_REPLAY_SUPPORTED_CONSUMED_PROXY`
- `CLOCK_DOMAIN_BINDING_REQUIRED_AND_VERIFIED`
- `REAL_TOF_REGISTRATION_NOT_EVALUATED`

## Decision

Retain the following conditional candidate for hardware integration:

```text
calibrated external RGB
  -> DA V2 Small Metric Hypersim 392x518
  -> left / center / right raw clearance
  + timestamped sparse metric-scale anchors
  -> missing, future, stale, or empty input => UNKNOWN
  -> scaled three-band clearance sidecar (no alert)
```

This is the quality/cost candidate, not a claim that the smaller observer is
more accurate than Metric3D. Metric3D remains the stronger standalone observer
and current teacher/reference. The retained hypothesis is that a faster,
smaller observer plus a cheap recurring metric anchor can satisfy the frozen
task gates at lower deployment cost.

## End-to-end consumed replay

The new class-free sidecar recomputed all 120 consumed TUM RGB frames through
the `392x518` PyTorch observer and clearance geometry. It did not reuse stored
candidate clearance fields. One fixed anchor was materialized at frame 9 of
each 30-frame sequence from the already-consumed registered sensor-depth proxy;
frames 10-29 were evaluated.

| Measure | Result |
|---|---:|
| eligible / paired-valid frames | 80 / 78 (97.5%) |
| clearance MAE | 0.098145 m |
| collision agreement | 93.7729% |
| false-clear rate | 4.9451% |
| temporal clearance-delta MAE | 0.085803 m |
| frozen task gates | 5 / 5 |
| host depth median | 33.0685 ms |
| host geometry + scale median | 11.3147 ms |

The host timings are PyTorch replay diagnostics and cannot be added to or
substituted for the separately measured HTP model timing. They do not establish
camera-to-clearance latency, energy, thermals, or sustained frame rate.

## Clock and failure behavior

The first local integration attempt bound anchor timestamps to the absolute
image clock while the manifest used sequence-relative timestamps. It failed
closed: 119 frames were `UNKNOWN_NO_METRIC_SCALE_ANCHOR` and one was
`UNKNOWN_RAW_CLEARANCE`; no incorrectly scaled output was emitted.

The materializer now binds anchor time to the manifest authority. The corrected
replay produced 36 pre-anchor `UNKNOWN_NO_METRIC_SCALE_ANCHOR`, 83 `VALID`, and
one `UNKNOWN_RAW_CLEARANCE`, with no `VALID` row before the anchor frame.
Monotonic ordering is enforced, and unit fault injection verifies that an
expired anchor becomes `UNKNOWN_STALE_METRIC_SCALE_ANCHOR`.

The replay used an explicit `5000 ms` maximum anchor age only to keep the fixed
single-prefix proxy available over each short sequence. That value was not
selected as a deployment TTL and must not be inherited by real hardware.

## Claim ceiling and next step

The anchor is a registered sensor-derived proxy, not real multi-zone ToF. The
observer ran on the host, not through an end-to-end HTP camera pipeline. This
result therefore supports interface causality and consumed task quality only;
it does not validate final-camera optics, RGB-ToF spatial registration, anchor
availability, synchronization jitter, device cost, alerts, safety, production,
research-mainline promotion, or the default App.

The next valid experiment is a real multi-zone ToF/RGB registration adapter and
device replay with a prospectively fixed expiry policy. Another depth-model
search is not authorized by this result.

Machine-readable result: `SPARSE_SCALE_CLEARANCE_SIDECAR_R0_RESULT.json`.
