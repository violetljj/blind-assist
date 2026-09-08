# ASE BODY/HEAD coverage: fixed subset lacks HEAD-only cases

Status: `ASE_BODY_QUERY_PILOT_COVERAGE_INSUFFICIENT_FOR_HEAD_ONLY_TEST`.
Completed 2026-09-09. EXPLORE synthetic visible-surface geometry only.

The [predeclared pilot](ASE_BODY_QUERY_PILOT_20260909.md) completed on the official
ASE source. Geometry decoding and camera-anchored query labeling worked, but the
fixed subset supplied **zero HEAD-only frames**, so the model-test admission
criterion was not met. No model inference, fitting or automatic data expansion
followed. This is a source-coverage result, not a negative model result.

## Observed coverage

| Scene | Sampled / source frames | BODY-only | HEAD-only | Both | Neither detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 24 / 350 | 7 | 0 | 15 | 2 |
| 4 | 24 / 538 | 7 | 0 | 12 | 5 |
| 8 | 24 / 93 | 1 | 0 | 20 | 3 |
| Total | 72 / 981 | 15 | 0 | 47 | 10 |

Each scene used 24 uniformly spaced source-frame indices with no outcome-based
selection. Independent layouts are not verified asset-disjoint scenes. Scene 8
has a shorter sequence; temporal samples within a scene are correlated.

At least 8 HEAD-only frames spanning 2 scenes and 8 BODY-only frames were needed
to admit a follow-up model/calibration check. BODY-only coverage met that count;
HEAD-only did not. In all 47 observed HEAD-positive frames, BODY was also positive.
This subset therefore cannot isolate head-height hazards from whole-height
barriers or supply the intended HEAD-only recall denominator.

All frames are native 704x704 RGB and 16-bit ray-distance depth. The conservative
calibration validity mask excludes 22.7271% of pixels in every sampled frame;
there are no fully invalid frames. Excluded pixels remain UNKNOWN, and the ten
neither-detected frames are not certified free space. BODY visible counts range
0–61,599 and HEAD counts 0–127,708 per frame; these are source-resolution counts,
not the frozen V1 pinhole count contract.

The nine predetermined first/middle/last overlays were visually inspected.
The BODY mask occupies lower wall/furniture surfaces, with HEAD above it on
walls, doors and upper visible structure. No obvious coordinate-axis inversion
was seen. This is a visual plausibility check, not SDK parity or validation of
the actual camera-to-body/floor offset. No new physical floor estimate was made.

## Identity, execution and evidence

- Source: official TRAIN `train_chunk_0000000.zip`, scenes 0, 4, 8 only extracted.
- Official SHA1: `1bbfa8b3261d99a67e076a5797c2b583fb3c9496` (verified).
- Archive SHA256: `3f3250dddc2ffff77f4fa1634938720fa34076328e9ba071a7d243a9e24bdc46`.
- Archive bytes: 2,318,138,707; selected extracted bytes: 243,426,919 in 1,968 files.
- Download, hash and extraction: 103.031 s; 72-frame decode/audit: 26.735 s.
- Execution: existing Python 3.11 research runtime, NumPy CPU geometry/image ETL.
  No model, CUDA kernel, UE process or worker allocation was started.
- Registration: `ase-body-query-pilot-20260909`, protocol `ase-body-query-pilot-v1`,
  registered before source image decoding against implementation `e4e1c81b`.

Logical source root:
`artifacts.local/downloads/ase-body-query-pilot-20260909/`.
Logical evidence root:
`artifacts.local/evidence/ase-body-query-pilot-20260909/run-01/`.
Both resolve through the canonical junction onto F:, not an E: data copy.

`download-receipt.json` records the official archive check and source extraction.
`run-01/inputs.json` records SHA256 for all 72 RGB/depth pairs and 3 trajectories.
`result.json` retains all per-frame raw counts/labels and the declared assumptions.
`completion.json` adds the nine-image review and process completion receipt.
`viewer.html` displays all nine prescribed previews; it is local-only.
Retain the verified source and evidence for disclosed reuse; no process remains
allocated for this experiment. User-specific signed download URLs stay out of Git.

Reproduction is the download/audit command in the pilot brief. Do not rerun a
completed output directory or call the consumed cohort fresh confirmation.

## Disposition

Retain the verified ASE source adapter and count/preview tooling as
`COMPONENT_OR_CHALLENGER / COMPONENT` for camera-anchored external geometry
diagnostics. The native occupancy task is feasible. This particular subset is
not suitable for the intended HEAD-only comparison; it gives no evidence that
ASE fixes the current cross-region representation failure.

Keep frozen BODY-QUERY B and the R1 negative-control status unchanged. Preserve
UE controlled HEAD-only/BODY-only/miss pairs for height-selective evidence.
A future source-selection question would need demonstrable HEAD-only geometry
and reviewed model-compatible projection; it is not opened automatically by
this result. No claim about the whole 100K-scene ASE corpus follows from 3 scenes.

## Delivery validation

An isolated snapshot excludes concurrent registry/tooling WIP. The ten knowledge
unit tests and library validation passed; invalid experiment count is zero.
The global decision-engine history-recall check remains 0.70 versus its 0.80
gate, with the same three failing cases on the unchanged parent and this change.
This pre-existing retrieval-test failure is recorded rather than repaired as
part of the ASE data pilot. Both outputs are retained in the delivery logs.
