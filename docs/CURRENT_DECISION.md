# Current decision: unseen-location evidence routing

Status: `MSLS_SOURCE_ADMITTED / ROUTER_DEVELOPMENT_GATE_NOT_MET / TEST_UNOPENED`

## Authorized question

Can a quality-conditioned Evidence Router improve reference-conditioned
held-out-location retrieval over the strongest static learned fusion when the
model receives actual image evidence and a coarse-GPS local candidate set?

The phase predicts only candidate identity and confidence. It does not perform
named-POI grounding, target boxes/masks, video persistence, navigation, entrance
finding, or product/safety decisions.

## Frozen surface

- Protocol and code: `research/active/unseen-location-router/`
- Successor source: Mapillary Street-Level Sequences (MSLS)
- City split: 22 training cities / Copenhagen and San Francisco Development /
  six unopened test cities
- Candidate adapter: centre of a frozen 100 m metric GPS grid cell, K=8
- Actual providers: local DINOv2-S and RapidOCR on sampled image pixels
- Primary comparison: static learned fusion versus quality-conditioned Router
- Frozen seeds: 1701 and 2701
- Advancement target: at least +8pp in both seeds without increased local error

No goal string, location ID, evaluator target, annotation text substituted for
OCR, test image, or test metadata is a model input.

## Source admission

MMS-VPR remains rejected because its capture-level coordinates were not aligned
to extracted frames. MSLS replaced only the source, not the question or scorer.

The MSLS metadata canary passed before image extraction. Query GPS was present
for 100% of frames. After frozen 100 m coarsening, target coverage was
449126/449323 (99.96%) for train and 9732/9732 (100%) for Development at K=8;
at K=16 it was 449271/449323 (99.99%) and 9732/9732 (100%). Test data remained
unopened. Exact admission evidence is in
`research/active/unseen-location-router/msls_source_canary_v1.json`.

## Development result

The bounded real-image run extracted DINO, real OCR, blur, and coarse-GPS
evidence for all 11,587 selected images with zero missing files. It evaluated
826/826 Development queries inside their K=8 candidate sets.

| Arm | Top-1 |
| --- | ---: |
| Visual only | 43.22% |
| OCR only, missing counts wrong | 33.17% |
| GPS only | 79.66% |
| Fixed equal available fusion | 71.91% |
| Static learned fusion, seed 1701 | 79.66% |
| Dynamic Router, seed 1701 | 79.18% (-0.48pp) |
| Static learned fusion, seed 2701 | 80.27% |
| Dynamic Router, seed 2701 | 76.15% (-4.12pp) |

The Router increased local-hard-negative error in both seeds. It assigned a mean
93.7% and 99.7% target-candidate weight to geography in the two seeds, so the
dynamic mechanism mostly collapsed onto the already strong GPS cue rather than
routing visual/OCR evidence usefully. Exact results and hashes are in
`research/active/unseen-location-router/msls_development_result_v1.json`.

## Decision

The Router missed the frozen `+8pp` advancement gate in both seeds and is closed
on this consumed Development cohort. Do not rescue it with backbone, threshold,
fusion-weight, quality-feature, candidate, or seed sweeps. The test cities remain
unopened.

MSLS did solve the original source-admission problem and made the Router genuinely
evaluable. The resulting negative Development outcome rejects this dynamic
routing formulation in the bounded setup; it does not establish a universal VPR,
named-POI, grounding, navigation, Android/device, product, or safety conclusion.
Only five Development queries were labelled night, and OCR correctness has no
independent transcription truth.

## Preserved prior terminal

GRAIL R1C-L remains closed as
`STOP_R1C_L_WITHOUT_FINAL_TEST / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`.
Its final test was not accessed, rerun, tuned, or fused into this route.
