# Current decision: unseen-location evidence routing

Status: `MMS_VPR_COARSE_GPS_SOURCE_REJECTED / ROUTER_NOT_EVALUABLE / TEST_UNOPENED`

## Authorized question

Can a quality-conditioned Evidence Router improve reference-conditioned
held-out-location retrieval over the strongest static learned fusion when the
model receives actual image evidence and a map-derived local candidate set?

The current phase predicts only candidate identity and confidence. It does not
perform named-POI grounding, target boxes/masks, video persistence, navigation,
entrance finding, or product/safety decisions.

## Frozen surface

- Protocol and code: `research/active/unseen-location-router/`
- Source: MMS-VPR images, text metadata, and graph metadata
- Identity split: 146 train / 31 Development / 31 unopened test locations
- Reference/query isolation: capture-group disjoint
- Actual providers: local DINOv2-S and RapidOCR on sampled image pixels
- Primary comparison: static learned fusion versus quality-conditioned router
- Frozen seeds: 1701 and 2701
- Advancement target: at least +8pp in both seeds without increased local error

No goal string, location ID, class index, manual store/sign list, evaluator target,
box/mask truth, or annotation text substituted for OCR is a model input.

## Data and canary result

The complete 110,529-image / 208-location source passed identity and capture-group
materialization. A bounded real-image canary then extracted DINOv2, actual OCR,
blur, and available coarse-GPS evidence for 1,060 images; no test pixels were read.

The source failed at candidate construction before the router. MMS-VPR coordinates
describe source videos/captures rather than each extracted frame. Those frames are
distributed among many location folders along a pedestrian route. Query GPS was
about 363 m from evaluator target location at the median.

Consequently, target coverage in GPS-nearest candidate sets was:

- K=8: train 53/566 (9.36%), Development 45/121 (37.19%);
- K=16: train 102/566 (18.02%), Development 77/121 (63.64%).

The K=8 scorer therefore had only 53 train and 45 Development queries. On that
diagnostic subset, static learned fusion scored 13/45 (28.89%). Dynamic routing
scored 13/45 (28.89%, +0.00pp) for seed 1701 and 14/45 (31.11%, +2.22pp) for
seed 2701. These values are diagnostic only: they neither advance nor reject the
router because the candidate source destroyed the denominator upstream.

Exact counts, hashes, failure mechanism, and prohibited rescues are recorded in
`research/active/unseen-location-router/development_canary_result_v1.json`.

## Decision

Reject MMS-VPR as the sole data source for the frozen coarse-GPS Evidence Router
experiment. Do not repair the consumed canary by injecting evaluator location,
forcing target inclusion, expanding K to the full split after outcome access,
using manual annotation text as OCR, or tuning the scorer.

A new run requires either:

1. a source with frame-aligned coarse position and independent references; or
2. separate authorization for a no-GPS global-retrieval task with a newly frozen
   question, baselines, denominators, and claim ceiling.

## Preserved prior terminal

GRAIL R1C-L remains closed as
`STOP_R1C_L_WITHOUT_FINAL_TEST / DEVELOPMENT_GATE_NOT_MET / FINAL_UNOPENED`.
Its final test was not accessed, rerun, tuned, or fused into this route.

## Claim ceiling

This run proves that the new data contracts, location/capture isolation, real-image
feature providers, learned baselines, router diagnostics, and evaluator can execute
on a bounded MMS-VPR canary. It does not prove or disprove dynamic routing on a
valid local candidate task, unseen-city or named-POI generalization, grounding,
tracking, navigation, Android/device behavior, product effectiveness, or safety.
