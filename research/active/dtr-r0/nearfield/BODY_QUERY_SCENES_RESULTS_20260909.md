# BODY-QUERY new-scene collection results

2026-09-09. Source engineering only, following
[the collection contract](BODY_QUERY_SCENES_COLLECTION_20260909.md).
No model inference, threshold selection or training was performed.

**Completed:120/120 captured and labeled frames,24/24 complete accepted groups,
zero intent mismatches or rejected groups.** Each region has four actual
HEAD-near-positive groups and four actual HEAD-far-positive groups, spanning
all four families. These are fixture groups within one scene per region.

| Role | Frames | BODY positive/negative | HEAD positive/negative | Accepted groups | HEAD near/far positive groups |
| --- | --- | --- | --- | --- | --- |
| TRAIN |40|16/24|16/24|8/8|4/4|
| DEV |40|16/24|16/24|8/8|4/4|
| EVAL |40|16/24|16/24|8/8|4/4|

All120 camera-position floor probes match the declared floor (reported maximum
Z error0m). Continuous floor-patch checks are explicitly disabled; this proves
the camera-height anchor, not walkability or complete ground smoothness. All
capture logs have zero matched Nanite-invalid/VT-fetch-failure signatures.
All120 RGB images were reviewed through the three complete contact sheets;
fixture interventions and distinct building/ground layouts are visible, without
obvious broken render resources. The near/far and extra negative states are
retained, including their physical posts/backboards; CLEAR means no native
BODY/HEAD witness, not an object-free image.

Capture plus native-world verification took216.78/120.27/218.56s for the three
roles. Subsequent label/ownership materialization took9.38s on CUDA, NVIDIA
GeForce RTX5060 Laptop GPU. The twelve raw/capped query counts and support masks
exactly reproduce each frame's native world labels. The generator's two tests
and label tool's five focused tests passed.

## Source and intended coverage

Three new BigCity placements reuse eight fixed-camera fixture groups from the
previous320-frame source-v3 generator. Every region contains four families,
near/far target centers1.0/2.4m, all four primary BODY/HEAD relations, and four
extra crossbar controls at each distance. Exact poses and roles were saved in
`artifacts.local/work/body-query-scenes-20260909/placements-v1.json` before
capture. All three source specifications were generated and hashed together.

| Role | Region | X,Y metres | Floor Z | Yaw |
| --- | --- | --- | --- | --- |
| TRAIN | dense_candidate_01 | -679.304091,-2062.533588 |1.065249|0 |
| DEV | dense_candidate_07 | -1548.846018,-3082.840065 |0.700000|0 |
| EVAL | dense_candidate_02 | 165.414739,846.718531 |0.730000|90 |

Each role has one scene/pose and eight fixture groups, not eight independent
scenes. The fixtures are reused across roles; local geometry hashes agree.
These are controlled simulator placements with explicit supports/backboards,
not naturally occurring obstacles. The previous source-only scouts exposed
all three views and labelled them TRAIN; the new Development roles above are
declared before this capture and before any model access. No asset-disjoint,
strict visible-background-disjoint or protected TEST claim is made.

## Reproduction and durable outputs

Canonical artifact root: `artifacts.local/work/body-query-scenes-20260909/`.
`specs-v1/manifest.json` binds120 intended frames and all source specs.
`run_capture.py` checks host availability and the existing native plugin build,
then runs the repository's owned-process capture launcher and CUDA native-world
verifier per region. Existing City project/cache are reused with an isolated
Zen service port; the cache is durable shared source infrastructure.

The raw `capture-{train,dev,eval}-v1/` folders retain640x360 RGB, float32 native
depth, source snapshots, input/payload hashes, original receipts, native floor
probes, world-support masks, render-health logs and process-release records.
Two focused source-generator tests pass: complete relations/fixed calibration,
and preserved local fixture geometry under region rotation/translation.

`labels-v1/manifest.json` is the dataset label entry point, SHA256
`6f40a3a9a87a0d53dd7f8f50c0bb9d884248a314cc061053fc1ea740129224d8`.
`labels-v1/evaluator/index.json` maps all120 frames to source roles, raw RGB/depth
hashes and per-frame NPZ labels. All roles remain evaluator metadata in this
source delivery, not automatic optimizer input. The NPZ files retain UNKNOWN,
native support, pooled support, raw/capped query counts, exact-ray ownership,
and maxpool/bilinear local membership with the existing R1/probe UNKNOWN mask.
Local membership is weighted block presence, not native-pixel probability.
`labels-v1/result.json` retains per-region/family/condition/range counts and all
24 complete-group decisions; `receipt.json` binds the result and manifest.

`train-overview.png`, `dev-overview.png`, and `eval-overview.png` show every
captured frame with its condition and intended distance. `completion.json`
records the three successful capture/verifier runs. Their process-release
receipts all report released=true with no survivors; the isolated Zen port was
28656. Durable City cache/source data remain for reuse. No capture is queued.

The next dataset-size or training decision must use actual source acceptance
and range/group coverage below. This collection changes no retained model.
