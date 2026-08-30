# DTR CARLA X27-X30 occupancy-lineage results — 2026-08-30

## Current decision

`DTR_CARLA_X30_ADAPTIVE_SURFACE_CONTACT_INTERVAL_SOURCE_DISJOINT_PRETRUTH_DEVELOPMENT_GATE_NOT_MET`

X30 is directionally better than X24 on fresh C7, but it is not strong enough to
promote. It improved frame F1 from `0.6000` to `0.6376`, raised precision from
`0.7931` to `0.8488`, raised recall from `0.4825` to `0.5105`, and removed the
one adjudicated SAFE false-alert segment produced by X24. The frozen gate still
failed because X30 completely missed dynamic contact `ep_03`, retained only
`0.3333` future-positive recall on `ep_07`, and remained below the `0.95`
precision and `0.80` F1 requirements.

Do not rescue X30 by changing the detector threshold, route tube, confirmation
duration, HOLD duration, association distance, seed, weather, target placement,
backbone, or C7 cohort. C7 is consumed pre-truth Development evidence.

## Structural sequence

The X27-X30 sequence replaced fitted detector-box velocity with increasingly
explicit world-occupancy authority:

1. X27 decomposed track state into `EGO_CARRIED`, `STATIC_SCENE`,
   `RIGID_DYNAMIC`, and `UNAUTHORIZED_MOTION`; only rigid dynamic state could
   sweep velocity into future route risk.
2. X28 added a persistent occupancy core. On fresh C5 it removed all SAFE risk
   segments but collapsed F1 from X24's `0.6087` to `0.3077`; the core was too
   brittle to retain authority through surface changes.
3. X29 transported world-lattice occupancy ancestry instead of a current-cell
   core. On C6 it achieved precision `1.0000`, zero SAFE risk segments, and F1
   `0.6606` versus X24's `0.6180`, but recall stayed `0.4932` and the retained
   occlusion pair failed. The scorer required a post-source mechanical binding
   correction, so C6 was not pristine confirmation.
4. X30 replaced the global `160x90` depth support test with a fixed `32x32`
   detector-box-normalized surface lattice, fused overlapping within-frame
   branches into a set-valued world component, retained semantics only as a
   provenance label set, and scored current/future route-contact intervals.

X28 result SHA-256:
`235B55E2FDDEBE312DF2AFB6B1D84A0AB238516B4B88CE63B446AF6591F15765`.
X29 result SHA-256:
`E9C33AA2C4CDAB646CAC8556F206BA9758447C1090BC8C51FBBF151EB30DE410`.

## Frozen C7 source and model replay

- Cohort: `DTR_CARLA_C7_X30_SOURCE_DISJOINT_CONFIRMATION_V1`
- Source run: `E:\linnan\CARLA\experiments\dtr-carla-c7-x30-source-disjoint\evidence\c7-x30-source-20260830-173540`
- Protocol SHA-256: `822976A7F34379CFCA48C800A829704C6774A5D8BFC3FC80A596E50C8B5EF55A`
- Source result SHA-256: `7B4C6F8A9A40B1028F94EC44053CB1428D4D03A2A679006470648E86049202BE`
- Capture: eight episodes, four layouts, four sensors, 708 frames per sensor,
  2,832 raw payloads, 1280x720, and 75 unique actual blueprints
- Model run: `E:\linnan\linnan\artifacts.local\evidence\dtr-carla-x30-adaptive-surface\c7-x30-source-20260830-173540`
- YOLO candidates: 7,011; candidate manifest SHA-256:
  `02D9A2D35CC50F5FDC6BDF0C87299574FA3FD268B7583B04BAE8C9B7C41AE834`
- Detector weight SHA-256:
  `55ED65C56C91713D23E8402371C6C49A6FD84F257F7DCE452E8D70E41DCBE152`
- X24 freeze SHA-256:
  `EC437426D6F90231027A4BDE42484EDBBD6CDC234787E21DFF909AA8F5002E55`
- X24 predictions SHA-256:
  `2B832B64EA00EF5B8CC64E1F64AEF5E248C0BCFAB03D3FDF670D2F6B69747431`
- X30 predictor SHA-256:
  `004419CEA2716D7E7BE8FA7BA41450DC6C0481DD3DF55F5D9DC9C08DEEF06ECA`
- X30 freeze SHA-256:
  `28CCE2085FA471F57A319621434DE97DD67DA2C9B39F45D640CBBE4CE2BC2B59`
- X30 predictions SHA-256:
  `87CA7EA0BD29E92DE012E8EB811E4249506D27294C88428B95CF5E2896C1DE6F`
- X30 result SHA-256:
  `F29346ABD7A75F1F21FEAA15898040F1DAAD3361568D68386E2D968BCE4E6D2B`

The first X30 prediction attempt stopped before writing predictions or opening
the evaluator because the descriptive fixed-constant key `semantic_role`
collided with the model-root forbidden-key list. The field was mechanically
renamed to `semantic_label_policy`; no algorithm value, model input, candidate,
or threshold changed. The failed freeze is preserved as
`freeze-x30.pre-sanitize-failure.json` with SHA-256
`F0FE2EC0E31CC99471BDDCA1F06975A53156370A15A97360EB531D036C247E83`.
Because this correction occurred after source capture, C7 is explicitly
pre-truth Development, not source-before-frozen confirmation.

## C7 metrics

| Metric | X24 | X30 | Delta |
|---|---:|---:|---:|
| Frame precision | 0.7931 | 0.8488 | +0.0557 |
| Frame recall | 0.4825 | 0.5105 | +0.0280 |
| Frame F1 | 0.6000 | 0.6376 | +0.0376 |
| True positives | 69 | 73 | +4 |
| False positives | 18 | 13 | -5 |
| False negatives | 74 | 70 | -4 |
| True negatives | 307 | 312 | +5 |
| Adjudicated SAFE risk segments | 1 | 0 | -1 |

| Episode | X24 lead | X30 lead | X24 recall | X30 recall | X30 result |
|---|---:|---:|---:|---:|---|
| `ep_01` retained occlusion | 3.0 s | 3.1 s | 0.9375 | 0.9375 | retained `5/5` occlusion frames |
| `ep_03` vehicle contact | none | none | 0.0000 | 0.0000 | complete miss |
| `ep_05` pedestrian contact | 2.4 s | 2.8 s | 0.7576 | 0.9394 | gate-level behavior |
| `ep_07` motorcycle contact | 1.1 s | 2.6 s | 0.3889 | 0.3333 | early alert, then 1.5 s gap |

X30 produced zero false-alert segments in all four SAFE episodes. All general
authority invariants passed: no ego or unauthorized track entered risk,
non-dynamic authorities had zero velocity, HOLD never promoted motion authority,
and route risk always referenced a confirmed eligible track.

## Post-score failure attribution

The evaluator was opened only after X24 and X30 predictions were sealed.

### `ep_03`: false contradiction removes the responsible target

The responsible target was carried by `temporal-lineage-000011`. It was born at
sample `i1`; its quantized center transport changed from `[-1,+2]` at `i2` to
`[-1,-1]` at `i3`. X29's inherited single-vector dot-product rule treated that
surface-center jitter as a direction reversal, permanently marked the lineage
conflicted, and X30 therefore made it `UNAUTHORIZED_MOTION`. The component
continued to be measured and intersected the responsible truth polygon through
`i33`, but risk geometry never received an authorized target branch.

The nine `RIGID_DYNAMIC` risk-eligible frames reported for `ep_03` belong to an
unrelated lineage at `t=5.8..6.6 s`, after the target's future-positive window;
its footprint remained `5.44..6.42 m` to the route's right and never appeared in
`candidate_risk_track_ids`. The failure is not confirmation hysteresis or route
geometry. It is premature motion-authority destruction.

### `ep_07`: one-frame surface rebound permanently kills continuity

The responsible target was `temporal-lineage-000018`. It became rigid dynamic at
`i4`, produced a candidate at `i4`, and produced confirmed route risk from
`i5..i16` (`0.5..1.6 s`). Between `i16` and `i17`, the single robust-OBB center
transport jumped from `[-4,0]` to `[+1,0]`, while the same frame's world-lattice
shift was `[0,0]`. Truth continued monotonically forward with no lateral move,
and transport returned to `[-4,0]` at `i18`; the `+1` vector was a surface-center
alias, not physical reversal.

The inherited permanent-conflict rule nevertheless demoted the lineage to
unauthorized at `i17`. It remained measured through `i30`, entered HOLD at
`i31`, and was measured again at `i32`, but could never regain authority. This
exactly creates the `i17..i31` 15-frame pre-contact alert gap and leaves only
`12/36` future-positive frames detected.

## Next structural route

The next successor should be one change:
`X31_AMBIGUITY_PRESERVING_SET_VALUED_SURFACE_TRANSPORT_ANCESTRY`.

Replace each single quantized `TransportEvidence.shift_xy` with a finite
non-dominated shift set or transport cone derived from overlapping world-lattice
surface correspondences. Temporal identity becomes a branch graph: a conflicting
center hypothesis terminates only that branch, while compatible occupancy-shift
branches retain their prior authorized motion cone. A real motion-epoch
contradiction exists only when every feasible current shift is separated from
every authorized prior branch. Route risk consumes only still-authorized branch
footprints and velocities.

This directly targets both consumed failures without changing thresholds or
information sources: `ep_03` keeps a compatible forward branch despite the
`+2 -> -1` lateral quantization swing, and `ep_07` discards the isolated `+1`
center rebound while retaining the supported forward branch. Freeze X31 and its
scorer before capturing a fresh C8 cohort, then run exactly once. Do not iterate
on C7.

## Claim boundary

X28 is fresh scripted-CARLA Development. X29 is a source-disjoint replay with a
post-source binding correction. X30 is fresh scripted-CARLA pre-truth
Development because of the post-source, pre-evaluator metadata-key correction.
None is pristine source-disjoint confirmation, real-world evidence, product
validation, default-App authority, or a deployment/safety claim.
