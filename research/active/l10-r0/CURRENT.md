# L10-R0 current

Updated: 2026-08-31

Status: `L10_R0_ACTIVE`

## Question

Can BlindAssist keep the user's exact destination bound through active
observation, identify its functional entrance or endpoint, and hand off only
when the required evidence is actually present?

`referent != affordance != waypoint != arrival != handoff` remains the route
contract.

## Current decisions

- **Controller:** seek, guide, reacquire, and the causal action-belief handoff
  guard remain implemented. Controlled controller results are mechanics
  evidence, not real-camera or product evidence.
- **PanoLab active observation:** entrance-ray recovery passed `4/4`. This
  authorizes an entrance ray geometrically, not a pixel portal or arrival.
- **SEVN address-door backend:** the frozen V2 stack passed all six Development
  gates on its consumed 24-address panel (`14` correct, `2` wrong, `8`
  `UNKNOWN`; visible-number OCR `19/22`). It did not confirm on 21 fresh
  addresses and 28 fresh panorama frames: OCR was `12/20`, binding was
  `9/0/12` correct/wrong/`UNKNOWN`. Do not tune that confirmation panel.
- **Metric portal extent:** 3RScan registered extent established a strong
  synthetic/registered Development ceiling. The latest source-distinct spatial
  mask reached `0.5403` complete IoU and `0.422 m` centroid error, but stayed
  below the `60%` ceiling-retention gate and confused an overlapping doorframe.
  Exact-instance and portal-set binding remain the information gap.

## Next admissible work

1. For SEVN, change the OCR observation representation and evaluate it on fresh
   addresses and frames; do not rescue the consumed confirmation panel with
   thresholds, tiling, or ranking changes.
2. For metric portals, add exact-instance or portal-set authority before
   another endpoint-mask successor; do not reinterpret overlap with a nearby
   frame as a correct entrance.
3. Keep active actions tied to the actual deficit: `APPROACH`, `SIDESTEP/PAN`,
   `SWEEP`, or `HOLD`. An action proposal is not an arrival or handoff.

## Stop and claim boundary

- Generic Panoramax pixel-portal mining and the consumed SceneFun3D ordinal
  source are closed.
- `UNKNOWN` and `NOT_EVALUABLE` are neither failure nor known-safe.
- Synthetic, registered, replay, and curated Development results do not prove
  natural-camera performance, user benefit, navigation, reliability, or
  safety.
- Device/demo integration is not reopened by an algorithm-only result.

## Detail and evidence

- Detailed route ledger and reproduction commands: [README.md](README.md)
- SEVN fresh confirmation result:
  [l10_sevn_pixel_topology_confirmation_result_v1.json](l10_sevn_pixel_topology_confirmation_result_v1.json)
- PanoLab active-ray result:
  [l10_panolab_active_ray_recovery_result_v2.json](l10_panolab_active_ray_recovery_result_v2.json)
- Latest 3RScan spatial-mask result:
  [l10_3rscan_spatial_reference_mask_result_v1.json](l10_3rscan_spatial_reference_mask_result_v1.json)
