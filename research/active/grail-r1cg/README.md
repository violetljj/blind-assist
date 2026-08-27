# GRAIL R1C-G0

Status: `STOP_G0_POSE_TRANSPORT / DEVELOPMENT_GATE_NOT_MET / NO_FINAL_TEST`

R1C-G0 is the smallest geometry-observable owner-orientation probe. It does not
train another RGB orientation network. On a fresh ProcTHOR train-split roster
that excludes all R1C-L train and validation houses, it compares:

- always-PRESERVE;
- a fixed camera-yaw transport rule: PRESERVE when the cosine of the
  source-native query-minus-reference camera yaw is non-negative, otherwise
  FLIP.

The 90-degree boundary is fixed by the coordinate transport hypothesis and is
not selected on Development outcomes. Primary evidence is the discriminative
subset, especially FLIP-only cases, plus Drawer and Doorway results. Ambiguous
pairs on which both modes are valid are reported separately.

This first probe deliberately uses source-native cross-session relative camera
yaw. It is a mechanism oracle, not a deployable BlindAssist input. It uses no
owner yaw, owner position, object coordinates, depth, model training, threshold
sweep, R1C-L final data, or protected test data. Passing can authorize an active
multiview/viewpoint-transport successor; it cannot establish natural-scene,
device, Android, product, or safety capability.

## Development result

The frozen 24-house roster excluded all 180 R1C-L train/validation houses. It
produced 1,157 views and 2,562 ordered pairs with no runtime timeout. Of these,
2,094 were discriminative: 514 FLIP-only and 1,580 PRESERVE-only.

| Arm | Discriminative | FLIP-only | PRESERVE-only |
| --- | ---: | ---: | ---: |
| Always PRESERVE | 1,580/2,094 (75.45%) | 0/514 | 1,580/1,580 |
| Fixed pose transport | 1,576/2,094 (75.26%) | 82/514 (15.95%) | 1,494/1,580 (94.56%) |

The fixed pose rule was `-0.19pp` overall, `0.00pp` on Doorway, and `-0.25pp`
on Drawer versus the prior. It missed every advancement condition and closes at
`STOP_G0_POSE_TRANSPORT`. Relative camera yaw alone does not expose the missing
owner-canonical sign: it recovers too few FLIP cases while creating more false
flips on PRESERVE cases.

This result does not evaluate active multiview appearance. A future G1 would
need a separately versioned question in which the additional reference views
provide real occlusion, side-panel, foreshortening, or other asymmetric visual
evidence. Do not rescue G0 with a yaw-threshold sweep or fuse it into R1C-L.

Exact roster and outcomes are in `grail_r1c_g0_manifest_v1.json` and
`grail_r1c_g0_development_result_v1.json`.
