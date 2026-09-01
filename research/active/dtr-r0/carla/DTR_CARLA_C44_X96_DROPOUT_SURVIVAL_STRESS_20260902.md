# DTR CARLA C44 / X96 dropout-survival stress — 2026-09-02

## Terminal decision

`DTR_CARLA_C44_X96_PARTITION_NOT_EVALUABLE`

C44 admitted a complete source and sealed all `48` four-arm predictions before
opening evaluator truth. The frozen `B_PRE_ONSET` and `C_RELEASE_BOUNDARY`
placements did not both satisfy their preregistered truth semantics, so the
joint score is terminal `PARTITION_NOT_EVALUABLE`. The primary gate is not met.

The result also gives no reason to promote the current X96 implementation:
X96 emitted zero survival frames, recovered fewer positive dropout frames than
X94, and reduced frame F1 from `81.76%` to `81.42%`.

## Source and authority

C44 restored the last source-admitted C41 `c8_*` target, occluder, alias, and
wearer trajectories byte-for-byte after C43's source-gate failure. It changed
seed `441096`, weather assignments, plan receipts, and pixels. It is therefore
a fresh scripted render and truth-blind Development run, but **not**
trajectory-disjoint confirmation.

The first wearable-server attempt exited with zero frames. The single permitted
zero-frame retry resumed the unchanged run. All four shards then passed with
`728` frames per sensor, `73` unique actual blueprints, and all `8 / 8` local
occlusion contracts.

- Protocol SHA-256:
  `7C940C55362585D553BF86E198BEA42B641F29AA0AD730B99E4E7479EF881555`
- Source result SHA-256:
  `1F67C6363ECD7D86E1EB1009776AA513F1F008AC92E1B7EF7AE35C8B554FCA00`
- X24 freeze SHA-256:
  `66DD4E1A5552419B81C2027AF2AF139B5CE6842393EB12FA9F5E4EB207E1A4E6`
- Sealed predictions SHA-256:
  `2BF2B729E616F4E072336E6C3CD3B3A5F5AEDF8ED393D901DF147D6C450DCB29`
- Summary SHA-256:
  `62EB04B8A12B8AE3C3D18CDAF17E4410E15549755529BB84107AE8114CCA557B`
- Frozen X96 SHA-256:
  `C155911FDADA84EDAE31F417F0B68C8D2747304A86C5D85F04360D471E0FB0D2`

## Partition evaluability

| Partition | Evaluable | Meaning |
|---|---:|---|
| `A_ACTIVE_MIDDLE` | yes | dropout truth remained positive |
| `B_PRE_ONSET` | no | `ep_05` already had positive truth before the frozen placement |
| `C_RELEASE_BOUNDARY` | no | no contact episode transitioned positive-to-negative near sample 44 |
| `D_PLAN_CONFLICT` | yes | controlled route-authority conflict was evaluable |

The placement indices were frozen before capture and may not be moved after
truth opened. Metrics below are diagnostic Development outputs, not a valid
joint generalization adjudication.

## Diagnostic four-arm result

| Arm | TP / FP / FN | Precision | Recall | F1 | 2 / 3 / 6-frame positive recovery |
|---|---:|---:|---:|---:|---:|
| Recursive forward-fill | 1592 / 3 / 472 | 99.81% | 77.13% | 87.02% | 100 / 100 / 100% |
| 0.60 s hysteresis | 1773 / 89 / 291 | 95.22% | 85.90% | 90.32% | 100 / 100 / 100% |
| X94 one-frame continuity | 1428 / 1 / 636 | 99.93% | 69.19% | 81.76% | 75.0 / 66.7 / 58.3% |
| X96 bounded survival | 1418 / 1 / 646 | 99.93% | 68.70% | 81.42% | 62.5 / 58.3 / 54.2% |

All arms hit all 48 event windows, so event F1 was `100%` and was not
discriminative in this controlled construction. X96 preserved zero false births,
zero plan-conflict carries, zero negative-dropout persistence, and zero median
release overshoot, but it failed the frozen frame-F1 and 2/3-frame recovery
requirements. The naive baselines' higher recovery is not a promotion result:
forward-fill carried `44` plan-conflict frames, while hysteresis added `89`
frame false positives and `0.15 s` median release overshoot.

Mechanism counters were:

- zero-detector-and-metric dropout frames: `176`;
- X94 continuity frames: `10`;
- X96 conflict rejections: `64`;
- X96 survival frames: `0`.

Thus the current X96 credential requirements prevented the proposed extension
from exercising on this admitted source. It must not replace X94, and C44 may
not be retuned, rescored, or converted into confirmation.

## Claim correction and next decision

The sealed summary contains a generic claim-boundary field named
`new_actor_trajectory_seed_weather_and_pixels=true`. For C44 that field name is
incorrect: the protocol explicitly records
`RESTORED_BYTE_IDENTICAL_C41_TRAJECTORY_BINDINGS_AFTER_C43_SOURCE_GATE_FAILURE`.
The hashes, predictions, truth alignment, metrics, and terminal decision are
unaffected. The reusable runner now records trajectory authority separately.

Keep X94 as the cumulative main arm and X73 as the latest complete
source-disjoint authority. Close the current X96 bounded-survival challenger for
this role rather than adding another dropout-duration rule. A successor would
need new information reachability, especially anchorless current residual
occupancy, not a longer history-only bridge. C44 is scripted synthetic
Development evidence only, not natural-dropout prevalence, real-world,
deployment, reliability, user-benefit, or safety evidence.
