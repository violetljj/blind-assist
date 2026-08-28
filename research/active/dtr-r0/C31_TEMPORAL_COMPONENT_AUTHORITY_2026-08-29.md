# DTR-C31 temporal velocity-component authority

## Decision

C31 meets the frozen consumed-cohort Development gate:

`DTR_C31_TEMPORAL_COMPONENT_AUTHORITY_DEVELOPMENT_GATE_MET`.

This is the first DTR arm in the C25 line to retain `12/12` CONTACT recall,
recover at least `30/36` induced track-dropout misses, remain at no more than 21
false segments, and keep every event no later than M1-PDC.  It is the next
development representation baseline, not fresh confirmation or a production
safety claim.

## Mechanism

C30 proved that confidence-aware local direct motion can suppress much of the
static pseudo-motion, but the raw trace had zero-filled `dp_m/dv_mps`; it did
not contain genuine cross-frame consistency.  C31 adds the missing causal state:

1. all supported reciprocal raw-flow rows vote into local position/velocity
   components;
2. a soft component is matched to the previous component at its predicted
   location `c + v * dt`;
3. the match accumulates positive dynamic evidence only when the observation is
   closer to the transported location than to the old static location;
4. only a positive, two-hit hard component can grant authority to its members;
5. during a short `OCCLUDED` interval, the last observed component support
   footprint is transported for at most 0.25 seconds with confidence decay;
6. `KNOWN_FREE` revokes the component, while `HIT` and `UNSENSED` cannot refresh
   motion identity.

The signed transport update is structural rather than a learned score:

`dynamic_evidence = q * (static_residual^2 - transport_residual^2)`.

Static pseudo-motion stays near its old position and drives the score negative;
a genuine mover better follows its reported velocity and drives it positive.

## Result

| arm | CONTACT recall | false segments | event F1 | median first lead | induced dropout recovery | every event no later than PDC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M1-PDC | `12/12` | 21 | 53.33% | 1.624 s | `25/36` | yes |
| C30 point confidence + local consensus | `12/12` | **20** | **54.55%** | 1.734 s | `29/36` | yes |
| C31 soft-to-hard component without static/dynamic evidence | `12/12` | 28 | 46.15% | **2.732 s** | **`33/36`** | yes |
| **C31 signed transport authority** | **`12/12`** | **21** | **53.33%** | **2.667 s** | **`30/36`** | **yes** |

Relative to M1-PDC, C31 recovers five more dropout misses and improves median
lead by `1.043 s` without increasing false segments or losing an event.  Relative
to the broad soft-to-hard component arm, signed transport removes seven false
segments while retaining enough causal mover evidence to pass the recovery
gate.  Relative to C30, it trades one additional false segment for one required
dropout recovery and `0.933 s` more median lead.

The final authority processed 66,501 eligible raw motion rows, granted 11,020
component-member extensions, emitted 341 short-occlusion support predictions,
and retained 9,823 mature-track observations.  These counts are diagnostic
exposure, not independent successes.

## Backend receipt

The shared research profile was checked before launch.  Representative 48x12
component matching selected CPU with explicit reason `CPU_FASTER_MEASURED`:
NumPy CPU median `0.0095 ms` versus real RTX 5060 Torch CUDA `0.1712 ms`.
JSON loading and scalar scoring also remained CPU tasks.

## External mechanism reserve

Exa reviewed 125 returned candidates across 16 searches in three independent
angles and deep-read nine high-relevance primary papers/pages.  The mechanisms
used here are intentionally smaller than their full systems:

- [VoteFlow](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_VoteFlow_Enforcing_Local_Rigidity_in_Self-Supervised_Scene_Flow_CVPR_2025_paper.html): local translation voting and rigidity; C31 uses local velocity components rather than a learned scene-flow network.
- [ICP-Flow](https://openaccess.thecvf.com/content/CVPR2024/html/Lin_ICP-Flow_LiDAR_Scene_Flow_Estimation_with_ICP_CVPR_2024_paper.html) and [RigidFlow](https://openaccess.thecvf.com/content/CVPR2022/html/Li_RigidFlow_Self-Supervised_Scene_Flow_Learning_on_Point_Clouds_by_Local_CVPR_2022_paper.html): shared local motion as mover evidence; C31 fits only causal 2-D translation because the trace cannot support honest SE(3).
- [Let It Flow](https://arxiv.org/html/2404.08363v3): overlapping soft hypotheses grow into hard components; C31 adopts soft-to-hard component birth without offline test-time optimization.
- [Transitional Grid Maps](https://arxiv.org/html/2401.06518v2): separate static and transported dynamic belief; C31's signed residual is its smallest falsifier.
- [DSP-map](https://arxiv.org/html/2202.06273): predict/update/birth/death with occlusion; C31 keeps one deterministic component hypothesis rather than a particle map.

Flow4D, GenFlow3D, SceneTracker-style long-sequence methods, full ICP/SE(3), and
whole-sequence Eulerian optimization were not admitted because they require
future frames, long offline context, unavailable geometry, or a much larger
trajectory model than M1 needs.

## Evidence and next action

This result uses the consumed five-sequence C25 Development cohort: 3,358
frames, 12 bounded CONTACT events, 36 induced dropout trials, and 130.98 seconds
of known non-contact exposure.  The policy is causal and truth-blind, but its
structure was developed on this cohort.  The gate authorizes a fresh
source-disjoint confirmation of the frozen C31 mechanism; it does not authorize
more C25 threshold, decay, component-radius, route, or lifecycle tuning.

Result SHA-256:
`1787d88a13c5dcc689dc28ce8a4f46c2d7ae6b0c3114ffab2e05d6c5acfe1e8d`.
Predictions SHA-256:
`9415defe2abbbb6ee6e8ac117f601c6c779127f6ad166b4522c3331a3c179184`.
Backend receipt SHA-256:
`5128af6522b80e229002e2a6950b28b22bd75c47ee6b256508193d5b8ef6d5dd`.
Executable candidate SHA-256:
`51c97b4dfdab54053cea525f810cede74da4500bd362a891c8d0567f2ca64112`.
