# DTR-CARLA-N1 native-dynamics pilot result — 2026-08-30

## Decision

`DTR_CARLA_N1_NATIVE_DYNAMICS_TRACE_MATERIALIZED`

The first N1 pilot successfully moves the CARLA source line from fully scripted
per-frame actor transforms toward native traffic and crowd behavior. One seeded
Town10HD_Opt realization ran CARLA Traffic Manager, `controller.ai.walker`,
three group reroutes, and four authored long-tail interventions under one
synchronous tick owner. The complete actor state was frozen for later replay.

This is a synthetic Development source/mechanism result. It is not an obstacle
algorithm result, a source-disjoint confirmation, a natural-traffic
distribution, real-device evidence, product benefit, or safety evidence.

## Frozen run

- Plan: `dtr-carla-n1-seed-20260830`
- Plan fingerprint: `A97BD37A912BD46EEF59DD16A7479CE1EC3FA7E9635F4B7E9C9FF57DD7E219A8`
- Evidence root: `E:\linnan\CARLA\experiments\dtr-carla-n1-natural-dynamics\evidence\n1-native-pilot-v4-20260830-2345`
- Result status: `DTR_CARLA_N1_NATURAL_DYNAMICS_MATERIALIZED`
- CARLA map/version: `Carla/Maps/Town10HD_Opt`, `0.9.16`
- Duration/timestep: `24.0 s`, `0.05 s`
- Trace: `481/481` frames, SHA-256 `794C7C909D77976C157A51A6D21A337C1EAB2520CDEAAC00BE59FAF19251E21A`
- Actor manifest SHA-256: `0DB62C8BCDE48A7101D48FB46B18DC185B45D700A7DAC8FF15835F8D2E256106`
- Event receipts SHA-256: `7C9EB51EB221C9CFF462EE2339BE942A5F788319E136386D0F939450C8F02211`
- CARLA recorder: `20,919,440` bytes, SHA-256 `9E799BD66F2ADFBA6F42F8D118D8CFBE5075A1635E24C3970010D82173736A7F`
- Contact sheet: `10,930,898` bytes, SHA-256 `99870A84656132BC57699575D5EB7ABA8B40EAD519052ABF39683CD99A64109C`

All 13 result checks are true, including exact actor denominator, complete
trace and preview, three traffic profiles, native vehicle/walker motion, group
reroutes, close crowd encounters, all event API calls, and observed event
effects. The runner released the task-owned CARLA processes and ports
`26000--26003` after completion.

## Observed behavior

| Surface | Frozen plan | Observed result |
|---|---:|---:|
| Traffic Manager vehicles | 6 | `6/6` moved |
| Driving profiles | cautious / nominal / assertive | all 3 realized |
| AI pedestrians | 15 | `15/15` moved |
| Crowd groups | 5 | all materialized |
| Seeded group reroutes | 3 | `3/3` executed |
| Pedestrian pairs within 2 m | not a fixed target | 28 distinct pairs |
| Long-tail events | 4 types | `4/4` API and effect checks passed |
| RGB witness preview | 9 frames at 1280×720 | `9/9` present |

The closest pairwise pedestrian distance in a read-only audit of the frozen
trace was `0.6008 m` (`n1_pedestrian_06` and `n1_pedestrian_07`). This supports
the narrow statement that the pilot produced actual close crowd encounters;
it is not evidence of realistic social-force calibration.

| Long-tail event | Trigger | Primary actor | Observable effect |
|---|---:|---|---:|
| occluded jaywalk | `5.05 s` | `n1_pedestrian_03` | `2.00 m` event-window displacement |
| sudden brake | `8.65 s` | `n1_vehicle_assertive_01` | `7.30 m/s` speed drop |
| reverse pullout | `13.35 s` | `n1_vehicle_cautious_01` | `2.79 m` event-window displacement |
| roadside door open | `17.75 s` | `n1_vehicle_nominal_02` | open/close API receipt passed |

## Structural contribution

N1 separates stochastic behavior generation from future formal sensor replay:

1. A pure compiler turns a registry plus master seed into explicit traffic,
   crowd, group-reroute, and event intents with independent subsystem seeds.
2. A single CARLA materialization runs native controllers and records every
   actor transform, velocity, acceleration, control, and event receipt.
3. The frozen behavior trace is the deterministic input for a later
   physics-off four-modality replay. Natural controllers are not rerun once per
   sensor shard, so their cold-start drift cannot masquerade as RGB-D alignment.

The current pilot stops after step 2. It deliberately does not alter the sealed
C4 scripted source, X24, X31, or C11.

## Reproduction

Compile the plan without starting CARLA:

```powershell
python -B .\research\active\dtr-r0\carla\dtr_carla_n1_natural_dynamics.py `
  --seed 20260830 `
  --output .\artifacts.local\tmp\dtr-carla-n1-plan-20260830.json
```

Materialize one new, non-overwriting native-policy trace:

```powershell
pwsh -NoProfile -File .\tools\run_dtr_carla_n1_natural_dynamics.ps1 `
  -RunId <new-unique-run-id> `
  -Plan .\artifacts.local\tmp\dtr-carla-n1-plan-20260830.json
```

## Next decision

The next useful increment is one `FrozenBehaviorTrace -> C2-compatible replay`
adapter, followed by one four-modality RGB/depth/instance/witness join. That
replay must preserve model/evaluator separation and bind the trace SHA-256. It
should not tune X24/X31 or expand the event taxonomy before the cross-modality
replay contract is demonstrated.
