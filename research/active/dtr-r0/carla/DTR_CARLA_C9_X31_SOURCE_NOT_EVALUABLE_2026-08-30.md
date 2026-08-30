# DTR CARLA C9 X31 source result — 2026-08-30

## Decision

`DTR_CARLA_C9_X31_COLLISION_DECOUPLED_SOURCE_NOT_EVALUABLE_OCCLUDER_ROUTE_INTERSECTION`

C9 removed the C8 scripted-pose drift, but it did not reach source admission.
The single frozen capture stopped after the complete `instance` shard because
the physical occluders themselves entered the wearer collision corridor.  No
wearable, depth, witness, model, predictor, joiner, evaluator, or scorer run
was started.

## Frozen identity

- Git commit: `92b84abd813145fd133319cad5f7543d9de26de3`
- Run ID: `c9-x31-collision-20260830-200002`
- Cohort: `DTR_CARLA_C9_X31_COLLISION_DECOUPLED_SOURCE_DISJOINT_V1`
- Capture seed: `130364`
- Protocol SHA-256:
  `0CE038F4CE411F6C01B0C6718FCD346E54681149E9F8201AA1FACA498486F9F2`
- X31 predictor SHA-256:
  `28E777021FC6AA39129480B069B544B562F559FD447CCE164706B6DCAC3A9F18`
- Source root:
  `E:\\linnan\\CARLA\\experiments\\dtr-carla-c9-x31-collision-decoupled\\evidence\\c9-x31-collision-20260830-200002`

## What C9 established

All `728` expected `instance` payloads were captured at `1280x720`.  Every
scripted-pose asset matched its authoritative transform; the maximum planar
residual was `1.57284e-05 m` against the frozen `1e-04 m` tolerance.  Sensor
and world frames aligned, all eight episodes materialized, all blueprints and
bounding boxes passed, and no blueprint fallback occurred.  Therefore the C8
pose drift was causally removed by collision decoupling plus hard physics
disablement.

## Source-admission failure

The four planned CONTACT episodes remained CONTACT, but each gained its
layout occluder as an extra responsible asset.  The four planned SAFE twins
all became CONTACT with the occluder as their responsible asset:

| Episodes | Frozen expectation | Observed source truth |
| --- | --- | --- |
| `ep_01`, `ep_03`, `ep_05`, `ep_07` | CONTACT, target only | CONTACT, target plus occluder |
| `ep_02`, `ep_04`, `ep_06`, `ep_08` | SAFE, no responsible asset | CONTACT, occluder only |

The source result therefore failed exactly
`all_expected_outcomes_match` and
`all_expected_responsible_sets_match`.  This is a scene-geometry defect, not
an X24 or X31 false positive or false negative.

## Occlusion reachability diagnostic

After the source had already failed, the frozen joiner's physical-occlusion
function was applied read-only to the captured `instance` rows.  This did not
create a formal evaluator package or metric result.  It showed that all eight
planned occlusion contracts would also have failed: the main zero-visibility
runs were `2.2`, `2.3`, `2.5`, `2.7`, `0.1`, `0.3`, `1.8`, and `2.6 s`, while
the frozen admissible interval was `0.6..1.3 s` with at least ten trackable
frames before and eight after.  Thus merely moving the current occluders out
of the route would not make this source representation evaluable.

## Evidence and cleanup

- Instance result SHA-256:
  `768195525ED120919769656886FF643DD0BB09605E48A869E286390AA439D6CF`
- Client stdout SHA-256:
  `08134CCE49C77F4C951F88C0314B19A843FEBF63CB84007DFC9984370D890432`
- Client stderr SHA-256:
  `E3B0C44298FC1C149AFBF4C8996FB92427AE41E4649B934CA495991B7852B855`
- Captured tree: `777` files, `93,013,858` bytes
- Root source `result.json`: absent
- Model package: absent
- Evaluator package: absent

The runner cleaned its owned CARLA processes and ports `2000..2002`; both
process and listener counts were zero after failure.  The C9 source tree is
retained as terminal failure evidence and must not be deleted, resumed,
replayed, reseeded, partially selected, joined, modeled, or scored.

## Next legal route

Use a fresh source identity and seed.  Change the information source once:
replace the lingering two-pass occluder crossings with a deterministic early
single-pass construction that (1) is outside the wearer collision corridor at
every sample, (2) creates one `0.6..1.3 s` complete target-loss interval with
the frozen pre/post trackability, and (3) leaves target, alias, wearer, model,
route, detector, and score gates unchanged.  Prove those two source properties
with a frozen prelaunch geometry falsifier; the fresh formal `instance` shard
remains the pixel-level occlusion authority inside the next one-shot source
capture.

## Claim boundary

C9 is `NOT_EVALUABLE`.  It supplies source-causality evidence for the collision
decoupling fix, but no Development metric, confirmation, generalization,
product, deployment, or safety evidence for X31.
