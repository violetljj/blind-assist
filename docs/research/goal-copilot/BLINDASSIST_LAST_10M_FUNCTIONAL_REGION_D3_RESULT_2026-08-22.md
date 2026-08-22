# BlindAssist Last-10m functional-region D3 confirmation (2026-08-22)

## Outcome

The frozen SAM 3 plus metric-ground functional-region rule failed independent confirmation:

- opportunity count: `24`;
- completion decisions: `1`;
- correct completions: `1`;
- false completions: `0`;
- correct coverage: `0.0416667`;
- terminal: `CONFIRMATION_SAM3_FUNCTIONAL_REGION_NOT_PASSED`.

The result is sealed and must not be replayed. It says only that exact `door` text proposals plus the frozen
ground-contact rule did not establish synthetic exact-door approachability coverage in JapaneseAlley. It does not say
that generic object detection, entrance understanding, navigation, or real-world assistance failed.

## Fully automated independent-environment selection

The official TartanAir V2 repository revision was frozen at
`0d2d145e973832742a2aaa04b7d2ebffc8d82817`. Selection ranked untouched exact-door environments by public
segmentation+depth archive bytes. Each denominator was checked before RGB access or provider calls:

| Attempt | Environment | Near | Far | Disposition |
|---|---|---:|---:|---|
| D2 | AbandonedFactory2 | 0 | 85 | denominator fail |
| D2B | OldBrickHouseNight | 8 | 361 | denominator fail |
| D2C | CoalMine | 10 | 196 | denominator fail |
| D2D | HQWesternSaloon | 251 | 2506 | qualified; runtime stopped before evaluator |
| D3 | JapaneseAlley | 59 | 3287 | qualified and sealed |

D2D stopped at case 30/48 because the constrained ground-plane estimator raised `no supported near-horizontal ground
plane`. No prediction or evaluation receipt was produced. It is `NOT_EVALUABLE_RUNTIME`, not a negative. The provider
was then changed mechanically to fail closed to no completion when ground is unavailable. D3 was untouched by all
earlier provider observations, and its public-depth ground preflight passed `48/48` before the only model run.

D3 used official SAM 3 source revision `8f0b7f4d4e7eda2ed606ebde6702c93359ad01da`, checkpoint SHA-256
`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`, exact prompt `door`, confidence
`0.105`, mask-height minimum `0.4`, ground-contact minimum `50` pixels, and contact-depth maximum `2.0m`.

## Topology-runtime follow-up

To replace bbox+depth proxy truth with actual room topology and reachable positions, a task-owned Docker runtime was
built from AI2-THOR `5.0.0` and the public ProcTHOR-10K test split at revision
`439193522244720b86d8c81cde2e51e3a4d150cf`. The machine's RTX GPU was visible to CUDA in Docker. AI2-THOR's
797MB CloudRendering build downloaded and unpacked successfully. Two Docker Desktop compatibility issues were isolated:
the default FIFO server was replaced by the supported WSGI server, and single-process release locking used the same
no-op policy as AI2-THOR on Windows.

The remaining terminal was not repairable inside the task container: `vulkaninfo` exposed only CPU `llvmpipe`, not the
WSL `/dev/dxg` GPU, and Unity CloudRendering exited with code `-11`. Ubuntu 24.04 Mesa 25.2 produced the same Vulkan
device result. Therefore ProcTHOR rendering/reachable-position evaluation is
`NOT_EVALUABLE_VULKAN_RUNTIME`; no topology algorithm verdict was produced.

## Decision

`GOAL_SEMANTIC_SAM3_GROUND_CONTACT_APPROACHABILITY_NOT_CONFIRMED`.

Do not tune or replay D3. The next scientifically supported algorithm lane needs an action-responsive/topological
observer on a runtime that can actually render ProcTHOR (native Linux Vulkan GPU or another public simulator), or a
public trajectory dataset whose future motion/topology can serve as private truth. The claim ceiling remains:

`SYNTHETIC_EXACT_DOOR_APPROACHABILITY_ONLY_NO_REAL_WORLD_ENTRANCE_TRAVERSABILITY_NAVIGATION_PRODUCT_OR_SAFETY_CLAIM`.
