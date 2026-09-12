# Current research decisions

Updated: 2026-09-07

Status: `L10_R0_ACTIVE / DTR_R2_DYNAMIC_RETAINED`

This page owns cross-route priorities and decision criteria. Each route current
owns its latest baseline, exact metrics, source boundaries and experiment status.
Historical gates and successor suggestions apply to named scopes; new hypotheses
or evaluation criteria need a rationale and a useful check, not prior success.

Across routes, prioritize useful effect over degree of novelty. Mature methods,
integration and incremental improvements remain eligible alongside new mechanisms.
Retain changes for demonstrated task benefit with acceptable errors, coverage and
cost; prefer simplicity and stability when effects are comparable. Major innovation
is not a prerequisite, and novelty alone does not justify retaining a method.

## L10: useful commitment under partial evidence

The present question is how to retain correct target recovery while reducing wrong
commitments without hiding abstentions or observation/reference costs.

Use the [L10 current](../research/active/l10-r0/CURRENT.md) for the paired fixed,
triggered and geometric-verifier baselines. Keep the triggered policy and episode
harness available within their recorded Development limits. An unconditional
verification gate must justify the correct coverage it removes.

The [consumed paired diagnosis](../research/WORKFLOW_UPGRADE_20260905.md) separates
two lost correct bindings with target-box support from three without it; error
reduction has no baseline error opportunities. Explore an evidence representation
that distinguishes availability, contradiction and extent. Integrate only a demonstrated useful
change; seek appropriate unchanged-method confirmation before expanding its claim.
No new street source, threshold rescue or protected outcome access is authorized
by this navigation update. The existing frozen result remains unchanged.

## Forward obstacle awareness complementary to a cane

**Active architecture, corrected 2026-09-13: one RGB camera + ToF + Radar + IMU,
simulation only.** The next capability is observable cross-sensor association and
current walking-corridor localization. Compare matched ToF+Radar+IMU against the
same pipeline with RGB spatial association. Preserve independently valid sensor
support and UNKNOWN. RGB is not a second camera, stereo depth, or a substitute
for Radar. IMU rotation is not metric translation or future walking intention.
See the [four-sensor mainline and stop points](../research/active/dtr-r0/nearfield/FOUR_SENSOR_MAINLINE.md).

The depth-front-end diagnostics described below are historical and scoped to
their named branches. MZ101--106 omitted Radar and used stereo; their failures
must not be presented as the bottleneck or ceiling of the four-sensor mainline.

The current goal is cane-complementary, class-agnostic forward obstacle
awareness (盲杖互补的类别无关前视障碍感知). Prioritize walls/large forward barriers,
body/head protrusions and suspended hazards; then multi-height poles/supports.
Knee-height hazards remain relevant. Ultra-low obstacles are secondary
compatibility evidence, not a headline optimization target. The wearer chooses
movement. Earlier warning and dynamic coverage still require temporal validation.
Use the [current route](../research/active/dtr-r0/CURRENT.md) for active evidence.
Historical results retain their original denominators. Do not infer actual cane
coverage or retroactively relabel old cells as VISION_COMPLEMENTARY successes.

Compare sampling, spatial aggregation and alert logic on identical inputs.
Report missed/false alerts, direction/height, UNKNOWN and processing cost. Keep
native UE depth and simulator calibration explicitly separate from predicted RGB
depth. Geometry-reference agreement is not independently labeled obstacle
accuracy. The first probe preserved more procedural thin-surface evidence but
exposed a correlated-artifact false alert and complete loss of the native near
alert opportunities with the fixed existing RGB depth model. This supports a
frontend/geometry diagnosis before training or further compression. The subsequent
[cached four-arm probe](../research/active/dtr-r0/nearfield/GROUND_ANCHOR_20260907.md)
recovered 21/26 native-positive observations with ground-relative height, versus
1/26 with scale alone and 0/26 raw, all with 0 FP. Low-boundary retention remained
123/807; retain this as secondary structural diagnosis. All 11 cached
frames were processed without inference or real-time replay; these are consumed
synthetic reference diagnostics, not independent obstacle accuracy. The subsequent
[support comparison](../research/active/dtr-r0/nearfield/SURFACE_SUPPORT_20260907.md)
found no gain from elevation or dual support: all 21/26, 0 FP. Four misses have no
eligible near candidate; the remaining weak candidates lack eligible triples.
Keep depth support. On [18 distinct views](../research/active/dtr-r0/nearfield/DISTINCT_VIEWS_20260907.md),
ground geometry improves22/87 to37/87 with0 FP, but failed ground fits suppress
15 raw wall positives. Retain the component; ground availability must not be
the sole gate for near alerts. [Evidence fusion](../research/active/dtr-r0/nearfield/EVIDENCE_FUSION_20260907.md)
preserves these positives: union and conditional switching both reach52/87
legacy diagnostic cells with0 FP, while reporting unknown height separately.
The suspended-bar reference at1.96 m is estimated around14.29 m; prioritize
this forward-geometry failure over the downward-pitch low-block failure.

Historical DTR/CARLA terminals remain unchanged. Their capture and protected-run
continuations are parked; this perception question does not reopen frozen runs.

## Shared execution choices

- **Explore:** test one explanatory hypothesis; necessary coupled edits and disclosed
  consumed Development are allowed. Keep comparisons reconstructable.
- **Confirm:** fix the method, comparison, criteria and retry/access rules before
  outcomes; choose independence appropriate to the claim.
- **Engineering:** resolve observed correctness, runtime or recovery bottlenecks.
  Classify implementation/source failures separately from method evidence.
- Each outcome determines continue, revise, integrate or stop. Complete remaining
  authorized delivery and resource release after the individual experiment ends.
- Choose baselines, controls and ablations for the discrepancy being tested; their
  historical labels alone do not make them universally mandatory.
- L10 and DTR do not wait for, modify or validate one another. Preserve concurrent WIP.

The [research workflow](../research/WORKFLOW.md) contains the brief and command.
Do not add governance or tests without a decision benefit or named material risk.
No local result establishes arrival, handoff, natural-camera reliability, user
benefit or safety beyond its own evidence. `UNKNOWN` remains distinct from CLEAR.

Detailed trajectories remain in owning ledgers/results and Git
`daf5720064d98a93b75336469d18e9a2fe0023e5:docs/CURRENT_DECISION.md`.
