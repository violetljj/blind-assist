# Selective LOCAL rescue: causal gate experiment

**RESCUE_GATE_NEW_INSTANCE_NEGATIVE.** The unchanged scalar gate preserves all
41 new-instance rescued true frames and every LOCAL event onset, but rejects
only 4/15 incremental false frames (26.7%), below the fixed 50% requirement.
Retain the measured partial filtering effect; this exact package does not
provide the proposed reliable rescue admission. No cutoff retry or App change.

This report separates consumed-data mechanism discovery from one unchanged
new-instance test. A, LOCAL, their original thresholds and UNKNOWN remain
unchanged. This experiment changes only whether LOCAL may add a reminder when
A is silent; it cannot repair A's original false alerts or all remaining misses.

## Why this gate

[Fixed public/causal controls](LOCAL_RESCUE_PROTOCOL_20260923.md) use all 1,152
frames from prior transfer and composite stability. Five fixed admission rules
test temporal persistence, nominal regional range support and interval upper
bounds. None meets the joint no-event-loss/no-delay criterion on both cohorts.

On composite stability, two-frame confirmation removes 6/8 incremental false
frames but keeps only 5/12 rescued true frames, loses three LOCAL-detected
events and delays three. Nominal-range admission keeps 10/12 rescued true
frames and removes 6/8 false frames, but delays two event onsets; it removes
neither false frame in the original transfer cohort. Full per-arm results and
every incremental frame remain under `ba-local-rescue-20260923-run`.

The diagnostic finds that winning-query possible-only pixel fraction (existing
public representation slot 939) separates true and false incremental frames
within each cohort, with lower values among the true frames. This is regional
RGB/ToF interval compatibility, not measured obstacle area or return ownership.
Feature selection inspected both consumed cohorts; it is posthoc discovery.

[One scalar gate](LOCAL_RESCUE_GATE_PROTOCOL_20260923.md) selects its cutoff only
on transfer's 36 incremental true and two false frames: the midpoint between
maximum true fraction and minimum false fraction, **0.02239669393748045**.
No stability threshold adjustment, model fitting, neural training, or threshold
sweep occurs. Preserve A OR (LOCAL AND fraction <= cutoff), with no added wait.

## Consumed Development replay

| Cohort / arm | TP / FP / FN | Detected events | False segments |
|---|---|---|---|
| Transfer A |192 / 4 / 64|32/32|4|
| Transfer A OR LOCAL |228 / 6 / 28|32/32|5|
| Transfer scalar gate |228 / 4 / 28|32/32|4|
| Stability A |107 / 12 / 117|22/32|12|
| Stability A OR LOCAL |119 / 20 / 105|27/32|18|
| Stability scalar gate |119 / 16 / 105|27/32|14|

The gate retains 36/36 and 12/12 incremental true frames and removes 2/2 and
4/8 incremental false frames, with zero lost or delayed LOCAL events in either
cohort. Stability still adds four false frames and two false segments over A;
the old full-stability failure remains. This is feasibility, not independent
validation. In particular, only two transfer negatives choose the cutoff.

## New-instance design

[The frozen source](LOCAL_RESCUE_FRESH_PROTOCOL_20260923.md) changes all eight
composite sizes, elevations, lateral overlap distances and both twelve-frame
sampled distance schedules. It retains four shape families, two sizes, three
lateral relations and two approach/return schedules: 576 frames, 48 clips,
eight geometry groups. Relative signatures are disjoint from four old source
specifications. There are 288 distinct relative geometries because of paired
paths and dwell samples; 576 frames are not 576 independent scenes.

The gate JSON and selection protocol are sealed before capture. All original
model probabilities and candidate decisions are sealed before gate evaluation
opens labels. Full rendered cube-union truth retains holes and UNKNOWN. Same
renderer, map, materials and hypothetical sensor law limit source independence;
posed 0.2-second timestamps do not measure physical motion or endpoint latency.

The fixed new-source criterion requires >=8 incremental true and >=2 false
frames for evaluability, >=80% true-frame retention, >=50% false-frame removal,
and no lost or delayed LOCAL event. All A decisions are structurally retained.
It tests improvement over LOCAL, not compliance with the old full-system gate.

## Unchanged new-instance result

All 576 frames and 3,456 query labels are valid; 224 positive and 352 negative
frames form 32 positive events. All 1,728 rendered cuboids match planned bounds
within 2 mm. No frame, family, relation or trajectory was excluded.

| Arm | TP / FP / FN | Recall | Events | False segments |
|---|---|---|---|---|
| A |118 / 7 / 106|52.68%|24/32|7|
| A OR LOCAL |159 / 22 / 65|70.98%|29/32|22|
| Two-frame confirmation |149 / 13 / 75|66.52%|29/32|13|
| Frozen scalar gate |159 / 18 / 65|70.98%|29/32|18|

The scalar gate removes four false frames/four segments without losing any
true frame, detected event or onset versus LOCAL. It retains all five events
that LOCAL adds over A. Nevertheless it still adds **11 false frames/segments
over A**, and three events/65 positive frames remain missed. Minimum event
coverage stays zero; mean positive-event coverage is A 52.68%, LOCAL/gate 70.98%
and two-frame confirmation 66.52%. UNKNOWN remains 538/576 in every arm; it is
not a validated free-space decision. The gate alerts on 139 UNKNOWN frames.

Both paths remove two false frames: approach-return LOCAL 79TP/14FP becomes
79/12, and approach-dwell-return 80/8 becomes 80/6. All four removed false frames
are depth-step INSIDE cases: large entry frame 02 (3.08 m) and small exit frame
10 (3.26 m), one of each per trajectory. No posthoc subgroup replaces the full
failed gate. A single scalar feature does not resolve the remaining mechanisms.

This independently sampled source confirms a partial lossless filtering effect
in this controlled scope, but not the planned >=50% rejection effect, stable
system performance or a deployable alternative. Record the exact package as a
NEGATIVE_CONTROL while retaining its diagnostic feature and all partial gains.
Any successor needs a separately justified mechanism and new evaluation; do
not lower the acceptance criterion or tune this cutoff on these outcomes.

## Verification and resources

Seven causal/range/gate tests and five new-source tests pass. Independent
consumed auditing passes 1,403 assertions. Independent new-source auditing
passes 15,629 counted assertions: all rendered cube-union class/distance labels,
saved score/gate readouts, events/onsets, false segments and UNKNOWN; four public
feature frames are re-extracted and all 576 A decisions recomputed. Native
pixel validity is tied to the sealed materialization audit, not independently
regenerated; do not interpret the geometry audit as a second sensor simulator.

The one capture completes in 380.375 s within its 900 s budget using the logged
renderer adapter. All task-created actors/processes are released, with no
Unreal process remaining after capture. Subsequent materialization, prediction,
gate and evaluation stages succeed without retry. No paid worker, model fit,
post-fresh cutoff selection or continuous background process remains. Raw
capture, features, predictions, metrics, seals and audit evidence are retained
under the canonical F:-backed artifact tree for reproduction.

## Reproduction and durable evidence

Scripts: `local_rescue_diagnostic.py`, `local_rescue_gate.py`,
`run_local_rescue_fresh.py`, `local_rescue_fresh_evaluate.py`.
All experiment runs use `pwsh -NoProfile -File tools/ba.ps1 run research-ue
-RunSpec <saved-spec>`. Preserve saved specs and terminal receipts; replay needs
a new output/run identity, never overwrite an existing terminal.

Artifact roots under `artifacts.local/evidence/`:

- `ba-local-rescue-20260923` and `-run`: diagnostic plan, public candidate seal,
  distributions, all frame/event metrics and causal controls.
- `ba-local-rescue-gate-20260923` and `-run`: single selection and consumed replay.
- `ba-local-rescue-audit-20260923`: independent consumed recount, 1,403 assertions.
- `ba-local-rescue-fresh-20260923` and stage siblings: new source and gate test.
- `ba-local-rescue-fresh-audit-20260923` and `-run`: independent rendered-label,
  public-readout and event recount. Reusable audit scripts are
  `local_rescue_audit.py` and `local_rescue_fresh_audit.py`.

Initial diagnostic management failures (unregistered plan and nested output
overlap) occurred before scientific execution. Their receipts are retained;
the unchanged diagnostic succeeded under v3 with sibling output. Preparing an
already-existing fresh capture RunSpec returned FileExistsError without changing
it; capture used the previously frozen v1. These are not scientific retries.
