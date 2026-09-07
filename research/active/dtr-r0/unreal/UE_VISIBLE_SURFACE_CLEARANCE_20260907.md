# UE visible-surface clearance Development

## Question and fixed experiment

Does representing extra clearance around visible depth surfaces and offering a
wider bypass resolve the late-stop contact without losing useful completion?
This is consumed, controlled synthetic Development, not fresh confirmation.

The existing 10 Hz late-stop run first contacts at 4.91514552 s in both candidate
arms, after moving to y=-0.72 m. Its front depth is absent while center-route
depth still sees the stopped obstacle. Candidate DTR makes zero interventions
over 97 frames; a current measured footprint at 4.8 s conflicts with right-side
alternatives but not the selected left bypass. The fixed evaluator requires
.70+.28=.98 m center separation for this pedestrian/wearer pair. These facts
support a geometry/action-space gap rather than absent perception alone.

Implement `CANDIDATE_CLEARANCE` as a separate depth-only selector. Increase the
visible-surface corridor half-width from .30 to .70 m and add +/-1.20 m targets
to the existing targets. The additional .40 m is an explicit heuristic under
test, not a measured whole-body estimate or a safety bound. Forward distance,
height filtering, sample count, velocity, braking, detector, DTR predictor,
collision geometry, scene, cadence, goals and time limits remain fixed.
DTR candidates are computed/logged but do not select actions in this arm.
The two coupled geometry changes form one mechanism; this run cannot attribute
their individual contributions or establish DTR gain.

Run once on all eight existing Development scenes (16 actual branches), with
fresh straight controls. Compare against the retained 20260906 depth and DTR
results, explicitly a historical rerendered comparison. Also replay both depth
policies on identical retained depth/poses to check where decisions change;
that replay does not establish counterfactual trajectory outcomes.

Retain as a challenger if late-stop becomes contact-free with goal completion
and previous depth successes remain. Report all contacts, goal/time/stationary
cost, missing footprint support and regressions, including obstructed bypass.
No gain means reject this fixed variant; no threshold search follows. Source
failure is NOT_EVALUABLE, with durable receipts and normal owned-process cleanup.
No default-controller promotion or protected split access is part of this run.

Output: `artifacts.local/unreal/ue-visible-clearance-20260907-a`.
Runtime: existing project CUDA PyTorch on RTX 5060 Laptop; scalar geometry and
receipt analysis are TASK_NOT_GPU_SUITABLE. The canonical artifact junction
was verified before execution. The launcher binds sources, bank and map hashes
and releases its owned process trees, lock and worker port in `finally`.

## Completed result

All 16 actual branches completed; all eight fresh straight controls passed.
The fixed variant achieved **2/8**, versus retained depth **5/8** and retained
DTR **6/8**. No scenario gained success; three former depth successes regressed.
All contact counts below refer only to declared scenario proxies.

| Scenario | Previous depth | Buffered depth | Time, previous / buffered (s) |
| --- | --- | --- | --- |
| Late crossing, intercept | Contact | Contact | 10.6 / 9.8 |
| Late crossing, phase miss | Success | Success | 8.1 / 8.1 |
| Late stop | Contact | No goal, no declared contact | 9.6 / 14.0 |
| Continued walking | Success | No goal, no declared contact | 8.8 / 14.0 |
| 12 cm obstacle | Success | No goal, no declared contact | 9.6 / 14.0 |
| 4 mm relief | Success | Success | 8.1 / 8.1 |
| Left bypass open | Success | Contact | 9.6 / 11.7 |
| Bypass obstructed | Contact | Contact | 11.7 / 11.7 |

The late-stop and continued-walking runs both stopped forward progress at
x=3.70 m, y approximately -1.20 m from t=5.0 through the 14 s limit. `BRAKE`
retains an exponentially small lateral centering velocity, so the evaluator's
strict applied-stationary measure is 3.5 s, not the 9 s without forward progress.
The low-obstacle run records 2.1 s under that same stationary definition.
There were 922 assisted observations: 397 had usable action footprints and
525 lacked them. These are logged support counts, not CLEAR labels or an
assertion that footprints controlled this depth-only arm.

### Static-scene attribution changes the interpretation

At t=5.0, both stop/continue recordings have the same front depth, 1.37501384 m,
despite different pedestrian motion. Independent depth reconstruction finds
1,129 samples in the buffered front query, including 248 below 1.4 m. The
blocking plane is world x=31.075014 m, matching the static tree planter front
at x=31.075 m in `build_street_lab.py`. The native sensor image was inspected.

That planter spans world y=[-3.025,-1.175] m, so the proposed -1.20 m path
actually enters it. Braking is appropriate; this is not merely a false positive
caused by margin expansion. The fixed bypass target is not replanned once
blocked. This run therefore fails as a usable controller, but does not isolate
or falsify all full-body clearance representations or sudden-stop perception.

The late-stop evaluator actor list contains only the pedestrian. Static
planters are rendered and sensed but absent from its declared-contact score.
Consequently, zero declared contact cannot certify the expanded path against
all scene geometry. This newly demonstrated coverage gap is scoped to physical
scene/action support; it does not erase the preserved eight-scenario results.

## Decision and next discriminating work

**Reject this fixed variant; keep it as a negative control with no default
promotion.** Preserve both the original proxy-contact failure and this blocked
bypass. Do not rescue it by changing the consumed margin or shrinking bodies.
The next useful comparison needs static-scene contact coverage and a traversable
witness for the admitted action space, followed by a controller that can replan
a blocked bypass. A native contact/occupancy check and a new, explicitly versioned
comparison can resolve that gap; this run does not perform them.

The structured pending disposition is in
`ue_visible_surface_clearance_disposition_20260907.json`. Normal registration
was attempted and failed on the pre-existing `experiments/index.jsonl:252`
input-fingerprint mismatch. The current registry files contain unrelated WIP;
no manual index append, inheritance publication or unrelated repair was made.
This result is a recorded Development finding, not a newly registered terminal.

## Evidence and validation

- `artifacts.local/unreal/ue-visible-clearance-20260907-a/evaluation.json`:
  full outcomes, applied dwell, source support and measured worker latency.
- `artifacts.local/unreal/ue-visible-clearance-20260907-analysis/comparison.json`:
  input-identity checks, all outcomes, regressions and receipt hashes.
- `artifacts.local/unreal/ue-visible-clearance-20260907-analysis/fixed-pose-replay.json`:
  all 1,425 historical depth frames reproduced with zero baseline action
  mismatches. Counterfactual commands are not scored as trajectories.
- `artifacts.local/unreal/ue-visible-clearance-20260907-analysis/static-blocker.json`:
  independent depth-plane attribution, geometry definition and source hashes.
- `artifacts.local/unreal/ue-visible-clearance-20260907-analysis/late-stop-diagnosis.png`:
  declared pedestrian clearance and actual forward progress.

The three new clearance counterexamples, 12 existing candidate-geometry tests,
five resume tests, five evaluator tests and four entrypoint-routing tests passed.
The existing source/model/map identity comparison passed. The native launcher
completed and its process-release receipt reports all owned trees and the worker
port released. Payloads remain in the canonical ignored artifact tree.
