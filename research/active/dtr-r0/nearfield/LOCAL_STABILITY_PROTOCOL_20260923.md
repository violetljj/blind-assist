# Frozen LOCAL: composite shape, size and posed approach stability

EXPLORE: one new 576-frame controlled Development capture. Keep A, RAW, LOCAL,
all six queries, public RGB/ToF extraction, model hashes and thresholds frozen.
No training, interval/cutoff/feature change, event smoothing or App integration.

Prior transfer showed LOCAL's useful early detections and 36 rescued frames,
with two added false frames and one false segment. The last correspondence
check did not remove those errors even with exact contributor association.
This experiment changes the source to test the retained capability, rather
than pursuing an error-removal filter. Earlier results remain in their scope.

## Source and opportunity

Eight new composite geometries: four original family identifiers, each with
small and large dimensions. Shape forms are L plate, depth step, inverted L
plate and T bar, each rendered as the union of two opaque axis-aligned cubes.
They test geometric composition, not semantic recognition or natural objects.
Labels intersect each actual rendered cube separately; enclosing empty space
must not become positive truth. Retain all partial visibility/UNKNOWN frames.

Each geometry crosses INSIDE (14 cm intrusion), BOUNDARY (1.2 cm intrusion)
and OUTSIDE (10 cm gap), with two fixed 12-pose schedules at nominal 0.2 s:

- approach_return: 4.1, 3.5, 3.15, 2.9, 2.55, 2.0, 1.45, 1.05, 1.8, 2.65, 3.2, 3.9 m;
- approach_dwell_return: 4.1, 3.5, 3.15, 2.9, 2.55, 2.55, 2.55, 2.55, 2.55, 2.9, 3.2, 3.9 m.

Depth is the foremost component surface's nominal optical Z. Objects stay fixed
within each clip. Dwell samples have distinct sensor-noise draws. Timestamped
poses are not continuous physical motion, frame-rate evidence or real stopping
time. No temporal model is added. Background/target materials remain fixed
across the two schedules and sizes within each family. Shapes, dimensions and
paths jointly change from the old cohort: this is transfer, not causal isolation
of one factor. Shared renderer, primitive assets and sensor law limit independence.

The full source has 48 clips, 224 positive/352 negative frames and 32 positive
events, 16 per trajectory. Source-only geometry checks precede capture, with
exact relative-signature comparison to old query, coverage and LOCAL sources.
The protocol fixes all dimensions and schedules in local_stability_spec.py.

## Evaluation and decision

Prediction sees only public RGB/ToF and fixed queries. Identity, shape, size,
trajectory, object geometry and timing metadata never enter learned features.
Seal public inputs, model/source hashes, A states and all probabilities before
opening evaluation labels. Recompute event truth from full rendered cube union;
native visible masks audit geometry/visibility, not independent metric support.

Report A, A OR RAW, A OR LOCAL and standalone controls: TP/FP/FN, recall,
precision/FPR, all UNKNOWN/silent frames, positive-event detection, first-alert
times, false segments/inclusive sampled duration, positive-frame coverage and
internal alert gaps. Include every family, size, trajectory, layer, relation and
all eight base geometries, plus individual gains and losses against A and RAW.

Primary stability requires all of:

1. LOCAL advances detection by at least one sample (0.2 s), or recovers a missed
   event, versus A on >=8/32 events, including >=3 in each trajectory. A frames,
   detected events and first alerts are retained by OR. If fewer than eight
   events offer opportunity (A not alerting at true entry, including misses),
   the timing clause is NOT_EVALUABLE, not evidence against early benefit.
2. Keep the preceding 48-clip cost budget: LOCAL versus A adds <=2 false frames
   and <=1 false segment in total. Also each trajectory adds <=2 false frames
   and <=1 false segment. These are research usefulness costs, not safety bounds.
3. LOCAL versus RAW union gains >=5 percentage points recall overall, with
   <=1 added false frame and <=1 added false segment; positive-frame net gains
   in >=4/8 geometry groups, and neither BODY nor HEAD recall drops >5 points.

Frame/event timing and error distributions, rather than PASS alone, govern
interpretation. A valid failed clause is a scoped negative stability result;
it does not erase old-source component value. All passing retains a broader
controlled COMPONENT, never a default product or safety claim. No subgroup
exclusion, new operating point, fresh confirmation claim or rescue run.

## Execution and stop

Capture budget: one 576-frame run, 900 s, the previous map/plugin/render readiness
and 32/16 settle calls, on an available local GPU with recorded backend and
process/actor release. All payloads remain in artifacts.local, an existing F:
junction. Fit-free HGB inference and evidence accounting use the prior CPU
implementation with TASK_NOT_GPU_SUITABLE recorded. No paid worker allocation.

Every scientific stage uses research-ue RunSpecs and exact input role contracts.
One materialization, one frozen prediction pass, one evaluation and a focused
independent geometry/output/event audit. Mechanical failures keep receipts and
partial output; repair only an observed implementation/source-identity defect.
Never replace completed outputs, expand the source or tune to their outcomes.
Finish result/inheritance/ledger and scoped delivery, then stop this experiment.
