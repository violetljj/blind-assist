# CITY-CROSSREGION-V1

2026-09-08. Next static data-engine stage. The 36-frame field remains a usable
engineering release; this protocol does not reopen its labels or model results.

## Question and comparison

Does repeated BODY/HEAD conflict geometry across physically separated city
backgrounds improve unseen-region HEAD recall and localization, without raising
false positives on clearance-safe horizontal structures?

The inspected coverage1198 baseline has HEAD TP 12/32, FP 1/32 and positive
support IoU 0.185173 on its **consumed** EVAL64. The later Dice recipe reaches
TP 11/32, FP 3/32 and IoU 0.19299, so it is not the default successor. These
numbers identify the gap; neither this result nor field capture proves its cause.
See [coverage pilot](../../research/active/dtr-r0/nearfield/CITY_COVERAGE64_20260908.md)
and [Dice comparison](../../research/active/dtr-r0/nearfield/CITY_SUPPORT_DICE_20260908.md).

## Fixed first design

| Split | Regions | Frames | Intended HEAD positive / negative |
| --- | ---: | ---: | ---: |
| TRAIN | 4 | 192 | 96 / 96 |
| DEV | 1 | 48 | 24 / 24 |
| unseen-region TEST | 2 | 96 | 48 / 48 |

Each region has sidewalk, intersection and narrow-passage sites. Each site has
four identical-camera quartets: NONE, BODY_ONLY, HEAD_ONLY, BOTH. The four
small perturbations are fixed lateral offsets ±0.08 m and yaw offsets ±2 degrees.
The main fixture's geometry and relative anchor are reused across regions;
configured instance IDs and physical surroundings remain distinct.

The [machine-readable design](crossregion-v1.json) budgets 336 keyframes.
Expected class bits are **placement intent**, never evaluator truth. Native
depth, independent target identity and measured floors decide label admission.
Any incomplete/unreliable quartet is excluded as a unit from comparative metrics,
with exclusions and surviving region/scene/condition coverage reported.

## Counterfactual contract

Camera, light configuration, scene, context objects, carrier posts and clamps
remain identical within each quartet. BODY changes only its target arm; HEAD
changes only its target crossbar. Support structures do not appear/disappear
with the target. Changing a target can legitimately change shadows/reflections;
record background pixel differences rather than asserting bit-identical images.

HEAD comparisons are NONE→HEAD_ONLY and BODY_ONLY→BOTH. BODY comparisons are
NONE→BODY_ONLY and HEAD_ONLY→BOTH. Score only verified 0→1 ground-truth pairs:

`CFA = count(prediction_negative == 0 AND prediction_positive == 1) / eligible_pairs`

Always-positive and always-negative predictors both score zero. A reversed flip
is wrong. Removing the obstacle is the same static pair in reverse order, not
another independent sample. Report mean probability change, recall/FPR, support
IoU, both TEST regions separately and a region-macro summary. Frames/perturbations
within a route are correlated; do not present 96 frames as 96 independent regions.
The primary CFA denominator is at most 48 HEAD pairs on TEST, not the old 32
positive frames. Empty eligibility yields null, not zero accuracy.

CFA measures response to a controlled visual intervention. It does not by itself
prove that the model reasons about head clearance rather than object presence,
shadows or other cues caused by the intervention.

## HEAD hard negatives

Before model results, assign and source-check clearance-safe horizontal context
across sites: lateral signs/arms outside the corridor, sufficiently high canopies,
far connectors, central image projections outside the physical corridor, facade
lines/windows/shadows, and BODY occupancy with horizontal background structure.
Keep each site's context unchanged across its four conditions. The shared high
carrier beam is one controlled high-clearance negative; it is not a substitute
for admitting the broader hard-negative strata. Native labels must verify that
NONE and BODY_ONLY remain HEAD-negative. Record actual stratum coverage; do not
claim all listed contexts have been captured merely because they are planned.

## Geographic and observation isolation

Seven disjoint coordinate boxes are necessary but insufficient. Source admission
must inventory visible physical background instances across all sampled views,
including distant buildings, road/sidewalk pieces, lamps and HLOD provenance.
No such instance may cross TRAIN/DEV/TEST. Descriptor membership and a near-camera
loaded-component list are not visibility proofs. Whole regions and complete
quartets stay in one split, with a check against all earlier training-source views.

City Sample shares its mesh/material library. Physical-instance and region
separation therefore does not imply unseen meshes, textures or architectural
modules. Report shared assets separately; never silently advertise the stronger
asset-disjoint claim. If the installed maps cannot satisfy the physical-view
isolation contract, retain candidate status and change the source layout before
training rather than weaken the claim after outcomes.

## Training and evaluation stage

After source admission and completed acquisition, compare the existing
coverage1198 recipe against the same recipe with the new 192 TRAIN frames.
Keep original initialization, full-parameter optimizer, preprocessing, loss and
2000-step budget matched; record actual old/new sample exposures. Freeze the
exact baseline checkpoint/input hashes and DEV-only threshold selection before
evaluating either new TEST region. No TEST-based site, threshold, checkpoint or
hard-negative choice. A fixed TEST pass follows training; no automatic extra fits.

Use the existing coverage1198 checkpoint as A without refitting. B starts from
the original G13 seed17 initialization, not A. Both A and B receive the same
selector on the new DEV48: empirical FPR≤10%, maximum recall, lower FPR, then
higher threshold. Fix the support threshold at 0.5. Recalibrating A here means
the historical 12/32 result is context, not an interchangeable new TEST baseline.

Treat the proposed targets as decision targets, not predictions: HEAD recall
≥75%, FPR≤6.25%, positive support IoU≥0.35, alongside improved CFA. With complete
new TEST coverage this corresponds to ≥36/48 positives and ≤3/48 false positives.
The old 24/32 and 2/32 example cannot be copied as this dataset's denominator.
Failure to improve retains the data-engine capability and narrows the hypothesis;
it does not invalidate the already delivered collection field.

## Current implementation and boundary

Engineering preflight on 2026-09-08:

- [Source audit](CROSSREGION_SOURCE_AUDIT_20260908.md) identifies seven Big City
  descriptor candidates with a minimum 1,531 m gap between their 240 m boxes.
  All remain `NOT_ADMITTED`: descriptors do not establish walkability, the three
  actual route types, or isolation of distant buildings and HLOD constituents.
- The first 16-frame TRAIN-only canary captured successfully but admitted only
  2/4 quartets: the crossbar penetrated its clamps. Shortening the crossbar to
  a 1.56 m butt joint repaired the geometry without relaxing label thresholds.
  The fresh second capture admits **16/16 frames and 4/4 complete quartets**;
  all native BODY/HEAD bits match their intended conditions.
- Both attempts are retained (32 engineering frames, outside the 336 cohort).
  Evidence: `artifacts.local/nearfield/city-crossregion-v1-20260908/`, including
  `canary-capture-v2/`, `canary-labels-v2/` and `pair-check-v2/result.json`.
  The four `pair-check-v2/cxr_canary_sidewalk_p*.png` sheets show actual pairs.
- Outside the union of target projections, mean RGB absolute differences are
  0.83–1.10 on the 0–255 scale; at most 0.241% of pixels have mean RGB difference
  above 10. These are measured drift diagnostics, not a predeclared acceptance
  threshold or a claim of identical pixels. Camera and common context checks pass.

Next is source-only admission of actual routes and conservative visible physical
instance sets, including distant/HLOD background and overlap with prior training.
The full 336-frame acquisition, model training and model CFA measurement have
not started. No cross-region improvement is claimed by this engineering canary.

`tools/city_crossregion.py` provides the fixed intervention compiler and paired
metric. Full compilation refuses unadmitted visible-background isolation,
overlapping regions, cross-split instance reuse and incorrect 336-frame coverage.
Eight focused tests cover carrier/camera invariance, budget, leakage, constant
predictors, missing/UNKNOWN quartets and context mismatches.

A separate 16-frame engineering canary uses an already consumed TRAIN sidewalk
to check capture/label/pair mechanics. It is not a new TEST region, not part of
the 336-frame research cohort, and is not evaluated by a model. Its budget does
not authorize further training or enlarge the research cohort.
Mechanical scene repairs retain failed receipts and replay the same 16-position
engineering layout; they do not select viewpoints using model outcomes.

Commands:

```powershell
python tools/city_crossregion.py compile --source <admitted-source.json> --template <capture-template.json> --output <fresh-artifact-spec.json>
python tools/city_crossregion.py compile --source <admitted-source.json> --template <capture-template.json> --output <fresh-artifact-directory> --by-region
python tools/verify_city_counterfactual.py --capture <capture> --labels <native-labels> --output <fresh-pair-checks>
python tools/city_crossregion.py score --rows <verified-prediction-rows.json> --output <fresh-score.json>
```

Use `--by-region` for acquisition: it writes seven 48-frame specifications and a
cohort index manifest. Each session loads its own region; the capture launcher
rejects a combined multi-region specification. This preserves the 336 sample
identities without asking one UE process to keep seven distant districts loaded.

Source scouting and secondary-worker provisioning are recorded in
[the source collection report](SOURCE_COLLECTION_20260908.md). They are separate
from the formal cohort and do not change region admission automatically.

The scorer takes one split at a time. Each row joins `region_id`, `pair_id`,
`condition`, `split`, `eligible`, verified `truth`, `context_hash`, `prediction`
and `probability` (BODY/HEAD arrays). The source verifier writes metric-input
rows without predictions; inference must join them by preserved sample identity.
Do not feed evaluator truth/context hashes as model observations.

Mass traffic/crowds, DTR events and L10 remain deferred. Epic documents both
Small/Big City and Mass AI traffic/crowds, but their availability does not mean
this project's dynamic collector has been built or validated.
[Epic City Sample documentation](https://dev.epicgames.com/documentation/unreal-engine/city-sample-project-unreal-engine-demonstration).
