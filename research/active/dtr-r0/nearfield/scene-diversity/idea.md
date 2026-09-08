# Equal-budget scene coverage contrast

2026-09-08. Candidate design after HEAD-P1; not a frozen run protocol and no
acquisition, training or held-out model evaluation has started for this design.
Coverage1198 remains retained. This proposal is separate from CITY-CROSSREGION-V1
and does not modify its existing336-frame source contract or concurrent work.

## What the evidence currently supports

| Experiment | Retained inference | Not established |
|---|---|---|
| Coverage64 | Limited gain from additional local coverage on consumed EVAL | Independent-region generalization or a unique failure cause |
| Support Dice | Better TRAIN overlap did not deliver the needed EVAL recognition | All support supervision or all Dice formulations fail |
| HEAD-S1 | Tested max/top6/LSE did not rescue current HEAD transfer | Backbone alone causes failure; learned readout is harmless |
| HEAD-P1 diagnosis | Appearance-capture interventions produced heterogeneous damage; A already failed substantially | Pure appearance shift dominates cross-region errors |
| HEAD-P1 matched fits | Fixed .1 consistency did not outperform matched paired ERM on main EVAL | All consistency or augmentation fails; sampling caused BODY loss |

See [completed HEAD-P1 results](../HEAD_PAIRED_RESULTS_20260908.md),
[HEAD-S1](../HEAD_S1_20260908.md),
[Coverage64](../CITY_COVERAGE64_20260908.md), and
[Dice](../CITY_SUPPORT_DICE_20260908.md).
Correct signed relations, positive support and actual alerts must precede any
claim based on small appearance differences. Equality can represent shared failure.

## Question and minimal candidate comparison

At equal new-image count, geometry-template distribution and training exposure,
does spreading ordinary supervised examples across more distinct source regions
improve transfer compared with concentrating them within fewer regions?

Use the existing four proposed TRAIN-region candidates as the initial source
pool, subject to source admission. Do not invent eight available TRAIN regions.

| Surface | Concentrated ERM | Distributed ERM |
|---|---|---|
| New RGB images | 256 | 256 |
| Complete geometry quartets | 64 | 64 |
| Proposed TRAIN-region count | 2 | 4 |
| Quartets per region | 32 | 16 |
| Relation totals | CLEAR/BODY_ONLY/HEAD_ONLY/BOTH:64 each | Identical |
| Fixture families | Same four families and counts | Identical |
| Model/init/loss/optimizer | Original G13 seed17, original full-fit BCE recipe | Identical |
| Candidate fit budget | Final2000 steps, batch32 | Identical |

Freeze a shared list of64 geometry templates before assigning source placement.
Match fixture dimensions, range, body/head clearance, relative target bearing,
camera height/pitch/FOV, and nuisance lighting/material settings. Balance route
types and viewing-direction strata across arms where physically feasible. No
arm-specific light/material randomization or added algorithmic objective.
Global placements and resulting backgrounds are the experimental treatment;
they are not expected to have pixel-equal masks across regions. Native labels
must independently verify all four relations and preserve UNKNOWN handling.

Select the concentrated two-region subset by a documented source-only rule
before any new model predictions. Freeze exact source identities and poses;
do not search subsets based on task outcomes. Distribute independent camera
positions within the admitted regions rather than calling tiny yaw jitter new
scenes. Region count alone is not visual-diversity evidence: retain annotated
RGB overviews, route/heading strata and visible-background inventories.

## Matched sampling, including BODY exposure

Candidate common schedule:20 uniform original1198 draws plus three complete
new quartets per step (12 images). Both arms consume the exact same sequence
of original indices and paired geometry-template indices. Select three distinct
quartets per batch and include all four relations in each selected quartet.
This gives40000 original and24000 new draws per fit, with exactly12000 BODY
positives and12000 HEAD positives within new-image exposure. Original draws
also match arm to arm. Record per-image, template, family and region exposure.

This deliberately keeps an explicit fixed new-data fraction; it does not claim
37.5% is optimal. Comparing these two arms isolates their allocation policy
within this design. Comparison to historical Coverage1198 remains contextual:
new images, sampling and exposure differ from that old fit. Do not attribute
any BODY recovery uniquely to balanced sampling without a separate contrast.

## Common evaluation and limits

Reserve the same two proposed unseen regions for both arms, independently of
all old and new TRAIN views. Candidate assessment is96 frames:12 complete
quartets per region, balanced across the four families;48 positives and48
negatives per head overall. Exact poses and the valid denominator must be
admitted and frozen before model access. Do not drop failed/UNKNOWN frames
after seeing scores. Source failures require repair before the run, not a
post-result denominator change. Reserve candidate05 rather than implicitly
consuming it as another training location.

Keep the original DEV128 threshold selector unchanged for both final models
(per-head empirical FPR<=.1 and the existing tie rules). Old DEV/EVAL64 remain
consumed Development/regression; neither becomes blind again. Report new-region
results using those fixed DEV cutoffs, along with threshold-independent AUC/AP,
signed quartet relations, positive support recall, IoU and false activation.
Do not tune on the new regions, even if their FPR is high. If a different DEV
allocation is chosen before source freeze, version the protocol explicitly.

Report BODY and HEAD separately, each held-out region separately, and pooled
counts with actual denominators. Two held-out regions are two region clusters,
not96 independent generalization trials. Do not derive image-iid confidence or
significance from these correlated variants. No success claim from improved
HEAD alone if false alarms or BODY regress materially. AUC, relation correctness
and positive support should explain any selected-threshold gain; equality alone
is never success. Freeze concrete decision criteria before training.

A single concentrated subset versus all four regions also changes which source
identities occur. It can support a practical allocation result on this source
pool, not separate generic diversity from the quality of particular regions.
If the result is decision-changing, a complementary concentrated subset would
be a distinct follow-up; it is not automatically added to this two-fit budget.
Shared City Sample assets further limit claims to regional transfer within the
simulator, not independent asset libraries, cities or real devices.

## Source feasibility before freezing this candidate

Current source audit identifies seven descriptor-backed BigCity candidates,
all NOT_ADMITTED. The16-frame engineering canary validates mechanics on an old
TRAIN sidewalk; it does not admit the proposed regions. The background audit
still reports visibility isolation UNVERIFIED, including HLOD source membership.
Coordinate-box gaps do not establish nonoverlapping rendered backgrounds.

Read the current source work instead of duplicating it:

- [Cross-region source design](../../../../../experiments/city-field/CITY_CROSSREGION_V1.md)
- [Source audit](../../../../../experiments/city-field/CROSSREGION_SOURCE_AUDIT_20260908.md)
- [Background audit](../../../../../experiments/city-field/BACKGROUND_FRUSTUM_AUDIT_20260908.md)

Next source-only evidence is actual floor/obstruction and street-content
reconnaissance, then final-pose visible-background/physical-instance isolation
against old TRAIN and the new evaluation regions. Match scene-content strata
before outcome access. If fewer than four visibly distinct TRAIN regions or
two isolated evaluation regions are available, the proposed comparison is not
ready; do not silently relabel nearby views as new regions or weaken the claim.

Before execution, replace proposed region counts with admitted exact sources,
freeze templates/splits/schedule/evaluator and register a new versioned run.
This file records the candidate and its unresolved source dependencies only.

Implementation and source-only evidence are recorded in
[source progress](SOURCE_PROGRESS_20260908.md). The shared allocation, exposure
schedule and geometry library exist; this does not change the candidate's
NOT_ADMITTED status or start either fit.

On2026-09-09 the user authorized a separate small BODY-QUERY counterfactual
source pilot. Its [120-frame result](../BODY_QUERY_SCENES_RESULTS_20260909.md)
owns that completed collection. It is not execution of this256+256+96 candidate
or a claim that this candidate's strict background-isolation condition passed.
