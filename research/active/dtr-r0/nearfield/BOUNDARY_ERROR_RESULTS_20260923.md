# Boundary-error audit: coarse supervision and constraint violations coexist

2026-09-23 completed EXPLORE diagnostic under the
[bounded protocol](BOUNDARY_ERROR_PROTOCOL_20260923.md). No model was trained,
invoked or recalibrated. The [contact comparison](CONTACT_BOUNDARY_RESULTS_20260923.md)
remains a NEGATIVE_CONTROL; A, LOCAL, UNKNOWN and App behavior are unchanged.

The decision is to investigate metric supervision/constraint fitting before
adding model complexity. Coarse labels leave wide boundary intervals, yet many
errors also violate those available intervals. Native returns often include
boundary-near points, but their public spatial support is broad. This audit
does not uniquely identify feature loss, optimization or calibration as a cause.

## Cohort and definitions

Same1728consumed simulation images, original864train/288dev/576evaluation split.
The evaluation cohort has528finite interior width boundaries and352horizon
boundaries across BODY/HEAD. They are image/layer cases, not independent scenes;
both arms share them. Width means full cross-section width, twice lateral gap.
Thus5cm width error corresponds to2.5cm lateral-gap error for a fixed edge.
Horizon means axial first-contact coordinate, with sweep starting atZ=.3m.

All three splits use saved predictions and original dev-selected thresholds.
Evaluation-image labels in this diagnostic are hypothetical constraints for
analysis, never supervision that was supplied during training. All624width and
800horizon right-censored evaluation truths remain in the detailed rows.

## What the training labels constrain

For a width boundary at h=3, monotonicity gives a lower-open/upper-closed bracket
from the largest negative width and smallest positive width. The analogous
bracket holds for horizon at w=.6. The full Cartesian query grid supplies no
tighter off-axis constraints for these two included slices.

| Finite interior evaluation boundaries | Width | Horizon |
| --- | ---: | ---: |
| Cases |528|352|
| Per-image label bracket width |24 or36cm|30cm|
| Median minimax half-width from labels alone |12cm|15cm|
| Brackets at most10cm wide |0|0|
| Same complete binary-grid signature, observed true span>10cm: affected interior cases |152|0|

The width signature comparison supplies concrete different-boundary examples
with identical binary labels. It excludes RGB/ToF and is not an observation
collision. The horizon signature comparison has760affected finite cases only
beyond the evaluation sweep; none belong to the352interior denominator.

These brackets describe information in each image's supplied binary grid.
They are not a lower bound on learned error: images, shared geometry and other
examples may resolve the interval. Conversely, the exact-truth step sampled on
the original fine curves succeeds within5cm on every finite interior case;
evaluation discretization cannot explain the large observed errors.

## Errors beyond the available coarse constraints

The diagnostic compares the interval between the last sampled negative and
first sampled positive with the label bracket. It preserves left/right-censored
predictions and original5cm failure denominators. This is a sampled first
transition, not proof of no hidden between-sample excursions in the direct MLP.

| Split / axis | Direct: all5cm errors / bracket-incompatible errors | Geometry: all5cm errors / bracket-incompatible errors |
| --- | ---: | ---: |
| Train / width,840truths |746 /496|716 /259|
| Train / horizon,560truths |526 /465|385 /41|
| Evaluation / width,528truths |491 /322|485 /292|
| Evaluation / horizon,352truths |333 /303|291 /156|

Geometry usually respects the coarse training horizon bracket even when its
metric answer fails5cm. Its evaluation horizon violations grow substantially.
Direct has many horizon violations already on train. These are different
limitations; neither can be summarized as only insufficient sensor resolution
or only unseen-layout generalization. Remaining bracket-compatible errors are
not proved to be *caused* by label sparsity.

Original evaluation successes reproduce exactly: width37/528direct,43/528geometry;
horizon19/352direct,61/352geometry. Predictions already positive at the sweep
minimum are separately counted as left-censored: width215direct/156geometry,
horizon22direct/0geometry across all1152image/layer cases per axis. The original
metric still treats those endpoints as operational finite predictions. Missing
crossings still fail the finite-truth metric; no denominator was repaired.

## Native support and public spatial ambiguity

All576evaluation ToF frames reproduce byte-for-byte using the unchanged
winning-bin simulator and original noise identities. For each finite true
boundary, the audit searches native pixel points, the sensor-zone lattice,
and actual observed contributors for a contact coordinate within5cm.

| Closest-point support partition | Width,528 | Horizon,352 |
| --- | ---: | ---: |
| Present among actual returned contributors |463|277|
| Present on zone lattice, absent from returned contributors |21|7|
| Present in native image, absent from zone lattice |44|50|
| No native point within5cm |0|18|

First-contact minima and closest-to-truth points are both retained. Their5cm
agreement counts happen to coincide on this cohort; the mechanisms are not
interchangeable in general. Native points near the coordinate do not establish
public object association, exact full extent or unique boundary ownership.

Even with returned-near support, geometry fails5cm on424/463width and227/277horizon
cases. Presence of a useful native contribution is not successful public use.
For the privileged nearest contributor, preserving the original public distance
interval gives the following support spans:

| Median coordinate span | Entire zone + range interval | Exact contributor angle + same range interval |
| --- | ---: | ---: |
| Width,463cases |52.71cm|11.31cm|
| Horizon,277cases |48.09cm|26.74cm|

Exact-angle spans are at most10cm in197/463width and35/277horizon cases.
The horizon calculation retains the candidate tail beyond3m in21cases.
No selected near contributor is outside its public distance interval here;
this does not make the interval a guarantee under unbounded Gaussian noise.
These are single-return support projections selected with evaluator truth,
not RGB+ToF information ceilings, posterior confidence or full-scene bounds.
No interval was narrowed or installed as a new predictor.

## Cached feature evidence and next decision

All384ToF feature components match public tokens exactly. No exact duplicate
RGB rows or combined feature rows occur among1728images. The1728matched lateral
pairs all retain a nonzero visual-feature difference; train-standardized visual
RMS median=.26651, combined=.26531. This excludes literal feature equality in
these comparisons, not inadequate resolution, entangled nuisance or poor use.
ToF alone has32exact-duplicate groups/71rows;33first-versus-other comparisons
have differing finite/missing boundaries or >10cm finite differences. This
includes outside-sweep cases and is not a primary-cohort impossibility count.

The most targeted next *proposal* is a separately scoped test of whether
boundary-localized supervision improves metric fitting with a fixed public
representation and fixed model structure. It should distinguish train fitting
from held-layout transfer and retain censored cases and false-positive costs.
The current audit does not authorize or run that fit, choose its recipe, or
retune this consumed contact package. Observationally unresolved cases must
remain explicit even if future supervision becomes more precise.

## Validation and retained evidence

Eight synthetic tests pass. Primary checks reproduce eight original boundary
metric blocks,13824monotone label brackets,13824sampled monotone curves and
5440exact-truth resolution cases (arm repetitions included). Counts are checks,
not independent evidence. All declared old inputs and saved predictions remain
unchanged. Scientific CPU runtime19.99s, TASK_NOT_GPU_SUITABLE; no UE, training,
paid allocation or continuing task-owned process.

Independent audit PASS:13824rows,12split/arm/axis summaries,eight original
metric blocks,48fixed native frames spanning all16evaluation groups and three
lateral relations. It recomputes world-space labels/boundaries, contributor
selection and support spans without diagnostic-producer imports. Seven analytic
audit fixtures pass. Complete corrected version2 and version3 rows have the
same SHA256. The final expanded-seal audit takes2.94s; its separate receipt is
`artifacts.local/evidence/ba-boundary-error-20260923-audit-v2/result.json`.

Evidence root: `artifacts.local/evidence/ba-boundary-error-20260923-v3`.
It retains13824per-image/layer/axis/arm rows, input/code hashes and summaries.
Version1 and its receipt remain after independent review found minimum-versus-
nearest support and float-endpoint issues. Version2 preserved complete corrected
rows but failed JSON receipt serialization on a NumPy integer; version3 fixes
serialization. These are diagnostic repairs, not additional model attempts.

Terminal `terminal-boundary-error-20260923` retains this diagnostic as
COMPONENT_OR_CHALLENGER/COMPONENT. Archived experiment `ba-boundary-error-20260923`
is registered through the supported ledger, anchored to source commit27040e82.
All prior terminal dispositions remain unchanged.
