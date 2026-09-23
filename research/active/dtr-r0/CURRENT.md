## Boundary-aware query sampling: metric gains with costs (2026-09-23)

[Fixed-budget result](nearfield/CONTACT_SAMPLING_RESULTS_20260923.md):
Fixed72query Development: geometry evaluation width within5cm43/528 ->134/528,
horizon61/352 ->168/352; train horizon175/560 ->467/560. Recall86.88 ->78.49%,
FPR9.48 ->5.16%, horizon falsecross119/800 ->146/800. Retain sampling COMPONENT;
neither full package passes.13tests/777024world-label auditPASS. Old controls,
A/LOCAL/UNKNOWN preserved; two fits complete, no automatic successor.

## Boundary-error attribution: coarse labels and violated constraints (2026-09-23)

[Saved-output diagnostic](nearfield/BOUNDARY_ERROR_RESULTS_20260923.md): width label
brackets24-36cm, horizon30cm; geometry held-layout errors violate even these
brackets in292/485width and156/291horizon cases. Returned-near native support
463/528width,277/352horizon retains broad public spans. Training horizon violations
41/385 versus held156/291 distinguish coarse fitting from transfer. Diagnostic
COMPONENT; no refit/cutoff change. Preserve failed contact packages and A/LOCAL/UNKNOWN.
Independent13824-row and48native-frame auditPASS. Next proposal: separate metric
supervision from representation/optimization; no automatic successor started.

## Contact-boundary generalization: ranking without accurate metric crossings (2026-09-23)

[Fixed two-arm Development](nearfield/CONTACT_BOUNDARY_RESULTS_20260923.md) reuses
1728consumed frames; whole-layout24/8/16split and novel width/horizon queries.
Direct3346TP1162FP1136FN versus monotone geometry3894TP2196FP588FN on576held images:
74.65/86.88%recall,5.016/9.479%FPR. Pair ordering99.80/99.20%does not yield joint
pair decisions47.02/36.45%. Within5cm width37/528 versus43/528, horizon19/352 versus61/352;
missing boundaries count as failures. Both exact packages NEGATIVE_CONTROL;
preserve partial error reduction and A/LOCAL/UNKNOWN.22tests and independent
714816-label/count/selection auditPASS. Frozen features and unequal effective
capacity limit attribution. No capture, post-evaluation fit or App promotion.

## Tiny-training query fitting: added spatial branch not retained (2026-09-22)

[Fixed paired check](nearfield/QUERY_SPATIAL_FIT_RESULTS_20260922.md):
84 consumed train images,504queries. Original global FiLM179TP1FP1FN versus
spatial180TP0FP0FN; both540/540same-imagequerypairs correct. Both learn
image-dependent query answers; old fixed query ordering is not structurally
unavoidable. Macro maskIoU0.484823/0.481782 fails fixed0.50gate, with no
localization gain from the added branch. Exact package NEGATIVE_CONTROL;
preserve query-fitting evidence and old negatives. New shared optimization
and tiny cohort prevent historical-cause attribution. No dev/held access,
threshold selection, video or App change. A/UNKNOWN remain unchanged.

## Inheritance batch: three useful components, no default replacement (2026-09-22)

[Four frozen trials](nearfield/INHERIT_BATCH_RESULTS_20260922.md) are complete.
On the original576consumed frames, LOCAL plus A223TP8FP33FN versus
RAW200/12/56 and A193/6/63 remains a COMPONENT; its two added false segments
still exceed the old one-segment budget. [Frozen transfer](nearfield/LOCAL_TRANSFER_RESULTS_20260922.md)
on576new same-generator frames gives LOCAL228/6/28, RAW193/7/63, A192/4/64,
with5/7/4false segments. Task transfer, background tolerance and geometry rank
pass;14/16base rescues survive background changes, with2lostTP/6gainedTP/2removedFP.
This cohort passes the strict cost gate; the old failure remains. Retain LOCAL
COMPONENT, A and UNKNOWN; no appearance-invariance, hardware or App claim.
[Native support diagnostic](nearfield/LOCAL_SUPPORT_RESULTS_20260922.md):
winning-query witness retains 32/36 rescues; either-centre control 32/36,
primary PASS. Native false-alert removal is privileged geometry;
retain full support partitions and LOCAL onset costs, without public ownership
or filter promotion. Original LOCAL transfer remains a COMPONENT.
[Public-interval feasibility](nearfield/RETURN_CORRESPONDENCE_RESULTS_20260923.md),2026-09-23:
even perfect contributor angular association preserves all36rescues but removes0/2addedFP;
full and1024ray controls exactly reproduce LOCAL228/6/28. NativeXYZ32/36does
not establish this unchanged-interval filter opportunity. Exact readout NEGATIVE_CONTROL;
stop before student training, preserve LOCAL/A/UNKNOWN and prior diagnostic roles.
[Composite stability transfer](nearfield/LOCAL_STABILITY_RESULTS_20260923.md),2026-09-23:
new576frames LOCAL119/20/105, RAW109/16/115, A107/12/117; 8earlier/recovered events.
Frozen stabilityFAIL; retain full trajectory/size/shape costs and prior LOCAL value.
No fitting, source exclusion, cutoff change or physical-motion claim.
Additional hypothetical returns plus unchanged hold give 254/29/2 versus
225/26/31, with entry-sample detection 17/32 to 31/32; retain information and
its added costs, not a lossless upgrade. Support inheritance returns exactly
to A current, losing every hold TP gain: exact recipe NEGATIVE_CONTROL.
On 180 analytic geometries, hypothetical 1mm bins with unchanged shared bias
and initial query increase nominal correct decisions 97 to 109, no observed
wrong decisions or old losses: precision COMPONENT, not hardware evidence.
A, UNKNOWN and old negatives remain. Real detector-track seven-frame analysis
is NOT_EVALUABLE; physical CNH remains paused. No combined-system claim or retry.

## Frozen checkpoint diagnosis: query ordering does not follow the image (2026-09-22)

[Train/dev inference-only diagnostic](nearfield/QUERY_OCCUPANCY_DIAGNOSTIC_RESULTS_20260922.md)
finds the same strict6-query order in all1152train/dev images in both models:
HEAD-left > HEAD-centre > HEAD-right > BODY-left > BODY-centre > BODY-right.
Thus centre max always uses HEAD-centre. Occupancy foregroundAUROC0.544/0.576
and IoU0 show weak selected-state training localization; this is more than a
held-layout or0.5cutoff failure. Conditional occupied-bin accuracy85.64/70.45%
retains coarse distance signal but no reliable query localization. Diagnostic
COMPONENT, original model NEGATIVE_CONTROL and A/UNKNOWN unchanged. Eight
synthetic tests pass; original dev counts reproduce. No training, BN update,
held-label access, held-row inference, threshold selection or video successor.

## Single-frame query occupancy: exact recipe not retained (2026-09-22)

[New-layout paired prototype](nearfield/QUERY_OCCUPANCY_RESULTS_20260922.md)
completed1728new controlled frames and24epochs per arm. On576held frames,
classifier62TP34FP194FN becomes occupancy88TP44FP168FN,16 to29 of32events,
but false segments17 to29 and addedFP10 exceed the +1/+2 costs. Both are worse
than A current193TP6FP32events.880visible-positive queries have maskIoU0 and
13correct first-hit bins; conditional0.283m MAE is not reliable detection/range.
Preserve the exact recipe as NEGATIVE_CONTROL and this source as consumed
Development. A/UNKNOWN unchanged.37tests and independent metric, geometry and
component/decision audits PASS; inheritance and archived registration complete.
Task-owned UE/training processes released. No video, App promotion or extra fit.

## Retained A operating scope: broader-layout stability not established (2026-09-22)

[Consumed two-cohort scope audit](nearfield/OPERATING_SCOPE_RESULTS_20260922.md)
replays2304existing frames at nominal/left/right projection with unchanged A.
Actual Core positive target fronts span2.294-3.000m; the0.3-3m query definition
does not establish closer-range coverage. Nominal original173/33/0,16/16events
becomes159/25/17,15/16on broader layouts. A1.124mwide,7.6cmhigh HEAD-horizontal
object intruding13.1cm is wholly missed. Broader right shift165/63/11 detects16/16
but has0.6s onset and58.33%minimum coverage; all three broader conditions fail
historical Core usefulness. Nominal FP are range-entry/exit errors; shifts add
OUTSIDE FP. Broader nominal UNKNOWN681/768 includes97alerts and584silent frames;
silence never establishes clear space. Retain A and diagnostic COMPONENT only.
Public replay nominal parity2304/2304; independent saved-output auditPASS and
geometry-label parity2304/2304. No training/capture/App change or successor.
[Exact receipt recovery](nearfield/LEDGER_REPAIR_20260922.md) resolves ledger303;
the audit is now archived with explicit COMPONENT inheritance and a refreshed
global decision index. Historical rows and failed-command receipts are preserved.

## Initial-camera-relative query: semantic consistency without nominal gain (2026-09-22)

[One fixed375history replay](nearfield/INITIAL_RELATIVE_RESULTS_20260922.md) shifts
query constraints, labels and INprojection to the actual initialcamera, retaining
all shared sensor uncertainty, worlddomain and anchoredwall. Nominal51TP/46correct
negatives keep all97old identities with0wrong and0gains; boundary60remainUNKNOWN.
Range+2mm adds one correctnegative over fullglobalquery, already recognized by
oldrange-only; all375queries/2340mapped decisions equal that simpler oldcontrol.
Eight pose/combined arms each change30boundary truth labels; denominators80/100
or100/80 are explicit, oldlabels untouched. Drift retains92-96nominalcorrect,
without observedwrong;40globalgauge alternatives preserve relativequery labels
but do not yield40recognitions. Retain explicit initial-reference COMPONENT,
nominal-gain NEGATIVE_CONTROL; current-camera runtime and extra sensing untouched.
750MILPs,0newobservations,6tests and independent math/receipt/containment/metric
auditPASS. Local inheritance and ledger303/unknown-terminal receipts retained.
No successor tuning, continuing process or paid resource.

## Boundary observability: sampling gains and query-reference ambiguity (2026-09-22)

[Fixed angle/precision diagnostic](nearfield/BOUNDARY_OBSERVABILITY_RESULTS_20260922.md)
reuses180consumed geometries,13views/12cm path and adds known5.625degree yaw.
Coarse paired separation2/30 becomes5/30primary,4/30reverse; both lose one old pair.
Hypothetical1mm quantization gives18/30fixed,20/30primary; all10far pairs separate
only under fine quantization. These are observation differences, not TP/FP gains.
All40lateral cases retain opposite-label common-X-translation witnesses under
the fixed-global query, all3yaw schedules and both quantizers. More angles/range
resolution cannot break that coordinate symmetry within the declared model.
The adopted camera-relative corridor is a different query: common translation
preserves labels for all40saved witnesses. Do not infer a need for submillimetre
absolute-world localization; clarify initial/current reference camera before
new inference. Relative-pose and sensor-extrinsic errors do not automatically
cancel. Old labels/results unchanged.5tests,2independent audits and saved semantic
checkPASS;0MILP,56160base and12480counterexample rays. Local inheritance and ledger
block receipts retained, no further sweep/model, persistent or paid resources.

## Frozen model under slow drift: finite tolerance, boundary information gap (2026-09-22)

[One new-instance drift pilot](nearfield/BIAS_DRIFT_RESULTS_20260922.md) uses180
new synthetic geometries,13conditions and unchanged shared-bias models. Primary
nominal51TP/46correct negatives retains the same97cases in all6constant controls;
six small linear drifts retain92-96, no observedFP/falseOUT. No lossless drift
claim:1-5correct cases becomeUNKNOWN. General120cases contain all97nominal decisions;
boundary60cases are allUNKNOWN under fullmodel across every condition.28-30of30
opposite-truth boundary pairs have identical13view histories, so zero errors is
not boundary robustness. Existing nominal model is misspecified for drift; no
general safe-abstention, hardware calibration or independent-source claim.
Retain finite response diagnostic and scoped lossless/boundary NEGATIVE_CONTROLs;
prefer new distinguishable observations over more nuisance parameters if pursued.
375unique queries/1500MILPs,30420new simulated views.6focused tests and independent
capture/witness/containment/metric auditPASS. Local inheritance and supported
ledger303/unknown-terminal receipts retained; no model tuning or successor run.

## Shared sensor bias: primary collapse recovery with local identity retention (2026-09-22)

[Frozen shared-bias comparison](nearfield/SHARED_BIAS_RESULTS_20260922.md) reuses
749 public histories, two presets and2996MILP calls; no new observations. Primary
X-then-Z restores84/87correct decisions from the two prior all-UNKNOWN conditions.
Its nominal43TP+37correct negatives remain the SAME80cases across all5conditions;
no FP or falseOUT observed. Range-only and full range/pose models give identical
decisions on all900primary histories, so pose variables add no primary gain here.
Full pose uncertainty costs3correct negatives on straight and2TP on reverse order
per condition, all genuine opposing-witness UNKNOWN. Previous6straight-baseline
TP losses on the bend remain; this is not route-wide lossless recovery.
Bounds come from the frozen synthetic injector; shared constant bias only, no
hardware calibration, fresh confirmation, arbitrary-noise or safety claim. Retain
COMPONENT_OR_CHALLENGER/COMPONENT and simple attribution control.7focused tests
and2independent saved-output/containment auditsPASS. Supported ledger303/unknown
terminal failures and local inheritance retained; no continuing resources.

## Equal-cost bent-path observation: partial gain, exact-model stress failure (2026-09-22)

[One frozen3path x5condition comparison](nearfield/BENT_PATH_RESULTS_20260922.md)
keeps12cm/13views on180consumed singlebox scenes. PrimaryXthenZ gives43TP/37correct
negative versus34/45straight; gains15TP but loses6, so no lossless improvement.
Reverse-order control reaches48/35, gains23TP loses9; it does not replace primary.
Prior98constructive-ambiguity cases yield17/32correct conditional decisions on
the two bends: new viewpoint information exists. No nominalFP/falseOUT observed.
Range-2mm and pose+(1mm,1mm) make bothbends180UNKNOWN, all both-class infeasible:
backgroundwall45to44bin crossings cannotfit fixedwall or declaredboxdomain.
This is modelinconsistency, not complete loss of observationinformation. Retain
partial geometrycomponent and scoped stressNEGATIVE_CONTROL; no tolerance/path
retuning, Apppromotion or yaw/RGBsuccessor.9fixtures and2independent auditsPASS;
749queries/1498calls,3600replayed+23400new simulatedmeasurements. Localinheritance
andledger303receipts retained; originalexperiments and unrelatedWIP preserved.

## Numerical witness recovery exposes the remaining observation ambiguity (2026-09-22)

[One consumed258query replay](nearfield/CANDIDATE_RECOVERY_RESULTS_20260922.md)
reproduces every frozen baseline result and restores94of97rejectedIN witnesses
using closed-query center projection with unchanged forward validation. No added
MILP/observations and no classification gain. Fixed13view incomplete64to3;
98of101UNKNOWN have explicit compatibleIN/OUT geometries. Keep34TP/45correctnegative,
0FP/falseOUT and all79correct decisions.70failures arose at12decimaldecode,24at
face-to-center arithmetic;3bin mismatches persist and are not retuned. Initial42
witness cache recovers only3fixedpath cases. Prioritize new observable information
over solver-only recognition expectations; retain numerical repair as COMPONENT,
not App/alert promotion.13fixtures and independent saved-output auditPASS;
rawvectors, failures, local inheritance and ledger303 receipts preserved.

## Saved-trace deepening: continuous path constraints and feedback attribution (2026-09-22)

[Consumed180scene replay](nearfield/OBSERVATION_DEEPENING_RESULTS_20260922.md)
raises correct numerical single-rectangle decisions4to79 with fixed+X path2to13
views at unchanged12cm motion: TP2to34, correctnegative2to45, no observedFP/falseOUT,
all4old correct decisions retained. This costs11extra observations; nothardware
or formal exclusion evidence. Fixed13view UNKNOWN101includes37opposing-witness
cases and64numerical candidate failures. Single-pair guided13view path loses14TP
and gives64correct decisions; retain this selector as scoped NEGATIVE_CONTROL.
Public initial openloop planning reproduces all180feedback decisions at3views/
12cm:28TP/3FP/7falseOUT/36correctnegative, with5different paths but no decision gain.
Continuous3view readout removes10wrong commitments but loses22oldTP and adds5;
notlossless replacement. Preserve numerical limitations and separate ambiguity
from rejected candidates.15focused tests and independent saved-output audits;
258unique histories/516calls, no new actual views or retries. Components only,
no App promotion; scoped local inheritance and ledger303 block receipts retained.

## Five observation mechanisms: path information and constructive witnesses (2026-09-22)

[One shared180scene suite](nearfield/OBSERVATION_MECHANISMS_RESULTS_20260922.md)
finds fixed+X path information35to81/90pairs and finite-cohort purity30to156/180,
at2to13observations with unchanged12cm travel. Strict same-surface quantization
crossing has0gains on all4spokes; retain that exact check as NEGATIVE_CONTROL.
Continuous real-valued MILPs produce119validated witnesses from122calls/61public
signatures: initial180/180and two-view176/180cases have opposite-label witnesses.
All15old fixed-endpoint commitments admit opposite explanations, including10
correct ones; this is no selective veto or new classifier. Reasoned UNKNOWN stays
180/180at both modes, distinguishing witness-pair ambiguity and incomplete search.
Public3view two-step improves fixed24TP/3FP/8falseOUT to28TP/3FP/7falseOUT,
retaining24TP, correcting2wrongOUT but adding1newwrongOUT; preserve partial gain
without zero-new-error claim. Adaptive/nonadaptive evaluator ceilings both157,
so no isolated feedback-ceiling gain is shown. Retain scoped components and costs;
no aggregate system promotion, tuning or automatic successor.18tests and two
independent saved-output audits PASS; old evidence intact, no resources remain.
Per-mechanism local dispositions and ledger303/unknown-terminal receipts retained.

## Boundary-pair information exists beyond frozen selector coverage (2026-09-22)

[One fixed90pair check](nearfield/BOUNDARY_SEPARABILITY_RESULTS_20260922.md)
uses180new same-generator boundary scenes. Of58initially identical pairs,
38can separate under an allowed12cm action; one common action per initial
observation class gives evaluator ceiling32 versus12for all frozen policies.
This is not an implemented32-pair selector; the class oracle gains22and loses2
versus fixed. Twenty pairs stay aliased across all5poses:14through quantization,
6even in raw ray returns. Of26missed available pairs,20have no initial old-bank
match; six share a background signature. Full180-scene opposite-label ambiguity
is stricter than designated-pair separation:174initially mixed scenes have
class-common82/per-case88resolved ceilings. No alert-accuracy or universal
geometry guarantee follows. Retain diagnostic COMPONENT and original methods;
no new selector, training, bank expansion or automatic successor. Four tests and
independent900-view equation/count audit PASS; ledger303/unknown-terminal gap
receipts retained, no allocated resources remain.

## Frozen opportunity transfer: useful positive gain with new errors (2026-09-21)

[One frozen +/-1cm synthetic transfer](nearfield/OPPORTUNITY_TRANSFER_RESULTS_20260921.md)
uses348 shape and1304 active-view layouts derived from consumed parents; no
unused in-prior layouts existed. These are correlated same-generator geometries,
not independent-source confirmation. Shape point84TP/4FP becomes union84TP/6FP:
no new TP, two added FP. Active fixed504TP/4FP/2false OUT becomes positive-priority
512TP/6FP/4false OUT, retaining all504fixed TP but adding four wrong commitments.
Versus old adaptive, net+2TP conceals17gained/15lost TP and18lost correct negatives.
The old four wrong shape OUT remain; eight shifted descendants give6wrong OUT,
2UNKNOWN and0IN. Original off-grid/wall active descendants all64remain UNKNOWN.
Both full transfer criteria fail; preserve original consumed component gains in
their original scope, and retain this local-transfer diagnostic as COMPONENT.
No method/bank/tolerance change, retuning, successor or runtime promotion. Four
focused tests and independent saved-output audit PASS; old evidence unchanged.
Local disposition and ledger303/unknown-terminal receipts retained; no resource
remains allocated. New geometry can share old observations; report that boundary.

## Saved B/N disagreement: screening signal, binary-only budget failure (2026-09-21)

[Completed posthoc diagnostic](nearfield/BRANCH_DISAGREEMENT_RESULTS_20260921.md)
partitions A-excluded B_control/N outputs on the now-consumed1152-frame cohort.
Boundary current common/B-only/N-only yield41/3,49/4,45/5 added TP/FP: each bucket
alone exceeds the2-FP budget. All8fixed global binary retention masks are reported;
only A alone passes every cost. No mask, cutoff, verifier or expert is trained/selected.
Agreement still has real Core benefit: AND hold172/26/4 and16/16events versus
A159/25/17 and15/16; Core costs pass. Boundary AND62/10/114 and10/16events fails
current andheld budgets. Preserve this screening evidence and N's Core/HEAD
improvement/B's BODY coverage; do not call Core universally solved or a partial
gain a balanced upgrade. Layer labels remain evaluator strata, not routing inputs.
This narrow result does not exclude continuous-score, public-input or temporal
verification. A later method needs a justified within-pattern discriminator and
different isolated validation data; no automatic successor starts. A/UNKNOWN and
old B/N/R dispositions stay fixed. Diagnostic COMPONENT,4tests and independent
arithmetic checks PASS; local inheritance/global ledger303 gap receipts retained.

## Fixed B with broader training data: partial gains, joint budget failure (2026-09-21)

[Completed data-condition experiment](nearfield/DATA_COVERAGE_RESULTS_20260921.md)
captures3456 frames from48 distinct groups, fits one unchanged B recipe on1728,
and evaluates16 isolated groups/1152 frames. Relative to old B calibrated by the
same new-dev rule, N reduces Core held FP66 to34 and raises TP172 to175. Boundary
held counts remain106/19/70,60.23% recall; detected events12/16 to13/16. This
aggregate equality hides a trade: HEAD-horizontal12/1/34 to43/7/3, but BODY-plane
38/9/6 to9/3/35. N's costs fail; Boundary adds14FP/10segments over A versus caps4/2,
and Core segments also fail. Close exact condition as NEGATIVE_CONTROL without
erasing Core/HEAD improvements or calling them a balanced upgrade. Old consumed
regression N87/20/86 versus calibrated B145/20/28 is secondary evidence only.
Preserve A/UNKNOWN, all prior B/R/U/G and paused POINT/REGION dispositions. No
protected test, refit, threshold/loss retry, default promotion or successor.
Twelve tests, independent arithmetic/event audit and64-frame public replay pass;
task-owned processes released. Local inheritance and ledger303 gap receipts saved.

## Dense-depth alert branch paused; partial R benefit retained (2026-09-21)

[Completed POINT/REGION experiment and user review](nearfield/REGIONAL_DEPTH_RESULTS_20260921.md):
Only the training measurement-consistency loss differs: center sample versus
regional aggregation, same network/supervision/top16 inference readout. REGION
adds0TP/1CoreFP; POINT gives Boundary hold22/5/151,6/16events but fails coverage
and Core current costs. Both exact recipes remain NEGATIVE_CONTROL. Pause this
recover-dense-depth-then-alert branch without another diagnostic, loss/top-k/
aggregation change or longer fit. Core event/onset retention follows from A OR
plus unchanged hold; it is not independent new-model benefit. Regional fusion
in general remains unrefuted; this does not authorize a renamed retry.
[Earlier R](nearfield/CORRIDOR_RELATIVE_RESULTS_20260921.md) has useful partial
research value: Boundary hold14/5/159 to78/6/95, recall8.09% to45.09%, events4/16
to11/16, gains9/16layouts; Boundary/Core adds1/5FP and all cost caps pass.
Retain original nonpromotion: HEAD-horizontal0/43 in all4layouts, five wholly
missed events and1.8s maximum detected onset delay. Present gains/costs/failures
together, not all nonpromoted recipes as ineffective. These saved working points
on consumed data do not establish equal-realized-FP dominance.
A/demo, UNKNOWN and existing B/R/U/G dispositions stay fixed. Later work should
introduce new spatial cues or object/layout/background learning coverage for
specific misses; direct task alerts have priority and dense depth is optional.
This is an investment priority, not a universal architecture claim. The1152
consumed frames remain explanatory/replay evidence, never fresh confirmation.
No experiment starts here. Existing local dispositions/ledger303 receipts remain.

## Ordered spatial contrast: no added horizontal benefit (2026-09-21)

[Matched ordered RGB comparison](nearfield/SPATIAL_STRUCTURE_RESULTS_20260921.md)
reuses frozen 24-channel features with lossless zone/subcell spatial ordering.
Same 65825-parameter U/G heads, initialization,1200 batches, optimizer and CUDA;
only G's eight interval-relative geometry channels differ from U's zero ablation.
On consumed1152-frame transfer, horizontal96-frame decisions are identical:
15/43 true positives,2/4 timely events. Boundary hold U66/11/107 vs G68/9/105;
Core173/44/0 vs173/45/0. Both fail the fixed cost/coverage criteria; G's extra
horizontal TP is zero. Mark both exact recipes NEGATIVE_CONTROL and the geometry
hypothesis NOT_SUPPORTED. This does not prove spatial pooling caused earlier
misses or exclude other geometry representations. Existing A/B/R and UNKNOWN
are unchanged; no threshold/fit retry, protected test or automatic successor.
Local evidence and global index303/unknown-terminal receipts are retained.

## Corridor-relative representation: low-cost gain below coverage floor (2026-09-21)

[One fixed representation pilot](nearfield/CORRIDOR_RELATIVE_RESULTS_20260921.md)
uses native RGB statistics in full-range-conditioned corridor query bands and a
shared max MLP (3041 parameters), retaining original A/UNKNOWN and .2s hold.
On consumed1152-frame transfer, Boundary hold14/5/159 becomes78/6/95;
events4/16 to11/16, gains9/16layouts. Core173/33/0 becomes173/38/0, all added
FP in one HEAD hanging-plane OUTSIDE clip. Added-cost caps pass, but45.09%
recall misses the frozen50% floor; HEAD-horizontal0/43 across all four layouts.
Detected Boundary onset delay reaches1.8s; five events remain wholly missed.
NO_GO / NEGATIVE_CONTROL for this exact representation/fit/selection recipe,
not a claim of no RGB information. A/demo retained; no retry, protected test,
hardware claim or automatic successor. Five focused tests and independent audit
PASS. Structured local disposition and global ledger303/unknown-terminal receipts
are retained. User's current input scope remains simulation only.

## Core projection sensitivity: events survive with lateral FP cost (2026-09-21)

[One frozen +/-2-pixel projection check](nearfield/CORE_PROJECTION_STRESS_RESULTS_20260921.md)
reuses1152 consumed transfer frames and unchanged ranges. Nominal Core hold
173/33/0 becomes left172/36/1 and right172/64/1; all16events remain, with one
HEAD onset delayed.2s per sign. All added Core FP occur in OUTSIDE clips
(0 to11/34), while the same-condition Calibration-relative usefulness gates
still pass. Thus retained event coverage does not imply calibration-insensitive
specificity. Zero-offset all1152 scores/decisions reproduce exactly. Retain a
diagnostic COMPONENT, preserve A/demo and all old negatives; no offset/threshold
retry, training, protected test, hardware claim or automatic successor. The
user confirmed no connected hardware/measured inputs. Local evidence and global
ledger303/unknown-terminal receipts are retained.

## Fixed-entry conditional hold: no transferred continuation gain (2026-09-21)

[One separately authorized decoder contrast](nearfield/CONDITIONAL_HOLD_RESULTS_20260921.md)
keeps original entry15.769264 and selects keep15.535847 on consumed selection
layouts only. Selection gains one internal positive; evaluation matches every
original_current flag (Core82/11/2, Boundary16/0/68). The12-frame gain over A
belongs to the prior spatial supplement, not this decoder. Five unseeded events
account for51 missed Boundary positives;1 precedes a seed and16 are reachable.
The same selection event has an internal positive11.615569 below its exit
negative14.757340; a Core exit15.453609 further binds the shared keep threshold.
Close this original-score fixed-entry recipe as NEGATIVE_CONTROL; do not infer
all temporal information is absent. Baseline, UNKNOWN and protected test remain
unchanged. No entry/keep retry, training or automatic successor. Local evidence
and global registration/inheritance failure receipts are retained. Four tests and
independent audit PASS.

## Multi-zone RGB anchoring: insufficient coverage (2026-09-21)

[One frozen diagnostic](nearfield/RGB_MULTIZONE_RESULTS_20260921.md) on seven
remaining lateral-negative/matched Boundary pairs finds zero explained OUTSIDE
negatives. Only two selection positives meet the two-adjacent-whole-zone anchor
rule; their three CROSSING queries cover307/307 observed-return contributor
incidences. Eight frames have zero complete-zone anchors, four have one, two
have two. Four frames lack an eligible query and remain unresolved. This closes
the exact multi-zone anchoring recipe for residual lateral error, not RGB or
8x8 information in general. Local NEGATIVE_CONTROL; A/demo, UNKNOWN and source
alerts unchanged. No weaker anchors, segmentation retry, training, protected
test or automatic successor. Three focused tests and independent audit PASS.
Global metadata remains pending ledger303/unknown terminal; receipts retained.

## Public ToF axes: distance evidence with boundary costs (2026-09-21)

[Sealed1152-frame audit](nearfield/AXIS_EVIDENCE_RESULTS_20260921.md) finds
contained-depth evidence on273/346positives and0/633distance negatives, but
also137/173depth-valid lateral negatives. Every one of32positive clips lacks
this evidence at its first and last positive sample. Of82native-backed
balanced-suppressed positives,78have it; all15high-score distance negatives
lack it. This is useful axis-separation evidence, not a lossless distance gate
or lateral ownership solution. All frames have valid returns despite1040
baseline UNKNOWN frames. Retain COMPONENT diagnostics and A/demo, with no new
alerts, threshold, training, conditional hold or protected-test access.
Independent audit PASS. Existing global registration/inheritance blocker remains;
local receipts kept.

## Current-only audit: local rescue, no broad low-FP repair (2026-09-21)

[Fixed-score audit](nearfield/CURRENT_ONLY_RESULTS_20260921.md) removes hold
from cutoff selection without fitting. Current evaluation Core remains82/11/2;
Boundary A4/0/80 becomes16/0/68 original,12/0/72 uniform,9/0/75 balanced.
All evaluation gains are in one head_hanging_plane layout; selection gains
four frames in one body_protruding_plane layout. Zero added current FP,
but all fail the prespecified >=4/8-layout criterion. New heads underperform
original despite higher xAUC. The old hold constraint did suppress local
signal; it was not the sole limit on useful cross-layout repair. Keep A/demo;
NEGATIVE_CONTROL for this broad current-only role, local rescues diagnostic.
No conditional hold, training or protected test follows. Global metadata
pending ledger303/unknown terminal; independent audit PASS and receipts retained.

## Last-layer pilot: no useful repair under unchanged hold (2026-09-21)

[Frozen32D two-readout pilot](nearfield/LAST_LAYER_RESULTS_20260921.md) on fixed
consumed Development roles improves evaluation xAUC .8975 to .9118/.9133,
but both heads retain0/68 Boundary rescues at no-added-FP current+hold cutoffs.
Both exactly equal A: Core82/11/2, Boundary4/0/80; hold84/19/0 and6/2/78.
The binding cutoff sample is a true pre-exit Boundary frame whose hold would
add a negative next-frame alert. Failure therefore applies to this combined
recipe, not a proof against current-only linear separability. Audit PASS;
local NEGATIVE_CONTROL, global metadata pending. No automatic successor or
original-test activation. [Paired ordering](nearfield/LATERAL_PAIR_DIAGNOSTIC_RESULTS_20260921.md) remains173/173.

## Existing-data reuse: geometry usable, hard-negative coverage insufficient (2026-09-21)

[Read-only15000-row audit](nearfield/EXISTING_CORRIDOR_DATA_RESULTS_20260921.md)
verifies controlled-cube OBB geometry and indexed path availability. The750
lateral candidates are all crossbars from12 original fixture groups with
1.636-1.794m clearance, unlike recent6.6-11.9cm OUTSIDE failures. Ten fixed pairs
yield20payload checks (all hashes match), but only3 visibly supported target
pairs and zero certified full-scene negatives. Existing ownership means query
membership, not actor identity; preserve background/unseen UNKNOWN.
Retain geometry/input adapter COMPONENT and data inventory; no immediate binary
hard-negative fit. Seven geometry tests and independent saved-record recount pass.
Local disposition records existing global metadata blocker. No new source,
training, original-test access or automatic successor.

## Frozen supplement transfer: recall persists, specificity fails (2026-09-21)

[One new16-group/1152-frame verification](nearfield/SPATIAL_COMPLEMENT_TRANSFER_RESULTS_20260921.md)
keeps the head, high cutoff and original hold fixed. Boundary current10/1/163
becomes128/10/45, with gains in15/16 groups and event detection4/16 to15/16.
Core171/17/2 becomes173/36/0;17 of19 new current FP are OUTSIDE, concentrated in
three groups. Held Core173/33/0 becomes173/58/0; Boundary14/5/159 becomes140/19/33.
Rescue signal passes, both strict no-added-FP gates fail. Retain COMPONENT
evidence; no new operating point, runtime promotion or automatic successor.
Source/prediction seals and independent audit PASS; UE/model processes released,
original test still unactivated. Existing ledger303/unknown-terminal metadata
gap is documented locally. [Data reuse inventory](nearfield/SPATIAL_DATA_REUSE_INVENTORY_20260921.md)
records matched sources and15000 BODY-query frames requiring contract adaptation;
it authorizes no new fit or protected-test access.

## Frozen A/B complementarity: rescue signal with remaining costs (2026-09-21)

[Saved-output diagnostic](nearfield/SPATIAL_COMPLEMENT_RESULTS_20260921.md) reuses
eight consumed dev groups/576 frames. Fixed-high OR keeps every A alert and
changes current Core85/4/1 to86/4/0 and Boundary5/3/81 to71/3/15. Boundary gains
span seven groups. Whole-group cutoff calibration adds one Core pre-entry FP;
unchanged hold adds three Boundary exit FP. Calibration screening changes no
flags or cutoffs. No tested complete policy meets the zero-added-FP criterion.
Retain diagnostic COMPONENT; preserve the earlier full-replacement negative.
Four focused tests and independent audit pass, source hashes unchanged, test
unactivated. Local disposition records global ledger303/unknown-terminal gap.
No model fit, threshold retry, hold change, demo promotion or automatic successor.

## Spatial BCE: Boundary signal, no admissible Core replacement (2026-09-21)

[One frozen 40-group/2880-frame A/B pilot](nearfield/SPATIAL_BCE_RESULTS_20260920.md)
completed one ordinary BCE fit, with no ranking or second return. Train fits all
1728 labels; on eight dev groups the prescribed zero-logit diagnostic changes
Boundary current5/3/81 to78/3/8, but Core85/4/1 to83/15/3 and delays two onsets.2s.
All577 dev thresholds fail combined Core TP retention and FP/segment budgets;
current-only constraints also have no feasible threshold. Test inference remains
unactivated. Retain original Calibration/strong+hold; close this fixed full-alert
replacement recipe as NEGATIVE_CONTROL, preserving Boundary diagnostic scores.
Independent source/count/threshold audit PASS; UE/training released. No test,
retuning, demo change or automatic successor. Global ledger303/unknown-terminal
registration remains blocked, with local disposition and supported-CLI receipts.

## Complete-event transfer: same Core alerts, Boundary tradeoff (2026-09-20)

[One frozen24-clip/576-frame capture](nearfield/FULL_EVENT_RESULTS_20260920.md)
compares strongest+hold with closest-exported+identical hold. Core384 is exactly
identical96/14/0,8/8events, full coverage, zero onset delay/internal gaps; releases
are seven.2s and one.4s, with5pre-entry and9post-exit FP. One native-backed HEAD
current-frame recovery was already held by the baseline (held-only9to8).
The frozen information screen passes, but complete-Core task gain is absent.
Boundary0/0/96 becomes22/8/74;8/8events cover only1-4/12frames, five interruptions,
and one2.2s delayed onset. Retain COMPONENT evidence; no Core upgrade or default
promotion. Old demo/Calibration stay unchanged. Independent576-frame/native audit
passes; task UE released. End this round without tuning/training/new cohort.
Global metadata pending ledger303/unknown terminal; local disposition retained.

## Multi-return pilot: conditional synthetic information gain (2026-09-20)

[One frozen consumed432 simulation](nearfield/MULTIRETURN_PILOT_RESULTS_20260920.md)
retains every original first return and adds one declared separated synthetic bin.
Core strongest73/8/5 becomes77/8/1; with unchanged hold76/9/2 becomes77/9/1.
Four current HEAD recoveries have native support in the triggering second-slot
zones; one of two delayed onsets is repaired. Closest-exported and independent
two-return alerts coincide; no learned/joint-representation gain is established.
Boundary current6/1/72 becomes18/8/60, events3/12to9/12: seven extra FP remain.
Retain COMPONENT_OR_CHALLENGER / COMPONENT for this hypothetical input law only.
Hardware weak-return detectability and complete-event release remain unevaluated;
the74/10/0 demo, Calibration and defaults stay unchanged. End this pilot without
training, retuning or successor. Global metadata pending ledger303; sealed local
evidence and structured disposition retained.

## Consolidated Core policy: new-layout benefit, bounded challenger (2026-09-20)

[One frozen new36-layout/432-frame validation](nearfield/CORE_HOLD_VALIDATION_RESULTS_20260920.md)
compares Calibration, fixed strong T=.4071309640537889 and unchanged nonrecursive
one-frame hold. Core74P/214N:74/133/0 to72/10/2 to74/10/0. Hold restores two HEAD
gaps with no FP addition; the strong working point supplies most of the gain.
Core false segments26 to8, sampled duration26.6 to2.0s; BODY36/7/0, HEAD38/3/0.
All12 Core events have entry-frame alerts/full coverage;8 were already alarming
before entry. OUTSIDE FP88 to0; INSIDE-negative45 to10. Prospective bounded
tradeoff passes; independent432-frame recount, seals and process release pass.
Retain COMPONENT_OR_CHALLENGER / CHALLENGER for controlled Core demonstrations.
Boundary9/4/65 and4/12 events excludes full strict-task replacement. Original
Calibration, earlier76/9/2 with two delayed HEAD onsets and all prior dispositions
remain unchanged. No App promotion, retuning, extra cohort or automatic successor;
this completes the consolidation round. Global metadata pending ledger303/unknown
terminal, with supported-CLI receipts and local structured disposition retained.

## One half-bin phase: local proxy sensitivity without onset repair (2026-09-20)

[Frozen paired consumed432 diagnostic](nearfield/HALF_BIN_PHASE_RESULTS_20260920.md)
changes the 10 cm histogram origin by exactly5 cm, preserving paired dropout and
standardized noise. Neither original delayed HEAD onset's score/critical returns
changes. Across27648 zone-frames,4405 observed contributor sets change and75
paired ranges move >1 m. Core strong73/8/5 becomes72/8/6 (new f0007 loss),
hold76/9/2 becomes75/9/3 (f0008 loses preceding genuine strong), and Calibration
77/128/1 becomes77/129/1. Core12/12 events and first times persist; Boundary
flags are identical. Independent source and complete metric recount pass.
Retain original proxy, Calibration and hold; diagnostic COMPONENT only, no new
phase selection, task-contract change or automatic successor. Global metadata
pending ledger303/unknown-terminal; receipts and local disposition preserved.

## Onset64-zone audit: return selection loss and interval scoring are distinct (2026-09-20)

[One consumed432 audit plus six native frames](nearfield/ONSET_ZONE_AUDIT_RESULTS_20260920.md)
reproduces all original64-zone observations/scores. At f0005,28 sampled corridor
target points contribute to no selected return; dominant z35 chooses background,
leaving only z34 joint.026336. At f0293,48 target corridor contributors survive in
z36, but reported3.162859m gives depth fraction.245351 and joint.244035; this frame
is1mm inside the depth boundary. The two positive-zone masks also occur in21/4
Core negative frames; fixed support-magnitude tuple dominance counts78/7. Neither
observation proves useful rescue or full64-vector inseparability. No candidate,
retuning, new capture or automatic connected-support successor. Retain Calibration,
one-frame hold and this COMPONENT diagnostic. Independent feature/native checks
pass; global metadata remains pending ledger303/unknown-terminal with receipts.

## Causal readout: hold helps continuity, rising extrapolation fails onset rescue (2026-09-20)

[Two consumed432 fixed replays](nearfield/CAUSAL_EVENT_READOUT_RESULTS_20260920.md)
keep T=0.4071309640537889, scalar scores, UNKNOWN and all layouts. Primary Core
strong73/8/5; rising73/17/5; one genuine-strong-backed held frame76/9/2;
combined76/18/2. Combined retains12/12events and BODY36/36, restores worst HEAD
4/7 to6/7, but both HEAD first alerts remain delayed.2s. Rising adds9 CoreFP and
no CoreTP; hold adds1FP and recovers3TP, including one Calibration FN with no
current raw support. This is decision continuity, not recovered current evidence.
Old consumed Core similarly gives rising0TP/+11FP versus hold+1TP/+1FP.
Boundary and pre-entry costs remain explicit; no fresh transfer or promotion.
Retain Calibration. Hold is separately scoped COMPONENT evidence for continuity
(76/9/2, F1 93.25%), not current sensing or onset recovery; Boundary remains3/12.
Close exact rising onset rescue and the combined onset-recovery role as NEGATIVE_CONTROL.
The assumed smooth HEAD entry pattern and spatial sufficiency are unproven.
First-alert evidence remains unresolved. No new experiment, parameter sweep or automatic
connected-support successor. Independent864-frame five-arm recount, prefix causality
and seals pass; no persistent resources. Global metadata pending ledger303/unknown
terminal, with CLI receipts and local structured disposition.

## Core workpoint: strong FP reduction, fixed transfer fails onset/coverage (2026-09-20)

[Consumed-score diagnostic](nearfield/CORE_WORKPOINT_RESULTS_20260920.md) finds
0/126 definite-bypass Core FP. Full-Core-TP threshold control gives72/29/0;
event/onset ceiling gives71/10/1 with12/12 original onsets, one.2s trailing loss.
[One frozen new-layout transfer](nearfield/CORE_WORKPOINT_TRANSFER_RESULTS_20260920.md)
compares Calibration77/128/1 with fixedT=0.4071309640537889 candidate73/8/5 on
Core288 (78P/210N). FP26segments/25.6s falls to7/1.6s; OUTSIDE90FP falls to0,
INSIDE pre-entry38 to8. Both detect12/12 Core events, but two HEAD onsets delay.2s
and one horizontal HEAD event covers4/7. Boundary events12/12 fall to3/12.
Retain Calibration; close this exact transfer as NEGATIVE_CONTROL, retain the
consumed diagnostic as COMPONENT. No threshold retune, RGB reopening or automatic
successor. Seals and independent recount pass; task processes released. Global
registration/inheritance remain pending ledger303/unknown-terminal; local structured
dispositions and CLI receipts are preserved.

## Return-lineage audit: positive foreground survives but can lose selection (2026-09-20)

[One frozen Core432 + Thin96 diagnostic](nearfield/RETURN_LINEAGE_RESULTS_20260920.md)
reproduces all528 original observations/predictions. Four of five negative gap
frames retain the target but select backdrop; their three initial switches all
coincide with backdrop entering the proxy's8m candidate gate. The fifth retains
a target return. Do not repair these negatives or infer a hardware fault.
In positive sampled corridor zones, far non-target winners occur94/500 Core
and95/187 Thin zone-frames (55/144 and24/36 frames). Core retains all144TP;
Thin's six existing small-foreground FN all retain near candidates suppressed
by backdrop. This is candidate-selection loss, not demonstrated FN recovery.
One Core true near-to-far switch also exposes10cm bin-boundary sensitivity.
Retain Calibration and RGB/noRGB/Zone-Handoff closures. This evaluator-only
lineage evidence is COMPONENT_OR_CHALLENGER in COMPONENT mode; no new return
arm or automatic successor. Independent replay/recount passes; global metadata
remains pending ledger303/unknown-terminal, with receipts and local disposition.

## Core usefulness: existing results reassessed (2026-09-20)

[Saved-decision stratified review](nearfield/EXISTING_RESULTS_STRATIFIED_REVIEW_20260920.md)
recounts old BODY/HEAD A*70/1/2,12/12events; Raw/Single decisions are identical
there. Public-positive v2 retains one HEAD configuration's five-frame gain;
CCRL/tail provide no Core upgrade. These four-sensor results do not validate
the current RGB+ToF task. Current Calibration Core-layout288 is72/126/0,
12/12events,24false segments; retain it and prioritize false-alert burden.
NFO near-area gains remain pixel components, not physical Core event evidence.
Posthoc accounting preserves old full/challenge results, frozen gates and
inheritance roles. No model run, new threshold, capture or successor follows.

## Score-factor audit: dominant overlap sensitivity is not a causal geometry fault (2026-09-20)

[Four gaps/five frames,13-frame factor audit](nearfield/SCORE_FACTOR_COLLAPSE_RESULTS_20260920.md)
finds4overlap-replacement-only cases and1either-factor-sufficient case;0depth-only.
But fixed-zone tracking shows outgoing high-overlap winners become far-return
intervals (d=0) in those four frames; incoming winners retain depth support with
tiny overlap. Three fixed-gap-zone CFs are undefined, one reference itself fails
threshold, one is ambiguous. Neither dominant-depth collapse nor wrong geometry
is established. All gaps are strict negatives, not missed true obstacles.
Retain Calibration144/146/0,24/24and onsets; Zone-Handoff and RGB/noRGB stay closed.
No formula/output/threshold change or successor. If requested later, isolate the
outgoing-zone return lineage before choosing a range-quality or geometry method.
Frozen source/factor checks retained; global metadata still pending ledger303.

## Zone-handoff ceiling fails the gap-bridging condition (2026-09-20)

[One frozen consumed432 spatial diagnostic](nearfield/ZONE_HANDOFF_CEILING_RESULTS_20260920.md)
bridges only1/4 gaps (1/5frames), even though three gap frames have adjacent,
depth-compatible corridor-continuous top2 support. The other four frames remain
below threshold even under unconstrained all-positive-zone sum. Hypothetical
component sums give144/150/0 versus Calibration144/146/0, FP segments40->39,
OUTSIDE FP87->90;24/24events and onsets stay identical. Four new FP comprise one
bridge and three collateral frames. The >=3gap/<=37segment implementation gate
fails. Close this exact aggregation explanation as NEGATIVE_CONTROL; retain
Calibration baseline and existing RGB/noRGB closures. No threshold/overlap change,
training, runtime promotion, temporal layer or score-dip successor. Feature seals,
baseline parity and focused checks retained; global metadata pending ledger303.

## Calibration adopted baseline; proposed soft footprint already present (2026-09-20)

User adopts [Calibration as the camera-forward branch baseline](nearfield/CALIBRATION_BASELINE_20260920.md):
RETAINED_CORE144/146/0, F1 66.36%,24/24events and all first in-event alerts retained;
OUTSIDE-layout FP87, false segments40. Exact RGB/noRGB vetoes are disabled in the
branch's active method and retained as negative controls. Historical runners and
App defaults stay unchanged. The existing score already integrates full-zone
footprint overlap across depth; calibration selects a threshold, not extrinsics.
Sealed432-frame/22,371-zone identity audit passes. Raw37 FP segments map to one
removed,32single and four split-in-two segments, giving40; soft-score threshold
crossings and winning-zone changes are observed, angular jitter is not established.
No new soft predictor, repeated weighting, training or temporal successor ran.
Versioned local disposition records adoption; global inheritance remains blocked.

## Frozen Core transfer: calibration retained, RGB lossless-transfer fails (2026-09-20)

[One frozen36-scene/432-frame validation](nearfield/CORE_TRANSFER_RESULTS_20260920.md)
uses new BODY/HEAD targets, Brick/Wood backdrops and independent full-extent
INSIDE/BOUNDARY/OUTSIDE arrangements under the shared simulator. Raw144/181/0
becomes calibrated144/146/0 with all24events and onsets retained; false segments
increase37->40. Frozen noRGB137/138/7 and RGB134/135/10 both fail lossless transfer.
RGB improves OUTSIDE-layout FP87->79 (noRGB81), but loses10calibrated TP, one
contact event, delays two by0.2/0.4s and suppresses30native-contributing zone
samples. Core INSIDE events stay12/12 while boundary events drop12/12->11/12;
one INSIDE TP is lost, so Core event recall alone conceals the failure.
Retain calibration's scoped frame/duration benefit with fragmentation disclosed;
close exact frozen RGB/noRGB vetoes as negative transfer controls. Old96 fitting
and Thin-object Challenge remain unchanged. No fit, retuning or successor.
Source/seal/native audits and independent recount pass; task processes released.
Global metadata remains blocked by ledger303/unknown terminal; receipts and local
disposition retained. This is controlled new-arrangement evidence, not natural,
hardware, new-world, body-trajectory or safety validation.

## Local RGB head fits lateral attribution with matched-control increment (2026-09-20)

[One frozen local RGB pilot](nearfield/TOF_LATERAL_ATTRIBUTION_20260920.md) keeps
calibrated A30/15/6 and compares Otsu B30/15/6, RGB C30/10/6, matched noRGB30/13/6
on the same consumed96. C removes all5 eligible lateralFP, retains original30TP,
4/5 interior and5/6 total events, all first alerts, and every native corridor
contributor. Precision66.67%->75%, FPR25%->16.67%, false duration3s->2s; segments
stay4. noRGB reduces2FP but fragments false segments4->6. B suppresses4zone votes
without changing a frame. UNKNOWN stays86; all raw4912anchors remain unchanged.
This is a5915-parameter in-sample fit (60crops,400updates per matched arm), not
held-out transfer: all13OUTSIDE labels come from one physical arrangement.
Retain C only as a COMPONENT with3FP increment over matched noRGB. B is a scoped
negative control; A remains frozen. One3.007m lateral FP is ineligible;9depthFP
and the fifth missing event remain. No cutoff change, new capture or successor.
Core synthetic/CUDA checks pass; pre-fit backend API repair preserves sealed
inputs and prescribed fits. Global metadata still blocked by ledger303/unknown
terminal; local disposition/receipts retained and task compute released.

## Fixed ToF score reduces in-sample false alerts (2026-09-20)

[One calibrated readout on consumed96](nearfield/TOF_CORRIDOR_CALIBRATION_20260920.md)
retains every original30TP and reduces nominal45-FOV FP19->15, with6FN unchanged.
Recall stays83.33%, precision61.22%->66.67%, FPR31.67%->25%; false segments6->4,
sampled duration3.8s->3.0s. Interior events4/5 and all events5/6 retain first
in-event alerts; removed pre-entry false warnings move two clip-first timestamps.
All4,912 valid returns and original intervals remain; suppressed frames lose0
native corridor contributors. UNKNOWN stays86 and TN0. All6 lateral-only FP
remain, and the same-zone small-object event is still missed.
Retain only a consumed in-sample COMPONENT: the same96 labels select and report
one maximal TP-preserving cutoff on a frozen geometric score, not a probability.
Six synthetic checks pass; one evaluation schema repair preserves sealed scores
and predictions. No new observations, RGB, models, training or successor.
Global metadata remains pending ledger303/unknown-terminal with local disposition
and receipts. All task-owned compute is released.

## Nominal ToF FOV exposes returns but worsens false alerts (2026-09-20)

[One fixed96-frame nominal45x45 contrast](nearfield/TOF_FOV45_20260920.md)
changes only the8x8 bin layout on the original sampled depth. The center cylinder
and thin plate gain pure near returns in5 event frames each; the small same-zone
object remains unanchored. Raw ToF interior events rise2/5->4/5, TP12->30 and
FN24->6, but FP5->19, FPR8.33%->31.67%, precision70.59%->61.22%, and false
segments2->6. This retains a geometry-sensitive observation component, not an
overall detector improvement. Prediction-UNKNOWN remains86/96; TN remains0.
The narrower footprint loses21,032/31,416 old covered image samples; all target
and reference-corridor surface samples happen to remain covered in this cohort.
This is nominal geometry on an uncalibrated optical-Z/noise/return proxy, not
physical VL53 evidence. Nine synthetic checks pass; predictions were sealed before
ownership scoring. No model, new native capture, training or successor ran.
Global registration/inheritance remains pending ledger303/unknown-terminal;
local disposition and receipts preserve that metadata limitation.

## Cross-zone existing-return ceiling is narrow (2026-09-20)

[Fixed596-frame attribution audit](nearfield/CROSS_ZONE_ANCHOR_CEILING_20260920.md)
replays every saved scalar and baseline exactly, with no new models or observations.
Only1/4 originally missed corridor interior events has a pure same-frame near
anchor elsewhere: horizontal bar s02, five interior frames, first at1.4s versus
entry1.2s. None has an entry witness; center cylinder, thin plate and same-zone
small object have no target-owned return anywhere. Raw ToF already detects s02.
The separate500 VAL far_small component envelope covers4,539/6,079 FN pixels
(74.67%), but8-connected reference-near regions can merge different objects.
This is a privileged simulation availability ceiling, not a recovered prediction,
valid pixel ownership, FPR gain or physical weak-peak result. Original NFO remains
1/5 interior events; far_small precision11.73%, IoU11.13% and FP98,788 remain.
Retain only bounded component evidence; no automatic association/training run.
Sixteen synthetic tests pass; global metadata remains pending ledger303 and
unknown-terminal errors, with local disposition/receipts retained.

## Near-return raw input prerequisite absent (2026-09-20)

[Existing-input audit](nearfield/TOF_NEAR_COMPONENT_AVAILABILITY_20260920.md)
checks 1,044 records: corridor96 and NFO500 retain single simulated distances;
real ZJU160 retains Gaussian location/scale parameters, not measured time bins;
retained A*288 already includes403 double-return zones, all hypothetical.
No existing measured weak-foreground peak stream was found in these fixed inputs.
The user confirms no owned8x8 board/captures. Physical near-peak existence remains
NOT_EVALUABLE, not a negative result. The actual Android scalar ToF4M adapter and
research multi-slot schema do not establish an integrated8x8/CNH acquisition path.
No inference, training, simulator, device operation, new capture or successor ran.

## CPU contour-parallax recipe fails (2026-09-20)

[One fixed 12-window / 36-frame reuse of G6](nearfield/BA_CONTOUR_PARALLAX_20260919.md)
uses sparse line profiles and ideal metric poses, with no model, training, ToF
input or new capture. It recovers 0/4 original translational 2–3 m misses.
Far bars and wall mark have 21/3/34 target candidate samples and zero target
false-near; their sparse coverage and frequent abstention do not prove reliable
far ranging. Both pure-yaw windows keep every distance UNKNOWN. Across all
windows, 104 non-target false-near samples remain; the accepted native-depth
interval coverage is 5,853/7,695 (76.062%). Different proposal sets prevent paired
false-positive claims against old G6. The old cached baseline replays exactly.
Single-logical-core algorithm P95 is 554.568 ms against a frozen 200 ms allocation;
peak RSS is 67.484 MiB. This is a desktop CPU proxy, not edge-device validation.
Close the exact recipe as NEGATIVE_CONTROL; no automatic parameter, speed or
successor rescue. This does not reject every contour or layered representation.
Twelve predictions were sealed before native scoring. Actual work includes one
extra unsealed call lost to scalar JSON serialization, with the failure retained;
an old-report schema repair also preserves its failed code and logs. Global
registration/inheritance remains pending ledger303/unknown-terminal errors;
local disposition and receipts are retained. No owned compute remains active.

## Frozen camera-corridor replacement fails (2026-09-19)

[One new controlled 96-frame / 8-clip comparison](nearfield/BA_CAMERA_CORRIDOR_20260919.md)
uses the fixed camera-forward volume and retained weights, without training or
threshold search. Interior events: raw ToF 2/5, retained NFO 1/5, Depth Pro plus
existing global ToF correction 0/5; all events 3/6, 2/6, 0/6. Frame TP/FP/FN:
12/5/24, 7/5/29, 0/0/36. False-alert segments/duration: 2/1.0s, 3/1.0s, 0/0s.
The candidate's zero false alerts accompany zero detections; the frozen retain
gate fails. Raw ToF and NFO already alert before the large-plate entry, so their
zero entry-relative delay is not evidence of timely onset discrimination.
All three miss the thin center cylinder, thin plate and same-zone inside object.
UNKNOWN remains 96/1/0 frames; raw alerts are all spatially ambiguous.
Retain inherited NFO only as a comparator, not as an adequate solution; close
this exact global-scale corridor replacement as NEGATIVE_CONTROL. This does not
reject all spatial representations or reopen earlier near2m negatives.
Source admission passes all96; two failed mechanical/source attempts are retained,
including texture-loading rejection before model scoring. Predictions were sealed
before evaluator access. This is posed synthetic Development, not real-time,
hardware, natural-distribution or safety evidence. No automatic successor.
Global metadata remains pending ledger303/unknown-terminal errors; local structured
disposition and failure receipts are retained. All task-owned compute is released.

## Existing ToF helps Depth Pro; frozen layer recipe rejected (2026-09-19)

[Same500-frame global versus layered echo comparison](nearfield/BA_NFO_DEPTHPRO_ECHO_20260919.md)
reuses native Depth Pro and saved single returns, with identical predicted
histogram-peak pairs. Global improves bare Depth Pro far-small recall61.29%
->67.84%, IoU23.93%->42.62%, pure-far FP344,336->19,236; mixed recall88.08%
still misses94.5%. Layered recall58.66%, IoU38.81%, mixed90.22%, FP21,845:
neither passes the two recall gates, and layers do not dominate global.
Retain NFO as task baseline, global as descriptive calibration component,
and this exact bounded-region recipe as NEGATIVE_CONTROL. This revises the
prior range-priority interpretation: RGB-only failure did not exhaust existing
ToF. 6,094/7,942 layered far-small misses lie in algorithm-assigned regions;
assignment is not proof of correct near-surface anchoring. No automatic tuning,
new hardware inference, training, fresh test, App change or successor.
Old18-view G5 fusion is NOT_EVALUABLE_NO_FROZEN_TOF, not a negative result;
its prior RGB-only diagnostic remains separate. Four synthetic tests and an
independent500/4-arm/11-domain recount pass;350,140 UNKNOWN and9,903,793
unanchored prediction pixels are retained. CPU postprocessing7.606s, no models.
Global metadata remains pending ledger303 fingerprint/unknown-terminal errors;
structured local disposition and failure receipts are preserved.


## Depth Pro reference: contour gain, near2m task fails (2026-09-19)

[One frozen500-frame comparison](nearfield/BA_NFO_DEPTHPRO_20260919.md) completes
official Depth Pro with low/native RGB and public camera rectification, plus18
separate old G5 views. Native contour F1 improves (matched70 SI:25.21% UniDepth
to51.28%; near-mask boundary:4.61% NFO to44.43%), but far-small recall falls
68.36%->61.29%, mixed recall95.00%->81.64%, pure-far FP212,988->344,336.
Far-small IoU rises11.13%->23.93%; this does not pass the joint task. Low also
fails. Native detail and inherited sampling differences prevent attribution
solely to pretrained representation. Old bar optical-z median improves14.21m
to2.875m versus1.947m reference, but remains0/1,054 pixels below2m.
Retain NFO; exact standalone Depth Pro2m replacement is NEGATIVE_CONTROL,
native shape/range-error evidence remains diagnostic. Prioritize near-layer
range evidence over another segmentation-head round; no successor started.
Six focused tests and independent all500/3-arm/11-domain recount pass;350,140
UNKNOWN pixels preserved. No originaltest, training, threshold sweep or App
change. Global registration is blocked by ledger303; terminal is unregistered,
with structured local disposition and actual failure receipts retained.


## NFO full-training support contrast: recipe rejected (2026-09-19)

[Full3,000-frame paired training and500-frame Development check](nearfield/BA_NFO_FULLSUPPORT_20260919.md)
uses original NFO, shared from-scratch initialization, identical12epochs/1,500
batches and fixed0.081. Control vs full-support loss: far-small recall68.27%
->62.56%, IoU11.25%->12.30%, mixed recall95.05%->90.40%, pure-far FP208,202
->196,151. Both recalls decline in all6 Development families; net1,097 extra
far-small FN. Outside-ToF FP grows238,643. Candidate passes2/4 usertargets,
fails paired recall guards; retain original NFO and close this exact loss recipe.
Coverage includes254,276 small-near TRAIN pixels,133,697 at1.8–2m; no selected32
or depth-gap filtering. This rejects the full-coverage loss recipe, not all
spatial architectures or a causal proof that data coverage cannot matter.
Original baseline exactly reproduced; fresh control differs slightly. Two
loss/mask tests, all500 independent counts and44-frame three-arm reload pass.
Consumed synthetic Development only; no test, sweep, successor or App change.
Global registration/inheritance remain pending at existing ledger303 failure.

## NFO frozen cross-scene transfer check: adaptations rejected (2026-09-19)

[Three frozen models on all 500 original val/Development frames](nearfield/BA_NFO_FROZEN_TRANSFER_20260919.md)
close the 32-frame adaptation sequence. Far-small recall: original68.36%,
joint62.37%, late60.12%; IoU11.13%,10.89%,11.64%; mixed recall95.00%,87.80%,87.25%;
pure-far FP212,988,367,659,273,576. Joint meets0/4 user targets, late1/4. Both
candidates lose far-small and mixed recall in all6 scene families. Preserve
original NFO as reference (it still misses75%recall), reject these fixed checkpoints
as transfer upgrades, retain local fit/ablation evidence and all old gate failures.
This is separately authorized consumed-Development transfer, not new blind evidence
or a retroactive pass of14/16. All500frames are image/scene/family-disjoint from
both original training and32-frame fitting, but were used in prior calibration.
No training, cutoff change, original test access or successor. Exact baseline,
independent500-frame/domain/family recount and44-frame three-arm checkpoint reload
pass. Stop same-checkpoint/small-set repair; future work needs representative
training coverage and a distinct cross-scene question. Global metadata still pending.

## NFO fixed-checkpoint branch ablation (2026-09-19)

[Disable only the final late residual](nearfield/BA_NFO_LATE_BRANCH_ABLATION_20260919.md):
102/183 prior lost small-near hits recover; 81 remain missed. But target IoU falls
70.07% -> 29.05%, pure-far FP rises 9,496 -> 24,688 and zone coverage falls 12/16 ->
7/16. The branch provides material far rejection with near-retention costs; off is
not an adopted repair. All-small residual effect is 135 base hits suppressed and
36 base FN recovered. Selected TP totals both398 hide one hit lost and one rescued;
all four previous high-coverage FN are detected by base alone, only three with branch.
The off state uses co-trained final base weights, not an independent trained baseline
or a counterfactual frozen-base run. No causal training-history or raw-head sign claim.
All32 saved-full/zero-correction/base equivalence and independent set-count checks pass;
no training, val/test or successor. Original checkpoints and12/16 failure retained;
global registration/inheritance remain pending at the existing ledger error.

## NFO spatial tradeoff attribution (2026-09-19)

[Saved-output audit of all 126 small-support zones](nearfield/BA_NFO_SPATIAL_TRADEOFF_AUDIT_20260919.md)
resolves the net 122 extra FN into 183 previous hits lost and 61 misses recovered.
All losses occur outside the 16 selected zones, still on the same supervised TRAIN
images. Depth 1.8-2m contributes 138 losses/23 rescues; <=1.8m contributes 45/38.
The original depth-separation + far-ToF stratum (30 zones) has 0 losses/5 rescues,
but is not a new admission rule or promotion denominator. Keep all valid <2m
positives, other-zone costs and the failed 12/16 coverage gate. Selected-pixel loss
coefficients are about 101x other pixels under the existing half/half means, identically
in both arms; this is not a causal explanation or measured gradient ratio.
Independent all-zone set-difference and source/output checks pass; no new training,
inference, validation, test or successor. Global registration/inheritance pending.

## NFO late spatial-conditioning contrast (2026-09-19)

[One fixed 512-update architecture contrast](nearfield/BA_NFO_LATEFUSION_20260919.md)
adds a zero-initialized full-resolution RGB/ToF residual branch (+8,804 parameters).
Same 32 TRAIN cases and joint loss: target recall 98.75% -> 99.50%, IoU 66.72% ->
70.07%; full recall 98.89% -> 99.50%, pure-far FP 16,645 -> 9,496. Empty-native-near
FP 125 -> 96; three of four high-coverage FN recovered. Coverage rises 11/16 ->
12/16, still below 14/16; five original gates and all four paired guards pass.
Broader small-support recall declines 92.43% -> 90.23% (122 extra FN), despite
IoU improvement. Retain component evidence and this cost, not a promoted model.
Exact public zone lookup and all-32 zero-initial identity verified; two tests and
independent final-weight reload/count checks pass. Parameters and late conditioning
are coupled; no isolated mechanism or generalization claim. This recipe ends with
no val/test, sweep or successor. Original/joint checkpoints unchanged. Global
registration/inheritance remains pending at the existing ledger error.

## NFO native-support audit (2026-09-19)

[All 16 positive TRAIN cases inspected](nearfield/BA_NFO_NATIVE_SUPPORT_AUDIT_20260919.md):
original 1024x768 RGB/depth exactly reproduce 256x192 inputs. Only 38/400 near
labels cover a minority-near native footprint. Of 192 joint-fit FP, 125 have no
native near coverage; four FN have mean 96.875% near coverage, the fifth 12.5%.
Input averaging is not supported as the principal failure explanation.
Every case contains cross-zone near support; connectivity does not establish
object identity or invalidate the mixed-cell task. Keep all cases and the 11/16
failure unchanged. No automatic high-resolution fit or cohort replacement.
A future bounded spatial-localization contrast may address spill and missed
support; no successor was run. All source/array/confusion checks and all 16 visual
inspections pass. No training, model inference, validation or test access.
Global registration/inheritance remains pending at the existing ledger error.

## NFO fixed joint-supervision diagnostic (2026-09-19)

[Fixed 0.5 full + 0.5 target loss](nearfield/BA_NFO_JOINTFIT_20260919.md) on the same
32 TRAIN cases gives target recall 98.75%, IoU 66.72%, FP 192; full recall 98.89%,
pure-far FP 16,645 retain the original 32-frame baseline. Five of six gates pass,
but local coverage 11/16 misses 14/16. Validation is therefore skipped, 0 frames
read; no test/transfer claim. Retain joint-supervision contribution as COMPONENT,
not a promoted model. Three remaining zones have excess FP, two lose near support.
The fixed 512-step sequence ends with no ratio/threshold/budget sweep or successor.
Two tests and independent all-32 reload/count/batch/skip checks pass; original NFO
unchanged, global metadata pending at ledger line 303. Full-only fit retains stronger
global metrics, so the joint recipe does not dominate every previous control.

## NFO paired supervision-domain diagnostic (2026-09-19)

[Same32TRAIN cases, target-zone loss](nearfield/BA_NFO_TARGETFIT_20260919.md)
changes only supervised pixel support, retaining original initialization,
512batches/updates, original loss formula and0.081cutoff. Versus full-image
fit, target IoU25.83%->60%, FP1110->240, recall97.5%->96%, local passing
zones1/16->11/16. It still misses65%IoU and14/16gates. Full-image recall
falls99.40%->58.80%, IoU80.93%->29.34%, pure-far FP4937->70192.
Retain the supervision-domain contribution as COMPONENT diagnostic evidence;
do not replace NFO or infer an information/architecture ceiling or generalization.
Two loss-gradient tests and independent all32output/count/support/batch checks
pass. One fit ended, no tuning/successor; global metadata remains pending303.

## NFO difficult-case learning check (2026-09-19)

[One TRAIN-only32frame fit](nearfield/BA_NFO_HARDFIT_20260919.md) uses16depth-separated
small-support cases plus16textured pure-far controls, original network/loss/inputs,
512updates and frozen0.081. Target recall71.25%->97.50%, IoU13.62%->25.83%,
FP1693->1110; negative target FP0->13/7689. Only1/16positive zones meet local
recall90%/IoU50%; aggregate IoU65% and14/16coverage targets fail.
Retain the learnability diagnostic as COMPONENT, not a replacement model.
The fixed full-image-loss fit remains incomplete; neither RGB information
limits nor network incapacity is established. No validation/test evaluation,
recalibration or successor. Two tests and independent all32weight-reload/count
checks pass; original NFO unchanged, metadata pending ledger303.

## NFO-ZCR closes the current readout iteration (2026-09-19)

[One frozen-NFO zone-median contrast](nearfield/BA_NFO_ZCR_20260919.md)
gives far-small recall69.612%->99.740% but IoU12.090%->6.357%, mixed
recall94.699%->94.285%, pure-far FP157,187->9,876,975. Three user gates fail.
Validation alone selects residual cutoff-0.1943483. At this point mixed recall
is already below94.5% and pure-far FP above cap; monotonicity proves no global
cutoff can meet both on these fixed scores. AP14.701%->24.165% is retained
ranking evidence, not a successful near mask or proof all margins are weak.
Keep original NFO; ZCR and eight-parameter Zone Readout remain negative controls.
No MAD, regional threshold/weight, new fit or successor. Three focused tests,
independent public inference/NumPy median and exact baseline reproduction pass;
one NCHW layout repair is documented. Metadata remains pending ledger303.
The user's next direction readout is fixed image-center/left/right sectors;
defer implementation until a useful mask, superseding the 3D corridor proposal
below for that future readout. No body-frame reconstruction or alert claim.

## Adopted camera-forward task (2026-09-19)

[Camera-forward contract](nearfield/CAMERA_FORWARD_CONTRACT_20260919.md): optical axis defines forward; use RGB + 8x8 ToF and a fixed camera-frame corridor (0.3-3m forward, +/-0.3m lateral, explicit fixed vertical profile). Evaluate paired corridor alerts and UNKNOWN coverage; IoU remains a component metric. NFO threshold scores are not exact depth. No body-heading/IMU compensation, future trajectory or SLAM work is required. Definition adopted; runtime integration and alert validation are pending. Historical results below retain their original task.

## BA-NFO local diagnosis and conditional-readout contrast (2026-09-19)

[One local diagnosis and one frozen-backbone readout](nearfield/BA_NFO_ZONE_READOUT_20260919.md)
find NFO median within-zone AUC0.9465, selecting an8-parameter public-zone
offset rather than a decoder fit. Conditional AP14.701%->15.857%; at65,965FP
diagnostic recall69.612%->71.401%. At frozen0.081, however, far-small recall
falls63.596%, FP65,965->50,119, and validation recall94.598% fails95%.
Mixed IoU51.042%->51.906%; pure-far FP157,187->106,249. This is improved
ranking plus suppression, not foreground rescue. Keep NFO; fixed-cut replacement
is NEGATIVE_CONTROL with ranking/local-diagnostic evidence retained.
One12epoch CUDA fit complete; base/gate-out unchanged,4tests and independent
public inference pass. No recalibration, alternative branch or successor fit.

## BA-NFO conditional ranking diagnosis (2026-09-19)

[Four frozen-model curves and spatial audit](nearfield/BA_NFO_CONDITIONAL_20260919.md)
find Hybrid AP14.606% versus NFO14.701% in449far-return small zones. At80%
recall Hybrid needs96,944FP versus91,526. Positive/negative mean score shifts
are+0.00941/+0.01082; the lower Hybrid cutoff also contributes to near expansion.
Curves cross locally: no overall ranking gain, but no exact translation claim.
Local rescue spill exists, while35.44%of new FP occur in no-rescue frames.
Keep original NFO and Hybrid NEGATIVE_CONTROL; retain this diagnostic component.
No fifth fit, calibration adoption or local-branch implementation. A structured
local mechanism remains a hypothesis with an unresolved observable trigger.
All original2m counts reproduce,3focused tests pass; consumed Development only.

## BA-NFO auxiliary-depth matched result (2026-09-19)

[One hybrid fit](nearfield/BA_NFO_HYBRID_20260919.md) freezes train-only lambda0.89323,
adds17 training parameters and removes the auxiliary head for inference.
Far-return small recall69.61%->79.98%, but IoU12.09%->10.02%; mixed
IoU51.04%->49.34%, recall94.70%->93.28%. Joint target fails. Retain original
NFO and this recipe as NEGATIVE_CONTROL; no lambda tuning or successor.
Both controls reproduce all original counts; actual pruned inference passes.
One12epoch CUDA run complete and process released. Global metadata pending
ledger303/unknown terminal; local structured disposition and weights retained.

## BA-NFO trained RGB-only control (2026-09-19)

[Matched RGB-only fit](nearfield/BA_NFO_RGB_CONTROL_20260919.md) completes12epochs
with original split/loss/budget and identical surviving initial parameters.
Far-return small foreground recall97.89% but IoU6.35%,FPR96.26%, versus fusion
NFO69.61%/12.09%/31.79%. High recall is broad overprediction, not useful RGB
recovery; frozen Non-Veto trigger fails. Retain NEGATIVE_CONTROL for this recipe,
not RGB information impossibility or proven causal veto. No successor training;
actual weight and input-independent inference pass; metadata pending ledger303.

## BA-NFO small-area decomposition (2026-09-19)

[Fixed-model diagnostic](nearfield/BA_NFO_AREA_DIAGNOSTIC_20260919.md) reproduces
all original mixed counts. At2m <=20% area, near-return FN2542->1640 improves;
far-return FN1437->4213 causes the net recall regression. Recovering every
near-return miss alone reaches87.41%, below87.6% target. Do not train the
proposed near-return existence auxiliary as the presumed fix; no new fit or
threshold change. Retain decomposition as COMPONENT and matched NFO gain with
small-area limitation.1m recall improves in every area band while FPR increases;
ultra-near calibration/representation cause remains unresolved. Metadata pending.

## BA-NFO matched supervision completed (2026-09-19)

[Matched two-arm training](nearfield/BA_NFO_MATCHED_20260919.md) supersedes the
preflight-only status below. On 500 training-unseen Hypersim Development frames,
2m mixed IoU47.11%->51.04%, recall93.09%->94.70%, FP pixels536667->470344.
48/57 eligible scenes and all6 held-out families improve. Same trainable RGB/ToF
U-Net,3000train/500val,12epochs/arm; real weights and public-input inference pass.
Small-near-area recall87.62%->83.01% remains a material loss; no thin-rod or alert
benefit claim. Retain COMPONENT_OR_CHALLENGER/CHALLENGER for this synthetic
supervision comparison only. No further fit/oracle/assignment branch launched;
A/A*,Radar and App unchanged. Global registration/inheritance pending ledger303
and unknown terminal; local evidence and disposition retained.

## BA-NFO direct near-field occupancy preflight (2026-09-18)

[BA-NFO protocol](nearfield/BA_NFO_PROTOCOL_20260918.md) is frozen as the
last public-data training candidate, but preflight is BLOCKED_DATA_PREREQUISITES:
no local Hypersim/SANPO-Synthetic and no dense UE RGB+depth+ToF manifest. Existing
MZ120/MZ122 45-cell AABB occupancy is not interchangeable pixelwise near-mask
truth. No download/training/alert change. Resume only with exact data manifest;
one failed gate closes this route.

## BA-RPCC-O0 cardinality curve (2026-09-18)

[Rank-preserving cardinality oracle](nearfield/BA_RPCC_O0_20260918.md): with
GT count and frozen DEPTHOR ranking, best global c=1.20 at recall96.13%
gets IoU78.27% versus75.40% (+2.86), below the +5 training trigger.
Close RPCC training from this evidence; per-zone oracle remains descriptive.
No model/training/alert change; registration pending ledger303.

## BA-Depth 2x2 oracle decomposition (2026-09-18)

[Equal-support oracle](nearfield/BA_DEPTH_ORACLE_2X2_20260918.md) reproduces
DEPTHOR 75.40% IoU. GT count + DEPTHOR ranking reaches87.02% IoU but recall
falls93.06%; DEPTHOR count + perfect placement reaches76.17% IoU, recall96.24%.
Threshold bands show57.51% of current FN have both GT/prediction >20cm from2m;
small calibration is not dominant. Close ZPA/SCDE; count/fraction is only a
candidate question, with recall constraint. Registration pending ledger303.

## BA-ZMR fixed-mass placement (2026-09-18)

[160-frame ZMR](nearfield/BA_ZMR_20260918.md): IoU75.40% DEPTHOR falls
to74.22% mono rank/74.06% connected; precision and recall also fall. Exact
zone/union mass retained. GT-mass rank86.47% is privileged, recall92.74%.
No ZPA training trigger. User closes existence-only SCDE; preserve learned
geometry component evidence. Registration/inheritance pending ledger303.

## BA-Depth support-loss diagnosis (2026-09-18)

[Paired FN audit](nearfield/BA_DEPTH_SUPPORT_DIAGNOSTIC_20260918.md) finds8,468
new FN and4,627rescues (net3,841). All52affected near-return zones retain
predicted near surface;48already satisfy Q10<=d+.1m, covering95.61%of new FN.
Existence-only SCDE training trigger is not met; no training or alert change.
Preserve localization gain and prior negatives; metadata remains pending303.

## BA-Depth public diagnostic (2026-09-18)

[160-frame ZJU-L5 probe](nearfield/BA_DEPTH_PROBE_20260918.md) improves mixed-zone
2m IoU 63.18% to75.40%, but recall97.32% to95.69%; no joint replacement gain.
Retain descriptive localization only. No BODY/HEAD corridor authority without
pose/calibration; no training or alert changes. Prior UE negatives remain.
Registration/inheritance pending existing ledger303.

# Cane-complementary forward perception and DTR history

Updated: 2026-09-12

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).
Current work: cane-complementary, class-agnostic forward obstacle awareness.

## MZ180 public feasibility and corrected audit (2026-09-18)

[MZ180 corrected result](nearfield/corridor_fusion_v1/MZ180_RESULTS_20260918.md)
reproduces public merged UNKNOWN on1920 frames:1228/103/156,142 full TP additions
and44 new FP over A*. Historical privileged proxy has identical true-alert IDs
and first risk-alert times, but22 more FP. Both retain the same98 of103 recoveries.
835 merged slots expose no internal modes; stop this public decomposition recipe.
Withdraw old native-union/strict-oracle claims: the range proxy is not authenticated
component geometry; the oracle removed Radar and retained89/103, not66.
Keep A*; no training, capture, promotion or automatic successor.

## MZ181 fixed angular refinement (2026-09-18)

[MZ181 result](nearfield/corridor_fusion_v1/MZ181_RESULTS_20260918.md) retains all142
extra TP over A* but removes only5 of44 extra FP:1228/98/156 versus public UNKNOWN
1228/103/156. Fixed extra-FP<=25 gate fails (39 remain); stop this recipe, no
MZ182 or automatic geometry successor.8x8..64x64 outputs agree. Remaining32
ToF-only FP have verified feasible public-support points;7 are Radar-sustained.
All true-alert IDs/times and12 advance contacts remain.132 target-linked slots
are removed, including94 in true frames still alerting; do not claim native
contributor retention. Independent witness/count/Radar audit passes; A* remains,
UNKNOWN stays high-recall research control. Metadata pending ledger303.

## MZ182 angular dither ceiling (2026-09-18)

[MZ182 result](nearfield/corridor_fusion_v1/MZ182_RESULTS_20260918.md) captures24
matched scenes/144 native UE collision observations. Single and repeat0x3 both
12TP/12FP; yaw0,+quarter-zone,-quarter-zone gives12TP/8FP.4/12FP removal fails
frozen50% gate despite12/12TP retention. Stop fixed recipe, no phase expansion
or public-association successor. Ideal exact yaw/no packet loss and oracle actor
IDs define the ceiling; object association is not same-point registration.
Native extent coverage is incomplete; no surface/hardware claim. Prior MZ179/180/
181 remain closed. ST contract migration target documented, exact vendor figures
not independently verified (access blocked); no sensor-default change. Metadata
registration/inheritance pending ledger303/unknown terminal; UE resources released.

## MZ183 visible angular ownership ceiling (2026-09-19)

[MZ183 result](nearfield/corridor_fusion_v1/MZ183_RESULTS_20260919.md) applies a
reconstructed visible actor mask to MZ181 valid-ToF support. It retains142/142
extra TP but removes only3/32 ToF-only FP (3/39 total), leaving1228/95/156;
Radar-only7 FP remain. The >=24 FP ceiling fails. Close ownership, do not train
segmentation or start MZ184. Masks were reconstructed from controlled scene
meshes because no instance-ID render was saved; no native depth or hit point
entered readout. A*/Radar remain independent, baseline alerts/events retained.
This reconstructed-mask result does not establish a limit for all RGB observables.
ST multi-target/CNH contract is documented with user-provided official figures;
local vendor fetch was blocked, so no default sensor contract changed.

## MZ178 head/body current-frame diagnostic (2026-09-18)

[MZ178-A result](nearfield/corridor_fusion_v1/MZ178_RESULTS_20260918.md)
recovers103/106 head-turn supported FN (98 target-contributor linked), versus64
for same-rule head-frame control. Transitions93->41, but FP59->162 and side-pass
seconds1.2->11.2 fail the frozen gate. Existing A* already yaw-rotates supports;
this was a direct geometric union readout, not newly introduced compensation.
Stop A, do not launch memory or tune. Retain A*/raw; metadata remains pending.

## Temporal frontier evidence ceiling (2026-09-18)

[MZ179 fixed ceiling](nearfield/corridor_fusion_v1/MZ179_RESULTS_20260918.md)
finds past Radar features separate the six clear A*/raw disagreement cases,
but current Radar range and past model scores also separate them. Cell survival,
density and Doppler overlap; incremental temporal mechanism is unproven.
Withdraw ToF-survival rescue: none of the three TP has past native corridor ToF
support. Tiny source yaw and absent prior support do not establish head-turn
dropout here; keep the MZ177/178 question separate. Retain diagnostic tooling
only, freeze A*/raw, no training/capture successor. Metadata pending ledger303.

## Continuous approach diagnostic (2026-09-18)

[MZ177 complete continuous result](nearfield/corridor_fusion_v1/MZ177_CONTINUOUS_APPROACH_RESULTS_20260918.md)
finishes24 sequences/1920 frames with frozen A*, raw HGB and fixed hysteresis.
All12 moving contact events receive advance alerts, but all A* moving sequences
already alert at frame0: onset is left-censored. A* loses130 head-turn risk frames,
106 with valid target ToF support; hysteresis leaves98 supported head-turn FN.
Retain A*/raw; choose only rotation-conditioned body-frame evidence readout as
an unlaunched candidate. Release is mostly censored; recovered controlled
Development only. Existing ledger line303 fingerprint error leaves metadata pending.


## Saved-score residual budget diagnosis (2026-09-18)

The [posthoc budget diagnostic](nearfield/corridor_fusion_v1/RESIDUAL_BUDGET_DIAGNOSIS_20260918.md)
finds maximum Clear TP at FP<=3/5: raw99/99, A*98/99, B088/91, B197/98,
B293/95. All selected points retain18/18 core events and76/76 native Clear TP,
but residuals delay one rod onset0.25s and exchange strict events. Full plateaus
and event identities are retained. No demonstrated joint low-FP gain supports
threshold-transfer rescue; keep raw/A* and the fixed residual recipe closed.
Saved scores only, ten-point independent count/onset audit passes; no training,
new data or formal-threshold edits. Existing metadata errors remain pending.

## Texture counterfactual training pilot (2026-09-18)

The [fixed B0/B1/B2 comparison](nearfield/corridor_fusion_v1/BG_INVARIANCE_RESULTS_20260918.md)
fails the low-FP target. Old clear raw99/5/9 becomes B0100/16/8,B1103/21/5,
B2104/22/4. B2 retains raw clear TP but loses one strict event and increases
clear false episodes4to9. One288-frame texture-only capture authenticates144
sensor/label-preserving background pairs;192train/96held stay physical-group
disjoint. Held B2 logit drift drops29.14% versus B1 but alerts remain identical:
clear34/0/2,strict40/9/8; A*40/4/8 has fewer strict FP. Keep raw/A*, close this
fixed recipe as NEGATIVE_CONTROL; drift is diagnostic, not causal disentanglement
or alert promotion.30inner HGB/21residual fits and replay/count/event audit pass.
Worker processes/task/port released;455MB owned DDC removed, raw retained.
No tuning/capture successor. Registration/inheritance remain pending ledger303.

## Three-arm representation ablation (2026-09-18)

The [fixed same-data HGB ablation](nearfield/corridor_fusion_v1/REPRESENTATION_RESULTS_20260918.md)
does not establish multi-association superiority. Raw2224 and single2485 both
give clear99/5/9,F193.40%, versus multi/A*96/3/12,F192.75%. Core18/18 and
onsets are unchanged; strict events25/26/24 hide2/1/0 lost baseline events.
Raw loses5 and rescues5 native-supported boundary TP; single loses4/rescues1.
All preserve76/76 clear-native TP. Equal-FP3 report diagnostics give raw99TP
with23strict events, single96/24 and multi98/25: no joint dominance. Keep A*
unchanged and the controls as diagnostic components; the main contribution is
unproven. All21 fits,1632feature parity rows and independent score/count/onset
audits complete. Consumed Development only; no capture, tuning or successor.
Registration/inheritance remain pending existing ledger303/unknown-terminal errors.

## Minimal CCRL ranking result (2026-09-18)

The [fixed consumed CCRL comparison](nearfield/corridor_fusion_v1/CCRL_RESULTS_20260918.md)
retains A*. Clear A*96/3/12,F192.75% becomes101/10/7,F192.24% with matched
BCE residual and101/19/7,F188.60% with ranking. Both rescue6rod frames and
lose1BODY TP; ranking adds16clear FP and delays one retained boundary event
by.25s. Strict events24/30,27/30,26/30; core18/18 throughout. Pair ordering
also favors matched BCE over ranking. Keep the fixed ranking recipe as a
negative control, BCE recall/event tradeoff as diagnosis, and A* as the low-FP
baseline.672lateral pairs authenticate; zero exact nuisance pairs means
invariance is NOT_EVALUABLE.14residual/30inner HGB fits complete once, public
replay/native accounting audit passes, all owned processes exit. No sweep,
capture, App change or fusion successor. Structured metadata remains pending
the existing ledger303/unknown-terminal errors; receipts and weights retained.

## Standalone A* effect delivery (2026-09-18)

The [standalone runtime and complete demo](nearfield/corridor_fusion_v1/ASTAR_EFFECT_USAGE_20260917.md)
load only retained A*. Saved predictions, reminders and display share `alert`.
All288 scores are bitwise equal to the prior comparison's A* `control_score`;
clear96/3/12,F192.75%,P96.97%,R88.89%,core18/18 and strict107/6/37,24/30
are unchanged. Full48-episode A/A* replay includes four lost clear TP, three
rescued FN and one rod onset delayed.5s. Direct host decode/frontend/A* latency
mean66.67ms,p5066.04,p9574.64,max85.00 excludes capture, saving and display.
No speedup is established. Old entry semantics remain A OR positive / A* control.
Keep the public head separately: its posthoc OR had zero clear increment and
one boundary TP/FP tradeoff. No training, capture, threshold or App change.
This is engineering extraction of retained evidence, not a new experiment;
earlier scientific registration remains pending, without ledger edits.

## Single public version and frozen confirmation (2026-09-17)

[One model and one threshold](nearfield/corridor_fusion_v1/PUBLIC_SINGLE_RESULTS_20260917.md)
are implemented on all1344 existing frames, with pooled1152OOF calibration and
an independent same-data retrained-A* control. Both recipes freeze before one
new24-configuration288-frame capture. Clear A/OR97/25/11,F184.35%; all11 A
misses lack returned corridor support. The source therefore does not test clear
rescue transfer; no new clear FP, while boundary adds1TP/1FP. Preserve v2 gain
and the runnable public component without claiming new clear benefit.

A* clear96/3/12,F192.75%, removes22FP/rescues3FN/loses4TP. Core17/18to18/18,
one rod onset delayed.5s; strict107/6/37 versus A111/37/33, events24/30 versus
25/30. Retain this precision-oriented challenger with explicit costs, not an
unconditional replacement. Full host algorithm mean A63.95/OR66.15ms; branch
increment2.20ms. Capture authentication, independent geometry/accounting and
exact public replay pass; owned worker processes/cache/port released. Bundle
and public entry documented. No App promotion, retuning, combined A*+head or
successor run. Registration/inheritance pending ledger303/unknown terminal.

## Public positive pooled calibration (2026-09-17)

[Continued Development](nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_V2_RESULTS_20260917.md)
reuses anchor192 and four complete consumed288 cohorts, with nested scene-OOF
calibration and final refit. Unchanged public1537-parameter BCE head gives changed
clear103/27/5 (F186.55%), rescuing5HEAD+3rod with no added FP across1,152 report
frames. Old/MZ146/MZ158 alerts remain A; changed strict128/45/16, boundary25/18/11.
Core18/18 retained; changed strict remains29/30. Two observed onsets improve;
no A alert delayed. Matched peak loss gives1 more old TP but3 boundary FP.
Retain BCE plus pooled calibration as the simpler Development challenger.
All48 checkpoints and2,304 public replay scores audited; no App/fresh claim,
capture or single all-data model. Structured registration still pending ledger303.

## Public positive evidence implementation (2026-09-17)

The [public return head](nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_RESULTS_20260917.md)
implements a 1,537-parameter shared ToF MLP, native return-level supervision and
positive-only OR with frozen A. Six whole-scene outer folds use disjoint fit,
calibration and report groups; existing sources only, original test excluded.
Selected working points leave old clear 107/22/1 and changed 95/27/13 unchanged.
A single posthoc fixed logit0 diagnostic gives changed 103/33/5 (F1 84.43%),
recovering all 8 sampled-support misses with 6 new rod FP; old clear 108/23/0.
Boundary FP grow to old 23 and changed 33 (from 6 and 18). Keep the public
component and diagnostic signal, not a selected final-alert gain. Calibration
opportunity imbalance and false positive return maxima remain unresolved.
No new data/RGB/depth execution, veto or App change. Public replay and independent
group/weight/threshold audits pass. No automatic successor; ledger303 pending.

## Existing-evidence tri-state readout (2026-09-17)

The user-authorized [existing-data probe](nearfield/corridor_fusion_v1/TRISTATE_EVIDENCE_RESULTS_20260917.md)
finds privileged positive OR clear 103/27/5 (F1 86.55%) and joint positive plus
outside-only veto 102/17/6 (89.87%), versus A 95/27/13. All 72 zero-return
frames remain UNKNOWN and retain A. Joint readout recovers 8 FN, removes 10 FP,
but veto loses one rod TP and delays its first alert by 0.75 s despite 50 usable
background returns. Core events stay 18/18. Full-face strict joint 137/31/7,
30/30 events; sampled/full clear outputs are identical. Retain an evaluator-only
spatial-readout opportunity and positive OR comparator; outside-only returns do
not certify corridor coverage/free space. No public-input result, training,
capture or App promotion. Registration/inheritance receipts remain pending.

## Existing-return surface-support oracle (2026-09-17)

The [new-domain oracle](nearfield/corridor_fusion_v1/SURFACE_ORACLE_RESULTS_20260917.md)
preserves frozen A and replaces only per-slot support endpoints with native
sampled/full-face geometry. Clear95/27/13, strict120/45/24 and every alert remain
unchanged. This is not a global upper bound: A has only1of900splits on support
endpoints. Of13clear misses,8already have returned corridor points(5HEAD3rod),
3rod have no native ray hit and2rod have hits but no usable return. Full faces
add no clear reachability beyond sampled points;4gains are boundary-only.
Among27clear FP,17have no usable ToF returns; lack of support cannot veto
independent evidence. Exact face-only readout84/0/24 would lose19A true frames.
Do not commit to unsampled-extent completion as the main bottleneck from this
cohort. No model/CNH/DA-V2/capture successor. Keep5cm clear+coverage, strict
boundary and alert burden as separate reporting axes. Ledger303 remains pending.

## Existing-data task definition review (2026-09-17)

The user redirects the next step to [stored-prediction tolerance analysis](nearfield/corridor_fusion_v1/TOLERANCE_RESULTS_20260917.md),
separating clear corridor decisions, strict boundary pressure and observed alert
behaviour. Across fixed3/5/10cm lateral tolerances,5cm retains216/288frames in
each domain: old A107/22/1,F1 90.30%; changed A=S195/27/13,F1 82.61%.
All40remaining changed-domain errors are at least10cm from the nominal lateral
boundary. Thus boundary pressure is real but does not explain transfer failure.
Keep original labels/results and closed online DA-V2 disposition. The proposed
intrusion model remains untrained/deferred; capture was cancelled on task redirect
with verified process release, not a scientific failure. Prioritize existing
datasets and disclose their consumed/held-out roles before acquiring more data.
At5cm,18/18observed core events coexist with25%clear-negative alert burden in
the changed domain; pre-entry timing/release are not established by these short
left-censored trajectories. No model/default-App change or automatic successor.

## Balanced A and conditional depth specialist (2026-09-17)

The user adopts [frozen A](nearfield/corridor_fusion_v1/BALANCED_BASELINE_20260917.md)
as the balanced research baseline: consumed MZ170137/28/7,F1 88.67%,30/30events.
MZ129 stays a historical/native-support reference and App defaults stay unchanged.
The [S1 conditional specialist](nearfield/corridor_fusion_v1/SPECIALIST_RESULTS_20260917.md)
changes A to137/24/7,F1 89.84%, with every A true frame, event onset/release and
native-supported missed-warning case unchanged. Actual C calls10/288; matched
mean latency67.16to70.02ms. Gate-only loses6TP, so C contributes in this comparison.
Retain the fixed S1 as a consumed Development challenger for separately scoped
unchanged confirmation; no report tuning, automatic capture or App promotion.
A/C weights remain frozen. Registration/inheritance remain pending ledger303.

The [completed targeted confirmation](nearfield/corridor_fusion_v1/CONFIRMATION_FINAL_20260917.md)
keeps A and S1 identical at120/45/24,F1 77.67%,recall83.33%,29/30events.
C invokes14/288, retains8true and6false alerts, removes0FP. Calls shift to
11HEAD/2boundary/1BODY, with zero rod. Preserve the prior Development gain,
close this online DA-V2 recipe, and retain A with its observed transfer limits.
The arbitrary1200s acquisition cutoff was corrected; identical source completed
in23minutes. No threshold tuning, CNH successor or App change. Metadata remains
pending ledger303; all task processes released.

## Depth-prior E0/E1 exploration (2026-09-17, historical global roles)

The authorized [single intermediate-feature diagnostic](nearfield/corridor_fusion_v1/INTERMEDIATE_RESULTS_20260917.md)
also has no overall gain: positioned DA-V2 tokens/TRAIN-only PCA change A137/28/7
to103/3/41, F1 82.40%, recall71.53%, and miss9/30events. Rod36/0/0 is a scoped
consumed gain; HEAD and boundary collapse prevent a system gain. Pause these
two online DA-V2 feature adapters, preserve original references, and do not start
E2. Complete local p50 is155.2ms versus matched A81.6ms; no App change.

[E1 frozen relative-depth statistics](nearfield/corridor_fusion_v1/E1_RESULTS_20260917.md)
does not improve the matched HGB: consumed MZ170288 changes137/28/7 to135/32/9,
F1 88.67%to86.82%, PR-AUC .9615to.9266, complete p50 78.2to303.8ms.
The retrained A scores exactly reproduce original MZ145 static scores; its
working-point gain is a threshold/readout tradeoff, not new representation.
E0 reproduces all frozen expert scores/flags. Stop this E1 statistics integration;
no E2, fresh capture, or App promotion. Keep MZ129/default and prior references.
Intended NEGATIVE_CONTROL registration/inheritance remains pending ledger303.

## Latest corridor-alert exploration (2026-09-16)

[MZ175 conditional Radar consensus](nearfield/MZ175_RESULTS_20260916.md) retains
a small consumed recall component: TRAIN192 91/46/5 becomes92/46/4 and MZ146288
130/86/14 becomes131/86/13. All old warnings survive; one HEAD onset is0.25s
earlier. Multiple visible identities can agree on a task proxy without unique
association. This does not resolve null/ghost hypotheses or lower baseline FP.
Keep the fixed component for unchanged prospective confirmation; MZ129 remains
the research baseline and App defaults stay unchanged. [MZ174](nearfield/MZ174_RESULTS_20260916.md)
and [MZ173](nearfield/MZ173_RESULTS_20260916.md) stay closed. Two consumed cohorts
do not establish a broad breakthrough; structured metadata remains pending303.

[MZ172 spatial return context](nearfield/MZ172_RESULTS_20260916.md) changes
HELD pooled24/5/0 to registered24/7/0 versus MZ12922/6/2. All old HELD true
frames/onsets survive, but both FIT arms miss three native-supported frames
with33 corridor samples and a whole boundary event. Registered FIT95.83%
does not overcome that loss. Raw/contextual slots remain allnegative despite
parameter updates; no branch or joint warning gain. Close the exact contextual
representation/head/schedule/readout without tuning or capture rescue. Keep
MZ129; independent audit passes, structured metadata remains pending303.

[MZ171 direct return supervision](nearfield/MZ171_RESULTS_20260916.md) changes
matched HELD control24/7/0 to witness24/8/0 versus MZ12922/6/2. Both arms still
classify all382 FIT and168 HELD positive ToF slots negative; every final alert
is pixel-only. Witness FIT96.53% passes, but six boundaryFP remain and HEAD FP
increases1to2. All old HELD true frames/onsets and sampled native support survive.
No declared branch-learning or alert gain; close the exact loss/head/schedule/
zero-cutoff recipe without tuning or new capture. Keep MZ129. Registration
is pending the existing ledger303 error; inheritance is consequently pending.

[MZ170 unchanged mean fresh confirmation](nearfield/MZ170_RESULTS_20260916.md)
reduces MZ129136/74/8 to140/36/4 on288 new same-generator frames, but loses four
old true frames with54native corridor samples, delays two events0.50s/0.25s,
and raises HEAD FP1to6. Against static143/69/1 it loses threeTP and delays two
onsets0.25s. Independent audit confirms the losses and all288 source frames.
Close this exact mean as NEGATIVE_CONTROL, superseding MZ169's eligibility for
unchanged fresh promotion; preserve its historical consumed gain. No weights,
cutoffs, OR, padding or source rescue. MZ129 remains; no App/default promotion;
structured metadata pending303.

[MZ169 expert combination](nearfield/MZ169_RESULTS_20260916.md) retains its
predeclared equal-mean control as consumed Development: dev24/1/0 and MZ158
140/35/4 versus MZ129133/80/11, with all old true frames/event onsets, all-family
FP noninferiority (HEAD7to7) and sampled native corridor support retained.
The learned stack140/28/4 fails: three oldTP lost, two0.25s delays,21native
corridor samples suppressed and HEAD FP8. Close that fixed learned arm without
tuning. Mean cutoffs use separate MZ146 meta-training; four original TRAIN192
experts stay frozen. Retain only the mean as a Development challenger eligible
for unchanged fresh confirmation; no new capture or App/default promotion.
MZ129 remains retained; structured metadata pending303.

[MZ168 causal visual yaw](nearfield/MZ168_RESULTS_20260916.md) accepts26/120
noninitial RGB estimates and lowers all-frame yawMAE.15004to.14070deg, but
exact-range FIT144 remains69/1/3 with identical flags and0.50s boundary delay.
Only2/30boundary opportunities accept; all four original error frames fall back
under fixed inlier/ambiguity rules. BODY spatial precision/recall slightly worsen.
Keep the conditional pose effect, stop this exact estimator without tuning;
no alert or learned-range promotion. MZ129 remains retained; metadata pending303.

[MZ167 first-hit range feasibility](nearfield/MZ167_RESULTS_20260916.md) finds
an oracle public-pose floor before learning: native FIT144 72/0/0 becomes
69/1/3 with exact range and public pose, then 69/5/3 with the fixed6.25cm grid.
Three old boundary true frames are lost (35native corridor samples), and one
event is delayed0.50s. Saved-source audit confirms simulated noisy yaw increments
and correct public accumulation; no privileged correction is applied. Midpoint
and uniform-bin-mass masks coincide. Stop this fixed interface/grid; no learned
distribution was run. Keep MZ129 and prior scoped gains; metadata pending303.

[MZ166 dense virtual queries](nearfield/MZ166_RESULTS_20260916.md) improves
FIT changed-query pair decisions6/42to24/42 (all18gains rod), but worsens
central HELD24/8/0to24/12/0 versus matched central-only training; MZ12922/6/2.
FIT accuracy63.89% fails95%; six boundaryFP remain and BODY/HEAD nuisance grows.
All oldTP/events/onsets and sampled native support survive, including MZ165's
two recovered frames. Keep the scoped FIT query-response effect, not an alert
replacement or held-scene query claim. Stop this exact schedule/head/supervision
recipe without tuning; keep MZ129. Metadata remains pending303; no original
dev/test or fresh confirmation.

[MZ165 pretrained task context](nearfield/MZ165_RESULTS_20260916.md) improves
matched random-encoder HELD15/8/9 to24/7/0, versus MZ12922/6/2. All incumbent
true frames/events/timing and sampled native corridor support survive; both
recovered frames lack returned target ToF evidence. FIT93.06% misses95%, six
baseline boundary FP persist and one HEAD FP is added. Keep the measured direct-
risk pretraining effect, not the complete failed readout as an alert replacement.
Rod pixel precision31.42%/recall99.88%, boundary9.90%/98.93%, and BODY29.03%
recall limit extent claims. No epoch/threshold/encoder-size rescue; MZ129 remains
retained, metadata pending303. No original dev/test or fresh confirmation.

[MZ164 camera-aware metric prior](nearfield/MZ164_RESULTS_20260916.md) alerts on
all192 consumed TRAIN frames: MZ12991/46/5 becomes96/96/0, adding50 false frames
for five recovered true frames. Known-camera rod columns75/423 trail the
estimated-camera control141/423. With no returned target evidence,31/166 rod
columns agree, but74.09% of ring pixels are too near; one gained true frame has
1910 correct target pixels and29419 known false corridor pixels. This is limited
metric agreement without useful object/background or alert discrimination.
Keep MZ129 and the distinct MZ158 partial gain. The fixed metric-prior OR is a
negative control, with no model/threshold/resolution rescue; metadata pending303.

[MZ163 integration robustness](nearfield/MZ163_RESULTS_20260916.md) finds the
MZ155 rod scan VALID episode gain4to6 becomes4to4 under12x12 normalized integration
within the same8x8 zones; prefix24to34 becomes24to24, also without packet loss.
Native3x3 geometry, target identity and returned lineage reproduce. Boundary
valid episodes3to6 remain a conditional mixture/status opportunity, with inside
first-valid delay0.25s; no alert gain evaluated. The grids are nonnested and
neither is calibrated physical truth. Keep MZ129 and original source/default;
avoid another rod micro-scan justified by the old coverage gain alone. Metadata
pending303; consumed diagnostic only. All-packets MERGED-inclusive rod coverage
still improves4to5; this weaker opportunity is not erased by the VALID result.

[MZ162 returned-face correspondence](nearfield/MZ162_RESULTS_20260916.md)
stops before real training: even complete analytic visible faces linked to
returned ToF samples cover only46/152 rod target-risk half-columns (30.26%).
Twenty-three of48 rod frames lack resolved visible target correspondence.
There are54 informative same-zone dual-return pairs, but anchored-face extent
cannot fill unanchored surfaces. Retain MZ129; no alert change, no hardware
ceiling claim. The50% rule was written during the scan before aggregate output;
this is consumed EXPLORE evidence, not pre-parse confirmation. Metadata pending303.
[MZ161](nearfield/MZ161_RESULTS_20260916.md) retains its dense-task failure:
FIT78.47%, HELD22/6/2 to15/6/9, eight old true frames lost, one gained,
one missed event and two delays;34 FIT native corridor samples in nonalerts.
[MZ160](nearfield/MZ160_RESULTS_20260916.md) retains zero added dense-parallax
geometry and unchanged153/50/2. [MZ159](nearfield/MZ159_RESULTS_20260916.md)
retains its reflection failure; [MZ158](nearfield/MZ158_RESULTS_20260916.md)
retains fresh133/80/11 to140/49/4, with HEAD FP7 to16 failing its joint check.
[MZ157](nearfield/MZ157_RESULTS_20260916.md) retains failed global calibration;
[MZ156](nearfield/MZ156_RESULTS_20260916.md) its one-frame static-world component,
and [MZ155](nearfield/MZ155_RESULTS_20260916.md) its original failed gates.

[MZ154 finer RGB sampling](nearfield/MZ154_RESULTS_20260916.md) separates
direct256 guidance from enlarged128 at the same model scale32 and original8x8
ToF. Rod columns196/423 trail control202/423; current-return-absent cases stay
0/175 in all arms. Stop this fixed pretrained resolution follow-up; no alert gain.
Post-outcome source audit finds only6/24absent frames have earlier target returns;
native rod lateral motion follows the camera, limiting static-world parallax
interpretation. A more informative observation needs a distinct matched test,
not another resolution sweep. Keep MZ129; metadata pending ledger303.

[MZ153 causal video depth](nearfield/MZ153_RESULTS_20260916.md) tests frozen
DVSR with past/current RGB+8x8 ToF against repeated current observations on
TRAIN192. Rod surface columns fall36.88%to35.70%; boundary rises84.19%to87.85%
but local precision does not improve. Stop this fixed pretrained transfer.
No current rod ToF return means0/175 recovered columns across24frames in both
arms; this is scoped model/input evidence, not a fundamental sensing limit.
No dev/test model scoring or native-label access; old baseline cache parsed
after sealing uses only TRAIN corrections, with the original broad scope claim
explicitly corrected. Keep MZ129; metadata pending ledger303.

[MZ151 expanded training](nearfield/MZ151_RESULTS_20260916.md) combines original
TRAIN192 with consumed MZ146288. Same-frame OOF discrimination improves, but
dev24/8/0 and five false segments fail the first gate; no fresh capture occurs.
MZ146 is now TRAIN evidence for this model, not its validation. Two low-scoring
boundary/head TRAIN observations determine the permissive low/high cutoffs.
[MZ152 contained-return anchors](nearfield/MZ152_RESULTS_20260916.md) adds a
full-envelope ToF branch without contracting support or requiring RGB. It covers
106 TRAIN true frames with zero false anchors but neither cutoff-support case;
cutoffs and all dev flags stay identical to MZ151. Both exact recipes stop as
intended negative controls, metadata pending ledger303. MZ129 remains retained.
Partial-intrusion geometry remains unresolved; no new App/default/safety claim.

[MZ150 virtual corridor supervision](nearfield/MZ150_RESULTS_20260916.md)
gives24TP/10FP/0FN on consumed dev48 versus MZ14524/3/0: rod/BODY FP0,
HEAD FP4 and boundary FP6, with all five event times retained. Seven TRAIN
queries per image preserve central-query inputs bitwise but fail the first
gate; MZ146 is not scored. A TRAIN-only fixed-model diagnostic finds3 central
fit errors versus26 whole-scene OOF errors at the diagnostic .5 boundary;
cross-scene generalization is the next evidence gap, not a new alert cutoff.
[MZ149 state persistence](nearfield/MZ149_RESULTS_20260916.md) also stops at
its first gate:24/6/0, four false segments and one new HEAD false frame.
Both exact recipes are intended negative controls, metadata pending ledger303.
Keep MZ129 and prior scoped components; no App change or fresh capture.

[MZ148 background residuals](nearfield/MZ148_RESULTS_20260916.md) adds causal
dominant-layer image compensation without metric pose or native support pruning.
Dev48 improves MZ14524/3/0 to24/1/0. On consumed MZ146288 it gives142/43/2
versus MZ145143/57/1 and MZ129130/86/14, including rod FP25to18. All families'
FP improve against MZ129, but one incumbent boundary TP is lost. The two-panel
Development gate fails; retain only the image feature component, not the frozen
complete readout as a challenger. No fresh source or capture was created.

[MZ147 local query pixels](nearfield/MZ147_RESULTS_20260916.md) replaces pooled
RGB hypotheses with per-return native-pixel/body-query features under the same
HGB learner. Consumed dev48 gives23TP/10FP/1FN, versus MZ12924/13/0 and
MZ14524/3/0: a boundary TP is lost, HEAD FP rises0to1, rod FP stays6. Stop at
this first gate; MZ146 is not scored with this candidate. Its read-only rod
diagnostic finds correspondence errors before pooling and intermittent native
near returns; neither quantile pooling nor geometric overlap alone explains
all alerts. This carrier is a negative control, not a new alert challenger.

[MZ146 fresh confirmation](nearfield/MZ146_RESULTS_20260916.md) evaluates the
unchanged MZ143/MZ145 pipeline on288 newly captured same-generator frames.
MZ129130TP/86FP/14FN becomes143/57/1: precision60.19%to71.50%, recall90.28%to
99.31%, events29/30to30/30 and false segments28to21. No incumbent TP or event
is lost or delayed. Rod FP25to29 fails the pre-outcome family condition;
`FRESH_CONTROLLED_CONFIRMATION_NOT_MET`. Preserve this fresh aggregate gain
with its localized regression; keep MZ129/default. The exact confirmation is
closed without threshold/model/source rescue. Same simulator priors are not
natural-scene or device evidence. Returned native ToF corridor samples are
retained in alerted frames; native Radar per-return ownership is NOT_EVALUABLE.

[MZ145](nearfield/MZ145_RESULTS_20260916.md) remains a consumed Development
gain:24TP/3FP/0FN versus24/13/0, all5first-alert times unchanged, false segments
4to2. [MZ143](nearfield/MZ143_RESULTS_20260916.md) alone gave24/8/0 but extra
HEAD/fragmentation nuisance; [MZ144](nearfield/MZ144_RESULTS_20260916.md) direct
geometry gave24/17/0. Keep all frozen evidence and scoped failure boundaries.
Supported registration/inheritance remain pending the existing ledger303
mismatch, separate from these measured outcomes; no manual ledger bypass.

## Active mainline correction: one RGB + ToF + Radar + IMU (2026-09-13)

User priority correction,2026-09-14: fix inexpensive RGB + **8x8 ToF** +
same Radar + IMU; prioritize direct corridor-task learning over further
flow-to-return contraction. MZ129 remains the retained baseline. Native evidence
is preserved for audit; statistical candidates issue independent alarms and must
expose misses, nuisance, timing and UNKNOWN without incumbent OR fallback.

Latest matched learning result: [MZ136](nearfield/MZ136_RESULTS_20260914.md)
compares direct corridor BCE with BCE plus corridor hinge on24new procedural
scene groups,192/48/48train/dev/held-out frames. Both rank24/24dev pairs but
fail a useful common readout: exact dev-score diagnosis needs23FP versus MZ12913
at retained recall/timing. The frozen .01grid selects all-alert; held-out MZ129
22/12/2 becomes24/24/0 for both, with no pairing gain. Shifted ToF leaves
22/13/2 versus24/24/0; all-alert invariance is not robustness. New training fit
also remains incomplete. Keep MZ129; exact recipe intended NEGATIVE_CONTROL,
registration/inheritance pending ledger303. The original comparison is complete.

User-authorized [TRAIN-only fit repair](nearfield/MZ136_TRAIN_FIT_REPAIR_20260914.md)
now gets191/192 frames and95/96 pairs correct with the original zero-logit
criterion, using frozen existing features plus385direct-readout parameters.
The subsequent frozen original-dev48 check needs20FP versus MZ12913 at matched
recall/timing (zero logit misses12/24 positives). Retain only the fitting diagnostic;
scene-dependent readout offsets are the next bottleneck. No new capture or original
test inference; MZ129 remains baseline. See the linked report's transfer follow-up.

[Grouped readout check](nearfield/MZ136_GROUPED_READOUT_20260914.md) reduces
readout-only OOF instability (63.54% to79.17%; backbone saw all TRAIN), but its
frozen dev48 still needs22FP versus MZ12913 at matched recall/timing. Shallow
boundary decisions remain the dominant misses. Keep MZ129; no original-test run.

[Boundary-input audit](nearfield/MZ136_BOUNDARY_INPUTS_20260915.md): observed IMU
rectification and RGB gradients reduce MAE2.886 to1.227px on the same38 TRAIN
frames, but fine-edge coverage is38/48. Coarse seeds and independent native
evidence remain available. Geometry component only; no alert gain or dev/test
scoring. MZ129 retained; component registration/inheritance pending ledger303.

[MZ137 edge-to-corridor contrast](nearfield/MZ137_EDGE_CORRIDOR_20260915.md)
uses one fixed public-ToF plane and matched coarse/fine edges on consumed dev48.
All arms remain24/13/0TP/FP/FN with5/5events and unchanged timing.13geometry
changes yield one correct plane crossing but zero final changes; fine excludes
105previously enclosed native corridor contributors. Reject this exact readout,
retain MZ129 and MZ136's input component. No successor/test run; registration
and inheritance pending ledger303, separate from completed technical evidence.

[MZ138 oracle support ceiling](nearfield/MZ138_SUPPORT_CEILING_20260915.md)
finds complete native-face24TP/2FP/0FN at unchanged event times on consumed dev48.
Angular22/3/2 and returned-point21/2/3 miss unsampled corridor extent; native
point grouping adds zero gain. Full-face oracle restores three shallow positives,
preserves1046corridor contributors, and leaves two fixed-Radar FP. Retain only
surface-extent headroom evidence, with disclosed posthoc full-surface scope
completion; no observable algorithm promotion. MZ129 unchanged; metadata pending.

[MZ139 observable regional surface fit](nearfield/MZ139_SURFACE_FIT_20260915.md)
now implements joint ToF/RGB finite-slab fitting, frozen after TRAIN12. Consumed
dev48 remains24TP/13FP/0FN,5/5events unchanged:10accepted fits replace30returns
but0alerts change. One of3unsampled intrusions has estimated surface support;
2remain ambiguous.24native corridor contributors are excluded,12under a wrongly
nonalerting accepted surface. Keep MZ129; this exact estimator is a negative
control, not a route-wide impossibility claim. Metadata pending ledger303.

[MZ142 fixed-pretrained-BN training](nearfield/MZ142_RESULTS_20260915.md) completes the same
144fit/48scene-heldout,288updates per loss from original weights. BN buffers
stay fixed while affine parameters update. Neither arm restores joint FIT and
heldout geometry. Balanced rod MAE improves to0.915m FIT/0.722m heldout,
but coverage is0%/12.9%; heldout shallow1.651m and BODY/HEAD regressions
prevent admission. Raw outputs are finite and within range. Pause this exact
fine-tuning recipe; no epoch/weight rescue, dev/test or alert scoring. Keep
MZ129 and partial rod evidence; metadata pending303, no route-wide rejection.

[MZ141 paired same-domain DEPTHOR training](nearfield/MZ141_RESULTS_20260915.md) completes
144fit/48whole-scene-heldout frames,288updates per loss. Heldout rod MAE
1.091m frozen becomes1.723m pixel/1.721m surface; shallow1.034m becomes
2.088m/1.479m, with BODY regression and near-rod ring spill. Both also fail
fit geometry. A no-update TRAIN4 check exposes a substantial BatchNorm-mode
effect but does not repair thin surfaces. Stop this batch1-normalization recipe;
do not infer that surface balancing or learned RGB+ToF geometry is impossible.
Keep MZ129; no dev/test/alert evaluation or extended training. Metadata pending303.

[MZ140 pretrained full-RGB geometry](nearfield/MZ140_DEPTHOR_20260915.md) runs
official DEPTHOR-ZJU-Small on TRAIN48. Full RGB improves BODY/HEAD supported
column coverage to60.6%/71.3% versus35.2%/53.3%box-only; rod/shallow retain
0%target-depth agreement despite near ToF inputs. Stop frozen transfer before
dev; no new alert metrics, training, oracle or backend change. MZ129 retained;
MZ139's box/single-slab/frame-search is paused. Exact DEPTHOR adaptation is a
negative control with partial RGB benefit, metadata pending303.

Previous fixed-hardware temporal probe: [MZ135](nearfield/MZ135_RESULTS_20260914.md)
uses current plus four past frames and observed RGB flow, without metric
translation or future data. It narrows853 returns and removes17 possible bits,
but stays139TP/93FP/5FN (ToF115/89/29), with unchanged event times. It newly loses
225native contributors including31corridor samples. Reject this conditional
feature-to-ToF/local-flow transport rule; keep MZ129. No tuning or successor.

Supporting controls: [MZ134](nearfield/MZ134_RESULTS_20260914.md) adds16FP under
a free-unresolved joint model, which cannot identify mandatory visible sources
by construction; this is model inadequacy, not a sensor impossibility theorem.
[MZ133](nearfield/MZ133_RESULTS_20260914.md) regenerates admitted analytic rays:
same96x96 lattice, dense8 to fine32 full143/112/1 to143/102/1, chiefly HEAD gain,
BODY28 to31FP and rod27 unchanged. ToF loses one old rod TP/delays its event. A
hypothetical fixed-signal-budget stress loses every fine-zone detection. Retain
only as controlled capability context, not hardware or fixed-input algorithm gain.
All288 frames are consumed Development; registration/intended inheritance remain
pending the existing ledger303 fingerprint mismatch, with no ledger bypass.

Latest visible-contour diagnostic: [MZ132](nearfield/MZ132_RESULTS_20260914.md)
replaces RGB proposals with anonymous UE visible instances on the same288frames.
Both tight-box and mask arms stay139TP/93FP/5FN, BODY FP30, all15 event times
and22false segments unchanged. Masks create2553new ToF association proxies;
ten original BODY FP lose triggering bits, but only two alerts disappear and
two outside-RGB supports restore different FP. Merged returns and independent
nonvisual coverage sustain the remaining BODY errors. No new Radar association;
its masks have zero evaluation coverage. No native ToF samples newly excluded.
Keep MZ129; no frontend training, tuning or successor from this result. Intended
scoped NEGATIVE_CONTROL pending ledger303, not a general rejection of segmentation.

Latest native-angular falsifier: [MZ131-A](nearfield/MZ131_RESULTS_20260914.md)
tests one disclosed cohort center-span envelope on MZ129. It reaches triggering
footprints in 56/93 FP and changes actual support bits/scores in 23 frames,
removing six BODY and three rod FP. But 139/93/5 becomes 138/84/6 and 158 native
corridor contributor samples are lost across 24 frames. All 15 event times
remain identical; one rod TP is lost. Reject this exact uncalibrated envelope,
not all ToF-native angular inference; keep MZ129. Intended NEGATIVE_CONTROL
pending ledger303. No tuning or automatic successor.

Latest local-support falsifier: [MZ130](nearfield/MZ130_RESULTS_20260914.md)
keeps all four arms at 139/93/5 with identical event times and 22 false segments.
815 ToF returns and 103 Radar returns acquire local masks, but no new ToF
association or alert gain appears. Image clipping removes 67 native ToF samples
outside RGB coverage, including one corridor point hidden by retained alert
support. Reject this exact mask-as-complete-support rule; keep MZ129. Intended
NEGATIVE_CONTROL pending ledger303; no outcome-tuned repair or automatic successor.

Latest bounded correction: [MZ129 Radar extent](nearfield/MZ129_RESULTS_20260914.md)
uses MZ125 boxes in current Radar geometry while preserving raw support, guards
and complete original flow carry. Same consumed 288 frames: MZ128 139/103/5
becomes 139/93/5, precision 59.91%, false segments 26 to 22; all 139 TP and
15 event alert times retained. All ten exclusions are offroute HEAD. ToF
subdivision alone changes no frames and adds no combined benefit. Retain the
simpler scoped Radar component; MZ116/default unchanged, metadata pending
ledger303. Remaining BODY/rod/boundary nuisance persists; no automatic successor.

Latest diagnostic: [MZ128 remaining103 FP](nearfield/MZ128_FP_DIAGNOSIS_20260914.md)
reproduces139/103/5 on the frozen288frames. ToF-only67, Radar-only14, both22;
101FP have sufficient current real off-corridor support,2 require inherited carry.
422triggering ToF returns have0native corridor hits;30current real Radar supports
use offroute actor-to-visual-extent proxies. MERGED native-range oracle removes11FP,
while deleting all MERGED removes19 on this consumed panel; neither is a policy.
95FP belong to multi-frame segments. No ToF angular-centroid stage exists in this
alert path, so the separate depth-graph gain is not an explanation of these FP.
Prioritize corridor extent/association diagnosis. No algorithm change/successor;
existing MZ128 metadata remains pending ledger303.

Latest depth-connectivity component: [depth gate](nearfield/TOF_DEPTH_CONNECTIVITY_20260914.md)
reuses the same60 artificial packets. With the identical near-view selector,
equal/angular21/34/12 becomes45/10/0 for both fixed absolute and relative gates;
22/22 known-depth bridges cut,117/117 observed anchors retained. Raw all-support
instead19/36/12 to17/38/0: notice selection is an explicit separate contribution.
Four correct directions retain UNKNOWN depth; no within-zone recovery or RGB
fusion claim. Keep absolute as the simpler component, frozen equal and MZ116
unchanged. Registration/intended component inheritance pending ledger303.
No automatic successor or weight sweep.

Previous additive direction decision: [ToF stress](nearfield/TOF_DIRECTIONAL_STRESS_20260914.md)
freezes equal weights. On 60 artificial packets, equal and 1/0.5 both give
19/60 exact near-direction sets, 36 false CENTER frames and 12 bilateral merges.
The original disjoint-region demo remains valid; range-blind connectivity loses
modes when background bridges them. No weight gain, physical echo or fusion
validation; preserve explicit historical comparator, keep MZ116 unchanged.
Registration/intended inheritance pending ledger303; no automatic successor.

Latest weighted readout: [MZ128](nearfield/MZ128_RESULTS_20260913.md) keeps full
association with fixed symmetric zone weights. Original139/107/5but false segments
27to28; weighted MZ125 equals139/105/5. Declared binary-four alert-use control
retains full context and gives139/103/5,57.44%precision,15events,.25s delay,26false
segments. It preserves5HEAD exclusions lost by MZ127's context removal and adds
2BODY exclusions to MZ125. Retain scoped Development diagnostic/component;
MZ116 baseline unchanged, registration/inheritance pending ledger303. No successor.

Previous fixed combination: [MZ127](nearfield/MZ127_RESULTS_20260913.md) tests32zones
(four columns,28.41%image width). Original gives139TP/114FP/5FN; with MZ125 gives
139/108/5 versus full MZ125139/105/5. All15events and.25s delay retained; false
segments26/25versus27. Combined restores5HEAD FP when association loses its second
zone, while removing2BODY FP. Retain scoped Development components/interaction
diagnostic, keep MZ116 baseline; registration/inheritance pending ledger303.
No automatic successor or default promotion.

Previous width contrast: [MZ126](nearfield/MZ126_RESULTS_20260913.md) selects
whole ToF zones within central half/quarter image width, preserving vertical and
Radar policy. Half target (48zones) leaves139TP/116FP/5FN; quarter (16zones,actual
14.07% image width) yields130/58/14, precision69.15%,15/15events but delay1.25s.
All9 added FN are boundary stress; ordinary BODY/HEAD/rod recall stays unchanged.
Retain measured Development tradeoff/diagnostic component, keep MZ116 baseline.
Registration/inheritance pending ledger303. No automatic successor.

Previous focused correction: [MZ125](nearfield/MZ125_RESULTS_20260913.md) recovers
RGB foreground extent and retains the existing positive ToF association. On288
consumed frames,139TP/116FP/5FN becomes139/105/5; precision54.51%to56.97%,15/15
events and delay unchanged. All11FP removals come from two offroute HEAD episodes;
BODY/rod/stress unchanged. All1678 native samples in changed regions are retained.
Retain a scoped Development component, keep MZ116 baseline. No default promotion
or fresh validation claim. Registration/inheritance pending ledger303; no successor.

Previous five-direction package: [MZ124](nearfield/MZ124_RESULTS_20260913.md).
All bounded consumed diagnosis, measurement/correction, paired training and time
contrasts complete; no replacement. BODY score separation differs from rod's weak
response and HEAD's missing/low wrong-distance evidence. No old early threshold
matches MZ116 TP139/FP116. Strict correction leaves139/116/5; conditional3FP gain
inherits a Radar rejection and increases false segments. Matched ranking loses
13TP/11HEAD frames versus BCE on96consumed dev frames; no joint point. Temporal
hold adds FP; majority loses recall. Retain MZ116 and explanatory diagnostics;
exact recipes intended NEGATIVE_CONTROL. Worker released, evidence retained.
Registration blocked by ledger303; inheritance unknown terminal/pending. No successor.

[MZ123](nearfield/MZ123_RESULTS_20260913.md) remains frozen: early.58 gives
122/104/22 versus MZ116139/116/5 on288frames; precision54.51%to53.98%, events
15/15to14/15, HEAD33/36to22/36 with12frame full miss. No promotion; MZ122 spatial
findings remain scoped. These frames are consumed, never fresh validation again.

Latest paired representation experiment: [MZ122](nearfield/MZ122_RESULTS_20260913.md)
retains local RGB positions through ToF fusion before pooling. Shared21initial
tensors and exact paired800step schedule/augmentation. At>=90%grid recall,
train precision36.15%to39.74%, dev30.30%to39.89%; positive spatial evidence remains.
However late frame recall requires threshold<=.49while nuisance requires>=.60:
no joint point. Offroute FP23at.49versus early selected4; boundary-stress FN3.
Stop this exact joint-alert candidate, intended NEGATIVE_CONTROL for that role;
keep MZ116/MZ121 and the measured spatial findings. No transfer, tuning or capture.
The package adds1296parameters; pooling-order attribution alone is unproven.
Registration/inheritance pending the existing ledger303 fingerprint mismatch.

Latest readout diagnosis: [MZ121](nearfield/MZ121_JOINT_READOUT_20260913.md)
finds31sampled development thresholds.30–.60that jointly satisfy MZ120's original
retention criteria. Joint selection chooses.60:84TP/16FP/0FN, HEAD142/142 and
rod114/126,8/8events,0s delay. Versus.71, wrong grid cells899to1130 and precision
43.78%to40.34% expose the spatial tradeoff. Retain a separate Development joint
selector/profile; keep MZ116 and the original MZ120 terminal unchanged. No MZ119
retuning, new training, CNH or data expansion. Early-pooling remains a testable
hypothesis, not an established cause. Structured registration/inheritance remains
pending ledger303; intended COMPONENT_OR_CHALLENGER for this readout diagnostic.

Latest newly authorized direction: [MZ120](nearfield/MZ120_RESULTS_20260913.md)
implements a single-frame learned45-cell occupancy candidate, with MZ116 fixed.
336training/144scene-held-out Development frames:82TP/40FP/2FN becomes83/14/1,
but true rod cells102/126 fail spatial retention despite correct frame alarms.
Frozen threshold.71 on consumed MZ119 gives139/20/16 versus153/62/2; all16misses
are suspended HEAD and maximum delay grows.25s to1.25s. Stop this fixed pilot
recipe; intended NEGATIVE_CONTROL, no promotion or automatic CNH/capture.
This small from-scratch pilot does not establish a sensor information ceiling.
MZ116 remains runnable. Execution/boundary checks pass; structured registration
and inheritance remain pending the pre-existing ledger303 fingerprint mismatch.
The user's new direction superseded the earlier MZ119 stop only for this round.

Previous bounded attempt: [MZ119](nearfield/MZ119_RESULTS_20260913.md) completes
the user's last round. **USER_REQUESTED_STOP_AFTER_MZ119** supersedes the older
goal-active notes below: stop further optimization; no successor or capture.
New240frames: MZ116 and causal ToF-scaled RGB parallax both153TP/62FP/2FN at3.6m,
14/14events,0.25s maximum delay,18false segments/15.5s. At1.8m four bins improve,
but every gain has both real-actor and erroneous background support.55points on
four moving-barrier frames include13false context depths;17pose bounds exclude
commanded-reference translation with moving anchors. No localization promotion.
Controls/admission/sealed replay/audit complete; task-owned capture tree released.
Keep MZ116 runnable and the fixed parallax recipe as intended NEGATIVE_CONTROL.
No breakthrough achieved. Registration remains pending the existing ledger303
fingerprint mismatch, with attempted command and evidence preserved.

Latest surface-identity falsifier: [MZ118](nearfield/MZ118_RESULTS_20260913.md)
recovers border-touching RGB regions and tests interval-plane feasible sets on
480consumed frames. Primary283/133/6 becomes283/130/6, but native audits expose
551corridor-hit contributors excluded across44mixed-source frames; every frame
still alerts. A concrete wall-only-anchor fit cuts out near-rod hits. This defeats
localization promotion despite aggregate recall retention. Full distance curves
also lose warning bins. Preserve intended NEGATIVE_CONTROL for the fixed recipe,
retain frontend diagnosis/contrasts, keep MZ116 runnable; no fresh capture/tuning.
Eight focused tests and paired native audits pass; no persistent process started.
Registration remains pending the existing ledger303 fingerprint error; goal active.

Latest mixture/geometry study: [MZ117](nearfield/MZ117_RESULTS_20260913.md) closes
the source gap with35native near/far merged frames, including18weak-near frames.
All240labels match source geometry (145positive/95negative). Conditional plane
and nominal proxy refine0/1168merged components: the six fitted models occur
only in frames without mixed targets. All ten-distance decisions are unchanged;
primary141TP/64FP/4FN,14/14events,0.25s maximum delay,16s false duration.
This is zero operational coverage, not demonstrated safe localization. Preserve
the fixed recipe as intended NEGATIVE_CONTROL, keep MZ116 as runnable challenger,
and retain the now-consumed source/audits. No tuning or default promotion.
Native capture/resources are complete/released. Structured registration remains
pending the existing ledger line303 mismatch; larger optimization goal active.

Latest information/repair study: [MZ116](nearfield/MZ116_RESULTS_20260913.md) adds
a runnable resolution guard to preserve current independent Radar from tiny
RGB-box vetoes.1008consumed frames: the older768are unchanged; MZ115141/69/3
becomes142/69/2, no old alert lost or FP added. Retain as a Development component.
Merged-ToF oracle exposes29FP potential, but all429merged targets have one actor
(370singleface); there is no returned multi-actor mixture validation. A concrete
same-ToF-tuple/different-near-hit witness prevents treating mean range as a
clearance certificate. Next substantial geometry work needs a new complete
multi-actor mixed-return source. No fresh capture or default promotion; goal active.
Registration/intended component assignment remain pending the existing ledger
line303 fingerprint mismatch, with the attempt and diagnostic authority preserved.

Latest spatial-model study: [MZ115](nearfield/MZ115_RESULTS_20260913.md) replaces
center-only ToF observations with hypothetical finite-zone/multiple-return
tuples and compares RGB allocation against the same whole-zone measurements.
Fresh240frames:141TP/71FP/3FN→141/69/3 at3.6m, all14events alert at first positive
bin, no narrowed native contributor hit dropped. False duration17.75→17.25s,
but segments18→19;2.6m loses oneTP and max delay2→2.25s. A small spatial gain,
not a nuisance breakthrough or full-curve dominance. Intended Development
COMPONENT_OR_CHALLENGER for interface/allocation/audit, no alert promotion.
All frames now consumed; preserve merged/unseen uncertainty, separate Radar
HEIGHT_UNKNOWN and independent support. Native capture complete and UE/Zen
released. The larger goal remains active; do not tune the consumed source.
Structured registration remains pending the existing ledger line303 fingerprint
error; the attempted command and intended challenger inheritance are preserved.

Latest association falsifier: [MZ114](nearfield/MZ114_RESULTS_20260913.md) evaluates
joint null-inclusive assignment and an optical/range prior on768consumed frames.
375TP/138FP/21FN incumbent becomes377/142/19; threeTP gained but one lost.
A sealed posthoc matched-range-readout control restores that smoothing-related
loss (378/142/18), without changing any FP. Timing and post-exit burden do not
improve; the sole removed FP is wrong ghost-to-object association, and a ghost
prior reinforces another FP. Intended NEGATIVE_CONTROL for this fixed alert
replacement; retain MZ113 and do not tune consumed costs or expand capture.
Separately, suppressing unmatched Radar on480frames removes71FP but loses31TP,
including25correct hazardous-actor Radar supports. Missing center-ray ToF and
repeated compatible returns are insufficient discrimination. Next information
work concerns finite ToF footprints/multiple returns and local RGB allocation,
with explicit hypothetical detection limits. The goal remains active; structured
registration is pending the existing ledger fingerprint error.

Latest dynamic test: [MZ113](nearfield/MZ113_RESULTS_20260913.md) freezes optical
correspondence against a same-policy angular ablation. On new240frames, incumbent
84TP/60FP/4FN becomes87/61/1 with no old positives lost; angular remains84/60/4.
Three crossing-pole recoveries are correctly associated past range at0.25s age.
All16events alert at their first sampled positive bin, but one new FP tracks a
pole7mm outside the corridor and post-exit false support stays26frames/6.5s.
528 consumed static frames are unchanged frame by frame. Retain the optical
tracking component with its tradeoff; nuisance breakthrough is not established.
Native dynamic labels/Doppler and prediction supports audited; one capture
complete, UE/Zen released. Larger optimization remains active, with spatial
boundary ambiguity and persistent ghosts still material unresolved burdens.
Structured component registration is pending the existing ledger line303
fingerprint failure; command/error and intended inheritance are preserved.

Latest multi-method progress: [MZ111](nearfield/MZ111_RESULTS_20260913.md) combines
independent RGB/Radar association, a spatial range readout and short current-RGB
range persistence. Consumed288-frame nominal146TP/32FP/30FN becomes166/36/10,
with no nominal or baseline TP lost. The fixed primary challenger then reaches
122TP/41FP/10FN versus nominal97/38/35 on the new240-frame
[MZ112 panel](nearfield/MZ112_RESULTS_20260913.md), again retaining all positives.
Predeclared nonstress216 frames improve91/29/29 to113/29/7; maximum first alert
delay1.0s to0.25s. All three extra fresh FP are on the separate1cm stress object.
These are substantial controlled recall/timing gains, not a nuisance or natural
tracking breakthrough. Plane/filter variants add no fresh gain; zero-velocity
controls preserve all old and11/12 fresh temporal TP gains. Retain association
and visual-gated short persistence as intended COMPONENT_OR_CHALLENGER components,
with nominal, interval, simple-hold and ghost/boundary controls. No unconditional
alert promotion. New support identities/ages and native source bounds audited;
capture complete and UE/Zen released. Structured registration remains subject
to the existing ledger fingerprint failure. The larger optimization goal remains
active: dynamic/occlusion/reference errors and nuisance reduction are unproven.

Latest bounded diagnosis: [MZ110](nearfield/MZ110_RESULTS_20260913.md) reuses 288
unique MZ107--109 frames. Of 30 nominal RGB FN, 20 lack target Radar, six have
RGB+Radar pairs blocked by ToF and four fail geometry. Correct-association
nominal recovery ceiling is six frames; simple ToF-gate removal recovers four
but changes FP 32 to 33 (four removed, five added), with no TP lost. All nine
pole misses lack target Radar; five added FP have correct projected-box identity
but wrong extent geometry. Retain the diagnostic component, preserve nominal
and interval comparators, no alert promotion or automatic successor. See the
revised spatial-evidence responsibilities in the four-sensor mainline. Structured
registration remains blocked by the pre-existing ledger line 303 fingerprint
error; intended component disposition and failed attempts are retained locally.

Latest bounded optimization: [MZ109](nearfield/MZ109_RESULTS_20260913.md) propagates
working range/pose/box intervals into corridor geometry. Across288 frames it
retains all142 three-sensor baseline TP and reduces FP33to31, but it is not a
uniform upgrade over nominal MZ108. New-panel baseline42/8/22 becomes42/6/22;
nominal gives46/6/18. Interval loses5 nominal TP and restores1 other TP, all on
the1cm-intrusion stratum (baseline1/16,nominal5/16,interval1/16). Old MZ108 FP18
returns to17. Retain analytic interval states as COMPONENT_OR_CHALLENGER only,
no alert promotion. Baseline-only gate success is insufficient; future claims
must include absolute critical recall and challenger-specific positives.
18 tests pass; capture/analysis complete and resources released. Registration
pending the historical ledger fingerprint error; no automatic successor.

Previous bounded optimization: [MZ108](nearfield/MZ108_RESULTS_20260913.md) retains
stable image regions as a Development component. Projected-box recall rises
61/96 to96/96 on consumed MZ107 and42/96 to96/96 on the new controlled panel;
reciprocal association reduces wrong matches but abstains on some correct matches.
Combined association counts are44correct/2wrong and16/1. Spatial task gain is
absent: old61TP/8FP/3FN unchanged; new baseline39/17/9 becomes39/18/9. No old TP
lost, but one boundary FP removed and two added through noisy extent projection.
Keep baseline, no alert promotion. Next uncertainty is corridor extent after
association; no automatic next experiment. Intended COMPONENT_OR_CHALLENGER
for visual/diagnostic components, registration pending the historical ledger error.

The user-confirmed [four-sensor simulation mainline](nearfield/FOUR_SENSOR_MAINLINE.md)
supersedes historical successor suggestions. Prioritize algorithmic improvement
in deciding which measured object occupies the current corridor and which sensor
observations refer to it. MZ90--100 did not consume RGB; MZ101--106 consumed stereo
RGB+ToF and omitted Radar. Neither branch establishes a tested four-sensor system
or its ceiling. Preserve their results within those scopes.

[MZ107 protocol](nearfield/MZ107_FOUR_SENSOR_PROTOCOL_20260913.md) implements a
bounded fresh single-camera source and paired ToF+Radar+IMU / +RGB association
canary. Actual image pixels enter prediction; native geometry/identities remain
evaluator-only. No RGB proposal is not clearance; independent ToF survives.
No second camera, larger depth frontend, training sweep or automatic successor.

[MZ107 result](nearfield/MZ107_FOUR_SENSOR_RESULTS_20260913.md):96 fresh controlled
frames complete the four-sensor data path, but both arms remain61TP/8FP/3FN.
31 accepted associations include2 ghost-to-object mistakes; HEAD/turning image
proposal coverage is5/16 and4/16. All8FP are hypothetical persistent Radar ghosts;
this panel has no real-object offroute FP opportunity. Retain comparator, no
promotion or claim that RGB cannot help. Fixed rule intended NEGATIVE_CONTROL;
registration pending the historical ledger fingerprint error. Next decision is
visual-region coverage and competing identity explanations within this same
architecture; no automatic successor. Evidence and source retained.

## MZ106: temporal near/far contradiction fails retention (2026-09-12)

[MZ106](nearfield/MZ106_TEMPORAL_GEOMETRY_RESULTS_20260912.md) runs one fixed
past-frame LK geometric contradiction check on576 consumed frames, with RGB
stereo/PnP relative pose and separate ideal-pose control. Raw435/62/11 becomes
426/58/20 with RGB pose; final402/40/44 becomes388/39/58. smallHEAD retention
8/15 and1.75s extra delay fail. Ideal pose gives434/61/12 raw and401/40/45 final:
no finalFP reduction. Only19/62 false raw queries have any usable ideal-pose
check. Retain baseline; fixed verifier is an intended NEGATIVE_CONTROL, metadata
pending the historical ASE fingerprint failure. No future frames, threshold
rescue, integration or fresh capture; independent ToF preserved. Processes exited.

## MZ105: residual local matching scores lack lossless task benefit (2026-09-12)

[MZ105](nearfield/MZ105_RESIDUAL_MATCHING_RESULTS_20260912.md) diagnoses the576
consumed SGBM frames, reproducing cached depth and baseline support exactly.
Of62 rawFP,32 are exclusively native-far,19 mixed far/near-outside and11 entirely
near-outside; all are stereo-only despite existing LR/uniqueness/component checks.
Fixed5x5 ZNCC/near-versus-far/competitor score envelopes preserve all435 rawTP but
remove only1/0/0 FP. The sole raw removal loses2 finalTP with0 finalFP removed
and1.25s extra delay. Low-score actual HEAD support confirms a real retention
cost. Retain baseline; no filter integration, TAO rerun or fresh capture. Intended
NEGATIVE_CONTROL for these fixed local gates; metadata pending the historical
ASE fingerprint failure. Sealed features/audit retained; processes released.

## MZ104: pretrained stereo adds far-surface false support (2026-09-12)

[MZ104](nearfield/MZ104_FOUNDATION_STEREO_RESULTS_20260912.md) completes576
consumed RGB pairs with one official TAO FoundationStereo small dynamic v2
FP32 graph,0.8scale after a preserved full-resolution engineering OOM. Raw
SGBM+ToF435TP/62FP/11FN becomes437/273/9; final402/40/44 becomes399/242/47.
False sessions2to39, max extra paired delay1.25s; smallHEAD raw TP retention
13/15 fails95%. Both spatial and lifecycle gates fail.249/273 raw false queries
are supported entirely by pixels whose native depth exceeds4m. No uniform
metric-scale factor error identified; no new filter or model retry was run.
Mean full pair1.112s exceeds4Hz budget. Independent source/hash/metric audit
passes. Retain old baselines; intended NEGATIVE_CONTROL metadata pending the
unrelated historical ASE fingerprint failure. Scoped model/configuration result,
not a rejection of all learned stereo. Processes released, no automatic successor.

## MZ103: native depth separates spatial errors from alert-state costs (2026-09-12)

[MZ103](nearfield/MZ103_DEPTH_FRONTEND_RESULTS_20260912.md) replays consumed
MZ101/MZ102 576 frames with exact baseline support/prediction parity. Native
reference raw geometry446TP/0FP/0FN becomes390/16/56 through unchanged state;
SGBM union402/40/44. All56 native FN are event-entry confirmation, all16FP are
first-exit holding. All21 old TP lost by native borrowed earlier false spatial
support; native adds9otherTP. No lost events, false sessions2to0. The frozen
no-more-FN gate fails; FoundationStereo stage was not run. Retain the diagnostic
insight that frontend geometry and alert lifecycle need separate attribution,
not an achieved model improvement. Intended COMPONENT_OR_CHALLENGER/diagnostic
metadata is pending the unrelated historical ASE fingerprint error. No promotion,
no automatic model resumption; source/dependency acquisition processes released.

## MZ102: stereo span rejection loses real small support (2026-09-12)

[MZ102](nearfield/MZ102_STEREO_WITNESS_RESULTS_20260912.md) freezes query-local
disparity-coherent image-plane span>=0.10m as primary; +/-1px interval is only
a diagnostic. Consumed MZ101 improves FP25 to13 but loses7TP. New288-frame seed
102013: old union200TP/15FP/22FN becomes185/12/37, with1 lost small_head_flat
HEAD event and up to1.25s extra delay. IncrementalTP59/74 and thin33/37 fail90%
retention; extraFP7 to4 misseshalving. S+I losesmore. Intended NEGATIVE_CONTROL;
structured registration is pending an unrelated missing historical ASE receipt.
The experiment is stopped, not still running. No tuning or promotion.
Fresh ToF126/8/96 and union200/15/22 again
show sensor complementarity: stereo recovers2 smallHEAD events, ToF recovers
flatwall BODY/HEAD. Short visible support is not sufficient rejection evidence;
dev audit includes59/59 real head-surface pixels rejected. Known-pose synthetic
current-corridor scope only; historical core unchanged, no automatic successor.

## MZ101: spatial complementarity, direct union not promoted (2026-09-12)

[MZ101](nearfield/MZ101_STEREO_TOF_RESULTS_20260912.md) runs288 fresh rendered
stereo/full64-zone ToF frames,24 appearance-paired episodes/12 geometry families.
Same current BODY/HEAD corridor and common FOV: ToF148TP/6FP/76FN becomes
union202/25/22, F1.7831 to.8958; thin-pole TP36 to67 of68. Flat-wall stereo misses
are rescued by ToF. Union retains all148TP and adds54, but adds19FP and2 entirely
false sessions. ToF already covers28/28 events, leaving no new-event opportunity;
the separate FP gate fails.19 non-left-censored/non-carried event entry delays
improve.5658 to.1447s. Retain spatial complementarity as COMPONENT, not direct
union promotion or collision prediction. Known pose, point-ray ToF and posed
synthetic textures are controlled assumptions, not hardware evidence. No training,
threshold tuning or automatic successor; historical retained core unchanged.

## MZ100: observable angle anchors fail feasibility (2026-09-12)

[MZ100](nearfield/MZ100_CAUSAL_ANGLE_ANCHOR_RESULTS_20260912.md) freezes one causal
ToF/Radar unique-pair interval estimator on consumed MZ99. Only98/2875 Radar
fallback frames receive earlier calibration;80/98 estimates are within5deg.
Coverage and reliability fail the predeclared10%/90% gates. Actual real-return
MAE improves7.3679 to3.9260deg, but zero-bias MAE worsens2.9587 to4.3378deg.
Most uncovered fallback frames (2707) never established an anchor;70 had a prior
one that was unavailable. Stage1 stops the run before downstream predictions or
scoring. Retain this configuration as NEGATIVE_CONTROL; R/core unchanged. No
threshold relaxation, cache extension, new source or automatic successor.

## MZ99: angle information helps route relevance, not contact semantics (2026-09-12)

[MZ99](nearfield/MZ99_ANGLE_INFORMATION_RESULTS_20260912.md) uses one new seed99013
UE panel (128 episodes/5120 frames) with frozen baseline/R/R+F/A. Only Radar
bearing is replaced by evaluator-only pre-noise angle; ghosts, range/Doppler,
ToF and IMU errors remain. R/CURRENT_ROUTE meets predeclared gates:987TP/753FP
becomes1016/559; original TP retention980/987, no lost route intervals. This is
a diagnostic angular-information component, not achieved calibration or policy
promotion. Separate BODY_1S strict future TP barely changes124 to125/171. All
methods cover12/12 evaluable contacts through carried warnings;14 are right-censored,
so this timing measure does not distinguish predictive ability. Preserve route
awareness versus imminent-contact semantics and shared causal prompt evaluation.
The historical core is unchanged; no model, rule tuning or successor was launched.

## MZ98: A mechanism attribution correction (2026-09-12)

[MZ98](nearfield/MZ98_A_MECHANISM_AUDIT_RESULTS_20260912.md) exactly reproduces
sealed MZ96/MZ97 alerts and separates raw current route occupancy R, fitted-current
correction F and strict future entry P without changing tracks or downstream state.
Descriptive combined R578TP/284FP/future102 versus full A596/345/future106;
R+F588/299/future102. P adds8TP/46FP.120/173 future-only GT frames are already
inside the current extended wedge, so most future-only gain does not demonstrate
strict future prediction. Retain task-aligned route admission as the mechanism;
do not headline predictive crossing as its established main benefit. Exclusion
variants are consumed diagnostic components, not selected successors. Physical
ghost identities and wearer/target motion contributions remain unidentifiable.

## MZ97: residual filter disabled; fresh rule replication (2026-09-12)

[MZ97](nearfield/MZ97_RESIDUAL_SUPPRESSION_RESULTS_20260912.md) closes as
RESIDUAL_DISABLED_NO_VALIDATION_GAIN / NEGATIVE_CONTROL for the fixed A-positive
post-alert tree. No active validation threshold meets98% overall/future retention
and event constraints while reducing FP; fallback exactly preserves A.
Fresh seed97013 test32 scenes: matched_hold231TP/167FP/74FN F1.6572;
A254/169/51 .6978, future-only recall37.5% to64.3%; A+B272/211/33 .6904.
A recall benefit repeats but FP reduction does not. A+B adds18TP and42FP over A,
so consistent A+B superiority is unsupported. Keep A and A+B as Development
challengers with existing core unchanged. No test tuning, relaxed gates or refit.

## MZ94-MZ96: physical rules and a fresh UE decision head (2026-09-12)

[MZ94](nearfield/MZ94_RADAR_HORIZON_RESULTS_20260912.md) horizon A changes final
decisions but fails consumed proxy FP/fragment gates (.7440 F1; range-only .7547).
[MZ95](nearfield/MZ95_COVERAGE_AUTHORITY_RESULTS_20260912.md) standalone B releases
useful vetoes; retain as NEGATIVE_CONTROL for standalone authority replacement.
[MZ96](nearfield/MZ96_UE_DECISION_HEAD_RESULTS_20260912.md) uses5,120 fresh UE
geometry/proxy frames, scene split80/16/32. Held-out32 scenes: matched_hold
306TP/200FP/103FN F1.6689; A342/176/67 .7379; A+B368/205/41 .7495;
XGBoost297/84/112 .7519. Tree loses2 events and adds up to0.9s delay: no promotion,
NEGATIVE_CONTROL for complete risk-head replacement. A and A+B remain frozen
Development challengers, not retained-core or default-App changes. New test is
now consumed; no automatic refit. Native collision geometry with hypothetical
Radar is not RF, physical body collision, real hardware or unseen-family evidence.

## Frozen final-error diagnosis (2026-09-12)

[Final decision audit](nearfield/MZ90_FINAL_DECISION_AUDIT_20260912.md) separates
selected sensor from current/history support. Proxy151FP include108 current-Radar,
18 Radar-hysteresis-only,16 current-ToF and9 outer-hold FP, each with paired TP
counts.158FN include94 raw Radar gate failures,21 hysteresis entry failures,
20 nonalerting-ToF-over-Radar conflicts,17 other ToF nonalerts and6 no-return cases.
139FN have current hazardous-source returns by evaluator-only provenance.69FN
have all such Radar returns outside the raw range gate,66 with future-only truth.
Paired degradation is135 lost/11 recovered TP and128 added/18 removed FP.
Investigate range/horizon consistency and local ToF coverage before another
front-end filter. No new policy or change to retained matched_hold authority.

## MZ93 supported future occupancy (2026-09-12)

[MZ93](nearfield/MZ93_SUPPORTED_PREDICTION_RESULTS_20260912.md) closes as
SUPPORTED_PREDICTION_GATE_NOT_MET / NEGATIVE_CONTROL. Requiring observed inward
motion for future-only support adds8 raw-return exclusions, but only2 raw-support
and2 Radar-hysteresis frames differ; all differences disappear at ToF branch
selection before one-frame hold. Final333TP/151FP/158FN F1.6831 remains identical.
Keep matched_hold. Further upstream filters need evidence of final intervention
coverage; no tuning, promotion or automatic successor.

## MZ92 cross-sensor calibration (2026-09-12)

[MZ92](nearfield/MZ92_CROSS_SENSOR_CALIBRATION_RESULTS_20260912.md) closes as
CALIBRATION_GATE_NOT_MET / NEGATIVE_CONTROL. Unique co-observed ToF/Radar range
pairs reduce relative-angle MAE7.123 to2.128deg on424 active sensor-proxy frames,
but alerts still equal matched_hold and MZ91:333TP/151FP/158FN, F1.6831.
No85 real-only FP or45 persistent-only FP removed. This is a narrow consumed
calibration diagnostic, not an alert gain or a full spatial uncertainty result.
Keep the baseline; no tuning, promotion or automatic successor.

## MZ91 causal Radar localization (2026-09-12)

[MZ91](nearfield/MZ91_RADAR_LOCALIZATION_RESULTS_20260912.md) is closed as
LOCALIZATION_GATE_NOT_MET / NEGATIVE_CONTROL. Five-frame polar localization with
non-shrinking common angle uncertainty changes402 qualified Radar support scores,
but final alerts equal both matched_hold and the single-frame control:
333TP/151FP/158FN, F1.6831;29 false segments,49 fragments,1 missed event.
No real-only85FP or persistent-only45FP are removed. Ideal predictions also equal
the baseline. Preserve matched_hold for this comparison; internal score changes
are not task gains. No tuning, source expansion, model promotion or successor run.

## MZ90 fallback diagnosis (2026-09-12)

[Read-only provenance audit](nearfield/MZ90_RADAR_FALLBACK_AUDIT_20260912.md)
attributes126 fallback FP to78 real-only nonhazard supports,44 persistent-only,
1 transient-only and3 mixed.194of202 fallback TP have hazardous real support;
8 are coincident frame alerts. Exact raw/evaluator replay and22 sealed hashes pass.
Prioritize observable spatial localization diagnosis over extra persistence;
phantom and static-real kinematics overlap, but their full observation laws differ.
No new policy, tuning, source expansion or promotion; MZ90 remains negative.

## MZ90 common observable sensor contract (2026-09-12)

[MZ90](nearfield/MZ90_OBSERVABLE_CONTRACT_RESULTS_20260912.md) follows a
[literature-guided protocol](nearfield/MZ90_LITERATURE_AND_PROTOCOL_20260912.md):
48 common2D world scenes, ideal and sensor-proxy observations with identical truth,
no truth motion/height hints, raw-frame Radar and shared causal yaw. Joint versus
independent covariance adds78TP/0FP in ideal but14TP/6FP in sensor_proxy; false
segments rise23 to28. Record
`OBSERVABLE_CONTRACT_COVARIANCE_TRANSFER_NOT_ESTABLISHED` / NEGATIVE_CONTROL.

Proxy full joint+spatial reaches276TP/134FP/215FN, F1.6127, below matched hard
309/142/182, F1.6561 and simple hold333/151/158, F1.6831. One baseline-detected
event is lost and maximum added shared-event delay is0.3s.126of134FP (and202TP)
come from the no-valid-ToF Radar fallback, untouched by the valid-ToF spatial
gate. Preserve the shared raw-observation harness and narrow ideal covariance
benefit as diagnostics, not a promoted policy. Local observability and full
measurement uncertainty are unresolved; no soft-head, training or source expansion
is automatically launched. This is an uncalibrated horizontal eight-bin surrogate,
not actual8x8 ToF, real Radar, physical body-tube or safety evidence.

## MZ89 joint temporal geometry (2026-09-12)

[MZ89](nearfield/MZ89_JOINT_GEOMETRY_RESULTS_20260912.md) tests joint temporal yaw
covariance and spatially compatible Radar on36 new constructed episodes/1080frames.
Hard MZ85 reaches382TP/101FP/41FN, F1.8433; simple one-frame missing-ToF hold
reaches383/101/40, F1.8445. Joint+spatial reaches302/23/121, F1.8075: fewer false
segments (11 to5), but80 lost TP and2 additional missed baseline-detected events.
Nominal variants lose20TP and total within-event fragments rise24 to25. Record
`JOINT_GEOMETRY_FULL_GATE_NOT_MET`; retain full recipe as NEGATIVE_CONTROL.

The matched covariance ablation adds46 crossing TP (25 to71/98) with no changed
false frames; hard still reaches85/98. This supports only the isolated covariance
correction. Spatial Radar removes134FP but loses51TP versus joint+coarse Radar.
Preserve hard baseline and simple-hold comparator; do not tune on consumed MZ89 or
automatically launch another experiment. Exact analytic ToF, synthetic lateral
hint, stabilized Radar and yaw-only uncertainty remain source-aware Development
limits, not full body-tube, thin-pole, real-sensor or safety evidence.

## MZ88 uncertainty-aware geometric association (2026-09-12)

[MZ88](nearfield/MZ88_UNCERTAINTY_AWARE_ASSOCIATION_RESULTS_20260912.md) replaces
MZ85's hard stabilized corridor with a fixed scalar one-sigma tri-state
association on a new 36-episode/1,080-frame analytic source. The hard baseline
scores398TP/124FP/17FN, F1.850. Conservative `UNCERTAIN -> UNKNOWN` reduces FP
to48 but falls to295TP/120FN. Radar confirmation raises FP to151; the full
one-frame credentialed-hold policy reaches338/155/77, F1.744. Every performance
gate fails; the terminal is `SCALAR_UNCERTAINTY_ASSOCIATION_NOT_RETAINED`.

The failure separates two mechanisms. Scalar uncertainty plus conservative
forecasting delays lateral crossings by0.5--0.7s and loses weak-Radar evidence;
coarse Radar authority inside the uncertainty band changes outside-boundary FP
from48 to148. The hold itself remains credentialed and height-safe, adding12TP
and4FP, but cannot establish an initial hazard. Keep MZ88 as a negative control;
do not tune its margin, envelope, Radar gate or hold on the consumed source.
Any successor must change representation to a correlated occupancy/collision
tube or equivalent joint track distribution with spatially compatible Radar.
MZ86 remains paused and learned fusion remains lower priority.

## MZ87 IMU rotation realism falsifier (2026-09-12)

[MZ87](nearfield/MZ87_IMU_REALISM_FALSIFIER_RESULTS_20260912.md) freezes MZ85 and
runs36 analytic transform-error replays. Every smallest stress preserves
160TP/0FP/11FN, so rotation compensation is not smallest-perturbation-oracle
fragile. The declared medium tolerance region nevertheless fails: +2deg/s bias
and +2deg extrinsic error each add2 lateral-crossing FP, one of four2deg/s-noise
seeds adds1 and loses1 lateral frame, and both signed medium combinations restore
4 head-motion FP plus lateral FP or TP loss. Two-frame IMU dropout restores all8
head-motion FP. The terminal is `ROTATION_COMPENSATION_TOLERANCE_NOT_ESTABLISHED`.

Retain MZ85 as the ideal mechanism baseline, not a robust operating point. Hard
angular boundaries in a miscalibrated stabilized frame fail before or alongside
the original head-motion mechanism. The apparent +/-50ms stability is specific
to the locally flat constructed motion. Require fresh motion and calibrated
timing/extrinsic evidence before soft association or a state estimator; do not
tune thresholds or train on this consumed grid. MZ86 remains reserved for the
separate two-event continuity defect.

## MZ85 gyro rotation-compensated ToF state (2026-09-12)

[MZ85](nearfield/MZ85_ROTATION_COMPENSATED_STATE_RESULTS_20260912.md) preserves
the sealed MZ84 baseline exactly, then adds only causal yaw/pitch rotation of ToF
rays into a stabilized frame before association. Fusion changes from
160TP/8FP/11FN, F1.944 to160/0/11, F1.967: all8 constructed head-motion FP are
removed, while every non-head-motion bit, all14 positive-event first-alert
frames and all fragmentation counts remain unchanged. All four predeclared
mechanism gates pass.

Retain rotation compensation as a component, not a new-source confirmation.
MZ85 deliberately leaves the two multi-target gap-onset fragments untouched.
The consumed analytic source omits bias, clock/extrinsic error, translation,
vibration and measured sensor noise, so the next IMU claim requires fresh or
device-calibrated evidence; do not infer hardware, alert, user-benefit or safety
performance and do not launch learned state estimation from this result.

[Residual FN audit](nearfield/MZ85_RESIDUAL_FN_AUDIT_RESULTS_20260912.md) finds
no single dominant mechanism among the remaining11 generic FN:4 are lateral
episode-frame-zero history cold starts,4 have qualifying raw radar evidence but
no activated radar state, and3 have instantaneous ToF UNKNOWN plus no qualifying
radar return. Five are gap-related, but only the two already-fragmented events
have a prior hazard that an authority-aware hold can preserve. Separately,39
radar-only generic TP remain HEIGHT_UNKNOWN attribution debt. Bound any MZ86
continuity claim to those two frames/events; use fresh or perturbed evidence for
the larger robustness question.

## MZ84 bidirectional ToF/radar complementarity (2026-09-12)

[MZ84](nearfield/MZ84_BIDIRECTIONAL_COMPLEMENTARITY_RESULTS_20260912.md) runs a
zero-fit new analytic source: six stress families,24episodes and480frames. Fixed
`ToF OR (ToF_UNKNOWN AND Radar)` reaches160TP/8FP/11FN, F1.944 versus ToF
121/8/50,F1.807 and radar79/119/92,F1.428. ToF uniquely recovers weak reflectors
and low-radial-velocity crossings; radar uniquely recovers strong-light wall and
multi-target gap frames. Fusion FP equals ToF, all positive-event first alerts
match the earlier correct expert, and radar-only39 true frames remain
HEIGHT_UNKNOWN. All four predeclared complementarity gates pass.

Retain responsibility-separated late fusion as a controlled-simulation
challenger, not a sensor-performance result. Twelve of14 positive events have
one fusion segment; two multi-target episodes fragment once at gap onset. All8
fusion FP are ToF head-motion association errors, while valid ToF suppresses the
additional radar head-motion/clutter/ghost errors. Preserve both defects: do not
posthoc tune hold time or train learned fusion. The next state-estimation question
must add credible ego-motion information or measured/device-calibrated traces;
MZ84's RCS, multipath, sunlight, gaps and motion are analytic proxies, not
hardware, alert, deployment, user-benefit or safety evidence.

## MZ83 coarse radar information canary (2026-09-12)

[MZ83](nearfield/MZ83_RADAR_INFORMATION_CANARY_RESULTS_20260912.md) converts
MZ77 native scene depth into sealed radar-like `(range, radial velocity,
azimuth, valid)` packets through a fixed uncalibrated forward model with coarse
angular cells, support clustering, noise/quantization, misses, return merging,
FoV and clutter. The generic causal expert sees no depth, IDs, masks, source
cases or labels. Against the three MZ79 5-klux typical profiles it rescues
47/53/59 generic ToF FN at0 added FP, exceeding required12/14/15; fixed late OR
is77TP/0FP/1FN in each. Radar-only rescue remains HEIGHT_UNKNOWN.

Retain this as an information-sufficiency component and authorize one bounded
ToF+radar late-fusion successor on harder, preferably new scenes. Do not read the
77/0/1 as real-radar performance: MZ77 is one straight, constant-speed synthetic
scene with simple opaque targets and an easy control, while the forward model
lacks RCS/material response, multipath, antenna pattern, ambiguity, interference
and measured calibration. Before a radar network or ToF+radar+IMU state estimator,
challenge weak/small reflectors, near off-corridor clutter, wall/multipath proxies
and head motion while preserving generic HEIGHT_UNKNOWN output.

## MZ82 explicit temporal visual canary (2026-09-12)

[MZ82](nearfield/MZ82_TEMPORAL_VISUAL_CANARY_RESULTS_20260912.md) tests a fixed,
causal five-frame RGB-only flow/expansion/image-TTC score on the consumed MZ77
Development sequence. At added FP<=5, the three MZ79 5-klux typical profiles
rescue0/0/2TP versus predeclared10TP canary and22/25/28TP 25%-FN targets. The
only low-FP recovery is two HEAD_NEAR wall bits with one HEAD_FAR wall error;
head-bar and thin-pole recover none. At10FP the recipe recovers only4/5/7TP.

Close this explicit temporal recipe as a negative control. Do not tune its
corridor, flow, TTC, window, or cutoff on MZ77, and do not automatically launch
a learned temporal RGB expert. A revisit needs source-separated motion evidence
and a changed mechanism such as measured ego-motion compensation. Shift the main
research bet toward ToF+radar with IMU infrastructure, or a stronger depth
sensor; dual identical ToF remains secondary because it does not address shared
ambient-light range collapse. This is a consumed simulation priority decision,
not hardware, alert, deployment, user-benefit, or safety evidence.

## MZ81 conditional RGB feature probe (2026-09-12)

[MZ81](nearfield/MZ81_CONDITIONAL_FEATURE_PROBE_RESULTS_20260912.md) freezes the
current RGB backbone and trains only a4,652-parameter576-8-4 query head onMZ61,
selects epoch/cutoff on topology-disjointMZ67, then tests once on theMZ77
approach source. The locked low-FP MZ67 cutoff transfers as8/15/24 rescued TP
but211 added FP in the three5-klux profiles; MZ77 conditional PR-AUC is only
0.104/0.120/0.139. More decisively, the forbidden posthoc ceiling at added
FP<=5 is0TP in all three profiles. CLEAN remains exactly unchanged by `U_ToF`.

Pause the single-frame RGB residual route and retain the probe as a negative
control. Do not increase head/backbone size or retune on consumed MZ77. A future
restart must change the information source or geometry (temporal RGB, explicit
corridor/perspective representation, dual/higher-resolution ToF, ToF+IMU or
radar) under a separately bounded hypothesis. No successor is auto-started.

## MZ80 observability-conditioned visual rescue (2026-09-12)

[MZ80](nearfield/MZ80_OBSERVABILITY_RESCUE_RESULTS_20260912.md) evaluates
`ToF OR (U_ToF AND RGB)` with query-level `U_ToF` defined by whether the MZ79
profile cap covers the entire fixed query interval. It preserves all640 CLEAN
decisions exactly. Under5-klux white/light-gray/gray typical profiles, however,
the posthoc best frozen-RGB threshold at added FP<=5 rescues0TP in every case;
the required25%-FN targets are22/25/28TP. Even40 added FP rescues only0/0/2TP.

Retain the observability gate as a component and current RGB score as a negative
control for selective rescue. Threshold rescue is closed on consumed MZ79. A
successor must train a residual expert on ToF-UNKNOWN positives plus UNKNOWN,
pre-onset/out-of-corridor and wrong-band hard negatives with source-separated
evaluation. The RGB/ToF degradation matrix remains outstanding: clean RGB already
fails this score ceiling, so degrading it cannot rescue the frozen representation.

## MZ79 datasheet range-cap sensitivity (2026-09-12)

[MZ79](nearfield/MZ79_TOF_RANGE_CAP_RESULTS_20260912.md) applies VL53L8CX
DS14161 Rev12 Table20 inner/corner maximum-range endpoints to the consumed MZ77
packets without fitting or threshold search. MZ78 geometry drops from124/130 TP
ideal to87 dark-gray, then42/32/20 under5-klux white/light-gray/gray typical
caps; gray minimum leaves10TP. The corresponding all-invalid frame counts are
103,130,136,142 and150/160. Missing returns remain `UNKNOWN`.

Frozen RGB OR recovers2 dark-gray and8/9/13 stressed 5-klux typical true bits,
but adds64 false bits in every profile; current RGB/fixed OR is not a useful
rescue. Retain MZ78 as ideal-packet comparator and the cap operator as a
sensitivity component. This is hard-cap interpolation from15-Hz datasheet
endpoints on nominal10-Hz simulated poses, not probabilistic sensor physics or
hardware evidence. Do not auto-start dual-surface or ego-motion successors.

## MZ78 strong pure-ToF temporal baseline (2026-09-12)

[MZ78](nearfield/MZ78_TOF_TEMPORAL_BASELINE_RESULTS_20260912.md) challenges the
learned system on the consumed MZ77 approach replay. Fixed zone-center geometry
scores124TP/2FP/6FN under CLEAN versus DIVERSE79/33/51. A causal five-frame
ledger with at most three closing-range extrapolations preserves124/2/6 under
the central gaps, where frame geometry falls to94/2/36 and DIVERSE to60/30/70.
The fixed temporal-geometry OR DIVERSE rule reaches125TP but keeps30--33FP, so
do not retain that fusion. Retain temporal geometry as a controlled-simulation
challenger; current learned fusion has no demonstrated incremental value here.

All40 control frames have entirely invalid model-visible packets, so zero alerts
are `UNKNOWN`, not clearance. This consumed simulator-native packet replay is not
measured sensor physics, natural-scene, latency, hardware or safety evidence and
lacks an independent ordinary BODY-obstacle clip. Fourteen focused tests passed;
no fit, threshold search, new capture or persistent resource was used.

## MZ77 simulated approach completed (2026-09-11)

[Fixed-model approach result](nearfield/MZ77_APPROACH_SEQUENCE_RESULTS_20260911.md): 160 posed frames in four clips, original models/cuts, CLEAN and central return gaps. DIVERSE and OLD_NEG have identical true detections (79 CLEAN, 60 gap); DIVERSE adds five HEAD_FAR false bits (33 versus 28 CLEAN FP). Bar/pole head alerts start at 3.1 m but gaps interrupt them. Wall BODY_ANY covers only 9/26 CLEAN and 7/26 gap positives, first true distance 2.4 to 2.2 m. Near/far union coverage hides pole HEAD_FAR misses. Control has no alerts, not clearance.

Retain the sequence harness as a diagnostic component; preserve both models and thresholds with no promotion. Capture/native bounds, historical parity, eight metric tests and independent 7,680-query scoring passed; task processes released. This is settled-pose simulation, not measured walking, sensor physics, latency or hardware evidence. Bounded run complete; no automatic successor.

## Resumed engineering replay and simulated baselines (2026-09-11)

The user resumed diagnosis and chose simulation only. [Replay diagnosis](nearfield/MZ76_REPLAY_DIAGNOSIS_20260911.md) isolates implicit RGB encoder memory layout: padded historical batches use NCHW, unpadded MZ76 batches use channels-last. Explicit encoder NCHW restores all2,048 HELD frames under3 original profiles within unchanged tolerance (42 arrays, raw max3.815e-6, zero decision or winner changes). Original models/cuts and MZ76 invalid result remain unchanged; opt-in replay only, no new fit/source.

Same-input simulated MZ67 DROP: RGB90TP/139FP, ToF355/155, fixed MZ5 average89/78, OLD_NEG562/71, original DIVERSE753/71. HEAD_NEAR OLD_NEG158/15 toDIVERSE223/15. Keep DIVERSE's scoped challenger role and OLD_NEG; averaging does not replace them. IDEAL increases FP (MZ67+2, MZ61+8). Measurement gaps remain UNKNOWN; no temporal/hardware claim or model promotion. This bounded diagnosis/comparison is complete; no automatic successor training or collection.

## MZ76 final attempt: paused after strict parity failure (2026-09-11)

[MZ76](nearfield/MZ76_OBSERVED_RELIABILITY_RESULTS_20260911.md): Two matched heads completed256steps each from originalD,4096TRAIN frames each,2048HELD frames under3canonical ordinary/partial profiles,118.366s CUDA run. Strict historical baseline parity failed; unchanged scorer failure is preserved. Posthoc RELIABILITY versus BASE: five conditions identical, MZ67IDEAL874/45 versus875/45 TP/FP, one BODY_FAR positive lost. No added benefit or promotion; posthoc diagnostics do not replace the registered comparison.

User requested pause after this final attempt. Do not start a successor fit, recut, inference or collection until resumed. Preserve originalD and all cuts, both new checkpoints and strict score failure. On resume diagnose baseline reproducibility before any new comparison; neighbor geometry is not measured hardware confidence and missing observations remain UNKNOWN.

[Source-contact smoke](nearfield/SOURCE_CONTACT_SMOKE_20260911.md) separately completed4worker frames in83.837s;12foot traces matched planned sidewalk height. No source admission or model input; worker resources released.

## Rendered surface recovery and support correction (2026-09-11)

[Four-frame worker smoke](nearfield/SOURCE_SURFACE_SMOKE_20260911.md) actually rendered corrected1x landing/birch poses in both original context partners: HF231/1287 pixels and extra BF2014/2085, with matching isolated-target evidence and exact pair query depths. Capture/world exit0 in91.820s; resources released. Collision comparison UNKNOWN is not ToF invalidity:125/128 rays hit the correct component, but121 fail rendered-depth agreement. Ground grid mixes181 sidewalk and149 controlled-target hits; next terrain probes must ignore controlled actors. No dataset admission or new model fit.

[Existing-cube support payload](nearfield/SOURCE_CONTACT_REPAIR_20260911.md) closes measured wood/duct gaps with unchanged target transforms, four bracket/rail and six post/leg adjustments. Duct contact areas641.324/523.526mm2; wood remains point/line contact with stability unknown. CPU0.313s, no new assets or rendered verification of these support changes. Preserve MZ72 failure; validate the exact fresh support/ground changes before any broader source capture.

## Explicit packet-to-model adapter (2026-09-11)

[Adapter](nearfield/TOF_MODEL_ADAPTER_20260911.md) now canonicalizes usable observed returns while preserving raw order and aligned quality/status availability. The legacy bridge supports64 zones and1..2 configured slots without silently truncating4 targets or expanding16 zones. Thirteen input tests passed;49,152 transforms across8,192 existing frames preserve40,960 original/closest profile frames byte-exact and reproduce all2,048 saved MZ75 packed HELD inputs. No model fit, new score, synthesized quality or automatic frozen-run/application change; quality-aware neural consumption remains outstanding.

## Identical-distance slot control: MZ75 (2026-09-11)

[MZ75](nearfield/MZ75_RETURN_SLOT_CONTROL_RESULTS_20260911.md): At identical surviving FAR distance content, slot1-only to slot0 packing changes frozen DIVERSE new MZ67 HELD from760TP83FP to912TP56FP; old MZ61 from881/106 to987/31. New query TP deltas[-6,+45,-13,+126], FP deltas[0,+3,-21,-9] retain local regressions. Both CONTROL/DIVERSE weights and all cuts unchanged. 2048 CUDA inference frames plus32 TRAIN parity,42.433s; independent scalar score4.621s. No model promotion or measured sensor-physics claim.

Preserve MZ74 and MZ75. Normalize measurement packet representation explicitly at the model adapter while retaining raw device order/status separately; validate actual packet ordering and quality before any deployment. Train or test quality-aware handling only under separately specified conditions, keeping old/new per-query costs and UNKNOWN. Continue small actual-surface source repair before expanding collection; no automatic restart of failed MZ72 expansion.

## Return survival and slot representation: MZ74 (2026-09-11)

[MZ74](nearfield/MZ74_RETURN_SURVIVAL_RESULTS_20260911.md): Frozen MZ70 DIVERSE on new MZ67 HELD: DROP753TP71FP, closest-slot0 911/57, farthest-slot1 760/83; old MZ61 832/38,988/31,881/106. FAR new HEAD_FAR95/28 versus DROP158/19. Both original arms/cuts retained. MZ70 TRAIN slot1-only exposure is zero, so FAR mixes near-return absence with an unfamiliar ordered representation. Run8192frames plus32parity,153.116s; score14.719s. Worker source triangles exported with unchanged assets and no captures.

Preserve all MZ74 predictions and both endpoint scenarios; isolate slot representation using the same FAR distances packed into slot0 with fixed models/cuts before attributing this difference to sensor physics. Keep old/new per-query false/true costs and UNKNOWN; no strongest-return, real-hardware or model-promotion claim. Use exported real surfaces for corrected target/contact placement, then new bounded source capture; MZ72failed64 stays unadmitted.

## Partial-return inputs and compact RGB tools (2026-09-11)

[Paired reported-return sensitivity](nearfield/TOF_RETURN_SENSITIVITY_20260911.md) now keeps either the existing near or far return in unresolved dual zones, without inventing a midpoint or assuming a thin near target survives. Both inputs passed byte/validity replay on8192 existing packets;14 focused input tests passed. No new model inference, fit, cutoff or sensor-physics claim. Existing MZ70 MZ67 HELD DIVERSE IDEAL861TP/46FP, MERGE911/57 and DROP753/71 are re-extracted with per-query denominators; ordinary/partial input remains primary and original challenger dispositions remain.

[Lossless RGB view builder](nearfield/COMPACT_RGB_VIEW.md) implements explicit PNG-to-WebP mapping, input/output/pixel hashes and exact decoding. Five focused checks and compact archive reading passed. Eight TRAIN images:3546777 to2412046 encoded bytes (31.99% smaller),0.783s total. No whole-dataset conversion or deletion; consumers must explicitly adopt the new mapping.

## Quality-aware measurement packet interface (2026-09-11)

[Opt-in packet contract](nearfield/VL53L8CX_MEASUREMENT_PACKET_20260911.md) implements observed status/quality availability, declared target order/count and causal time alignment. Eight focused input/causality tests pass; legacy range/valid bytes are preserved. No quality is synthesized from truth and old static cases are not turned into temporal sequences. Official ULD documents8x8/15Hz; motion aggregation differs from a64-cell map. Normal/partial-ToF remains primary. This is an input interface, not new model training, device integration or sensor-physics validation.

## Physical-source canary stop and sensing priority: MZ72 (2026-09-11)

[MZ72](nearfield/MZ72_PHYSICAL_SOURCE_RESULTS_20260911.md): Dual-host64 canary:64 source-valid,48 required-focus matches,32 exact native pairs; independent native64 and actual visual64 reviewed.16 landing/tree HEAD cases miss intent, physical contacts unresolved and some duct HN pixels match cradle geometry. Main4032 never started; no source admission/model work. Original MZ70 weights/cuts retained.

Preserve this failed64 and complete frozen4096 recipe; do not replay or silently relabel it. A revised source must anchor actual target surfaces and support contacts, distinguish target from mount contributions, and preserve all extra native bits/UNKNOWN. Prioritize normal and partially available ToF; ALL_INVALID remains a secondary artificial stress test. Any later fit must retain old/new per-query error costs. Lossless WebP method0 saved31.99% on8 original TRAIN images with exact RGBA and0.195s encoding; sealed PNG sources unchanged. Missing-state all-negative supervision remains an unregistered idea, not the automatic next fit. [Execution](nearfield/MZ72_EXECUTION_20260911.md).

## Observed missing-state calibration: MZ71 (2026-09-11)

[MZ71](nearfield/MZ71_MISSING_STATE_CALIBRATION_RESULTS_20260911.md): Frozen DIVERSE on MZ67 HELD ALL_INVALID: original-cut58/0/966 to state-cut30/1/994 TP/FP/FN; BODY_NEAR57 to28, HEAD_NEAR remains0. CONTROL3/1 to0/1. Original MZ70 weights/cuts remain retained; MZ71 state calibration is a scoped negative control. Zero fits;9544 new inference frames plus32 parity;155.016s run and37.109s independent score.

Retain the original MZ70 cuts and both trained arms; record these two missing-state cutoff vectors as NEGATIVE_CONTROL for the proposed replacement role, without denying their query-specific tradeoffs. DIVERSE loses29 new-source BODY_NEAR true bits (26 native,2 known-wrong,1 UNKNOWN), gains one native HEAD_FAR bit and adds one HEAD_FAR false bit with an UNKNOWN winner. On old MZ61 HELD ALL_INVALID, DIVERSE103TP/5FP becomes72/7:53 near-body losses versus13 far-body,1 near-head and8 far-head gains; the near-head gain is known-wrong and7 far-head gains are locally UNKNOWN. CONTROL gains58 old-source BODY_FAR bits (54 native) but loses120 BODY_NEAR,9 HEAD_NEAR and6 HEAD_FAR. All original-profile changes occur only on naturally missing rows, including DIVERSE DROP legacy2747/51 to2746/55, MZ48 nonfit571/21 to570/21 and MZ55 held501/83 to493/83. The missing-state rule raises DIVERSE BODY_NEAR cutoff4.069206 to4.831364 while lowering its HEAD_NEAR cutoff4.233131 to3.043964 without recovering new-source HEAD_NEAR. Thus matching the calibration sensing state alone does not solve the tested problem. All eight cutoff maxima come from original DEV negative rows; excellent within-MZ67 BODY_NEAR AUC does not establish cross-source negative-tail separation. No raw score, weight or nonmissing decision changes, and no real-sensor or safety claim follows. [Execution](nearfield/MZ71_EXECUTION_20260911.md).

## Matched diverse-source learning: MZ70 (2026-09-11)

[MZ70](nearfield/MZ70_DIVERSE_LEARNING_RESULTS_20260911.md): Matched4096-step CONTROL versus DIVERSE: MZ67 HELD DROP TP/FP/FN664/71/360 to753/71/271; ALL_INVALID3/1/1021 to58/0/966. Missing HEAD_NEAR remains0TP, although raw AUC/AP improves0.606455/0.383314 to0.782557/0.550310. MZ61 HELD DROP858/38 to832/38 and ALL_INVALID212/7 to103/5 expose the old-source cost. Run540.456s; shared21269 RGB encodes;8040042496 allocated scratch bytes released.

Retain both trained arms as scoped CHALLENGERS, with MZ64, MZ68 and MZ66 original dispositions unchanged. Diverse exposure improves new-shape raw ranking in all four missing-input queries and actual DROP decisions, but does not produce all-missing HEAD_NEAR detections. The new-source ALL_INVALID gains include38 BODY_NEAR native winners,13 known-wrong and6 UNKNOWN, with3 native BODY_NEAR losses; the sole HEAD_FAR gain is native. DROP HEAD_NEAR gains4 native winners and loses3. Old-source exposure was halved at fixed total updates, so the old-source decline does not isolate catastrophic forgetting from reduced old training. Better global ranking is not a useful low-FP operating point or a model replacement claim. BODY_NEAR missing ranking is nearly perfect on new geometry (AUC0.999873), yet only57/256 positives pass the original DROP-calibrated cutoff; this makes sensing-state calibration a concrete remaining question alongside head representation and source balance. [Execution](nearfield/MZ70_EXECUTION_20260911.md).

## Frozen topology transfer: MZ69 (2026-09-11)

[MZ69](nearfield/MZ69_TOPOLOGY_TRANSFER_RESULTS_20260911.md): Frozen MZ64 GEOMETRY and MZ68 NULL transfer to all 4096 MZ67 frames, with no fit or recut. HELD DROP G 628/71/396 -> NULL 636/71/388; ALL_INVALID G 0/0/1024 -> NULL 0/0/1024 (TP/FP/FN). Run 76.231s, score 3.466s; RGB4096 once, three shared views, no dense cache. New-shape missing HEAD_NEAR ranking remains weak: NULL HELD AUC0.582977/AP0.337211; prioritize new-source training over a calibration-only explanation.

Retain the complete descriptive transfer as a COMPONENT. All native gains, near/head losses, newly false bits and UNKNOWN remain explicit; no fit, recut or model promotion. MZ64 CHALLENGER, MZ68 and MZ66 NEGATIVE_CONTROL remain scoped as recorded. [Execution](nearfield/MZ69_EXECUTION_20260911.md).

## Missing-input coverage: MZ68 (2026-09-11)

[MZ68](nearfield/MZ68_MISSING_INPUT_COVERAGE_RESULTS_20260911.md) is a scoped NEGATIVE_CONTROL.
One1536-step NULL_COVERAGE candidate, exactly384 ALL_INVALID training steps and384 each original profile; identical MZ64 GEOMETRY frame sequence/init/loss/Adam/calibration. Gate6/11 FAIL. MZ61held ALL_INVALID G{'tp': 43, 'fp': 4, 'fn': 981} -> NULL{'tp': 114, 'fp': 11, 'fn': 910}; DROP G{'tp': 795, 'fp': 38, 'fn': 229} -> NULL{'tp': 826, 'fp': 39, 'fn': 198}. DROP nativeBN additions beyondOLD 17, awning 4. Missing-input training reaches1192/2048 unique MZ61TRAIN IDs, not completeperIDcoverage. Retain MZ64 CHALLENGER and MZ66 NEGATIVE_CONTROL. Run322.964s, score12.176s; 6152605696allocated scratch bytes released.

The sole variable is25% ALL_INVALID steps applied to both native8 and
OLD8 inputs; frame/loss/optimizer/cut rules remain fixed. All11 clauses,
4profile newsource and3profile oldcohort counts, false-bit IDs, native
winners and UNKNOWN remain in the evidence. Numerical teacher-score
preservation is not part of this run. MZ64 CHALLENGER and MZ66 negative
control remain available; no hardware or safety promotion.

[Execution](nearfield/MZ68_EXECUTION_20260911.md) binds actual run, score,
input hashes, process release and cache cleanup.

## Controlled topology source: MZ67 (2026-09-11)

[MZ67](nearfield/MZ67_TOPOLOGY_SOURCE_RESULTS_20260911.md) retains 4096
controlled frames, 1024 geometry IDs and 2048
native event pairs as a source-only COMPONENT. Source-valid/intent matching:
4096/4096; all 80
fixed cases were actually viewed and 80 independently
audited. UNKNOWN 0 query bits / 12370308
full cells remains. Positive count/presence collisions with new and old TRAIN
are reported separately; `NOT_ESTABLISHED_BY_SOURCE_AUDIT`.

[Execution](nearfield/MZ67_EXECUTION_20260911.md) preserves original raw and
1703344513-byte compact archives. Lossless transparent compression saved
3206631424 actual allocated bytes; owners released. No model
fit/prediction or hardware/generalization promotion. Separate MZ68 uses old
MZ61/MZ48, not this source; later model use needs its own declared comparison.

## Replay step projection: MZ66 (2026-09-11)

[MZ66](nearfield/MZ66_REPLAY_PROJECTION_RESULTS_20260911.md) is a scoped
NEGATIVE_CONTROL, not a replacement for the MZ64 geometry-learning CHALLENGER.
One exact1536-step actual-Adam replay projection passes numeric audit but fails
the retaining gate5/11. OldFP173->167 (+2/-8) costsTP3789->3768; MZ61held
795TP38FP->775/38 (+1/-21TP), native BODY_NEAR additions30->20 andawning5->3.
All1536 scalar bounds and16 fixed independent vectors pass;361 steps project,
while actual replay loss rises292times. Average first-order protection does
not guarantee nonlinear/per-query/held retention. Full profiles and prior
comparators remain; no new cutoff, source or outcome-selected mode.

[Execution](nearfield/MZ66_EXECUTION_20260911.md):292.266s model run and9.977s
CPU score,2486 inherited arrays/18954 historical metric rows exact,UNKNOWN
preserved. Task-owned processes exited;6152605696 allocated temporary bytes
released. No hardware, natural-scene, clearance or safety promotion.

## Capture actor reuse: MZ65 (2026-09-11)

[MZ65](nearfield/MZ65_ACTOR_POOL_RESULTS_20260911.md) completes48 worker A/B/R
frames. Map-excluded A34.719/B31.125/R33.203s meets speed conditions, but RGB
repeat-envelope failures12/16 full,13/16 target,13/16 boundary prevent adoption.
Native depth/labels/UNKNOWN are exact. Float64 packet baseline differences are
only<=2.22e-15m and vanish in float32, but the frozen exact gate remains failed.
All16 actual A/B/R visual panels show consistent macro placement/support; they
do not override numerical failure. Keep the original collector and this strict
appearance-preserving pooling recipe as NEGATIVE_CONTROL.

[Execution](nearfield/MZ65_EXECUTION_20260911.md) retains the startup-only0-frame
driver repair, original inputs, all48 captures and28.747MB return. Driver302.423s
exits0; worker UE/Python/Zen, port and temporary uploads are released. No recapture,
new threshold, trained model, new training-source admission or hardware claim.

## Geometry learning: MZ64 (2026-09-11)

[MZ64](nearfield/MZ64_GEOMETRY_LEARNING_RESULTS_20260911.md) fits the new MZ61
TRAIN2048 source against matched continued MZ55 learning. Overall gate FAIL14/19:
all seven new-geometry checks pass, five old false-alert clauses fail. DROP held1024
frozen CONTROL756TP38FP/matched CONTROL761/39/GEOMETRY795/38; native BODY_NEAR
additions beyondOLD8/12/30, heldawning0/2/5. Retain GEOMETRY as CHALLENGER with
both controls and all old comparators, not a default baseline replacement.

Old DROP legacy2733/55->2748/62, MZ48nonfit550/24->554/25, MZ55held478/83->487/86
versus matched CONTROL. New false near alerts include old unsupported far awnings.
ALL_INVALID geometry248/4096 true events still leaves3848 misses. Both source
CAL/HELD roles remain excluded from fit/cuts; MZ61 TRAIN is now fit-consumed.
[Execution](nearfield/MZ64_EXECUTION_20260911.md) takes381.847s CUDA/I/O plus
13.483s CPU score, preserving2133 arrays,12150 old metrics and UNKNOWN. A single
shared cache avoids repeated encoding and releases7627165696 allocated bytes.
No threshold/profile selection, causal native-winner, natural/hardware or safety claim.

## Fixed geometry transfer: MZ63 (2026-09-11)

[MZ63](nearfield/MZ63_GEOMETRY_TRANSFER_RESULTS_20260911.md) completes zero-fit
transfer on4096 MZ61 frames. Both preregistered DROP held1024 flags pass:
CONTROL native BODY_NEAR+8 and COVERAGE+1, no added false-alert bits;
OLD/MZ57 740TP/38FP ->756/38 and743/38. All4096 DROP is2985/154 ->3034/155
and3005/154. Preserve the fixed heads as CHALLENGER evidence and all old
comparators; the earlier MZ62 assignment negative control remains scoped intact.

IDEAL and MERGE add false alerts; held awning has no native added contribution.
ALL_INVALID leaves4025/4076 misses of4096 positives despite CONTROL/COVERAGE
71/20 TP. No adaptive profile selection, threshold tuning, or hardware claim.
[Execution](nearfield/MZ63_EXECUTION_20260911.md) uses4096 RGB decodes and three
trained views shared across four profiles,79.087s CUDA and5.431s score receipt
time, zero fit/recut and no persistent dense cache. UNKNOWN retained;524288
scalar decisions and327680 candidate reconstruction checks pass.

## Geometry source: MZ61 (2026-09-11)

[MZ61](nearfield/MZ61_GEOMETRY_SOURCE_RESULTS_20260911.md) admits4096 controlled
frames from both hosts with1024 geometry configurations and no role-ID overlap.
All source/intent and2048 event-pair checks pass;80 native audits and80 actual
visual reviews pass. All192 held positive geometries differ from TRAIN in exact
count tensors, although some binary occupancy repeats. UNKNOWN12188903 cells
remain. Retain as COMPONENT only; this is neither model nor hardware improvement.

[Execution](nearfield/MZ61_EXECUTION_20260911.md) retains owner-local raw bytes,
1710092176-byte training archives, and original failure/repair evidence. Transparent
lossless compression saved3188998144 allocated bytes across8192 native/support
files. Sites and mesh identities are consumed; generalization remains unestablished.
MZ63 fixed-model transfer has a separate registration and result.

## Matched profile coverage: MZ62 (2026-09-11)

[MZ62](nearfield/MZ62_PROFILE_COVERAGE_RESULTS_20260911.md) compares two
1200-step heads with identical per-frame totals and original loss/cuts.
Complete IDEAL/MERGE/DROP coverage loses to random assignment: retaining
gate FAIL3/7, native new-family held BODY_NEAR17->6; finalOR held640
TP480/FP88->461/82, legacy2732/57->2726/50, MZ48nonfit537/24->522/24.
Preserve this assignment/order recipe as NEGATIVE_CONTROL, its matched
CONTROL, and all older comparators. Neither arm dominates OLD_NEG.

[Execution and diagnosis](nearfield/MZ62_EXECUTION_20260911.md) show awning
held27->10/40 TP despite native winners28->31. Most losses retain native
locations, with reduced raw scores and a higher original-rule cutoff.
Do not infer a trained ceiling or separate training/calibration causality.
GPU work282.924s and CPU scoring3.876s preserve1696 prior arrays,5286 prior
metrics and UNKNOWN. The temporary cache released5739728896 allocated
bytes. No hardware, natural-scene, safety or default-app promotion.

## Native positive-query pooling: MZ60 (2026-09-11)

[MZ60](nearfield/MZ60_NATIVE_QUERY_RESULTS_20260911.md) changes only positive
query training pooling to known native witnesses, with the exact MZ59
DIVERSE schedule, initialization, model and calibration. The retaining
gate fails (2/7): new-family held480 native BODY_NEAR gain1, final OR
held640 TP453/FP83, legacy TP2722/FP48, MZ48nonfit TP518/FP25.
Retain this objective recipe as NEGATIVE_CONTROL; old comparators remain.

[Execution and diagnosis](nearfield/MZ60_EXECUTION_20260911.md) record
192.819s GPU work and3.064s CPU scoring, unchanged prior arrays/metrics and
UNKNOWN. Against59DIVERSE, MZ55 gains4TP/loses9 and adds15FP/removes1;
all15 additions are unsupported HEAD_NEAR. Awning TRAIN16->17/100 and
held1->2/40 do not add native-winning TP. Only34/100 training positives
were presented underDROP, so this is not a fully trained ceiling.
The task cache released5542506496 allocated bytes; no owning process
remains. At MZ60 delivery, MZ61 was only a prepared4096-frame design;
no MZ61 registration or capture was claimed by that terminal. No hardware, natural-scene or safety claim follows.

## Matched diverse training: MZ59 (2026-09-11)

[MZ59](nearfield/MZ59_TRAINING_DIVERSITY_RESULTS_20260911.md) uses MZ55 train
rows in one of two matched 600-step MZ56 GLOBAL continuations. New-family
heldout480 DROP native BODY_NEAR gains are CONTROL0 / DIVERSE1, but final
OR FP on heldout640 increases 79 to81 and on legacy noncalibration 45 to51.
The declared no-added-FP gate fails. Heldout TP451 to454 is a measured
tradeoff; all2560 TP1836 to1852 includes fitting rows. MZ48nonfit CONTROL
514TP/23FP versus DIVERSE505/21 shows a further tradeoff. Retain MZ59 as
NEGATIVE_CONTROL for this coverage recipe, without ruling out other data
or training schemes. No threshold change or automatic budget extension.

[Execution and diagnosis](nearfield/MZ59_EXECUTION_20260911.md) record
252.975s GPU work, 3.829s independent CPU score, shared features, exact old
array/metric preservation and UNKNOWN. Original2560 native files on both
hosts now use863387648 fewer allocated bytes (36.4332%), all SHA/arrays/IDs
unchanged. MZ55 train rows have now been consumed for fitting; its named
heldout sites remain previously inspected Development. Real sensor
performance remains uncalibrated and neither result is a safety claim.

## Fixed diverse-shape transfer: MZ58 (2026-09-11)

[MZ58](nearfield/MZ58_DIVERSE_TRANSFER_RESULTS_20260911.md) evaluates all 2560
MZ55 frames with ten frozen methods and three profiles, no fit or new cutoff.
DROP old union gives 1832 TP/317 FP; LOCAL union 1836/319, GLOBAL 1835/319,
SUPPRESSED 1834/318. All three no-extra-FP checks fail. All additions occur
on retained rods; the three new families (1920 frames) gain zero events.
GLOBAL adds three HEAD_FAR TP, only one with a native winner, and two
HEAD_NEAR FP; no new native outside-field detection. Shallow awning BODY_NEAR
remains 7/160 TP. Retain this transfer as NEGATIVE_CONTROL while preserving
MZ57's earlier scoped challenger. At the MZ58 stage, new data had not yet been used for training.
[Execution and interpretation](nearfield/MZ58_EXECUTION_20260911.md) records
64.391s CUDA inference, 1.46s independent CPU score, shared RGB features,
released processes and no dense cache. Full-frame UNKNOWN remains 7417741.

## Diverse rigid-mesh source admitted: MZ55 (2026-09-11)

[MZ55](nearfield/MZ55_DIVERSE_MESH_SOURCE_RESULTS_20260911.md) completes the
original 2560 frames: 1920 new awning/sign/opaque-patterned-grille frames and
640 retained rods on eight consumed sites. All source/intent checks, 1280
paired event arrays, 80 independent native block-loop audits and 2560 RGB
hashes pass. Roles are 1600 TRAIN_CANDIDATE/320 CALIBRATION/640 HELDOUT_SITE;
actual query positives 640/784/640/800 retain extra far bits. Full-frame
supervision is 568029 B, with 7417741 UNKNOWN cells; original 45-degree ToF
inputs remain unchanged. Raw 4.739 GB stays on owner hosts, compact packages
1.100 GB. Primary completed three main shards (936); worker five (1560) plus
64 canary after explicit handoff of one unstarted shard. Primary's preserved
controller FAIL is the expected ownership stop, with both release receipts
PASS. Retain a source COMPONENT, not an algorithm or perforation claim.
Fixed-checkpoint transfer must preserve cutoffs and consumed-site limits.

## Fixed complementary scale union: MZ57 (2026-09-11)

[MZ57](nearfield/MZ57_COMPLEMENTARY_SCALE_UNION_RESULTS_20260911.md) combines
the unchanged OLD_NEG union with each fixed MZ56 readout. DROP nonfit retained
463 TP/20 FP becomes LOCAL 474/20, GLOBAL 487/20, SUPPRESSED 482/20. GLOBAL
BODY_NEAR rises 66 to 90; 23 of 24 new events have native outside-field winners,
one has a known but non-native winner. Old noncal FP rises 45 to 48. IDEAL
and MERGE GLOBAL give 678/36 and 692/21 versus retained 655/36 and 668/21.
This is disclosed posthoc consumed Development, with no fit or new threshold.
Preserve a CHALLENGER, all constituents, 80 UNKNOWN and old-FP costs. The
additional crop/full deployment paths have not been timed on Android. New
MZ55 source transfer can check the candidate without refitting on those results.

## Measured global context outside local sensor coverage: MZ56 (2026-09-11)

[MZ56](nearfield/MZ56_GLOBAL_ANCHOR_RESULTS_20260911.md) passes its fixed
matched continuation check: DROP nonfit native-outside BODY_NEAR additions
GLOBAL 26 versus LOCAL 11, total BODY_NEAR 66 versus 50, with equal 20 nonfit
FP and old noncal FP 32 versus 36. GLOBAL totals 410 TP/20 FP/582 FN; same
weights with global context suppressed give 404/20/588. Retained OLD_NEG union
still gives 463/20/529. Keep a COMPONENT_OR_CHALLENGER, with all query costs,
UNKNOWN and original measured coverage. A fixed union can inspect the 24
complementary TP at its disclosed old-FP cost; no cutoff rescue or default claim.
[Execution](nearfield/MZ56_EXECUTION_20260911.md): successful CUDA run 150.116s,
CPU score 2.700s, zero training feature recomputation; numerical preflight
failure preserved and sole 4.408GB cache link removed after completion.

## Wider RGB with local-only metric conditioning: MZ54 (2026-09-11)

[MZ54](nearfield/MZ54_FULL_RGB_RESULTS_20260911.md) completes matched600step
756cellCROP/3600cellFULL fits with original45degree ToF. DROPnonfit CROP
401TP21FP591FN,FULL384/20/608; BODYNEAR41each, primarygate fails. FULL gains
2native-grounded TP belowcrop/outside45 butloses19otherTP; fixedMZ53union
remains463/20/529. SameFULLlogits croppedpool gives382/20/610. OldnoncalFP
FULL31vsCROP39, with recallcost; preserve NEGATIVE_CONTROL and allcomparators.
562priorarrays,251424scalarbits,92160nativewinninglookups,2cutoffs and80UNKNOWN
pass. [Execution](nearfield/MZ54_EXECUTION_20260911.md):CUDA258.78s,CPUscore2.48s,
exact16TRAINcropfeature replay,10synthetic checks. Originaltask scratch removed
after4.41GBfeature ownership passed viahardlink toregisteredMZ56, no duplicate.
Next test legitimate global measured context; outsideToF remainsunmeasured.

## Complementary fixed readouts under missing echoes: MZ53 (2026-09-11)

[MZ53](nearfield/MZ53_DUAL_READOUT_UNION_RESULTS_20260911.md) tests uniform
OPEN|GATED composition with existing cutoffs. DROP nonfit fixedMZ50union
455TP/21FP/537FN, NEW_NEG443/23/549, OLD_NEG463/20/529; oldnoncal addedFP
21/12/16. Newpracticalgate passes; MZ51individualgate still fails. OLDOPEN
adds21exclusiveTP0FP with savednativewinning evidence,9withoutgatedcandidate.
Total81addedTP has63acceptednativewinners; union adds11oldFP overitsGATED.
IDEAL OLDunion655TP36FP trailsfixed677/34, so retain CHALLENGER with these
costs and matched unions. CPU1.05s,251424scalarOR checks,1320sealedcountrows;
allpriorpositives and400MZ36/80UNKNOWN remain. No newfit/inference/threshold.
Next compare full-RGB evidence using MZ52 labels; no default/hardware claim.

## Compact full-RGB supervision: MZ52 (2026-09-11)

[MZ52](nearfield/MZ52_FULL_FRAME_SUPERVISION_RESULTS_20260911.md) derives existing
2560native files on both owners into538415B of fullframe8x8-cell counts.
All2432positiveevents have fullframewitness, including352old45gaps;7562077
unknowncells remain. All2560SHA,10240eventcounts/labels,1280eventpairs and80
independent block-loop samples pass; fit/cal/site/family partitions stay.
CPU15.92s primary/24.19s worker,10.83s verify; returnedpayload2.83MB,native0B.
Bothhosts released; preserve the failed pre-work hiddenlaunch receipt.
Retain COMPONENT training/evaluator labels with original45degree ToF unchanged.
Next test full-RGB learning; source coverage is not model improvement or raw
lossless compression, and no capture/inference/fit occurred in this derivation.

## Negative coverage and crop observability: MZ51 (2026-09-11)

[MZ51](nearfield/MZ51_TRAINING_COVERAGE_RESULTS_20260911.md) completes matched
600step NEW_NEG/OLD_NEG continuations. DROP oldnoncal addedFP falls19to8/5,
but nonfit TP falls453to442/442; FP21to23/20. The primary recall gate fails.
OLD_NEG OPEN adds23 nonfitTP with nativewinning evidence,9without gatedcandidate,
but11oldFP; preserve the matched comparator and fixed MZ50/MZ37. Separate
MZ53 tests readout complementarity without rewriting this gate. All original
positives and400MZ36/80UNKNOWN remain;921888scalar checks and4cutoffs pass.
CUDA74.44s, no baselineinference/newdensecache. The CPUcoverage audit finds
352of512BODY_NEAR positives outside45degree supervision; all10gaps in80native
samples also lie below224crop. MZ52 derives full-RGB labels, keeping ToF's
original field and missing status; visible bottom-edge surfaces are a learning
opportunity, not proven RGB metric recovery or physical-sensor capability.

## Native spatial learning and same-weight readout: MZ50 (2026-09-11)

[MZ50](nearfield/MZ50_ECHO_INDEPENDENT_LOCAL_RESULTS_20260911.md) fits one small
spatial head on1280 richer frames;256cal plusold1000DEV,640site/384birch nonfit.
DROP nonfit MZ37=382TP/20FP/610FN; GATED=453/21/539, OPEN=386/20/606.
The same weights isolate mandatory echo gating: unrestricted pooling does
not win. All4 OPEN cutoff maxima are oldDEV negatives without gated candidates.
GATED52of71addedTP have nativewinningcell; OPEN4of4 includes2 witnessed events
with no gated candidate. Across noncal inclfit, GATED adds197TP20FP, OPEN18TP2FP;
retain CHALLENGER with oldfalsealert costs and original MZ37 positives, not
default promotion.400MZ36/80UNKNOWN remain;586656scalar decisions and2cutoffs
pass.5.58s fit,73.70s full CUDA pipeline,0.48s CPUscore,109KB checkpoint;
existing detail mmap reused and no persistent dense duplicate. Improve spatial
separation/observability next; no threshold rescue or hardware claim.

## Rich paired source at kilotier scale: MZ48 (2026-09-11)

[MZ48](nearfield/MZ48_RICH_KILOTIER_RESULTS_20260911.md) delivers2560 source-valid
frames across8 sites/5forms and1280 target/context pairs with identical native
event masks/depth.2432 intent matches; actual labels and352positive bits lacking
45degree local witness remain. All80 prescribed native/RGB samples independently
audited and viewed. The fixed1920/640 split stays; these are consumed sites.
Training ZIP1.095GB plus260KB native auxiliary replaces a4.740GB working raw
representation while full raw evidence remains on ownerhosts. Primary936 and
secondary1624 used warm cache and exclusive unstarted-shard handoff; owned
processes/ports released. Retain COMPONENT source. Next fit native spatial
intrusion evidence without mandatory echo candidates; no source-only accuracy,
sensor validation or natural-scene claim.

## Candidate-domain calibration tradeoff: MZ49 (2026-09-11)

[MZ49](nearfield/MZ49_BANK_DOMAIN_CALIBRATION_RESULTS_20260911.md) changes only
MZ47 REPLAY/ENRICH calibration to the banked candidate domain, using the same
old1000DEV DROP rule. Rich non-fit TP rises18to24 and14to22 without newFP;
outside calibration/fit, each gains42TP but adds37/27FP. Prior positives remain.
Retain CHALLENGER with these costs, original cutoffs and the matched comparator;
no default or zero-cost promotion. Candidate-domain consistency is useful but
does not resolve absent witnesses or true/false spatial separation. Exact old
composition and212352scalar known decisions pass;80UNKNOWN stay. CPU0.34s,
no training/inference. Explanatory metadata finalization is documented without
changing scientific inputs. MZ48 supplies the richer source and MZ50 tests native spatial supervision;
these remain controlled Development without physical-sensor or natural-scene claims.

## Matched local enrichment versus replay: MZ47 (2026-09-11)

[MZ47](nearfield/MZ47_LOCAL_ENRICHMENT_RESULTS_20260911.md) completes two300step
fits from MZ20, with matching old batches and extra event-label patterns.
Under missing close returns,32 non-fit positive events yield REPLAY18TP/0FP/14FN
and ENRICH14/0/18; original MZ37 is13/0/19. Ideal/merged transfer also favors
REPLAY. Both fit groups reach4TP/0FP/8FN. Old relation-DEV ENRICH adds11TP and10FP;
MZ36 missing-return counts do not improve. Retain NEGATIVE_CONTROL for this
12near-example enrichment with hard return eligibility, not all richer data.
Keep the matched replay arm and frozen local/context failure comparators.
All400 MZ36 attempts/80UNKNOWN stay;1,008,672 scalar decisions and both cutoffs
pass. CUDA preparation/fits/inference47.26s, CPU score0.42s; no new dense cache.
[Applied reading](nearfield/LOCAL_EVIDENCE_LEARNING_20260911.md) motivates native
slot-independent spatial supervision and candidates that survive absent echoes.
MZ48 is excluded from this fit; MZ50 subsequently tests that mechanism on
the expanded source. No hardware, natural-scene or default replacement claim.

## Fixed-target context dependency: MZ46 (2026-09-11)

[MZ46](nearfield/MZ46_SCAFFOLD_CONTEXT_RESULTS_20260911.md) removes12 ancillary
supports in four MZ44 rod states. Actual target receipts and all native event
masks/depths remain identical. MZ43 ensemble TP falls3to0 IDEAL and4to0 under
merge/drop, adding3/3/2 false bits. Crossed inputs identify both RGB and ToF
contributions; this is a fixed additive-threshold effect, not RGB-only failure.
MZ35/MZ37 retain3TP with FP1/1/0: keep the local comparators and prior MZ43 scope.
Retain the paired failure as NEGATIVE_CONTROL. Next test local spatial
attribution with paired support/appearance variation; shape/size/site transfer
is still unresolved. One4frame capture, exact old replay and independent native/
branch checks pass;12.41MB raw becomes4.44MB packaged. Owned worker execution
released; zero fit or recapture. No physical-sensor or deployment claim.

## Fixed new-form transfer and scaffold confound: MZ45 (2026-09-11)

[MZ45](nearfield/MZ45_OBJECT_TRANSFER_RESULTS_20260911.md) finds no MZ43 transfer
gain on the 32 new-form positive event bits: RGB and simple ensembles miss all
32 under ideal, merge and drop. The small overall gain is entirely on retained
rod controls. MZ28/MZ37 recover 13/17/6 true bits with zero false bits; retain
these local comparators. Mixed ToF has partial useful evidence suppressed by
the RGB mean, but unconditional union has substantial false alerts.
All 40 attempts, exact old replay and independent native/scalar checks pass.
New forms omit the rod's 12 supports and have different size/negative placement;
shape alone is not isolated. Retain this fixed failure as NEGATIVE_CONTROL and
MZ43's earlier measured scope. MZ46 subsequently tests fixed-target support
removal. No hardware or natural-scene claim.

## Rich-object far coverage: MZ44 (2026-09-11)

[MZ44](nearfield/MZ44_RICH_FAR_OBJECTS_20260911.md) adds 20 source-valid frames
and five complete far paired groups; BODY_FAR/HEAD_FAR each have 10 positives,
with zero near leakage or UNKNOWN bits. This fills MZ42's distance gap on the
same four new forms and rod control, bringing the near/far blocks to 40 frames.
Independent native-label and full byte/hash checks pass; the compact source is
20,622,239 bytes versus 42,308,036 raw. Worker execution resources are released.
Retain COMPONENT: one shared site, small scaled/floating opaque forms remain;
neither new block was used to fit or select MZ43. Fixed-model transfer on these
forms is the next capability question, not established by source admission.

## Matched restricted-input training: MZ43 (2026-09-11)

[MZ43](nearfield/MZ43_RESTRICTED_TRAINING_RESULTS_20260911.md) isolates a training
mismatch with exact old-fit reproduction and equal input-exposure budgets.
Mixed ensemble MZ36 missing-return FP/FN improves 9/21 to 7/14, merged 7/14 to
7/11, ideal 6/10 to 4/8. All old true positives survive with no added false bits
on this cohort. Old ideal EVAL FP/FN changes 50/114 to 52/104; retain CHALLENGER
with this cost, not universal no-regression. Original MZ5/MZ28/MZ37 remain.
Four fits and cached scoring took 8.11s CUDA, no backbone inference; independent
packet/count checks pass. A 534,669-byte compact ensemble reproduces all 16,140
cached decisions. Rich-object data remains outside fitting/selection. Next test
the fixed candidate on those sources; no physical-sensor or deployment claim.

## Richer rendered objects and compact source delivery: MZ42 (2026-09-10)

[MZ42](nearfield/MZ42_RICH_OBJECTS_20260910.md) completed one secondary-worker
capture: pipe, open ladder, irregular pouch and woody birch, each with HEAD,
BODY, BOTH and visible nonintruding variants, plus four retained rod controls.
All20 attempted frames pass source/native/floor checks; all5 paired groups are
admitted and independent geometry reproduces all80 event labels. This is a
COMPONENT source pipeline result, not model accuracy. Positives are near-only;
one shared site, scaled/floating fixtures and leafless opaque birch leave
far-range, natural-placement, foliage and cross-site evidence unestablished.

[Lossless compact sources](nearfield/DATA_LIGHTWEIGHT_20260910.md) preserve exact
bytes and permit direct reads without full extraction. MZ36's1,247 files shrink
from751,784,227 to307,320,958 bytes in the portable package; MZ42's92 files shrink
from42,326,459 to20,489,306 bytes. Original MZ36 source allocation also fell from
755,059,248 to424,073,776 bytes under transparent NTFS compression, saving
315.65MiB without changing paths or bytes. These are measured batch results;
packages remain additional recoverable copies, not claimed whole-drive savings.

[Applied reading](nearfield/RESTRICTED_OBSERVATION_LEARNING_20260910.md) directs
the next hypothesis toward matched restricted-input training and observation
quality, with mesh/site held-out evaluation. This remains untested; MZ41's
negative control and the fixed fusion comparators remain relevant. All owned
worker execution resources were released; raw source and receipts are retained.

## Query missing-observation guard: MZ41 (2026-09-10)

[MZ41](nearfield/MZ41_MISSING_GUARD_20260910.md) tests a zero-valid-query-zone
guard on saved outputs in 0.141s, zero inference/fitting. Under missing close
returns it recovers four true bits and adds four false bits: FP2/FN18/exact363
becomes FP6/FN14/exact361. Ideal/merged decisions are unchanged. Broad protection
recovers eight true bits at six new false bits and also adds false bits on the
other arms. Retain NEGATIVE_CONTROL: binary observation absence alone is not
sufficiently selective. Prior positives, all400 attempts and80UNKNOWN bits stay.
Continue richer observation-quality learning alongside diverse data and compact
source storage; no default replacement or physical-sensor claim.

## Frozen constrained-observation sensitivity: MZ40 (2026-09-10)

[MZ40](nearfield/MZ40_L8CX_CONSTRAINED_RESULTS_20260910.md) completes the two
fixed close-return proxies on the same 380 admitted MZ36 frames. MZ37 ideal
FP3/FN0/exact377 becomes midpoint FP4/FN2/exact374 and invalid-close-zone
FP2/FN18/exact363. In the latter arm, MZ35 removes eight true MZ28 bits while
removing seven false bits; MZ37 restores none. RGB bytes/scores, all weights,
bank and cutoffs are frozen. Exact 16-frame adapter parity and independent
packet/score/group audit pass. All 20 excluded frames / 80 UNKNOWN bits remain.
Retain diagnostic COMPONENT and ideal/RGB comparators. Prioritize a separately
registered matched restricted-observation comparison and observable-quality-aware
selection. Model training mismatch and information loss are not separated.
Midpoint and zone invalidation are named proxies, not calibrated VL53L8CX
responses. This two-arm experiment ends here; diverse paired object replacement
remains the data direction. No additional capture, fitting or hardware claim.

## VL53L8CX observation mismatch: MZ39 (2026-09-10)

[MZ39 saved-packet audit](nearfield/MZ39_L8CX_READOUT_AUDIT_RESULTS_20260910.md)
finds4234/4893dual-return zones(86.53percent) separated by<600mm, across346
of380MZ36frames. Independent24320zone recount and immutable-input checks pass;
zero inference or fitting. The10cm-bin/3pixel ideal extractor is not equivalent
to VL53L8CX sensing. Surviving separated pairs are not certified valid returns.
The user sets VL53L8CX as the simulation target and requests limited capability;
the current single-zone adapter does not override that target. Follow the
[observation contract](nearfield/VL53L8CX_OBSERVATION_CONTRACT_20260910.md).
Retain diagnostic COMPONENT and ideal comparators. Prioritize a declared
constrained readout before new-object capability claims; broad asset replacement
includes nonvegetation. No device result or degraded-model task score yet.

## Positive restoration tradeoff: MZ37 (2026-09-10)

[MZ37 fixed restoration](nearfield/MZ37_POSITIVE_RESTORATION_RESULTS_20260910.md)
recovers17 noncalibration true bits with2 newFP; placement FN67->52,
FP26->28, exact2910->2922/3000. MZ36's two motivating misses are recovered,
withFP3 unchanged; these cases are now consumed Development. All previous
positive score values and MZ28 additions survive. One oldDEV calibration and
saved-scalar replay, zero fit or neural inference; independent audit passes.
The zero-new-FP criterion fails; retain CHALLENGER, baselines unchanged.
One new error strongly prefers the wrong branch; BODY_NEAR/FAR calibration
has no eligible false-restoration negatives. Next examine action-conditional
coverage or paired new object shapes, preserving original controls and UNKNOWN.
No posthoc threshold rescue, hardware, temporal or safety promotion.

## Frozen new-XY transfer: MZ36 (2026-09-10)

[MZ36 fixed new-source comparison](nearfield/MZ36_NEW_SOURCE_RESULTS_20260910.md)
Frozen new-XY comparison: 380/400 frames in 76/80 groups admitted. MZ5/MZ28/MZ30/MZ35 FP=6/6/4/3, FN=10/2/2/2. MZ30: PASS, removed2FP, lost0TP; MZ35: PASS, removed3FP, lost0TP. No refit or calibration.
Shared prior regions/assets and ideal geometric readout limit this to controlled
Development. All attempted/excluded groups and UNKNOWN remain explicit.
Both remaining far-crossbar misses already have a correct positive modality,
but lie outside the negative-only selector scope. A separately defined restoration
test is the next mechanism question; no rule was added to MZ36.
Retain finite transfer/tradeoff evidence; MZ5 remains unchanged. No extra fitting,
source expansion, threshold rescue or device/safety promotion.

## Fixed responsibility optimization endpoints: MZ35 (2026-09-10)

[MZ35 fixed trajectory](nearfield/MZ35_RESPONSIBILITY_CONVERGENCE_RESULTS_20260910.md)
reproduces every MZ32 model tensor and1200loss/exposure rows bit-for-bit before
continuing the same shared selector to4800steps. TRAIN errors66->28, weighted
BCE0.160144->0.063869; placementFP30->26, FN67 unchanged, exact2908->2910/3000.
All normal MZ28 TP/additions and trained pole49/48 survive. Distance FP5->0,
relation25->26 and oldDEV17->18 show nonuniform transfer; BODY_NEAR TRAIN errors
7->8 despite lower loss. The prior crossbar case now correctly chooses RGB.
Independent prefix/schedule/optimizer/calibration and31200task-bit audit passes.
The joint<=20FP target still fails; retain4800 as CHALLENGER, MZ5/MZ20 unchanged.
One fixed trajectory is complete. Next diagnose responsibility transfer or test a
frozen candidate on new appropriate sources; no appended duration/threshold rescue.
Consumed Development only; no device, temporal, fresh-source or safety promotion.

## Query separation admission: MZ34 (2026-09-10)

[MZ34 admission](nearfield/MZ34_QUERY_SEPARATION_RESULTS_20260910.md) exits
NOT_ADMITTED_NO_FIT: final shared-gradient minimum+0.077761 fails the fixed
<=-0.1 criterion despite66remaining TRAIN errors. No model initialization,
architecture inference, optimizer, fit or task predictions were produced.
The prepared four-query prototype is untested. Retain the recorded admission
outcome as COMPONENT; this does not establish query separation is ineffective.

## Shared query gradient diagnosis: MZ33 (2026-09-10)

[MZ33 frozen gradients](nearfield/MZ33_QUERY_GRADIENT_RESULTS_20260910.md)
checks initial/final MZ32 on1165TRAIN disagreements with zero optimizer updates.
All six final combined-shared gradient cosines are positive, min0.077761;
raw TRAIN responsibility errors are7/13/22/24. Count-weighted query gradients
match the direct global gradient, and states/input hashes remain unchanged.
An initial readout-only negative pair does not imply final combined conflict.
Retain diagnostic COMPONENT; positive aggregate gradients do not prove harmless
sharing or remove conditional/capacity questions. MZ35 tests optimization separately.

## Expanded responsibility supervision: MZ32 (2026-09-10)

[MZ32 matched supervision](nearfield/MZ32_EXPANDED_RESPONSIBILITY_RESULTS_20260910.md)
changes TRAIN eligibility from168 to1165 branch disagreements, preserving model,
initialization, normalization,1200batches, optimization and runtime repair scope.
Inverse-frequency weights use expanded478/687 class counts with the same formula.
It retains all MZ28 true positives across normal cohorts and removes53placementFP:
totalFP83->30, FN67 unchanged, exact2864->2908of3000. All additions and pole49/48
survive. Relative to MZ30,1BODY_NEAR TP is restored but netFP rises20->30.
The predefined joint improvement gate requires<=20FP and fails. Retain as a
CHALLENGER with an explicit retention/false-alert tradeoff; MZ5/MZ20 unchanged.
Independent supervision, calibration and task audit passes; one fit complete.
Next test frozen candidate behavior on new appropriate sources or diagnose
responsibility transfer before a separately defined successor. No threshold rescue,
fresh-source, device, temporal or safety claim follows from consumed Development.

## Responsibility label coverage: MZ31 (2026-09-10)

[MZ31 TRAIN coverage](nearfield/MZ31_RESPONSIBILITY_COVERAGE_RESULTS_20260910.md)
finds1165 total frozen-branch disagreements among7562 original TRAIN frames.
Every query/branch-correct class has>=20examples and>=5explicit sites; original
168eligible supervision omits997available disagreements. Counts RGB/ToF are
BODY_NEAR152/46, BODY_FAR69/81, HEAD_NEAR65/239, HEAD_FAR192/321. Expanded rows
include supported and baseline-negative cases; coverage alone cannot prove their
transfer to the runtime repair domain. Independent count audit passes, no fit or
prediction change. Retain diagnostic COMPONENT supporting the matched MZ32 test.

## Latest branch responsibility candidate: MZ30 (2026-09-10)

[MZ30 branch selection](nearfield/MZ30_BRANCH_RESPONSIBILITY_RESULTS_20260910.md)
uses one8641parameter selector on frozen RGB/ToF branch disagreements without
geometric support. The pre-fit opportunity gate finds80of81 baselineFP; eligible
TRAIN target counts are120RGB-correct and48ToF-correct. One1200batch fit removes
63baselineFP: placement totalFP83->20 and exact2864->2915of3000, but FN67->68.
One relation BODY_NEAR baselineTP is lost; all MZ28 additions, including72far,
and trained pole49/48 survive. No newFP. OldDEV exact940->957, clean176->182,
stress175->181; these reused views do not provide independent confirmation.
Independent branch, calibration and31200task-bit audit passes. Zero-TP-loss gate
fails; retain only as CHALLENGER with explicit tradeoff, MZ5/MZ20 unchanged.
Frozen residual diagnosis identifies the lost alarm and remaining20FP separately.
Next address responsibility/coverage transfer using a separately defined mechanism;
no posthoc cutoff or seed rescue. TRAIN branch predictions are in-sample.
Fit complete; consumed Development only, no device/temporal/safety promotion.

## Baseline correction surface diagnosis: MZ29 (2026-09-10)

[MZ29 support partitions](nearfield/MZ29_BASELINE_SUPPORT_RESULTS_20260910.md)
finds80of81 placement baselineFP have no original geometric candidate, alongside
48correct placement and28correct oldDEV BODY_NEAR alarms. All81FP have no actual
selected-return query contributor, but54placementTP also lack such contributors.
MZ22's supported-only branch did not test this responsibility; MZ28 preserves
baseline alarms by construction. Missing support cannot justify cancellation.
Independent110387200geometry-bit and full exported-partition audit passes; no
fit or prediction change. Retain as diagnostic COMPONENT, not an inference veto.

## Latest full-task packet support comparator: MZ28 (2026-09-10)

[MZ28 packet availability](nearfield/MZ28_PACKET_AVAILABILITY_RESULTS_20260910.md)
uses7362 fixed TRAIN packet/known references, top5 per physical zone, majority
per angular cell; no current labels/site IDs or learned visual features enter
availability.375bank sites are disjoint DEV. Placement far72/75, addedFP2,
exact2864/3000 versus MZ26's71/3/2860; trained pole49/48 and MZ5 positives survive.
Compared with MZ26, placements gain6TP and lose4, remove2FP and add1. Residuals
are relation1488HEAD_FAR cabinet and1677BODY_FAR sign; totalFP83 includes81baseFP.
Retain packet support as COMPONENT; no-added-FP still fails, MZ5/MZ20 unchanged.
Independent task/geometry/bank/vote audit passes. Explicit stress echo alignment
has39moved zones but0changed witness/count elements; no task numbers changed.
Runtime bank38.2MB, cached batched retrieval2.859s/4400frames, not phone timing.
One fixed evaluation complete, no process remains. Next investigate complementary
learned/packet support with actual witness losses; no appended sweep or fusion arm.
Consumed Development only; broad goal active, no temporal/device promotion.

## Latest observable feature separation: MZ27 (2026-09-10)

[MZ27 frozen descriptors](nearfield/MZ27_FEATURE_SEPARATION_RESULTS_20260910.md)
compares110 fixed locations with same-site/group exclusions and label-blind
packet-matched TRAIN references:92scored,18class-insufficient, all75far witnesses
exist. Negative-tail known-label correctness is packet27/29, raw64=10/29,
raw3x3=8/29, learned40=4/29. Positive witnesses are49/61,49/61,53/61,51/61.
Thus raw visual expansion lacks the predefined conditional separation advantage;
information absence is not proven. One residual has0/128negative references and
remains unscored. Retain diagnostic/packet-coordinate comparator as COMPONENT;
next test actual task effects with witness losses and missing support explicit.
No model or cutoff changed; MZ5 baseline, MZ20 challenger unchanged. Receipt and
independent identity/descriptor audit pass; no fit or process remains. Consumed
Development only, broad goal active.

## Latest deterministic optimization comparison: MZ26 (2026-09-10)

[MZ26 fixed endpoints](nearfield/MZ26_DETERMINISTIC_CONVERGENCE_RESULTS_20260910.md)
completes one4800-step deterministic trajectory. Fixed1200->4800 comparison
improves placement far additions69->71of75 and addedFP7->3; exact2855->2860.
Pole49/48 and all MZ5 positive judgments survive. Zero-added-FP still fails;
total placementFP84 includes81 baseline false bits preserved by add-only design.
TRAIN negative activation29.43->18.34%, but relationDEV negative p99 rises
3.01->4.32 and positive witness retention87.00->85.90%; other DEV tails also worsen.
Retain optimization-response evidence as COMPONENT, not full model promotion.
MZ5 baseline and MZ20 challenger unchanged. Both endpoint audits pass; fits and
diagnostics complete. Next inspect observable feature separation before choosing
a new objective or spatial representation; no blind duration extension. Consumed
Development only, no fresh-source/device/temporal claim. Broad goal remains active.

## Latest optimization-prefix diagnosis: MZ25 (2026-09-10)

[MZ25 prefix and sampler](nearfield/MZ25_CONVERGENCE_RESULTS_20260910.md) stops
at1200 of planned4800steps: a separately trained prefix differs0.314403 despite
23shared input hashes, exact initialization and batches. No4800candidate or
convergence result exists. Repeated fixed CUDA backward demonstrates small local
gradient differences; strict mode rejects grid_sample backward. A fixed bilinear
matrix sampler preserves10577parameters and passes equivalent forward/adjoint
and bit-identical repeated gradient checks, with9.8MB extra matrix storage.
Retain sampler as engineering COMPONENT only. Next use preregistered endpoints
within one deterministic trajectory; do not reopen or waive MZ25. MZ5 baseline,
MZ20 challenger and MZ24 objective component unchanged. Broad goal active.

## Latest decisive availability objective: MZ24 (2026-09-10)

[MZ24 decisive supervision](nearfield/MZ24_DECISIVE_AVAILABILITY_RESULTS_20260910.md)
completes one matched1200-step fit with the same head, initialization, batches
and cutoffs as MZ23. Added placementFP falls8->4 versus MZ20 (7->4 versus MZ23),
retaining70/75 far additions and trained pole49/48. Exact2860/3000; no-added-FP
still fails. Retain the changed objective as COMPONENT only; MZ5 baseline and
MZ20 challenger unchanged. Clearing the learned tail by a common posthoc cutoff
would retain32/75far, so no threshold rescue is adopted. Independent scalar,
geometry, initialization,58frame inference and100352gradient-element audits pass.
TRAIN-extrema diagnosis reduces negative-availability bag activation85.37->29.43%
(1855/6303 remain); placement rates36..38%. The gap is already present on TRAIN,
not only transfer. Pole has0 such negative bags. Next compare bounded optimization
before attributing failure to capacity or acquiring more data; do not reopen this
completed fit. Consumed Development only; no App/device/temporal promotion.

## Latest angular availability experiment: MZ23 (2026-09-10)

[MZ23 availability](nearfield/MZ23_AVAILABILITY_RESULTS_20260910.md) completes
one1200-step fit on frozen MZ20/MZ5. It retains74/75 added placement farTP and
trained pole49/48 of49 clean/stress, but only removes1 of8 newFP; exact2861/3000.
Retain this exact recipe as NEGATIVE_CONTROL; MZ5 baseline and MZ20 challenger
unchanged. Availability precision71.3/75.0% on placements leaves strong false
tails. A posthoc common cutoff clearing all8 would retain only14/75 farTP and
pole40/38, so no cutoff rescue is adopted. Privileged known-availability oracle
retains71/75 farTP with0 addedFP and pole49/48, but is not an inference method.
Independent audit checks31200task bits; selected58frame inference also agrees.
Six of7 residual false tails lie about19..25native pixels from actual support,
so adjacent-cell boundary ambiguity alone is insufficient. Next mechanism must
improve decisive-negative separation without discarding true witnesses. All fits complete; consumed Development only,
no fresh-source, temporal, hardware or default-App promotion. Broad goal active.

## Latest obstacle evolution: MZ19-MZ22 (2026-09-10)

[MZ20 cross-frame ranking](nearfield/MZ20_RANK_OBJECTIVE_RESULTS_20260910.md)
restores trained pole BODY25/25 clean/stress (all queries49/48 of49), versus
MZ18 BODY4/1. PlacementDEV3000 adds75 farTP and8FP over MZ5; exact2862.
BODY local AP .801/.785 improves on MZ16 .708/.703, below MZ18 .838/.828.
Retain ranking candidate as a controlled challenger; no baseline promotion.

[MZ19 saved margins](nearfield/MZ19_MARGIN_DIAGNOSTIC_RESULTS_20260910.md)
finds correct within-frame ordering but lost cross-frame separation in MZ18;
the cutoff-driving task negatives frequently lack known-local samples.
[MZ21 geometry trace](nearfield/MZ21_FALSE_ADDITION_RESULTS_20260910.md)
finds all16/7/8 MZ16/MZ18/MZ20 newFP have no actual packet query contributor
and no known eligible local candidate. Evaluator-only support is not an input.

[MZ22 residual arbitration](nearfield/MZ22_EVIDENCE_ARBITRATION_RESULTS_20260910.md)
adds89 farTP but loses18 original farTP on placements, with7 addedFP and2864
exact. Pole falls to41/35 of49 clean/stress, although3 sequence BODYfarFP are
removed. Retain this exact recipe as NEGATIVE_CONTROL. MZ5/MZ9 unchanged.

Next active goal question: learn observable angular evidence availability or
source support while preserving true-witness strength, original near coverage,
and low false alerts. UNKNOWN stays unknown; geometry is not source availability.
All current fits are complete; no job left running. Broad improvement goal
remains active; no hardware/App promotion.

## Latest BODY witness-objective fit (2026-09-10)

[MZ18 BODY objective](nearfield/MZ18_BODY_OBJECTIVE_RESULTS_20260910.md) completes
one matched1200step fit. Placement BODY addedFP falls11->3 versus MZ16 HIGH_DETAIL;
relation BODY near/far actual-contributor AP improves0.708/0.703->0.838/0.828.
Pole BODY maxima now have true query contributors25/25 in both conditions, but
clean/stress BODY detections collapse25/25->4/1 as true scores fail alert cutoffs.
Full placement far gains/addedFP are70/7 over MZ5 versus80/16 for MZ16. Retain
attribution mechanism as COMPONENT only; the alert augmentation fails FP and pole
retention. MZ5/MZ9 unchanged; no rescue fit.31200 task bits independently audited.

## Latest witness-objective gradient diagnostic (2026-09-10)

[MZ17 gradient diagnostic](nearfield/MZ17_WITNESS_OBJECTIVE_RESULTS_20260910.md)
checks the final MZ16 HIGH_DETAIL snapshot on1024 TRAIN draws/873 unique frames.
Known wrong maxima55/779 positive supervised outputs all receive net upward
logit pressure. BODY near/far rates20.79/14.38%; HEAD2.07/2.46%;67 unknown maxima
remain unknown. Aggregate7.06% misses the predefined10% fit-admission heuristic,
so no fit was launched. This confirms a scoped conflict, not its dominance or
failure of the untrained witness objective. Retain diagnostic as COMPONENT;
MZ5/MZ9 unchanged.4096 output rows independently recounted, zero model updates.

## Latest controlled visual detail comparison (2026-09-10)

[MZ16 visual detail](nearfield/MZ16_VISUAL_DETAIL_RESULTS_20260910.md) completes
matched LOW/HIGH detail1200-step fits with identical ROI, readout and batch IDs.
Consumed placementDEV3000 adds61/80 farTP and22/16 FP respectively over MZ5.
HIGH pole clean/stress reaches49/48 of49, LOW47/46; HIGH actual contributor
macro AP declines on both placement cohorts. All added false winners lack
actual source at their selected cell. Both fail the fixed false-alert budget;
stop this pair, retain MZ5/MZ9. No new capture, temporal restart or promotion.
Independent audit covers62400 task bits and1324062 local candidate bits.

## Latest shared-support model comparison (2026-09-10)

[MZ15 shared support](nearfield/MZ15_SHARED_SUPPORT_RESULTS_20260910.md) completes
two matched1200-step fits. SHARED adds64 farTP and17 FP on consumed DEV3000;
QUERY adds37 farTP and15 FP. Pole clean/stress is12/18 of49 versus49/48 forQUERY.
Both fail the fixed error budget; all added false winners still lack actual
source at their chosen location. Stop this pair, retain MZ5/MZ9. No new capture,
temporal restart or model promotion. Stress diagnostic slot alignment repaired
without changing predictions;62400 output bits audited.

## Latest local evidence trace (2026-09-10)

[MZ14 evidence trace](nearfield/MZ14_EVIDENCE_TRACE_RESULTS_20260910.md)
replays4200 consumed frames with exact frozen SOURCE margin parity. All17 MZ13
added false bits select cells without actual return contributors (16 sign bits,
15 groups/12 sites). Source-location oracle removes them but also6 recovered far
TPs; all130 SOURCE far misses have correct candidates below threshold. Paired
pole features respond but fine localization is unstable. Prioritize shared local
return-support prediction; no fit, capture or temporal restart in this diagnostic.
MZ5 remains baseline and MZ9 a coverage-bounded component, no model promotion.

## Latest gate coverage intervention (2026-09-10)

[MZ13 training coverage](nearfield/MZ13_TRAINING_COVERAGE_RESULTS_20260910.md)
adds original relationTRAIN5000 and distanceTRAIN2500 to the unchanged20parameter
600step gate. Sign-added FP only17->16; all15 BODY sign errors remain. OldDEV
loses6 previously recovered nearTP (exact928->926), failing retention. Thin
improves46->47 clean and44->46 stress, still below prior48 admission. Stop this
fit; no extra capture or continuation. MZ5 and MZ9 remain unchanged.

## Latest existing-data inventory and transfer (2026-09-10)

[MZ12 existing-data replay](nearfield/MZ12_EXISTING_DATA_RESULTS_20260910.md)
accounts for all20000 accepted frames. Fixed relationDEV2000 + distanceDEV1000
replay adds57 far TPs with MZ11 but18 FPs,17 on hanging signs. Exact improves
1883->1905 and920->929, while FP47->56 and34->43. Thus existing placements expose
an attribution gap without new acquisition; unchanged error-budget transfer fails.
All data remain consumed Development. Preserve MZ5/MZ9; no refit or collection.

## Latest selective addition (2026-09-10)

[MZ11 selective addition](nearfield/MZ11_SELECTIVE_ADDITION_RESULTS_20260910.md)
adds32 oldDEV TP bits including17 far, exact912->928, with unchanged FP6/11/10/7.
All original positive judgments survive. Sequence exact174 clean and172 stress;
thin46/49 and44/49 fail the frozen48/49 requirement, wrong correspondence1/49.
The20-parameter gate shows useful consumed-Development recall recovery, but
preserves original premature near warnings. Stop after one600-step fit; no
new capture, threshold rescue or temporal reopening. Keep MZ5 and MZ9 separately.

## Latest availability composition (2026-09-10)

[MZ10 fixed availability rule](nearfield/MZ10_AVAILABILITY_RESULTS_20260910.md)
uses SOURCE where geometric candidates exist and original MZ5 otherwise, with
zero fits or threshold changes. It restores oldDEV BODY_NEAR169->197, but exact
912->900 versus MZ5 and FP6/11/10/7->11/20/19/14. All34 original false positives
survive and30 new ones enter. Sequence exact177 versus143, thin48/49, but fallback
restores near errors. Candidate availability does not establish reliability;
standalone FP budgets do not compose. Stop this rule; no new confirmation capture.
Retain MZ5 baseline and MZ9 component separately; no temporal or App change.

## Latest source supervision and equal-exposure controls (2026-09-10)

[MZ9 contributor supervision](nearfield/MZ9_SOURCE_SUPERVISION_RESULTS_20260910.md)
corrects all22 old false-positive winner labels. SOURCE_RGB gives198/200 targeted
training-regression exact, thin48/49 and pole-absent25/25; wrong correspondence
reduces thin to1/49. Equal-new-exposure MZ5 adaptation gives199 exact; NO_RGB166.
Neither result is fresh confirmation. OldDEV BODY_NEAR MZ5_ADAPT180 andSOURCE169
trail frozen187. All28 SOURCE lost baseline TPs lack geometric candidates; its
ceiling is170/200. Keep source supervision/readout as a coverage-bounded component,
not whole-baseline replacement. Keep MZ5, no confirmation capture or temporal
reopening. Three1200-step fits complete; no pooling or threshold rescue. A future
availability-aware combination is untested and must preserve existing judgments.

## Latest angular-return attribution probe (2026-09-10)

[MZ8 single-frame attribution](nearfield/MZ8_ATTRIBUTION_RESULTS_20260910.md)
recovers thin49/49 on explicitly targeted training/regression, but pole-absent
BODY_FAR activates22/25 and wrong-zone correspondence still recovers49/49.
Exact143->177 hides BODY_FAR FP3->22; do not retain this2149-parameter recipe.
All22 false-positive winning sampled native points are outside BODY_FAR despite
passing0.10m echo compatibility: native forward3.200m versus query endpoint3.18m.
The angular/range hypothesis moves them inside. Region-consistent attribution
remains unresolved. Keep MZ5; no fresh confirmation capture or temporal reopening.
One1200-step fit complete; no same-run tolerance/threshold rescue. Native data
supervises/scorers only, invalid packets remain UNKNOWN, no hardware/safety claim.

## Latest single-frame paired readout (2026-09-10)

[MZ7 paired diagnosis and calibration](nearfield/MZ7_SINGLE_FRAME_RESULTS_20260910.md)
separates the approaching bar (ToF HEAD_FAR14/14 correct, all suppressed by RGB)
from the thin pole (both branches negative on all49 far opportunities). Thin
BODY_FAR RGB delta decreases24/25; positive HEAD response is not reliable regional
readout. One old-DEV fitted calibration gives MZ6 exact141 versus143, thin0/49,
and BODY_NEAR FP7 versus5. Do not retain it. Keep MZ5 and temporal closed.
Next evidence question is position-preserving echo-to-query attribution, with
wrong-zone correspondence control; its causal role is not yet established.
The200 samples remain consumed Development. No fresh confirmation was warranted
after this failed comparator; UNKNOWN and sensor/safety boundaries are unchanged.

## Latest regression diagnosis and short-sequence pilot (2026-09-10)

[MZ6 regression diagnosis](nearfield/MZ6_REGRESSION_RESULTS_20260910.md) reproduces
52 gains/27 losses:26/32 wrong bits in lost-exact rows have one correct branch;
all32 far TP losses across EVAL retain native cropped support. MZ5's net+25 exact
is+34 negative controls minus9 elsewhere. Keep its error benefit/far-recall cost;
do not interpret wrong-far as recovered near misses or infer absent information.

[MZ6 short sequences](nearfield/MZ6_SHORT_SEQUENCE_RESULTS_20260910.md) completes
one admitted200-frame/8-clip Willow pilot, zero fits. CURRENT143 exact, MEAN3 142,
compatible143. Mean3 loses2 far TPs and delays the exit clip's first hit by2 nominal
samples. Compatible makes0 completions: required local support is absent. The
thin-pole clean baseline already misses BODY_FAR25/25 and HEAD_FAR24/24, so packet
restoration alone cannot repair it. Retain current MZ5 and this diagnostic control;
no temporal upgrade.84 packets have no valid return, never CLEAR. Failed source
attempts and receipts are preserved; accepted capture readiness passes and owned
processes/ports/cache/temp are released. No more fits or threshold rescue here.

## Previous fixed ensemble and spatial fusion (2026-09-10)

[MZ5 fixed ensemble](nearfield/MZ5_FIXED_ENSEMBLE_RESULTS_20260910.md) reaches1378/1500
exact (91.87%) versus MZ1 fusion1353; wrong-far12->4 and cross-body19->8. Retain
its132872-parameter compact readout as the next controlled learned-readout baseline;
CPU/CUDA5000-row sign parity passes. Far recall declines (BODY272->257/300;
HEAD270->255/300). No uniform or safety superiority, alert or hardware change.

[MZ5 spatial](nearfield/MZ5_SPATIAL_FUSION_RESULTS_20260910.md) gives local1355 versus
pooled1354 and local-RGB1298 exact; paired+16/-15, HEAD_ONLY near134 versus131/150.
The criterion passes narrowly; the weak increment trails ensemble1378. Retain as
a mechanism control. Three fixed fits and one permutation are complete and audited.

[MZ3](nearfield/MZ3_ERROR_ATTRIBUTION_RESULTS_20260910.md) and
[MZ4](nearfield/MZ4_MOTION_RESULTS_20260910.md) remain completed; MZ0 remains an
information component. No temporal fit follows; consumed controlled Development.

## Previous learned fusion and resolution result (2026-09-10)

[MZ1](nearfield/MZ1_RESULTS_20260910.md) reaches90.20% exact versus matched
RGB87.33% and ToF88.47%, but wrong-far12 and cross-body19 exceed retained MZ0
7 and15. Full replacement gate fails: retain a challenger, no continued fit.
[MZ2](nearfield/MZ2_RESULTS_20260910.md) matched1x1/4x4/8x8 fusion gives
88.93/90.13/90.20%; fixed8x8 noise/dropout/column-shift gives90.27/89.33/87.40%.
This supports coarse metric information and correct correspondence, not a need
for64 zones or hardware performance. Both budgets completed;5000 original
alerts unchanged and independent replay/scoring pass. All evidence remains
consumed controlled Development. MZ0 remains the retained information component.

## Previous clean multi-zone information result (2026-09-10)

[MZ0](nearfield/MZ0_RESULTS_20260910.md) passes the illustrative information gate:
spatial exact79.87% to87.33%,nearest wrongfar43/600 to7/600,crossbody72/600 to15/600,
and1500 original alerts unchanged. All three readouts have the same exact total,
not identical predictions. SameFoV full depth reaches92.47%; single-zone center
readout is worse than RGB. No claim that64zones equal full depth or optimal fusion.
User prioritizes generic MultiZone-ToF-64 clean information, then MZ1 learned
fusion and MZ2 resolution/corruption; physical hardware fidelity comes later.

[Motion-coded feasibility](nearfield/TOF_MOTION_SCAN_RESULTS_20260910.md) separately
shows known-pose scans can resolve271/300 ideal HEAD cases versus0 stationary,
while background-return laws yield no gain. Multiple source objects remain valid
explanations. It is not a real walking or VL53L8CX result.

## Previous simulated fusion and jitter contrast (2026-09-10)

The [static and rotational results](nearfield/BODY_QUERY_TOF_SIM_RESULTS_20260910.md)
retain ideal15degree positive near support: near hits51/53,95/106,94/106 with no
added false events. RGB conflicts remain; strict simplified pairs stay0, while
the naive threshold is stronger there. No general fusion superiority or training.

User-requested fixed-origin rotation adds no anytime coverage in this saturated
source, lowers mean fusednear89.00% to82.07%, and raises spatialflips8.76% to10.24%.
Conservative orientation-aware association contributes abstentions. This is a
static-scene simulation, not measured walking. All600 B alerts remain unchanged.

## Previous observation study: fixed ToF footprints and cone support (2026-09-09)

The [600-frame ToF footprint study](nearfield/BODY_QUERY_TOF_COVERAGE_RESULTS_20260909.md)
finds all frames mixed at both27deg and15deg ideal diagonal footprints; no95percent
target-dominance cases. This is source geometry, not sensor failure or fusion
accuracy. User confirms simulation-only hardware. Preserve mixed return behavior
as unmodeled until explicit hypothetical response assumptions are introduced.

A tested positive-only cone-support primitive can add HEAD_NEAR when every
possible return direction and bounded range lie inside that query volume. It
never erases visual events or asserts clearance. Four engineering tests pass;
no data-level fusion gain or actual VL53L1X calibration has been demonstrated.
Next: explicit ideal response/validity sensitivity, without oracle association.

## Previous fixed comparator and observation route (2026-09-09)

The [frozen LOCAL transfer contrast](nearfield/BODY_QUERY_LOCAL_TRANSFER_RESULTS_20260909.md)
does not remedy transfer: strict original-fixture pairs3/53 versus JOINT29/53,
and0/106 in each simplified-fixture arm. All600 original B alerts/probabilities
remain identical. This rules out simply substituting the existing LOCAL readout;
it does not prove RGB contains no distance information. No fitting was performed.

The user-linked discussion now motivates [RGB plus constrained VL53L1X-like input](nearfield/RGB_TOF_OBSERVATION_20260909.md).
First check surface coverage and ambiguity in a fixed sensor footprint; preserve
invalid/mixed returns and require calibration. No full UE depth or oracle target
association enters model inputs. A ToF fusion result has NOT yet been measured.
Keep JOINT as the scoped comparator and original B alerts unchanged.

## Previous diagnostic: original fixture under background translation (2026-09-09)

The [background-only comparison](nearfield/BODY_QUERY_BACKGROUND_RESULTS_20260909.md)
retains exact original13-part crossbars and camera-relative geometry. On53 matched
admitted pairs, strict JOINT accuracy changes41/53 (77.36%) to29/53 (54.72%);
near query hit150 to119/159, wrong-far2 to3/53. Original120-frame alerts are unchanged.
Both regions decline;16old-correct pairs become wrong,4old-wrong become correct.

Original full-extent admission remains NOT_EVALUABLE (clamp self-occlusion). A
separate visible-HEAD diagnostic, frozen before model access on the same120frames,
admits26/30 and27/30 pairs with exact in-corridor native target witnesses; BODY
contamination stays excluded. This is consumed Development, not a restored primary.
The strong attribution criterion remains INCONCLUSIVE because the matched old
control is below80%; observed degradation is retained, not promoted to a unique
causal explanation. Keep original alerts and scoped JOINT; no ordinal continuation.

## Previous falsifier: fresh region and fixture transfer fails (2026-09-09)

The [frozen fresh-size test](nearfield/BODY_QUERY_FRESH_SIZE_RESULTS_20260909.md)
captured 240 pairs in two geographically new regions. Source admission retained
212 pairs, below the frozen 90% coverage gate; the original primary remains
NOT_EVALUABLE. A disclosed zero-fit diagnostic on those 212 pairs finds JOINT
strict pair-correct 0/106 in BOTH ordinary and size-matched arms, near query hits
113/318 and 116/318, wrong-far 26/106 each. Original alerts retain 480/480 parity.
Independent scoring and old-source aligned-batch regression pass.

Do not promote or extend to an ordinal head. Ordinary also changes assembly,
width and material, so geography versus fixture versus size cannot be isolated.
Retain the previous controlled result and original B alerts; broader spatial
transfer is unestablished. Old 90% was frame-exact; strict paired accuracy 82.4%.

## Previous algorithm result: direct visual range evidence (2026-09-09)

The [frozen context decoder](nearfield/BODY_QUERY_CONTEXT_DECODER_RESULTS_20260909.md)
recovers HEAD-near native query TP115 to1680/1788, with false activations332 to112
and wrong-far564 to24/600 on consumed relation EVAL. The unchanged
[paired-distance diagnostic](nearfield/BODY_QUERY_CONTEXT_DISTANCE_RESULTS_20260909.md)
retains near2085/2250 versus1/2250 baseline, wrong-far45 versus730/750 and exact
two-range1350 versus638/1500, passing all3 spatial-transfer criteria. LOCAL is a
strong simpler comparator; gains are not uniquely attributable to joint context.

Retain original10k B alerts plus JOINT as a separate spatial-evidence component.
The new count-derived alert chain regresses and is not a full replacement.
The two-output research interface preserves all3000 EVAL alert decisions and
support exactly while exposing range disagreement/UNKNOWN. Warm CUDA batch1
median16.86ms becomes19.29ms. These are shared-asset consumed Development results,
not natural-source, continuous-approach, metric-depth, Android or safety claims.
Next decision: unchanged-method independent-source/view confirmation and mechanism
attribution; preserve all completed fits and thresholds without rescue sweeps.

## Collection and historical context


Next collection phase: [CITY-CROSSREGION-V1](../../../experiments/city-field/CITY_CROSSREGION_V1.md),
7 regions / 4 TRAIN + 1 DEV + 2 TEST, with 336 planned counterfactual frames.
The TRAIN-only engineering canary passes 16/16 frames and 4/4 quartets after a
crossbar/clamp geometry repair; failed receipts remain preserved. Seven Big City
descriptor candidates are not yet admitted routes or proven isolated backgrounds.
Full collection and model training have not started; this adds no model result.

[Source reconnaissance and worker handoff](../../../experiments/city-field/SOURCE_COLLECTION_20260908.md)
now retain 39 successful primary source views, seven native floor grids, and a
runtime-verified native HLOD export. Several candidates favor waterfront plazas;
three dense TRAIN candidates are queued for worker scouting. Asset transfer
passed; the first cold smoke timed out with zero frames. An isolated warm-cache
transfer enabled one worker frame, but Nanite/VT missing-resource warnings keep
it REVIEW. Logs trace this to another UE replacing its Zen service on port 8558.
The launcher now selects a separate cache-service port and rejects missing-resource
signatures for formal captures. Worker source scouts completed 15/15 frames with
zero matching resource errors, and the separate port-fix probe passed 1/1 on
actual port 25229. Owned worker processes/tasks/port were released. These remain
source reconnaissance, with formal route and split admission pending.
Source-graph closure and seven useful isolated routes remain incomplete.

Current engineering priority is the [City Sample collection field](../../../experiments/city-field/README.md):
complete loaded routes, representative supported obstacle conditions and clear
controls, synchronized geometry-labelled acquisition, and region/route/instance
split isolation. Model fitting is a separate workstream and is not a prerequisite
for accepting this collection infrastructure. Historical model diagnostics below
remain scoped evidence rather than the field's development objective.

The user reconfirmed field delivery as the priority: visibly complete and usable
sidewalk/intersection/narrow-passage segments, plausible street obstacles and
controls, and [one-command filtered data export](../../../experiments/city-field/COLLECT.md).
Abnormal frames are excluded with reasons; label-method expansion and detector
diagnostics are not prerequisites for this delivery.

[Field v2](../../../experiments/city-field/FIELD_V2_20260908.md) now completes
one-command capture, filtering and portable export: 36/36 usable frames across
three west segments, all four obstacle conditions and clear controls, with
12/12 active targets evaluable. Shifted sidewalk samples avoid inspected native
props; clear no longer includes a universal portal. The narrow aisle and overhead
supports remain declared controlled geometry. All images reviewed, generator
reproduction and exported payload checks pass, capture processes released.

The [field-v1 fixed-method diagnostic](../../../experiments/city-field/DIAGNOSTIC_V1_20260908.md)
now caches one frozen G10/G13 inference pass on all108 admitted frames. G10
misses30/45 reliable target-band opportunities with11/45 joint alert/support
hits; G13 misses45/45 with0 joint hits. True clear-control false alerts are3/7
and0/7 respectively. Excluding five floor-review frames changes no positive
or miss counts. Retain field/failure evidence; no model promotion, training,
threshold adjustment, or monocular-depth causal claim follows from this run.

The [City approach diagnostic](../../../experiments/city-field/APPROACH_V1_20260908.md)
adds 105 settled views on the same west route, with exact-XY floor preflight.
Fourteen hazard frames retain UNKNOWN target geometry. Frozen G10/G13 miss
39/64 and 64/64 eligible target-band opportunities; G10 has 23 joint hits but
18/21 clear-pose false alerts. Eight of nine sequences with a joint hit lose it
at a nearer eligible pose; the last hits only at the nearest sample. Retain
distance/view failure evidence; no reliable warning distance or model promotion.

The [target uncertainty audit](../../../experiments/city-field/UNCERTAINTY_AUDIT_20260908.md)
locates 15 unresolved collision rays across those 14 frames: 12 near other
components, two no-hit rays and one farther other-component hit. Render gates
pass, but instance identity remains unresolved; labels and diagnostic counts
are unchanged. Explicit reasons and per-ray overlays now support label repair.

Existing controlled Development environment: **self-built UE5 StreetLabV4**, with 10 Hz
observations and measured conservative envelopes. The historical motion runner
is `tools/run_obstacle_research.py`; CARLA is retained for history/supplementary
checks. See [UE acceptance](unreal/UE_LAB_ACCEPTANCE_20260906.md).

The first [original City Sample route](nearfield/CITY_NATIVE_ROUTE_20260908.md)
now captures 21 settled RGB-D observations along 10 m of native
`Small_City_LVL` sidewalk. Capture/transport and process release pass; strict
floor geometry remains REVIEW (19/21 within the 2 cm median criterion).
The follow-up [native-route validation](nearfield/CITY_NATIVE_VALIDATION_20260908.md)
restores surrounding blocks with bounded full-detail loading and distant HLOD,
and captures 40 settled views along 16.95 m. Independently checked bollard
BODY opportunities are missed 2/2 by both frozen G10/G13; supported-sign
HEAD is missed 2/2 by both. One unreliable floor frame stays UNKNOWN. Retain
this consumed Development diagnostic and fix source transfer before expanding
acquisition; independently suspended obstacles and temporal behavior are untested.
The [inference/cache diagnosis](nearfield/CITY_NATIVE_DIAGNOSIS_20260908.md)
reproduces 16 original VAL scores exactly for all six weights. Aligning CPU/GPU
pixel conversion changes no City alert decisions. G10 has diffuse support and
a bollard score below every known negative; G13 shows weak rank separation and
suppressed support. Threshold tuning alone is insufficient; no new fits/capture.
The subsequent [native adaptation](nearfield/CITY_NATIVE_ADAPT_20260908.md)
uses45 TRAIN/45 DEV views on disjoint original instances. One300step seed17 fit
improves DEV BODY/HEAD recall to46.4%/44.4%, below the50% retention criterion;
retain the original model. Willow alert decisions hold but support IoU drops.
Its geometry audit corrects same-instance early collision from occlusion to
UNKNOWN: the original route's meter is no longer independently verified and
must be NOT_EVALUABLE in target metrics. Rendered all-scene labels are unchanged.
The [native thin sampling plus Willow replay](nearfield/CITY_NATIVE_REPLAY_20260908.md)
tests one new300step fit from original seed17 with4 thin/4 uniform City/8 Willow
TRAIN per batch. DEV BODY recall improves to53.6%, but HEAD falls to16.7%;
route bollard joint alerts remain0/2. Willow BODY/HEAD support IoU recovers to
0.2312/0.0897, still only52.2%/44.0% of original. Retain original; no rescue fit.
The combined sampling recipe is diagnostic only; this does not isolate replay
from oversampling. Inspect target-view coverage and train-versus-route score
separation before expanding capture or changing the model.
The [inference-only fit/transfer audit](nearfield/CITY_NATIVE_SCORE_AUDIT_20260908.md)
finds replay/thin alerts4/4 on exposed TRAIN bollard positives but0/2 on the
consumed route under locked DEV cutoffs; native-only remains0/4 and0/2.
Route target size/distance lies inside the TRAIN range, while viewing direction
and surroundings differ. Replay route scores0.773/0.680 sit below5/17 and6/17
DEV negatives, so recovering them by a scalar cutoff would break the DEV10% FPR
constraint. Prioritize a bounded view/context coverage comparison; scale-only,
extra identical training and route-tuned cutoffs are not established repairs.
The [eight-view same-instance probe](nearfield/CITY_NATIVE_VIEW_20260908.md)
reproduces replay's locked-DEV control alerts4/4. Both reliably checked1.5m
transverse views lose alert/joint hits (2/2 to0/2), with scores0.973/0.961 falling
to0.758/0.489 despite target support overlap and peak hits. The two2.4m
transverse targets remain UNKNOWN, excluded. Same-instance view/context
sensitivity is now observed; yaw/background causality remains entangled.
Retain original weights; next bounded coverage test must preserve separate DEV
and old-scene localization rather than extend this completed eight-view probe.

The [Willow sample](unreal/UE_WILLOW_SAMPLE_20260907.md) provides native geometry
and sanitized RGB-D. Conservative tree AABBs still disagree with native sweeps;
its engineering checks are not dynamic algorithm scores.
The earlier pause applied to that saved sample. G14 procedural-world visual
development is now authorized, with primary-machine creation and visual/truth
acceptance before secondary-machine bulk capture or long training. See
[G14 realism acceptance](nearfield/WORLD_REALISM_ACCEPTANCE.md) for the intended
criteria and pending capabilities; this is not a passed visual or model result.
The [attached fixture suite](nearfield/CONTEXTUAL_HEADSPACE_20260908.md) replaces
the user-rejected floating proxies with grounded maintenance frames, a
wall-mounted hinged cabinet and a bracket-supported sign. Its 96 core plus
32 hard conditions pass native geometry agreement (128/128); a separate mounted
casement preview retains its window opening. This is one-site synthetic scene
engineering, not a trained-model result or independent-world benchmark.
The [G14 asset/scene iteration](nearfield/WORLD_REALISM_20260908.md) delivers two
target-free native previews, four CC0 asset imports and passing capture/truth
checks. The [first City Sample PCG plaza](nearfield/CITY_PCG_PLAZA_20260908.md)
now reuses an official graph with 545 generated instances and passes three
clear/BODY/HEAD native capture and geometry checks. Wider visual acceptance,
multi-world generation remains pending. The subsequent
[200 m street](nearfield/CITY_PCG_STREET200_20260908.md) now has a V7 follow-up
with darker asphalt, distant buildings, volumetric clouds and a reusable
24-frame obstacle suite passing native capture/geometry checks. Persistent
[project caches](nearfield/CITY_PCG_CACHE.md) and one-session batch capture
reduce repeated work. Visible-gap distance evidence retains documented visual,
collision and task-definition limitations;
no new model training or promotion was performed.
The [complex real-asset suite](nearfield/CITY_COMPLEX_ASSETS_20260908.md)
adds bicycle, scaffold frame, barricade, picnic table and park bench with
center/lateral controls at two existing scene poses; all 26 capture/geometry
checks pass. This is not an independent-world or trained-model result.
The [50-group pilot](nearfield/CITY_TRIAL50_20260908.md) completes 150 frames
with capture/geometry/group QA PASS in 185.219 s including editor lifecycle.
A six-frame paired check supports the faster settling profile; asset/shader/
streaming readiness remains enforced. The pilot is one map and is not balanced
across HEAD risk states; no model training or new promotion follows from it.
The [worker commissioning](nearfield/CITY_WORKER_20260908.md) now passes the
same 150-frame acceptance after fixing first-use assets missing from RGB.
The corrected cached baseline takes 184.904 s editor lifecycle. The subsequent
[throughput work](nearfield/CITY_CAPTURE_THROUGHPUT_20260908.md) separates
production export from reference/preview work and bounds verifier GPU memory.
Its optional fast profile completes 150 frames in 127.286 s including job/QA,
with all task risk states matching. It is not background-pixel/depth equivalent;
the standard profile remains the default for strict matched comparisons.
The subsequent [500-group collection](nearfield/CITY_COLLECTION500_20260908.md)
uses 8 routine settling ticks and 32 first-use/large-view-change protection.
The worker actually completes 1,500 frames / 500 groups with native, geometry
and group QA PASS in 346.527 s complete job, producing 2.87 GB. This is one
world, 20 camera poses and five meshes, not independent-world model evidence.
The [data audit and split](nearfield/CITY_DATA_AUDIT_20260908.md) rehashes all
4,500 payloads and assigns street TRAIN250 groups / plaza TEST250 groups.
HEAD has only 30 binary positives, all scaffold, with zero HEAD-only or HEAD
DANGER examples. Distance-grade training is not ready; City requires an
independent adapter for all-scene support and pixel UNKNOWN. No new fit ran.
The saved sample remains frozen at
`willow-finish-4k`; first-person acquisition uses an optical center **1.70 m
above the floor**, with `tools/run_sample_segment.py sensors`. The eleven-frame
`willow-eye170-first-person` passes replay and ground-depth optical-height checks.
This is an acquisition smoke test.

## Latest grouped object diagnostic

[NF-G13](nearfield/DECOUPLED_20260908.md) completed three fixed600-step deep-content
gate fits; consumed screening reused G12 B/C outputs without refitting/inference.
Fixed checkpoints were then inferred on new G13
same-Willow groups D primary joint is31/32 and diversejoint15/16;
BODY/HEAD fixed IoU 0.396/0.202, peaks
63/64 and54/64. The consumed screen passed before this cohort;
the prospective joint criterion passed. Retain D as a stronger Development challenger,
C as localization comparator, and G10 as full-contract baseline. No further
training/capture or App promotion; same-Willow Development only.

[NF-G12](nearfield/REPRESENTATION_20260908.md) completed six paired600-step fits
of RepViT and RepViT plus shallow detail, reusing G10 with zero refits. On32 new
groups, common-VAL recall-first primary ensemble joint is25/32 G10,30/32 RepViT,
27/32 plus detail; bar recall28/32 ->32/32 for both new arms. RepViT is retained
as a strong decision challenger, not a failed representation route. The full
joint replacement rule remains unmet: RepViT diversejoint stays14/16 and HEAD
peak falls32/64 ->29/64; detail raises HEADpeak45/64 but diversejoint falls12/16.
Detail's paired single-seed primary joints do not decline, and compatibility
ensemble ties29/32, so its aggregate tradeoff is operating-point dependent.
Retain G10 as the complete-contract baseline and the B/C checkpoints as disclosed
challengers. Worker native capture verified128frames; six fits total730.06s;
new models are about4.73M parameters versus45,646. No temporal/App promotion.

[NF-G11](nearfield/DETACHED_20260907.md) completed three fixed detached-gate fits
and 32 fresh paired TEST groups. Localization recovers (BODY/HEAD IoU
0.160/0.113 -> 0.252/0.230), but primary joint correctness falls 26/32 -> 17/32:
retain G10 as baseline. Common-VAL calibration gives ensemble 24/32 -> 28/32,
with bar recall 31/32 -> 28/32 and no paired-seed joint improvement; retain
detached only as a disclosed operating-point challenger. Consumed spatial
interventions reduce joint 21/32 -> 0/32, showing spatial dependence, not faithful
localization. No further sweep or video/App promotion; closing remains unscored.

[NF-G10](nearfield/DIVERSITY_20260907.md) completed the fixed data-expansion x
predicted-region-head comparison: primary ensemble joint correctness is
14/32 existing/plain, 14/32 existing/region, 13/32 expanded/plain, and 21/32
expanded/region. The combination improves all three paired seeds; HEAD box-only
FP falls 14/32 to 4/32 with bar-only recall 32/32 to 31/32. Diverse-stratum joint
correctness remains only 6/16. Retain the combined near-field Development
candidate, with plain comparators: support IoU actually declines, so decision
gain is not faithful-mask localization or natural/closing/App certification.
Native GPU capture completed 864 independent-state images in 154.94 s of
acquisition (223.14 s engine); twelve fixed fits totaled 47.90 s. Old G9-B TEST
is separately disclosed regression; new TEST is consumed and not retuned.

[NF-G9-B](nearfield/FACTORIAL_20260907.md) completed a data-only ordinary-model
comparison on 40/12/12 fresh TRAIN/VAL/TEST geometry groups. Full four-state
coverage gives primary frozen-threshold ensemble joint all-four correctness
4/12, versus single-object training 2/12 and frozen G8 1/12; consumed G9-A
regression is 6/12 versus 1/12 and 1/12. BODY improves, but HEAD box-only false
alerts remain 5/12 (single-object training 1/12), with substantial seed variation.
Retain full-state data and the model as Development ingredients, not a complete
grounding fix or App promotion. Six equal-budget fits totaled 12.60 s; 256-image
capture 199.05 s. Preflight byte-identical native conversion reduced paired
conversion time by 24.1%; no large new map or bulk acquisition was needed.

[NF-G9-A](nearfield/GROUNDING_20260907.md) froze the nine G8 checkpoints and
thresholds on12 fresh bar/box quartets (144frames,48windows). Ordinary HEAD
alerts12/12 withboth objects but still8/12 withboxalone, versus0/12 withneither;
both BODY/HEAD outputs are correct acrossallfour variants in only1/12 groups.
Selective object response is insufficient on this compound/box Development
source; this does not identify a shortcut cause or overturn G8's scoped signal.
Retain G9 as an intervention regression set; target source/body-part separation
before assuming maskgating or a larger backbone will fix the problem. No new
App/DTR promotion. UE now supports perobject native visibility, exact retained
geometry checks and measured settling preflight. Main capture100s, frozen
inference3.43s, no fitting; saved Willow unchanged.

## Latest RGB-only learning probe

[NF-G8](nearfield/WHISKER_20260907.md) completed 1,024 controlled clips with
24 held-out parameter groups on the shared frozen Willow background. Ordinary
video ensemble near TP is BODY192/192 and HEAD254/288, FP20/576 and6/480.
Closing TP120/120 falls to0/120 on repeated history; near heads remain current-
frame-only. Bio improves HEAD near recall to284/288 with FP29/480 (6.04%);
it is a recall/FPR tradeoff, not an unconditional replacement. Keep ordinary
as a compact Development comparator and bio as a challenger. A posthoc
pose-matched diagnostic supports temporal closing use by both video arms.
Auxiliary support does not establish causal object attribution. No new App or
retained DTR promotion follows. Capture efficiency and receipt-path recovery
without retraining are recorded in the report; failed source/logs remain.

## Capability question and current probe

Goal: **盲杖互补的类别无关前视障碍感知** under limited compute. Prioritize walls,
large forward barriers, body/head protrusions and suspended hazards; then poles
and multi-height supports. Knee-height hazards remain relevant; ultra-low
obstacles are secondary compatibility evidence. Earlier warning and dynamic
coverage need future temporal tests. The wearer chooses movement, not a planner.
Past denominators stay frozen; no assumed cane-coverage labels or retroactive
headline selection. Base near evidence survives missing optional ground/height.

The [representation probe](nearfield/REPRESENTATION_PROBE_20260907.md) retained
162/210/240 of240 analytic positives with stride/dense/supported methods; FP0/0/1
retains the coherent3x3 artifact. This is constructed component evidence.
The fixed Hypersim Small RGB model initially recovered0/26 on11 consumed frames.
Simulator height/pitch remain privileged; native agreement is not object truth.
Original GPU inference/postprocess P50 was86.17/3.91 ms, not phone latency.

Saved-prediction diagnosis found all 119,721 native near-surface reference pixels
lost before aggregation: two invalid, 67,956 over-range and 51,763 still near but
below the height filter. The [cached ground probe](nearfield/GROUND_ANCHOR_20260907.md)
then recovered 1/26 with scale only and 21/26 with ground-relative height, with
or without scale; all four arms had 0 FP. Fits used predictions/calibration only.
Low-boundary matches123/807 remain secondary diagnosis, not the main objective.
[Support comparison](nearfield/SURFACE_SUPPORT_20260907.md): all arms 21/26,0 FP.
[18 distinct views](nearfield/DISTINCT_VIEWS_20260907.md): raw22/87 vs ground37/87,
0 FP, but failed ground fits suppress15 raw wall positives. [Fusion](nearfield/EVIDENCE_FUSION_20260907.md)
restores them:52/87 legacy hypothesis cells,0 FP; direction-only near output can
retain UNKNOWN height. Body/head subgroup16/47 is not verified-height recall.
[Frontend comparison](nearfield/FRONTEND_DOMAIN_20260907.md): neither candidate fixes the bar.
[Sparse structure probe](nearfield/STRUCTURE_INFORMATION_20260907.md): ideal-pose geometry
recovers0/4 MDE-missed translational2–3 m bars; coverage and depth ambiguity remain.
P50 13.87 ms,45 native range mismatches; baseline retained. No App promotion.
Historical registry repair and NF-G0–G5 backfill: [record](nearfield/REGISTRY_REPAIR_20260907.md).

[NF-G7 contact learning](nearfield/CONTACT_RETINA_20260907.md) completes 48 paired
clips / 768 frames with group-separated training, validation and test. On 8 test
contact episodes, temporal-structure seeds recover6/6/7; repeated RGB/pose history
recovers0/0/0. The ensemble recovers7/8 with0/8 negative-clip false alerts, but
nine BODY@2s false-positive cells span3 true-contact clips. Single-frame ensemble
also recovers7/8 (one negative-clip false alert); ordinary video recovers0/8 at its
validation-selected operating point. This is useful joint-history candidate
evidence, not an overall replacement gain. Retain the existing baseline; next
research should resolve urgency calibration, weak near-contact coverage and
single-frame seed instability. Ideal metric poses and analytic task-box contact
remain synthetic privileges; no App, human-mesh or real-sensor promotion.

## Retained evidence and baseline

- **Public/JRDB X21:** same-live-track transport of an already authorized X13
  component; six consumed sequences reached `5/6` CONTACT, 11 false segments,
  45.45% Event F1, `3.061 s` median lead, and `8/18` dropout recovery. This is
  same-source Development; genuinely source-disjoint confirmation is pending.
- **CARLA X73:** the latest complete source-disjoint synthetic Development
  confirmation. On C35, parent-hull reconstruction improved X72 by
  `+6 TP / +0 FP / +2.24 pp` frame F1, reaching `132/18/40` TP/FP/FN and
  `88.00/76.74/81.99%` precision/recall/F1. The same-map, scripted-motion,
  detector and evaluator boundaries remain; successors lack its confirmation.
- **CARLA X94:** consumed post-hoc Development reference, `1478/84/417`
  TP/FP/FN and `94.62/77.99/85.51%` frame precision/recall/F1 across eleven
  cohorts. Its one-observation full-dropout bridge added six TP and no FP over
  X93; same-parent evidence and an unchanged valid issued plan remain required.
- **Credible simple comparator:** in the consumed C35 raw-input pilot, Kalman
  CV + route tube + `0.60 s` hold tied X94 at 66.67% Event F1 and four false
  segments, with two versus ten fragmentation runs. X94 retained better frame
  F1 (`84.62%` versus `81.52%`), median lead (`2.80` versus `2.65 s`) and CLEAR
  (`0.00` versus `0.20 s`). This tradeoff has no fresh eleven-arm adjudication.
- **X95 remains a challenger:**
  `DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_GATE_NOT_MET`. Its consumed replay
  gained `3.74 pp` Event F1 and removed five false segments versus X94, but lost
  `7.02 pp` frame F1 and added 30 fragment gaps. Simple `0.60 s` hysteresis
  reached 89.07% frame F1 on that replay; complexity has not established a win.

## Preserved source bottleneck

Frozen avoidance R1 remains `NOT_EVALUABLE_SOURCE_CAPTURE_INTERRUPTED`.
Subsequent startup engineering admitted a reused Development composite, but
FIT_ONLY S03 failed dropout credential readiness; no final method score followed.
See [startup and method diagnosis](CARLA_CAMERA_STARTUP_20260905.md). Source
engineering changes no algorithm inheritance. Old raw-input comparisons lack
sufficient retained payloads beyond C35 for fair reconstruction.

## Preserved controller decisions (parked during perception work)

1. **UE Development:** use fixed RGB-D replay for perception changes and V4
   closed loop for motion changes. The measured motion reference is DEPTH_ONLY;
   candidate DTR remains an optional challenger. The new 10 Hz Development
   bank completed 32 actual branches: both sets of 8 straight controls passed,
   candidate depth succeeded 5/8 and candidate DTR 6/8, with no success
   regression. The extra success was late crossing, with nine changed-action
   frames. This is rerendered Development, not exact-pixel pairing or promotion.
2. **Latest Development finding:** the separately identified
   [visible-surface clearance experiment](unreal/UE_VISIBLE_SURFACE_CLEARANCE_20260907.md)
   finished 16 actual branches. A fixed .40 m extra surface margin and +/-1.20 m
   bypass regressed depth success from 5/8 to 2/8, with no gains. Late-stop and
   continued-walking both halted before a static planter entered by the new
   bypass. Reject this fixed variant; keep the defaults and earlier evidence.
   Its structured disposition is pending normal registry publication because
   the pre-existing experiment-index fingerprint mismatch remains unresolved.
   The next useful comparison must cover static street geometry in contact
   evaluation, establish a traversable witness for its action space and test
   replanning when a bypass becomes blocked. Existing scenario-only contact
   scores do not cover the newly diagnosed planter. Preserve conservative
   contact, goal, time and missing-support reporting; do not shrink proxies or
   tune this consumed margin to rescue the result. The unchanged late-crossing
   DTR gain still awaits a scoped replication.
3. **CARLA preservation:** park the pending dropout-window continuation and
   eleven-arm execution during the UE migration. Do not restore missing CARLA
   assets or recapture by default. Keep complete source, frozen method snapshots,
   old failures and prior algorithm inheritance. UE results do not inherit CARLA
   confirmation authority. This switch starts no X97, learner or held-out run;
   JRDB source-disjoint confirmation remains a separate requirement.

## Boundaries

- Frozen R1 is not retried or reclassified as fresh confirmation. Preserve its
  seals, nine shards and crash evidence; no protected fit/final access follows
  from capture-engineering success.
- C35-C41 cannot be rerun as confirmation. C8-C11 admitted no evaluable X31
  occlusion source; N4 v1 cannot resume or retry. It is consumed incomplete,
  not a three-town result. Historical outcomes and source gates remain unchanged.
- Consumed diagnosis is Development, not fresh authority. Keep frozen thresholds,
  source gates, lifecycle, association, seeds and denominators with their original
  results; changes belong to a separately identified Development version.
- `UNKNOWN` and `NOT_EVALUABLE` are not `CLEAR`, negative evidence or safety.
  Historical DTR route conflict owns its event correctness; near-field alerts
  use their separate perception evaluator. Public replay and CARLA do not establish Android readiness,
  natural-distribution performance, user benefit, deployment or safety.
- Uncommitted candidates and outputs remain WIP. Existing structured inheritance
  roles and historical verdicts remain authoritative; this compaction changes none.

## Evidence links

- [HEAD-P1 appearance and matched consistency](nearfield/HEAD_PAIRED_RESULTS_20260908.md):
 96-frame native-equal diagnostic observes L damage3/16, M0/16. Two matched
 2000-step fits give identical original EVAL64 HEAD TP17/32,FP8/32,AUC0.702;
 .1 consistency reduces support recall58.39%→57.69%. Both new fits lose BODY
 TP16→10 versus Coverage and increase HEAD FP1→8. Paired damage3→1 also hides
 an oblique-rod support collapse5/6→3/6 becoming1/6→1/6. Retain exact paired
 diagnostics and the scoped negative recipe; preserve Coverage1198. No further
 fit, weight sweep, MIL/Dice/backbone change or App promotion from this result.

- [HEAD-S1 support-to-decision audit](nearfield/HEAD_S1_20260908.md):
  zero training. Coverage1198 HEAD near AUC0.930/0.709 on DEV/coverage EVAL
  beats max/top6/logit-LSE0.795-0.808/0.647-0.651. On consumed plaza all four
  scores have zero attainable recall at FPR<=10%; support AUC remains below0.5.
  Simple support-only pooling does not rescue HEAD. Prioritize cross-region
  evidence stability, without proving position-specific readout innocent or
  appearance causal. Preserve the baseline; no automatic MIL/consistency fit.

- [Masked support Dice loss comparison](nearfield/CITY_SUPPORT_DICE_20260908.md):
  same1198 TRAIN/schedule/init and one2000-step fit improve added64 HEAD IoU
  0.2055 to0.4421 with99.63% positive-pixel recall. Consumed EVAL HEAD IoU
  only0.1852 to0.1930 while positive-pixel recall55.94% to36.71%, nearTP12
  to11/32 and FP1 to3/32; DEV near recall also falls. Retain this fixed Dice
  recipe as a scoped negative control, not the new default. Training overlap
  responds to supervision, but cross-region positives are lost; no weight sweep,
  architecture-failure claim or extra steps follow.

- [Matched64-frame regional coverage pilot](nearfield/CITY_COVERAGE64_20260908.md):
  one same-budget full fit on1198 TRAIN improves new same-world EVAL HEAD
  TP8 to12/32, FP3 to1/32, AUC0.6553 to0.7090 and IoU0.1385 to0.1852;
  joint0 to3/16. Added64 fits128/128 near bits, but20/32 EVAL HEAD positives
  remain missed. Negative support activation increases and DEV HEAD IoU falls
  0.2032 to0.1852; BODY EVAL AUC also slightly falls. Retain a limited-coverage
  Development challenger, not a relation/generalization repair or promotion.
  The one2000-step fit is complete; no automatic budget extension.

- [HEAD-X0 cross-region audit](nearfield/HEAD_X0_AUDIT_20260908.md):
  zero training; F HEAD plaza AUC0.25751/AP0.01327, while all15 old/plaza
  paired positive pooled masks match. FP peaks concentrate near distant
  building context;35 peaks remain UNKNOWN. Blur and mean-fill controls
  disagree, so background causality is unproven. Conditional local affinities
  shift toward positive references for FP. Retain data-coverage diagnostic;
  next one source-separated small-data pilot, not reused-plaza held-out claims.

- [Full-parameter City fitting at matched2000 steps](nearfield/CITY_FULL_FIT_20260908.md):
  all2268 TRAIN near bits fit; DEV BODY/HEAD recall81.25%/71.875% at9.375%
  FPR each, IoU0.20745/0.20324 and joint6/32 improve over frozen B2000.
  Consumed plaza HEAD remains0/15 TP with171/735 FP at unchanged DEV thresholds;
  full updates are a stronger fitting/DEV candidate, not a generalization fix.
  Keep G13-D; next inspect cached regional/family error structure, no promotion
  or automatic budget extension.

- [Balanced tiny TRAIN fitting and gradient diagnosis](nearfield/CITY_TINY_FIT_20260908.md):
  B frozen/F full each fit all64 near bits and8/8 groups at2000 steps. Full
  updates improve BODY/HEAD IoU0.110/0.089 to0.555/0.488, though F HEAD misses
  the fixed0.5 strong-fit criterion. Gradient connections behave as expected.
  No held-out access or promotion. Next compare full updates on the same1134
  TRAIN and2000-step budget with original initialization and final-only DEV.

- [Frozen-B fit adequacy at2000 steps](nearfield/CITY_FIT_ADEQUACY_20260908.md):
  exact200-step weight parity passes; final-only DEV recall43.75%/31.25% at
  FPR4.6875%/6.25%, but joint1/32 and negative support activation expands.
  New TRAIN HEAD AUC stays0.616 to0.611 despite10x updates; joint0/96.
  No promotion or further budget extension. Next use a small balanced TRAIN
  fitting check with separate loss/gradient inspection before new mechanisms.

- [Additional relational TRAIN with frozen B](nearfield/CITY_RELATIONAL_TRAIN_20260908.md):
  old750 plus384 new frames across three same-map sites leaves DEV joint2/32;
  BODY/HEAD recall is37.50%/28.125% at FPR7.8125%/9.375%, with HEAD IoU0.00260.
  New TRAIN itself has weak separation (AUC0.644/0.616), HEAD IoU0 and joint0/96;
  this is not established high TRAIN fit followed only by regional failure.
  Retain data and diagnostics, no promotion or extra fit. Next establish bounded
  optimization/fit adequacy before adding generalization mechanisms.

- [Region-separated DEV and matched update-policy baseline](nearfield/CITY_DEV_BASELINE_20260908.md):
  128-frame same-map DEV is133.47m from TRAIN and has64positive/64negative per
  head. Fixed200-step A/B/C selection chooses frozen-backbone B, but DEV recall
  is only40.63%/25.00% at FPR9.38%/7.81%, joint2/32 and HEAD support IoU0.
  Its unchanged DEV thresholds yield zero HEAD recall on consumed plaza.
  Keep the selection workflow and G13-D; B is a weak diagnostic challenger,
  not promoted. Fresh TEST and B/C Willow regression remain unavailable.

- [Zero-training gate intervention and score separation](nearfield/CITY_GATE_INTERVENTION_20260908.md):
  original-gate replacement reduces plaza BODY FPR to 1.14% but recall collapses
  to 10.37%, HEAD to zero. No repair/promotion. Step200 BODY AUC nevertheless
  improves 0.6821 to 0.8290; separate fixed-point failure from score ordering.

- [Zero-training City FP diagnosis](nearfield/CITY_FP_ATTRIBUTION_20260908.md):
  step200 has only 2/2 BODY/HEAD FP on TRAIN but 561/348 on plaza; this is region
  transfer failure, not uniform City-wide activation. Existing negative support
  masks are zero, so GT mask subtraction adds no known positive target. No CF fit.

- [City-only G13 fine-tuning pilot](nearfield/CITY_FINETUNE_PILOT_20260908.md):
  one 200-step Development fit increased plaza BODY/HEAD false positives from
  3/4 to 561/348 out of 615/735 negatives. Keep the original baseline; the new
  weights are diagnostic only. The predeclared balanced-error/recall/Willow
  criterion passed but failed to constrain City false alarms. No rescue fit.

- [X21 result](X17_X21_TRACK_CARRIED_COMPONENT_ANCESTRY_2026-08-29.md), [X73 confirmation](carla/DTR_CARLA_C35_X73_FRESH_CONFIRMATION_20260901.md), [X94 result](carla/DTR_CARLA_X94_CONSUMED_ELEVEN_COHORT_DEVELOPMENT_20260901.md).
- [C35 raw-input pilot](DTR_BASELINE_RECKONING_C35_RAW_PILOT_20260905.md), [X95 result and simple controls](carla/DTR_CARLA_X95_CONSUMED_CROSS_VALIDATION_20260901.md).
- [Frozen comparison design](DTR_FINAL_RECKONING_ROSTER_R1_20260905.md), [latest source execution and crash](DTR_FINAL_SOURCE_EXECUTION_20260905.md); the later execution record owns the design's current execution status.
- [Completed RGB/depth and lossless-PNG probe](CARLA_RGBD_THROUGHPUT_20260905.md).
- [Failed Development composite and DX11 probe](CARLA_FAST_COMPOSITE_SOURCE_20260905.md).
- [Detailed ledger and reproduction](README.md), [formal research governance](../../../docs/formal/RESEARCH_GOVERNANCE.md).

The pre-scope-change current is preserved at Git
`fc70658e:research/active/dtr-r0/CURRENT.md`; linked reports retain the full history.


## BODY-QUERY-V1 matched comparison (2026-09-09)

[Completed result and reproduction](nearfield/BODY_QUERY_V1_RESULTS_20260909.md):320
fixed-calibration same-world frames; both primary A/B fits completed2000 steps.
Spatial-count B versus support-gated G13 A gives EVAL40 BODY TP4 to7 (FP0 both),
HEAD TP11 to8 with FP8 to2, and all-state groups0/8 to2/8. Both fit TRAIN perfectly.
This mixed result fails the predeclared HEAD-recall/evidence replacement criterion.
Retain the source, LOW/ABOVE/outside controls and count-query tooling as diagnostic
components; do not promote either matched fit or reopen this consumed budget.
This does not replace the separate City cross-region collection work above.
Offline all40-case viewer and RGB-only CLI are delivered with measured limits.


[Zero-fit query diagnostic](nearfield/BODY_QUERY_DIAGNOSTIC_20260909.md) clarifies
that B improves EVAL low-FP ranking (HEAD TP8 versus A6 at FP<=2), while DEV still
favors A15 versus B5. B query nonempty recall is97.05% TRAIN but26.88% DEV and
29.90% EVAL; all8 EVAL HEAD misses lack an active correct HEAD cell. Preserve
the partial gain, focus the question on evidence transfer, and do not attribute
the failure to positive-query underfitting or prioritize balanced CE without a
distinct test. No source, weights, cutoffs or training budgets changed.


[Q2 frozen point decomposition](nearfield/BODY_QUERY_Q2_RESULTS_20260909.md) finds
strong-point/mean-missed and all-weak cells together. EVAL top3 HEAD TP rises8 to15
but FP rises2 to17; all48 TRAIN BODY_ONLY frames become false HEAD alerts.
Retain the evidence-path diagnostic, not a pooling-failure attribution: strong
responses lack sufficient body-region selectivity. Weak readouts do not isolate
the backbone. One320-frame local GPU pass, zero training; no V2 or promotion.


[Q3 native attribution](nearfield/BODY_QUERY_Q3_RESULTS_20260909.md) verifies320
original count labels and full local HEAD coverage on DEV/EVAL, with weak owned
point response transfer. Retain native attribution tooling as a component.
[Attribution R1](nearfield/BODY_QUERY_ATTRIBUTION_R1_RESULTS_20260909.md) completed
one matched2000-step auxiliary-supervision fit without pooling changes. DEV HEAD
improves, but EVAL HEAD FP rises2 to17 and oracle TP at FP<=2 falls8 to6. Query
recall declines. Retain B; carry this R1 recipe as a negative control, with no
promotion, automatic weight sweep or budget continuation. Both runs are complete.


[Frozen B/R1 probes](nearfield/BODY_QUERY_PROBE_RESULTS_20260909.md) retain a narrow
R1 HEAD-near readability signal: EVAL ray AUC0.8079 to0.9314 and local0.8420 to
0.9377, but positive near points come from one EVAL counterfactual group. Overall
local R1 AUC declines and DEV point cutoffs do not transfer reliably. Retain B
and R1 negative-control status; keep probe/features as diagnostic components.
No main-model fitting, R2 promotion or native-count reinterpretation.


[Frozen direct readouts](nearfield/BODY_QUERY_DIRECT_RESULTS_20260909.md) complete
24 zero-fit combinations using fixed FOV masks only. DEV-selected ray-R1 top3
gives EVAL HEAD11TP/16FP; all tested direct HEAD oracle TP at FP<=2 are<=7 versus
B8. XYZ matched deltas are zero; visual matched responses do not yield a better
final tradeoff. Retain B; close these fixed pooling replacements as a negative
control, without k/fusion sweeps or promotion.


[ASE external geometry pilot](nearfield/ASE_BODY_QUERY_RESULTS_20260909.md) completed
72 fixed frames across3 synthetic layouts: BODY-only15, HEAD-only0, both47,
neither-detected10. Retain the verified ray-distance/fisheye geometry adapter
as a component, but this subset lacks the HEAD-only denominator needed for
model comparison. No model evaluation, training or automatic source expansion.
The1.70m camera-anchored proxy and native pixel counts are not actual-body or
V1 pinhole-equivalent truth; B and R1 dispositions remain unchanged.

[Native-depth routing](nearfield/BODY_QUERY_DEPTH_ORACLE_RESULTS_20260909.md) gives
no HEAD gain: fixed gates reach7/16 at EVAL FP<=2 versus B8/16. UNKNOWN causes
217/240 HEAD cells to pass through, so this conservative veto is not a general
depth-aware ceiling. Retain B and diagnostic arrays; neither promote a depth
model nor declare missing depth the cause or all depth methods disproven.
Seven fixed gates complete; no training or automatic monocular prior integration.


[BodyLift R0](nearfield/BODYLIFT_R0_RESULTS_20260909.md) completed matched2000-step
CONTROL/LIFT head fits on frozen B appearance. LIFT EVAL HEAD11TP/6FP versus
B and CONTROL8TP/2FP; at oracle FP<=2 LIFT retains6 versus8. Near-surface DEV/EVAL
depth CE is worse than uniform. Retain B; record this recipe as a negative
control, without broader depth-family rejection or automatic loss/bin sweeps.
Native export collapses nonfinite/out-of-range values to0; no FREE_RAY relabel.


[Expanded-source B](nearfield/BODY_QUERY_EXPANDED_B_RESULTS_20260909.md): Expanded TRAIN2500 with unchanged B and original initialization/2000-step budget: EVAL HEAD OLD 177TP/139FP, NEW 571TP/34FP; groups 1 to 228/300. Shared-asset Development; retain historical controls. Decision: RETAIN_EXPANDED_B_WORKING_BASELINE. One completed fit; no automatic extension. HEAD-near query recall remains79/894 despite strong final detection; correct depth attribution is not established.


[Range R0](nearfield/BODY_QUERY_RANGE_RESULTS_20260909.md): One matched2000-step range supervision fit: EVAL HEAD BASE571TP/34FP, R0557TP/35FP; HEAD-near query TP79 to1/894. Attribution criterion=False, alert preservation=False. Repeated-geometry Development only. Decision: RETAIN_EXPANDED_B_RANGE_R0_NEGATIVE_CONTROL.

## 2026-09-09: frozen B strict distance pairs complete

- [Protocol](nearfield/BODY_QUERY_DISTANCE_PAIRS_PROTOCOL_20260909.md) and [results](nearfield/BODY_QUERY_DISTANCE_PAIRS_RESULTS_20260909.md): 32/32 native-valid pairs, all 64 frames visually checked; frozen expanded B, original DEV thresholds, zero training.
- Far score increases in 25/32 pairs, both HEAD alerts in 29/32 (near 32/32, far 29/32). Seven reversals are large enough to make mean delta -0.057944 despite median +0.003414. big01 only 3/8 direction-correct; all three HEAD misses there. No HEAD-negative denominator.
- Some controlled within-object distance response exists, but near-frame far scores remain >0.87 and regional stability fails. Retain expanded B; range R0 remains negative. Pair source is a diagnostic component, not a distance-capability promotion. Fixed batch ends here, no automatic follow-on training or capture.

## 2026-09-09: 10000-source B comparison complete

- [Results](nearfield/BODY_QUERY_10000_RESULTS_20260909.md): the accepted source has 10000 frames, 2000 complete groups, 500 sites, unique RGB hashes, and TRAIN/DEV/EVAL roles 5000/2000/3000. Visual review covers 800 frames; background limitations remain explicitly retained.
- One fixed 2000-step NEW fit improves EVAL BODY AUC 0.98458 to 0.99712 and HEAD AUC 0.97484 to 0.98695; complete-group correctness is 439/600 to 508/600. HEAD-near query-cell recall falls 7.83% to 6.43%, so this is alert-scope evidence rather than solved query geometry.
- The separate 2500-pair distance diagnostic has far-higher direction 557/750 to 574/750 on EVAL, but HEAD-near nonempty-query recall falls 0.267% to 0.044% and BODY false alerts rise 54/1500 to 113/1500. Retain NEW only for measured controlled-Development alert scope; keep OLD as historical comparator. No distance, deployment, natural-scene or safety promotion and no automatic follow-up.

## 2026-09-09: 10k B frozen TRAIN diagnostic

- [Diagnostic](nearfield/BODY_QUERY_10000_FROZEN_DIAGNOSTIC_20260909.md): zero fits, unchanged checkpoint. Shared readout alert/count gradient cosine -0.9634 on 128 balanced HEAD_ONLY frames; most wrong far outputs already receive a correcting combined output gradient.
- Frozen cross-region retrieval within consumed TRAIN: raw HEAD-near857/1000, pre-readout827/1000, shuffled498/527. Retain readable range-associated features and local objective tension as component evidence; no unique pooling/backbone attribution, model replacement or automatic training.

## 2026-09-09: 10k B readout-only continuation closed

- [Results](nearfield/BODY_QUERY_10000_READOUT_RESULTS_20260909.md): trained10k B with only readout updated for2000 steps yields HEAD-near115 to105/1788, complete groups508 to509/600, HEAD TP1150 to1148 at35FP. Retain original10k B and mark this exact continuation NEGATIVE_CONTROL.
- Frozen parameter/buffer identity and initialization parity are verified. All21 sampled readout alert/count cosines remain negative. Invalid G13-start run-v1 is preserved separately; no further fit or unique pooling/backbone claim.
