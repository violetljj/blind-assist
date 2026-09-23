2026-09-23 [contact error decomposition](../research/active/dtr-r0/nearfield/CONTACT_DECOMPOSITION_RESULTS_20260923.md):
No training: exact conditional position hits404/560train ->100/352held; binary
held also100/352. Only31/165blocked exact cases hide accurate median positions.
Width order alone does not solve localization. Diagnostic COMPONENT; old negatives,
A/LOCAL/UNKNOWN preserved. No q-unity or checkpoint promotion.

2026-09-23 [selective LOCAL rescue](../research/active/dtr-r0/nearfield/LOCAL_RESCUE_RESULTS_20260923.md):
One frozen ambiguity-fraction gate on576new controlled frames keeps41/41
incremental LOCAL TP and all event onsets, but removes4/15added FP (26.7%),
below50%. A118/7/106 versus LOCAL159/22/65 and gate159/18/65; events24/29/29
of32. Retain partial filtering and exact NEGATIVE_CONTROL; no cutoff retry,
App change or safety claim.12tests and independent15629assertion auditPASS;
one capture complete and all owned processes released. A/LOCAL/UNKNOWN retained.

2026-09-23 [exact contact supervision](../research/active/dtr-r0/nearfield/EXACT_CONTACT_RESULTS_20260923.md):
Same CDF plus exact train labels: selected held Z5cm117/352 ->27/352,
recall72.45 ->49.53%,missing94 ->165. Train trajectory can fit Z precisely,
but selected transfer fails; no proof metric selection fixes generalization.
Exact package NEGATIVE_CONTROL; no posthoc checkpoint promotion or App change.

2026-09-23 [direct metric contact result](../research/active/dtr-r0/nearfield/METRIC_CONTACT_RESULTS_20260923.md):
One same-binary-label CDF fit: held Z5cm168/352 ->117/352,width134/528 ->57/528;
false Z crossings146 ->111 but missing47 ->94,recall78.49 ->72.45%. Joint
upgrade NEGATIVE_CONTROL, preserving sampling COMPONENT and A/LOCAL/UNKNOWN.
No exact-distance labels, fresh confirmation, retry or App change.

2026-09-23 [dense boundary supervision](../research/active/dtr-r0/nearfield/BOUNDARY_SUPERVISION_RESULTS_20260923.md):
Same representation/model and equal query budget improve held width5cm8.14to36.36%
and horizon17.33to34.66%. FPR9.479to6.013%,but recall86.88to81.44% exceeds the
3point loss budget. Retain boundary gains and all costs; exact joint-upgrade
NEGATIVE_CONTROL. One fit and independent audit complete; no automatic retry.

2026-09-23 [boundary-aware contact sampling](../research/active/dtr-r0/nearfield/CONTACT_SAMPLING_RESULTS_20260923.md):
Fixed72query Development: geometry evaluation width within5cm43/528 ->134/528,
horizon61/352 ->168/352; train horizon175/560 ->467/560. Recall86.88 ->78.49%,
FPR9.48 ->5.16%, horizon falsecross119/800 ->146/800. Retain sampling COMPONENT;
neither full package passes.13tests/777024world-label auditPASS. Old controls,
A/LOCAL/UNKNOWN preserved; two fits complete, no automatic successor.

2026-09-23 [boundary-error source audit](../research/active/dtr-r0/nearfield/BOUNDARY_ERROR_RESULTS_20260923.md):
Coarse width24-36cm/horizon30cm labels coexist with geometry held-layout bracket
violations292/485width errors and156/291horizon errors. Returned-near native support
463/528width and277/352horizon does not establish precise public localization.
Independent13824-row/48native-frame auditPASS; diagnostic COMPONENT. No refit,
cutoff change or App promotion; old contact negatives and A/LOCAL/UNKNOWN remain.

2026-09-23 [counterfactual contact-boundary pilot](../research/active/dtr-r0/nearfield/CONTACT_BOUNDARY_RESULTS_20260923.md):
Consumed1728-frame source, frozen shared visual features and new width/horizon queries.
On576layout-held images, direct3346TP1162FP1136FN versus geometry3894TP2196FP588FN:
recall74.65/86.88%,FPR5.016/9.479%. Both rank lateral pairs>99%, but width within5cm
is37/528 versus43/528 and horizon19/352 versus61/352. Neither contact package passes;
retain exact NEGATIVE_CONTROL and partial boundary/ranking evidence.22tests and
independent714816-label auditPASS. No capture, retry or App change; A/LOCAL/UNKNOWN retained.

2026-09-22 [tiny-training spatial-query fitting check](../research/active/dtr-r0/nearfield/QUERY_SPATIAL_FIT_RESULTS_20260922.md):
84 consumed train images,504queries. Original global FiLM179TP1FP1FN versus
spatial180TP0FP0FN; both540/540same-imagequerypairs correct. Both learn
image-dependent query answers; old fixed query ordering is not structurally
unavoidable. Macro maskIoU0.484823/0.481782 fails fixed0.50gate, with no
localization gain from the added branch. Exact package NEGATIVE_CONTROL;
preserve query-fitting evidence and old negatives. New shared optimization
and tiny cohort prevent historical-cause attribution. No dev/held access,
threshold selection, video or App change. A/UNKNOWN remain unchanged.

2026-09-22 [frozen-checkpoint train/dev diagnosis](../research/active/dtr-r0/nearfield/QUERY_OCCUPANCY_DIAGNOSTIC_RESULTS_20260922.md):
Both learned arms rank all six queries identically in every864train/288dev image:
HEAD-left > centre > right > BODY-left > centre > right. Occupancy maskIoU0
already on train, foregroundAUROC0.544train/0.576dev; conditional occupied-bin
accuracy85.64%/70.45%does not repair occupancy or spatial query use. Retain this
diagnostic COMPONENT; original recipe remains NEGATIVE_CONTROL. No refit,
cutoff change, held-label access or held-row inference. A/UNKNOWN unchanged.

2026-09-22 [single-frame query occupancy result](../research/active/dtr-r0/nearfield/QUERY_OCCUPANCY_RESULTS_20260922.md):
New1728-frame grouped Development,576held frames. Matched classifier62/34/194
TP/FP/FN becomes occupancy88/44/168; events16/32 to29/32, but false segments17
to29 and addedFP10 exceed the declared costs. All44occupancyFP are OUTSIDE.
Visible maskIoU0 across880positive queries; only13positive distance bins correct.
A current193/6/63 with32/32events remains stronger. Preserve this exact recipe
as NEGATIVE_CONTROL; retain A/UNKNOWN.37tests and independent geometry, output,
component and decision audits pass. Explicit inheritance and archived ledger
registration complete. Same-generator simulation only; no video or App change.

2026-09-21 [saved B/N disagreement diagnostic](../research/active/dtr-r0/nearfield/BRANCH_DISAGREEMENT_RESULTS_20260921.md):
Consumed1152-frame posthoc partition, no refit/cutoff change. Boundary current
common/B-only/N-only add41/3,49/4,45/5 TP/FP; each exceeds added-FP cap2.
All8global memoryless binary retention masks reported; only A alone passes all
costs. AND retains useful Core gain: hold159/25/17 to172/26/4, events15/16 to16/16,
with Core costs passing. Boundary AND62/10/114 fails added FP5/cap4 and segments3/cap2;
current FP3/cap2 already fails. No whole-fusion impossibility claim or mask selected.
Core is not solved on wider geometries; retain N Core/HEAD gains and B BODY coverage.
No new verifier/expert training: within-pattern separation needs independently
justified public-input evidence and different validation layouts. A/UNKNOWN and
all prior dispositions unchanged. Diagnostic COMPONENT;4tests and independent
counts/event/cost checks pass; local inheritance and ledger303 receipts preserved.

2026-09-21 [fixed-B broader-data result](../research/active/dtr-r0/nearfield/DATA_COVERAGE_RESULTS_20260921.md):
One new3456-frame source,1728-frame fit and isolated1152-frame same-generator
Development evaluation completed. N versus same-dev-calibrated old B preserves
aggregate Boundary hold106/19/70, but reduces Core FP66 to34 and increases Core
TP172 to175. HEAD-horizontal12/1/34 becomes43/7/3; BODY-plane38/9/6 becomes9/3/35.
These are real partial gains and coverage losses, not lossless B rescue retention.
Boundary added FP/segments14/10 exceed caps4/2; Core segments also exceed budget.
NO_GO / exact data-condition NEGATIVE_CONTROL; no further fit/cutoff/loss retry.
Old consumed regression N87/20/86 versus calibrated B145/20/28 confirms a coverage
cost, not fresh confirmation. Twelve tests and independent64-frame public replay
pass. A/UNKNOWN and legacy B/R and paused POINT/REGION dispositions stay fixed.
All task-owned processes released; local inheritance and ledger303 receipts saved.

2026-09-21 [dense-depth alert branch paused](../research/active/dtr-r0/nearfield/REGIONAL_DEPTH_RESULTS_20260921.md):
POINT/REGION completed a negative algorithm test, unlike the unevaluable oracle.
Their sole matched change was center versus regional training consistency loss;
no usable gain under this network/supervision/top16 readout. Both remain exact
NEGATIVE_CONTROL. Pause dense recovery then alert aggregation; no extra diagnostic,
loss/top-k/aggregation/longer-fit retry. Core retention is structural A OR retention.
[Prior R](../research/active/dtr-r0/nearfield/CORRIDOR_RELATIVE_RESULTS_20260921.md)
retains useful partial evidence: Boundary hold14/5/159 to78/6/95,4/16 to11/16
events,9/16layouts gain, added Boundary/CoreFP1/5 within all cost caps.
Its45.09% recall, HEAD-horizontal0/43, five missed events and max detected delay
1.8s retain nonpromotion; no useful-rescue evidence is erased. Saved working
points are not equal-realized-FP comparisons. A/UNKNOWN and B/R/U/G unchanged.
Later research prioritizes task alerts via genuinely new spatial cues or learning
coverage across objects/layouts/backgrounds, not mandatory dense depth recovery.
The1152 consumed frames are not fresh confirmation. No experiment starts here;
existing local dispositions and ledger303/unknown-terminal receipts remain.

2026-09-21 [ordered spatial geometry contrast](../research/active/dtr-r0/nearfield/SPATIAL_STRUCTURE_RESULTS_20260921.md):
One matched U/G comparison preserves ordered frozen RGB, with only eight
interval-relative geometry channels added in G. Horizontal decisions are
identical: 15/43 TP, 2/4 events. Boundary hold U66/11/107 vs G68/9/105;
Core173/44/0 vs173/45/0. Both exceed transfer cost caps and fail coverage.
Geometry NOT_SUPPORTED; both exact recipes NEGATIVE_CONTROL. A/B/R retained.
No fit/cutoff retry, protected test, demo change or automatic successor.
Local structured evidence and global ledger303/unknown-terminal receipts retained.

2026-09-21 [corridor-relative representation](../research/active/dtr-r0/nearfield/CORRIDOR_RELATIVE_RESULTS_20260921.md):
One fixed 60-feature interval-band/shared-max MLP gains Boundary hold14/5/159
to78/6/95 on consumed1152-frame transfer; events4/16 to11/16, gain9/16layouts.
Core173/33/0 becomes173/38/0. Added-cost caps pass, but45.09% Boundary recall
misses frozen50% floor; all four HEAD-horizontal layouts remain wholly missed.
NO_GO / exact-recipe NEGATIVE_CONTROL, preserving measured low-cost gains.
No threshold/fit retry, default promotion, protected test or automatic successor.
Independent audit and focused tests PASS; global ledger303 gap retains receipts.

2026-09-21 [Core projection sensitivity](../research/active/dtr-r0/nearfield/CORE_PROJECTION_STRESS_RESULTS_20260921.md):
One frozen consumed1152-frame check shifts reported boxes by +/-2 lattice
pixels with ranges unchanged. Core173/33/0 becomes172/36/1 or172/64/1;
16/16events remain, one.2s HEAD onset delay per sign. OUTSIDE FP0 becomes11/34.
Both conditions pass inherited usefulness gates versus matching Calibration,
but right-offset false-alert duration rises6.6 to12.8s versus nominal Core.
Retain diagnostic COMPONENT, not calibration robustness or hardware evidence.
A/demo, thresholds and protected test unchanged; no automatic successor.
Global ledger303/unknown-terminal gap has local disposition and receipts.

2026-09-21 [fixed-entry conditional hold](../research/active/dtr-r0/nearfield/CONDITIONAL_HOLD_RESULTS_20260921.md):
Original score only, entry15.769264 frozen; selection-only keep15.535847 adds
one internal Boundary frame but evaluation adds zero TP/FP. Core82/11/2 and
Boundary16/0/68 exactly match original_current. Of68 Boundary misses,51 have
no clip trigger,1 precedes the trigger,16 are reachable by continuation. Internal
positive11.615569 versus exit negative14.757340 shows conflicting keep limits;
Core exit15.453609 tightens the shared threshold. NEGATIVE_CONTROL for this
specific fixed-entry/scalar-keep recipe; no threshold retry, new head or successor.
A/demo unchanged. Four tests and independent audit PASS; global ledger303/unknown-terminal receipts retained.

2026-09-21 [multi-zone RGB anchoring](../research/active/dtr-r0/nearfield/RGB_MULTIZONE_RESULTS_20260921.md):
Fourteen consumed diagnostic frames (seven negative/Boundary pairs, three layouts)
yield zero explained OUTSIDE negatives. Only two selection positives have the
required adjacent whole-zone anchors; their three CROSSING queries retain
307/307 observed-return contributor incidences. Eight frames have no whole-zone
anchor, four have one, two have two. Close this exact recipe as NEGATIVE_CONTROL;
UNKNOWN is unresolved, not an RGB information ceiling. Keep A/demo unchanged;
no weaker-anchor retry, training or automatic successor. Independent audit PASS;
global ledger303/unknown-terminal receipts and local disposition retained.

2026-09-21 [public ToF axis evidence](../research/active/dtr-r0/nearfield/AXIS_EVIDENCE_RESULTS_20260921.md):
All16 consumed transfer layouts/1152frames: contained-depth evidence exists on
273/346positives,0/633distance negatives,137/173depth-valid lateral negatives.
It excludes15/15high-score distance negatives but loses4/82native-backed
balanced-suppressed positives, and is absent at both endpoints of all32positive
clips. Distance information exists; hard qualification has boundary costs and
does not settle lateral ownership. Retain diagnostic COMPONENT, no new alerts,
gate, training or test. Independent audit PASS; global ledger303/unknown-terminal
receipts retained.

2026-09-21 [current-only fixed-score audit](../research/active/dtr-r0/nearfield/CURRENT_ONLY_RESULTS_20260921.md):
Removing hold from selection releases Boundary rescues, but gains remain in1/8
layouts. Evaluation current Core stays82/11/2; Boundary A4/0/80 becomes16/0/68
with original,12/0/72 uniform,9/0/75 balanced. Selection gains4frames in1/8layouts.
New readouts underperform original at matched no-added-current-FP cutoffs;
all fail the fixed cross-layout diagnostic. Retain A/demo and local rescue
evidence; stop without conditional hold, fitting or protected-test activation.
Independent audit PASS; local NEGATIVE_CONTROL; global metadata pending
(ledger303/unknown terminal).

2026-09-21 [frozen32D last-layer pilot](../research/active/dtr-r0/nearfield/LAST_LAYER_RESULTS_20260921.md):
Two33parameter heads retain0/68 Boundary rescues on eight consumed evaluation
layouts; both equal A (Core82/11/2, Boundary4/0/80). xAUC .8975 becomes .9133,
but the no-added-FP cutoff is bound by a true pre-exit frame under unchanged
hold. This rejects the combined recipe, not current-only linear separability.
Independent audit PASS; local NEGATIVE_CONTROL, global metadata pending; stop.
[Prior paired ordering](../research/active/dtr-r0/nearfield/LATERAL_PAIR_DIAGNOSTIC_RESULTS_20260921.md) remains173/173; original test closed.

2026-09-21 [existing-data corridor feasibility](../research/active/dtr-r0/nearfield/EXISTING_CORRIDOR_DATA_RESULTS_20260921.md):
All15000 BODY-query metadata geometries verified; all indexed payload paths exist.
750 lateral candidates across750 sites are crossbar-only,12 original fixture
groups,1.636-1.794m outside versus recent6.6-11.9cm false-positive clearances.
Only3 of10 sampled pairs have both targets visibly supported; all20payload
hash checks pass, but no full-scene negative is certified. Retain input/OBB
adapter COMPONENT; direct hard-negative training unsupported. Seven geometry
tests and independent recount pass. No training/capture/protected-test access.

2026-09-21 [frozen supplement new-layout verification](../research/active/dtr-r0/nearfield/SPATIAL_COMPLEMENT_TRANSFER_RESULTS_20260921.md):
16 new groups/1152 frames, unchanged model/high cutoff/hold. Current Boundary
10/1/163 becomes128/10/45, gains in15/16 groups; Core171/17/2 becomes173/36/0,
including17 additional OUTSIDE FP. Held Core gains no TP and adds25 FP; Boundary
adds126 TP and14 FP. Rescue signal transfers, low-FP specificity does not.
Retain COMPONENT only; independent audit PASS, original test unactivated, stop.
[Existing-data inventory](../research/active/dtr-r0/nearfield/SPATIAL_DATA_REUSE_INVENTORY_20260921.md)
identifies4752 matched prior frames and15000 BODY-query adaptation candidates;
reuse before routine expansion, with reserved test and label-contract limits.

2026-09-21 [frozen A/B complementarity diagnostic](../research/active/dtr-r0/nearfield/SPATIAL_COMPLEMENT_RESULTS_20260921.md):
Same consumed eight dev groups/576 frames, no new fit or test access. A OR
previously disclosed high B scores changes current Core85/4/1 to86/4/0 and
Boundary5/3/81 to71/3/15. Whole-group cutoff diagnosis retains these rescues but
adds one Core FP; unchanged hold adds three Boundary exit FP. Calibration screen
is redundant. Retain COMPONENT evidence, not a complete zero-added-FP upgrade;
original full-replacement negative remains scoped. Independent audit PASS;
stop without tuning, release changes or automatic successor.

2026-09-21 [one spatial RGB-ToF BCE pilot](../research/active/dtr-r0/nearfield/SPATIAL_BCE_RESULTS_20260920.md):
40 new groups/2880 frames, one fit, no ranking. Prespecified dev zero-logit
Boundary current5/3/81 becomes78/3/8, but Core85/4/1 becomes83/15/3 and two
onsets are delayed.2s. No admissible threshold among577; current-only gates fail
too. Stop before test inference; retain Calibration/strong+hold and close this
fixed full-alert replacement as NEGATIVE_CONTROL with Boundary signal retained
as diagnostic evidence. Independent audit PASS; no retuning/demo/successor.

2026-09-20 [frozen Core policy new-layout validation](../research/active/dtr-r0/nearfield/CORE_HOLD_VALIDATION_RESULTS_20260920.md):
One new36-layout/432-frame cohort supports the simple Core candidate:
Calibration74/133/0, strong72/10/2, strong+hold74/10/0; FP26 to8 segments,
26.6 to2.0s. All12 Core events have full sampled coverage and entry-frame alerts,
although8 alerts were already active before entry. Boundary remains9/4/65 and
4/12 events. Retain a scoped controlled-demo CHALLENGER; full-task Calibration,
old onset failures and challenge limits remain. End this algorithm round.

2026-09-20 [single half-bin phase diagnostic](../research/active/dtr-r0/nearfield/HALF_BIN_PHASE_RESULTS_20260920.md):
One frozen 5 cm histogram shift changes some proxy returns but neither delayed
HEAD onset. Core strong73/8/5 becomes72/8/6; hold76/9/2 becomes75/9/3;
Calibration77/128/1 becomes77/129/1. All12 events and onset times persist.
Retain original proxy/Calibration/hold and scoped sensitivity evidence; no phase
replacement, threshold search, contract change or automatic successor.

2026-09-20 [delayed-onset64-zone evidence audit](../research/active/dtr-r0/nearfield/ONSET_ZONE_AUDIT_RESULTS_20260920.md):
The two delayed HEAD frames differ: one loses actual corridor contributors during
return selection; the other retains a target return but has low interval depth
fraction. Neither shows large dispersed positive mass hidden by max. Full210 Core
negative comparisons overlap the fixed descriptors; no full64 information ceiling
or repair is established. Retain Calibration/hold and bounded diagnostic evidence.

2026-09-20 [fixed causal event readout](../research/active/dtr-r0/nearfield/CAUSAL_EVENT_READOUT_RESULTS_20260920.md):
Two consumed432 replays separate temporal effects. Primary Core strong73/8/5,
+rise73/17/5, +hold76/9/2, combined76/18/2. Hold improves local continuity,
but neither delayed HEAD onset is restored; rise adds9 CoreFP without CoreTP gain.
Keep nonrecursive hold as a scoped continuity COMPONENT; close rising onset rescue
and the combined onset-recovery role as NEGATIVE_CONTROL. First-alert evidence
remains unresolved; neither smooth-entry dynamics nor spatial sufficiency is proven.
Retain Calibration. No new experiment, tuning or automatic successor.

2026-09-20 [Core operating-point diagnostic and one fixed transfer](../research/active/dtr-r0/nearfield/CORE_WORKPOINT_TRANSFER_RESULTS_20260920.md):
The unchanged score has substantial consumed-data headroom (Core72/126/0 to71/10/1;
all12 onsets retained), and0/126 old FP use the definite bypass. One frozen
threshold on36 new layouts gives77/128/1 to73/8/5: FP falls93.75%, but two HEAD
onsets delay.2s and one event has4/7 coverage. Close this candidate as a negative
control; retain Calibration and the positive scoped diagnostic. No retuning,
RGB reopening or successor. Severe boundary costs and metadata gaps are retained.

2026-09-20 [existing-result stratified review](../research/active/dtr-r0/nearfield/EXISTING_RESULTS_STRATIFIED_REVIEW_20260920.md):
Core usefulness governs future comparisons under the [task contract](../research/active/dtr-r0/nearfield/CAMERA_FORWARD_CONTRACT_20260919.md#useful-core-capability-and-challenge-coverage).
Old four-sensor A* has strong BODY/HEAD performance; this does not transfer its
numbers to the current two-sensor task. Calibration stays the current baseline;
its complete Core layouts retain high false-alert burden. Preserve useful
components, separate challenge costs, and scope old lossless gates to their tests.
No new method or experiment is started; frozen dispositions remain unchanged.

2026-09-19 user decision: adopt [the camera-forward corridor contract](../research/active/dtr-r0/nearfield/CAMERA_FORWARD_CONTRACT_20260919.md). Immediate scope is RGB + 8x8 ToF with a fixed camera-frame volume; body-heading compensation, future trajectories and SLAM are excluded. This supersedes the older four-sensor requirement for this task; frozen results and defaults remain unchanged.

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

2026-09-18 [standalone A* effect delivery](../research/active/dtr-r0/nearfield/corridor_fusion_v1/ASTAR_EFFECT_USAGE_20260917.md):
use one retained A* for the host effect version. Its canonical `alert` drives
saved predictions, reminders and display; the old comparison entry is preserved.
All288 scores/events match frozen A*: clear96/3/12,F192.75%,core18/18;
strict107/6/37,24/30. Complete48-episode A/A* replay retains four lost clear TP,
three rescued FN and the.5s rod delay. Direct mean66.67ms,p5066.04,p9574.64
is a measured host algorithm path, not a demonstrated speedup or phone latency.
Keep public positive as a research component and OR as a boundary tradeoff.
No new fit, capture, threshold, default-App promotion or accuracy confirmation.

2026-09-17 [single-version frozen confirmation](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_SINGLE_RESULTS_20260917.md):
new clear A/OR97/25/11, no supported clear A miss; retain the public component
without claiming confirmed clear rescue. OR adds1 boundary TP/1FP. Same-data
A*96/3/12,F192.75% is a useful precision challenger:22 fewer clear FP,3 rescued
FN,4 lost TP; core18/18 but one rod delayed.5s and strict events24/30 versus25/30.
Preserve v2 Development, frozen A reference and all costs. No automatic App
replacement, A*+head combination or threshold repair; one capture completed.
Runtime/bundle delivered, audits pass, resources released; registration pending.

2026-09-17 [public positive v2](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_V2_RESULTS_20260917.md):
prefer ordinary return BCE plus pooled scene-OOF calibration as this round's
Development challenger: changed clear103/27/5, no added FP across four cohorts,
all A alerts/onsets retained. The matched peak loss adds1 old TP at3 boundary FP.
Keep both controls and earlier null result; the repair opportunities remain only
three scene groups, with no fresh or final deployment claim. Ledger303 pending.

2026-09-17 [public positive readout](../research/active/dtr-r0/nearfield/corridor_fusion_v1/PUBLIC_POSITIVE_RESULTS_20260917.md):
the trained public-only MLP demonstrates return discrimination, but grouped
calibration yields no added final alert. Preserve that primary null result.
Posthoc fixed logit0 gives changed clear 103/33/5 with 8 rescued FN and 6 added
FP, plus boundary degradation. Retain the reusable component/diagnostic evidence;
calibration and false activation remain open, without automatic successor work.

2026-09-17 [tri-state existing-evidence probe](../research/active/dtr-r0/nearfield/corridor_fusion_v1/TRISTATE_EVIDENCE_RESULTS_20260917.md):
retain privileged positive-readout headroom (clear 103/27/5) and the measured
joint tradeoff (102/17/6, F1 89.87%). All UNKNOWN frames retain A; outside-only
veto still loses one rod TP with 0.75 s later first alert. Prioritize the evidence
readout question over unsampled extent, while keeping negative coverage authority
unproven. No public-input method promotion or automatic training/capture.

2026-09-17 [surface-support oracle](../research/active/dtr-r0/nearfield/corridor_fusion_v1/SURFACE_ORACLE_RESULTS_20260917.md):
fixed A support-field replacement yields no alert gain. Native audit finds
8clear FN already have returned corridor points,5rod FN lack corresponding
returns;complete extent adds0clear and4boundary opportunities. Preserve this
scoped negative and A's separate input/readout limitation. No automatic RSSF,
CNH,new model or capture. Existing-return absence is not clear-space evidence.

2026-09-17 task-definition update: [stored native-geometry re-evaluation](../research/active/dtr-r0/nearfield/corridor_fusion_v1/TOLERANCE_RESULTS_20260917.md)
keeps strict results and separates clear corridor decisions with coverage,
boundary pressure and observed alert behaviour.5cm covers216/288frames per
domain; A90.30%old versus82.61%changed F1, with40clear errors beyond10cm.
No definition change rescues S1 transfer. Defer untrained intrusion work and
prioritize reusable datasets; current capture cancelled/released on user redirect.
No automatic training, CNH, new capture or App promotion.

2026-09-17 user decision: adopt [A as the balanced research baseline](../research/active/dtr-r0/nearfield/corridor_fusion_v1/BALANCED_BASELINE_20260917.md),
with its disclosed recall/native-support tradeoff. The [conditional depth specialist](../research/active/dtr-r0/nearfield/corridor_fusion_v1/SPECIALIST_RESULTS_20260917.md)
is a Development challenger:137/28/7to137/24/7 on consumed MZ170, unchanged30/30
events/timing and3.47%actual invocation. No default-App change or fresh claim;
the owning route records frozen identities, costs and next confirmation scope.
The [completed targeted confirmation](../research/active/dtr-r0/nearfield/corridor_fusion_v1/CONFIRMATION_FINAL_20260917.md)
finds no transferred benefit: A/S1 both120/45/24,29/30events;14C calls keep
8TP and6FP. Preserve prior Development gain, close this frozen online DA-V2
recipe without tuning, and keep A with disclosed transfer limits. The original
engineering cutoff was corrected without changing models or scenes. No CNH run.

Latest,2026-09-16: [MZ146](../research/active/dtr-r0/nearfield/MZ146_RESULTS_20260916.md)
confirms fresh aggregate improvement from the unchanged MZ143/MZ145 pipeline:
130TP/86FP/14FN to143/57/1 on288 same-generator frames, all incumbent true
frames/events retained, false segments28to21. Rod FP25to29 violates the frozen
family condition, so joint confirmation fails. Retain the scoped aggregate gain
and [MZ145's Development result](../research/active/dtr-r0/nearfield/MZ145_RESULTS_20260916.md),
close this exact confirmation without tuning, and keep MZ129/default. This is
controlled simulation evidence; the per-return native Radar audit is unavailable.

**Active architecture, corrected 2026-09-13: one RGB camera + ToF + Radar + IMU,
simulation only.** Current work tests direct walking-corridor classification
from fixed cheap sensors, without requiring unique return attribution first.
The [MZ136 paired-training comparison](../research/active/dtr-r0/nearfield/MZ136_RESULTS_20260914.md)
finds no joint alert gain; retain MZ129 and the distinction between pairing,
training fit and a usable common score threshold. Historical association work
compares matched ToF+Radar+IMU against the
same pipeline with RGB spatial association. Preserve independently valid sensor
support and UNKNOWN. RGB is not a second camera, stereo depth, or a substitute
for Radar. IMU rotation is not metric translation or future walking intention.
See the [four-sensor mainline and stop points](../research/active/dtr-r0/nearfield/FOUR_SENSOR_MAINLINE.md).

The user-authorized [TRAIN-only fit repair](../research/active/dtr-r0/nearfield/MZ136_TRAIN_FIT_REPAIR_20260914.md)
passes the original fitting criterion:191/192 frames and95/96 pairs correct,
with a standardized385-parameter readout over frozen features. Retain this as a
Development fitting diagnostic: its subsequent frozen dev48 transfer needs20FP
versus MZ12913 at matched recall/timing. Scene-dependent residual offsets remain
unresolved. No new capture or original-test scoring; MZ129 remains retained.
The [grouped readout follow-up](../research/active/dtr-r0/nearfield/MZ136_GROUPED_READOUT_20260914.md)
improves readout-only cross-group stability but still needs22FP versus MZ12913
at matched dev recall/timing. Do not promote this fixed-feature linear correction.
The [boundary-input audit](../research/active/dtr-r0/nearfield/MZ136_BOUNDARY_INPUTS_20260915.md)
reduces conditional TRAIN edge MAE2.886 to1.227px on the same38 frames; fine-edge
coverage is38/48. Keep it as a geometry component, with no alert promotion.

The [MZ137 end-to-end contrast](../research/active/dtr-r0/nearfield/MZ137_EDGE_CORRIDOR_20260915.md)
then tests a shared public-range plane with coarse/fine edges on consumed dev48.
All arms remain24TP/13FP/0FN:13 geometry changes reach one corrected plane
crossing but no final alert change, with native support loss. Reject this exact
integration; retain MZ129 and the MZ136 edge component. No further tuning/test run.

The [MZ138 support ceiling](../research/active/dtr-r0/nearfield/MZ138_SUPPORT_CEILING_20260915.md)
finds evaluator-only full native-face24TP/2FP/0FN with unchanged event times on
consumed dev48. Sample-only/angular supports lose unsampled corridor extent;
ownership grouping alone adds no gain. Retain surface-extent headroom as a
diagnostic, with explicit full-surface scope correction; keep MZ129, no new algorithm.

[MZ139 joint regional-ToF/RGB finite-surface fitting](../research/active/dtr-r0/nearfield/MZ139_SURFACE_FIT_20260915.md)
is now implemented and frozen after TRAIN12. Consumed dev48 remains24/13/0,
5/5events unchanged:10/48accepted surfaces replace30returns but change0alerts.
One of3unsampled intrusions has estimated continuous support; two stay ambiguous.
Native exclusions include12corridor samples under a wrongly nonalerting accepted
surface. Retain MZ129; this exact estimator/readout is a negative control,
not a refutation of observable surface recovery. Metadata remains pending303.

[MZ140 DEPTHOR Small](../research/active/dtr-r0/nearfield/MZ140_DEPTHOR_20260915.md)
uses the full RGB image and frozen pretrained weights. TRAIN48 shows useful
BODY/HEAD extent gains versus box-only RGB, but zero rod/shallow target-depth
agreement and meter-scale errors despite near ToF cues. The frozen transfer
stops before dev; no alert gain was tested. Keep MZ129, pause MZ139's specific
box/single-slab/search recipe, retain MZ140 as scoped negative control with
partial visual benefit. No model/threshold rescue; metadata pending303.

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
