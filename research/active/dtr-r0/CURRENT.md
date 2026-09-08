# Cane-complementary forward perception and DTR history

Updated: 2026-09-08

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).
Current work: cane-complementary, class-agnostic forward obstacle awareness.

Current engineering priority is the [City Sample collection field](../../../experiments/city-field/README.md):
complete loaded routes, representative supported obstacle conditions and clear
controls, synchronized geometry-labelled acquisition, and region/route/instance
split isolation. Model fitting is a separate workstream and is not a prerequisite
for accepting this collection infrastructure. Historical model diagnostics below
remain scoped evidence rather than the field's development objective.

The [field-v1 fixed-method diagnostic](../../../experiments/city-field/DIAGNOSTIC_V1_20260908.md)
now caches one frozen G10/G13 inference pass on all108 admitted frames. G10
misses30/45 reliable target-band opportunities with11/45 joint alert/support
hits; G13 misses45/45 with0 joint hits. True clear-control false alerts are3/7
and0/7 respectively. Excluding five floor-review frames changes no positive
or miss counts. Retain field/failure evidence; no model promotion, training,
threshold adjustment, or monocular-depth causal claim follows from this run.

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
