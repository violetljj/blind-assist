# Cane-complementary forward perception and DTR history

Updated: 2026-09-11

Status: `DTR_R2_DYNAMIC_RETAINED` (historical algorithm; no new promotion).
Current work: cane-complementary, class-agnostic forward obstacle awareness.

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
