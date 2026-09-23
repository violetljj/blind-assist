# Fixed nearfield stereo update adaptation

2026-09-23 EXPLORE. User authorizes a full-resolution backward feasibility check,
then one ordinary-versus-region-balanced supervised adaptation comparison if
feasible. Preserve VPP's historical camera-forward contact-distance gains and
the closed local-surface recipe. No new backbone or contact-classification head.

## Candidate and scope

Original23-51-11 ViT-Large, upstream6e8806816b533e4d13ddbb95ffa907b797060a62,
same checkpoint and original640x360 stereo padded384x640. VPP upstream fixed
ecf12d40a816286d541c96c46746f43401836cfa, rnd3x3 blend0.4, deterministic per-ID
seed, original public64idealray ToF only; no native-derived hints or confidence
deletion. ToF remains ideal point evidence, not a real8x8zonal sensor.

Freeze all model parameters except update_block,spx_2_gru,spx_gru. Keep modules
in eval mode to freeze stochastic behavior and running buffers; gradients are
enabled for selected parameters. Feature extraction,cost aggregation,initial
disparity and context remain fixed. This candidate can change refined disparity
but does not test adaptation of every upstream matching feature.

Engineering canary uses synthetic full-resolution RGB/plane target, at most two
optimizer steps. Check finite gradients, selected weight changes, frozen
parameter and buffer hashes, CUDA memory/timing. Discard canary weights. If
normal backward OOMs, one transparent activation-checkpoint implementation is
permitted, keeping resolution, model, training iterations and objective fixed.
Do not count a reduced-resolution substitute as the requested feasibility pass.

## New data and separation

72new geometry-layout groups,4poses each;48train/24heldout-development groups,
192/96frames. Six families:thin_near,small_head,ordinary_positive,near_outside,
far_background,empty. Eachfamily8train/4heldout layouts; appearance balanced.
Each layout's geometry/appearance and allposes stay in one split. No historical
named failure copy or random frame split. Scene generation is fixed before
candidate results. Actual engine cuboid bounds and full native axial depth are
retained, including background farther than4m.

Materialize separate TRAIN supervised examples (RGB,ToF,explicit native-depth
training targets), heldout public observations (RGB,ToF only), and heldout truth.
Training labels are declared supervised target authority, never sensor evidence.
Do not reclassify historical evaluator bundles or heldout truth for training.
Existing576pairs are historical regression only, never optimization/checkpoint
selection data. Heldout96frames/24layouts are Development transfer, not protected final
or hardware/general distribution confirmation.

## Two training arms, one fixed budget

Frozen original VPP baseline; ordinary adaptation; balanced adaptation. The two
trained arms start from identical weights and use identical sample order,192
updates (one pass),batch1,8refinement iterations,AdamW lr1e-5,weight_decay1e-4,
gradient norm cap1 and identical AMP/seed. No augmentation or checkpoint choice
from evaluator scores. Final checkpoint only. Evaluation uses32iterations as
the retained frontend. Training budget is a small feasibility pilot, not an
optimization ceiling; no extra epochs or unfreezing sweep after this result.

Target disparity=fB/native_visible_axialZ. Admit finite positive depth, target
disparity(0,416), projected right x>=0. No4m truncation. Occluded hidden-object
depth is not painted over a visible foreground surface. Initial disparity loss
is not an optimization objective for its frozen head; supervise8unpadded refined
outputs with normalizedgamma0.9 sequence L1.

Ordinary: uniform mean over valid supervised pixels. Balanced:0.5uniform plus
0.5equal mean of nonempty per-frame strata, with disjoint precedence:
(1)near Z<=4m pixels next to native3x3 depth jump>.05m or missing neighbor;
(2)remainingnear inside defaultcamera BODY/HEAD corridor;
(3)remainingnear outside corridor;(4)allfar Z>4m. Missing centers excluded.
Corridor uses samefixedcameraheightbands andfullwidth.56/.36m; no predicted-FP
mining, historical hard-ID weighting or heldout-derived region selection.
Save valid/stratum counts and losses,gradients,memory,time and frozen-state proof.

## Output and evaluation

Infer frozen baseline and both final adapters on new96public pairs. Infer both
adapters on historical576pairs; reuse exact oldVPP outputs as historicalbaseline.
Preserve raw disparity,rawdepth,filtereddepth and public hints. Same eligibility
0.5..4m/rightprojection, same directpoint geometry (closed surface module unused),
same publicToF union, defaultwidth/bodyheight and two-on/two-off.

Seal all predictions before loading heldout/historical evaluator truth. Report
camera-forward distance0.5..4m for BODYwidth.36/.56/.76 andHEAD.24/.36/.48;
criticalfullwidth<=1m within3m, positive-width separately. Exactactualcuboid LP
truth and independent native-visible reference retain hidden-obstacle misses.
Fixedtruth-contact denominator includes missing predictions as failure; report
conditionalerror,negativequery falsecontact,raw/finalalerts, wall/otherfamilies,
thin andHEAD, noframehint cases. Separately account for far-native surface
predictednear and true near-native support/5cm coverage; compare corrections
and regressions. Rawdisparity versus finalmask attribution remains explicit.

Keep only demonstrated heldout benefit beyond frozen baseline; attribute any
extra balancing benefit against ordinary adaptation. Regressions and historical
failures remain. If engineering fails, report not-evaluable and preserve why;
do not claim learned adaptation fails. If pilot fails, close this fixedbudget/
scope without concluding all stereo adaptation impossible. No threshold rescue.

All cohort access uses explicit research-ue RunSpecs and reviewed input scopes;
retain seeds,checkpoints,receipts,hashes and failures under canonicalartifacts.
Release owned processes promptly. No deployment,novelty or safety claim.
