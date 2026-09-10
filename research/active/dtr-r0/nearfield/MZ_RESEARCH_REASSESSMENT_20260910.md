# RGB + multizone ToF: evidence and literature reassessment

2026-09-10. Discussion and primary-source review after MZ13. No experiment,
training, capture, cutoff change or successor launch. MZ5 remains the controlled
baseline; existing negative terminals retain their original scope. This note
contains hypotheses and proposed decision logic, not new experimental results.

## What our evidence establishes

MZ5 equal-logit fusion improves complete correctness mainly through fewer false
activations, with a far-recall tradeoff. MZ6 showed that restoring clean packets
cannot fix a readout that misses the thin pole with those packets already present.
MZ9 actual-contributor supervision makes its local chain sensitive to wrong RGB
correspondence, while an equal-new-example adapted MZ5 also solves the trained
clip. Thus correct correspondence matters for that chain, but it is not yet a
unique algorithmic contribution or independent thin-pole generalization result.

MZ12 shows real but imperfect transfer: MZ11 adds57 far TP bits on existing3000
DEV frames, plus18 FP bits,17 on hanging signs. MZ13's expanded TRAIN coverage
reduces sign FP only17->16 and loses six previously recovered old-DEV near TPs.
It trained a20-parameter gate, not the complete fusion network. The visual
encoder and SOURCE readout stayed frozen. This does not show that existing data
cannot train a better spatial representation, or that RGB+ToF itself is futile.

The additional7500 TRAIN frames supply only109 positive and132 negative eligible
gate bits because training is restricted to MZ5-negative/SOURCE-positive outputs.
Dataset frame count, useful training opportunities, configuration variety and
independent confirmation are distinct quantities. There is no demonstrated need
for another large collection batch at present.

Sources: [MZ9](MZ9_SOURCE_SUPERVISION_RESULTS_20260910.md),
[MZ12](MZ12_EXISTING_DATA_RESULTS_20260910.md),
[MZ13](MZ13_TRAINING_COVERAGE_RESULTS_20260910.md).

## Where the current representation loses explanatory power

The compact MZ5 heads independently map RGB and ToF to four decisions, then
average them. SOURCE improves locality but still scores each angular/range
hypothesis with four independent outputs, then max-pools eligible hypotheses
for each query. A single spurious local maximum can dominate. Different queries
are not required to arise from one coherent scene interpretation. These are
structural facts, not proof that max pooling caused the sign errors.

The subsequent gate receives SOURCE margin, candidate count, zone count and
return count. It retains some geometric summaries but not candidate coordinates,
which local evidence won, the distribution of competing hypotheses, or relations
between surfaces. More examples cannot literally restore distinctions already
discarded by these summaries, although learned correlations may still help.
The current results show overlap, not a formal impossibility theorem for all
functions of the inputs.

The add-only rule preserves every MZ5 positive. Consequently it preserves every
MZ5 false positive too, including premature near warnings. This is a useful
retention control with a fixed performance ceiling, not a complete distance
attribution solution. An old positive judgment is not equivalent to an observed
physical fact. Future replacements should be evaluated for both correction and
retention rather than requiring all old predictions to be immutable forever.

Spatial resolution is another unresolved constraint. For the existing3.5cm pole
at approximately1.9m, a pinhole estimate using100-degree horizontal FoV gives
about1.98 pixels after resizing RGB to256 pixels wide. The32-wide dense feature
grid corresponds to about0.25 cells across that width. This does not prove that
features lose the pole (they can encode subcell cues), but sampling49 interpolated
positions per ToF zone does not create49 independent high-resolution observations.
We have not established whether local foreground/background separation survives
the frozen visual representation sufficiently for the attribution task.

## Primary sources and what they actually contribute

### DELTAR, ECCV2022

The [author project](https://zju3dv.github.io/deltar/) and
[paper](https://arxiv.org/pdf/2209.13362) explicitly distinguish regional depth
distributions from depth at known pixels. A distribution encoder and patchwise
cross-attention allow RGB locations to query depth hypotheses belonging to the
calibrated ToF zone; image attention propagates information beyond that zone.
This directly addresses the representation distinction relevant here.

The [author loader](https://github.com/zju3dv/deltar/blob/main/src/utils/dataloader.py)
uses mean/std statistics and Gaussian quantile sampling. Its simulated input
processing bins depth and retains the strongest contiguous cluster before fitting
statistics. Our first/last supported0.1m-bin means plus valid flags are a different
observation interface. Borrowing regional association is reasonable; inventing
a Gaussian width for our two returns or importing the model unchanged is not.

### CFPNet, 3DV2025

[Full paper](https://arxiv.org/html/2411.04480v4) and
[author code](https://github.com/denyingmxd/CFPNet) extend DELTAR-style regional
distribution fusion to RGB/ToF FoV mismatch. Direct cross-zone attention and
large-kernel propagation help image regions outside sensor coverage obtain
features from inside it. This is relevant to coverage gaps, but does not show
that in-zone thin-pole/sign return ownership is solved. Copying its propagation
modules before identifying whether the failure is in-zone would target a
different problem.

### NLSPN, ECCV2020

[Paper](https://arxiv.org/abs/2007.10042) and
[author implementation](https://github.com/zzangjinsun/NLSPN_ECCV20) use RGB and
sparse depth to predict nonlocal neighbors, affinities and confidence, reducing
mixing across depth boundaries. Relevant lesson: select compatible spatial
evidence rather than indiscriminately smoothing it. Its input is spatially
registered sparse depth; it does not locate an unassigned regional return for us.

### Prompt Depth Anything, CVPR2025

The [primary paper](https://openaccess.thecvf.com/content/CVPR2025/html/Lin_Prompting_Depth_Anything_for_4K_Resolution_Accurate_Metric_Depth_Estimation_CVPR_2025_paper.html)
conditions a depth foundation model on low-resolution LiDAR at multiple decoder
scales. It is useful evidence for combining visual shape with metric observations
and for considering a mature dense-geometry comparator. Its spatial depth-map
prompt differs from our unresolved angular return sets; performance on ARKitScenes
and ScanNet++ is not evidence of our BODY/HEAD obstacle performance.

### SelfToF,2025

The [paper](https://arxiv.org/html/2506.13444v1) uses regional depth consistency,
scale recovery and self-supervised image training; its sparse variant addresses
missing ToF zones. It offers supervision ideas if dense truth is scarce. We
already have native geometric supervision, so this does not justify introducing
photometric/pose assumptions or restarting temporal work at present. Its regional
mean interface also differs from our two-surface return set.

These papers mainly evaluate dense depth quality. None establishes our desired
far recall at a fixed false-alert budget. Whole-image depth error can hide thin
objects and body-region boundary errors; task metrics must remain explicit.

## Observation semantics must come before hardware-oriented claims

The [official ST UM3109 Rev12](https://www.st.com/resource/en/user_manual/um3109-a-guide-for-using-the-vl53l8cx-lowpower-highperformance-timeofflight-multizone-ranging-sensor-stmicroelectronics.pdf)
states that VL53L8CX defaults to one reported target per zone and strongest-first
ordering; up to four targets are configurable, with600mm stated minimum target
separation. Section5.1 defines range sigma as noise in the reported target
distance. It also exposes signal/ambient rates, target status and related data.
Range sigma must not be silently identified with geometric variation of all
surface depths inside a zone. Standard output containing these fields does not
by itself imply access to a full spatial/depth histogram.

Our simulation's clean first/last bin returns, exact valid bits and no reflectance
or multipath physics are an explicitly different interface. For now it is valid
to ask whether the algorithm uses those given observations correctly. Any later
real-sensor bridge must state which range/confidence/missingness measurements
are actually available and validate that mapping separately. No hardware purchase
or complete photon simulator is required by this reassessment.

## Revised working hypothesis and decision order

The most plausible next research object is a shared local spatial interpretation:
RGB retains foreground/boundary evidence, ToF supplies angularly bounded range
hypotheses, and the model represents supported surface locations or depth
distributions before applying the fixed BODY/HEAD near/far geometric queries.
This is a hypothesis, not an implemented method or a claim that dense depth is
necessary. A small task-relevant field may suffice. Multi-surface/extended objects
must remain representable; do not enforce one return equals one pixel or make
BODY and HEAD universally mutually exclusive. Preserve missing evidence.

Before choosing another architecture, answer three questions with the existing
data and frozen code: (1) Is a useful return present and geometrically compatible
with the true obstacle? (2) Do the frozen local RGB features distinguish its
location from the background? (3) Does correct local support survive query
aggregation? Existing native depth/ownership is diagnostic or supervision only,
never an unlabelled inference input. A few complete sign/pole/control case
traces can distinguish these failures better than another aggregate score.

Only after locating the failing stage should one choose between improving local
visual representation, return association, or query aggregation. Include a
credible spatial/depth baseline as a mechanism comparator, with its sensor input
semantics made explicit. Avoid another automatic gate/threshold/pooling variant.
Do not equate a failed frozen acceptance check with a universal prohibition on
all future versions, nor relax a completed experiment's criteria retrospectively.

No additional experiment is authorized or started by this discussion note.
