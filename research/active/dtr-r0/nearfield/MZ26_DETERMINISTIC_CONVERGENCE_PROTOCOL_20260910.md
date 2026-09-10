# MZ26: two fixed budget endpoints within one deterministic trajectory

2026-09-10 EXPLORE, consumed Development. MZ24 left29.43% negative-availability
TRAIN bags active and4 placement addedFP. MZ25's separate-run1200-prefix failed;
that failed run remains closed and provides no4800-step task result. Its fixed
bilinear sampler passed strict repeated-gradient and selected real-input parity.

Run one fresh4800-step trajectory using StableAngularAvailability and strict
deterministic algorithms, CUBLAS_WORKSPACE_CONFIG=:4096:8, recorded Torch/CUDA/
cuDNN/device/TF32 settings. Preserve the10577 learned parameters, seed123 MZ24
initial weights, unchanged MZ24 objective (dense1,negative0.25,positive0.25),
Adam0.001 with continuous optimizer state, features, normalization and labels.
Four exact copies of the original1200x16 batch stream give76800draws over the
same7562uniqueTRAIN frames. No new input, LR schedule, seed/weight search or
extra fitting. The fixed9.8MB interpolation matrix is an engineering change
shared by both endpoints; comparisons to the old nondeterministic run are
descriptive, not a clean attribution to training duration.

Save actual checkpoints at steps1200 and4800 within this same trajectory.
Both endpoints are predeclared and evaluated; do not select a favorable snapshot
or use intermediate DEV results to alter training. Complete all4800steps before
task evaluation. The duration comparison is4800versus1200 within this run, not
MZ25's failed separate-prefix comparison. Save first/1200/4800 loss inputs and
availability gradients; hash both checkpoints and the single batch stream.

Inference stays fixed availability logit>=0, frozen MZ20 task cutoffs and add-only
MZ5 fallback. All baseline positives must remain exact; no positive may be created
beyond MZ20. Known0 is missing native<=4m evidence, never no obstacle/CLEAR.
Native labels enter only training/evaluation, never learned inference inputs.

Evaluate both endpoints on oldDEV1000, clean/stress200, relationDEV2000 and
distanceDEV1000 plus the same both-chain wrong-zone controls. Report all gains,
losses, addedFP, complete-frame/group/site results, availability and contributor
rejection. Useful effect requires zero addedFP on normal cohorts, >=68/75
placement far additions with gain on both placements, trained pole>=48/49
clean/stress and exact original-positive retention. Pole has no negative-
availability bags and cannot alone prove difficult-negative separation.

Compare the two fixed endpoints' extrema on all7562uniqueTRAIN plus disjoint
oldDEV1000 and consumedplacementDEV3000. This diagnosis performs no updates or
threshold selection. Duration supports a useful direction only if it improves
error/coverage together; deleting real witnesses is not a successful solution.
One finite budget does not prove a capacity ceiling or universal convergence.

Fresh outputs under `artifacts.local/work/mz26-deterministic-convergence-20260910/`.
Independent checks cover scalar losses/gradients, both endpoint outputs/masks,
fixed initial/batches/cutoffs and selected real inference. Mechanical failures
are retained and diagnosed; no silent gate relaxation or fit restart. At4800,
finish report/terminal/scoped delivery and release task resources. No protected
EVAL, capture, App/device/temporal promotion or safety claim. Broader goal active.
