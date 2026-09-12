# MZ103: replace depth generation, preserve the risk readout

Phase: EXPLORE, consumed Development. Date: 2026-09-12.

Question: does better stereo depth retain small/thin hazards while removing
SGBM spatial artifacts, without modifying BODY/HEAD geometry or alert state?
MZ102 rejected real small surfaces; another size filter is not the intervention.

Inputs are all 288 MZ101 plus all 288 MZ102 captured frames, already consumed.
Keep their original ground truth, per-episode resets, known camera/body pose,
common FOV, 0.5--4 m depth eligibility and two-on/two-off alert state. Use the
unmodified mz101_spatial.py and run_mz101_spatial.py readout/evaluator.
Compare ToF, SGBM, and their original raw-support union, reproducing saved
predictions exactly before evaluating a replacement frontend.

Stage 1 is an evaluator-only native-left-depth diagnostic, plus unchanged ToF.
It uses all eligible rendered depth pixels, never object masks or GT-conditioned
selection. It is a reference intervention, not a sensor candidate or mathematical
upper bound (rasterization, occlusion and state remain). Proceed with the model
if the combined native union has fewer FP, no more FN, and no lost union events;
otherwise explain the readout/input gap before model work. Report both panels
separately, not only a pooled number, and direct-support versus held FP.

Stage 2, conditional on headroom: one public frozen FoundationStereo checkpoint,
no training. Prefer the official small variant for local feasibility. Freeze the
exact source, weights, license, preprocessing, inference iterations and validity
policy in an engineering manifest before the full replay. An unlabeled pair is
allowed for runtime/shape checking; no selection by task scores or native depth.
Preserve metric depth using f * 0.10 / disparity and the original intrinsics;
padding must be removed. No output surface-size filters, threshold scans,
GT-guided filling, temporal model, or downstream policy changes. If a mechanical
runtime failure requires a backend change, record it before replay. Model output
is sealed before task evaluation and the producer receives RGB and calibration
only. The native diagnostic and candidate output directories stay separate.

Retain as a Development challenger only if on each panel the replacement union
has fewer FP, no more FN, no lost baseline-union events, no more false sessions,
and no paired detected event is delayed by over one frame (0.25 s). Report exact
TP retention/new TP/lost TP, thin and small_head slices, BODY/HEAD, timing and
actual backend/runtime even if these criteria fail. F1 alone cannot pass.
No claim of fresh confirmation, future collision prediction, hardware accuracy,
or real-time execution follows. Keep the existing baseline if no useful gain.

Budget: one native replay, one engineering model canary, one 576-frame candidate
replay if feasible. No new capture, training, successor sweep or default-App
promotion. Preserve hashes, logs, summaries and pending metadata; release only
task-owned processes. An unrelated missing ASE receipt currently blocks the
global registry validator; attempt the supported registration command and record
its outcome without editing historical rows or bypassing validation.
