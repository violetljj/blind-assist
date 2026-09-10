# MZ22: arbitrate global and local evidence instead of only adding alerts

2026-09-10 EXPLORE under the active obstacle-improvement goal. MZ20 restores
trained pole recall and improves placement error tradeoff, but8 false additions
remain and add-only composition cannot remove baseline premature near alerts.

Hypothesis: the joint observable pattern of global scores, local candidate
strength/dispersion, and metric coverage can distinguish when local evidence
should override a global judgment. A residual arbiter can correct both false
positives and false negatives while preserving exact baseline behavior when
there is no geometric candidate. This is not a claim that availability proves
reliability; MZ10's failed availability switch and MZ13's20parameter gate remain
comparators with their original limits.

Freeze MZ5 and MZ20. Extract only observable features: four MZ5 logits, four
MZ20 raw maxima, per-query top3/mean/std, eligible count, occupied-zone count,
return count, geometric coordinates of winning hypothesis, and packet valid
fraction. No native pixel/contributor/known label or source identity enters
inference. Missing candidates are explicit support masks; unsupported queries
return original MZ5 logits exactly. Geometry assumes the existing level camera.

One residual MLP per query (hidden32, ReLU, zero-initialized final layer),
TRAIN-only feature normalization. Train1200 updates with the exact MZ20 saved
1200x16 frame IDs, Adam .001, binary cross-entropy on supported task queries.
MZ20 remains frozen and its TRAIN predictions are in-sample base predictions;
report this stacking limitation. No model/cutoff fitting on placements, no new
capture/EVAL, no seed/architecture/budget search. The clean200 diagnostic
sequence already participates in training; its results are regression only.

Calibrate per-query cutoffs on oldDEV only: maximize supported true positives
subject to at most the original MZ5 supported false-positive count; break ties
by fewer false positives then higher cutoff. Unsupported outputs are unchanged,
so oldDEV total false positives cannot increase. Apply cutoffs unchanged to
relationDEV2000, distanceDEV1000, clean/stress200 and wrong-zone controls.
Calibration is not a guarantee of achieved transfer FP. Report actual retained
and lost MZ5 positives, removed/added FP, recovered/lost TP and net per query,
complete frames, pole retention and changed unsupported outputs (must be0).

An upgrade requires no FP increase per query on each normal cohort, no loss
of baseline near TPs, far TP improvement on both placements and pole>=48/49
clean and stress. Report gains even if the conjunction fails, with exact
tradeoffs; do not deploy a failed candidate. Comparisons to MZ20 test a whole
additional arbiter, not an equal-capacity architectural attribution.

Validate baseline identity at initialization, unavailable fallback, observable
feature boundaries and saved-output scalar replay. Use a fresh
`artifacts.local/work/mz22-evidence-arbitration-20260910/` directory. Freeze code,
protocol, input hashes and batches before training. Stop the fit after1200
updates, preserve failures and output evidence, release task-owned processes,
then report/record/deliver. No default-App, real-sensor or safety claim follows.
