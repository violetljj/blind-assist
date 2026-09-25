# Standardized scanner Development reproduction — frozen before outcomes

Status: implementation/protocol prepared; outcome execution requires this protocol and code committed first. EXPLORE, bounded 1800 seconds local CPU wall time plus one worker source-extraction job bounded1800 seconds; at most two concurrent source jobs. No capture, network training, City or protected test access. This is a newly authorized diagnostic on consumed old data, not reopening Track A G2 or claiming independent generalization. No 30-second performance guarantee.

## Inputs and provenance

Reference is the advisor's scratch `C:/Users/26442/AppData/Local/Temp/claude/E--linnan-linnan/bde48e47-fc5b-4b4a-91ff-4c1aecfc0b40/scratchpad/scan_precheck.py`, not project evidence. Read frozen alley 960 = 480 train / 480 dev (408 positive, 2472 negative dev queries), Street 1920 descriptive only through R1 identity/hash-checked loaders. Raw128 alley is reconstructed by original R1 `synthesize` using original frame seeds and canonical source bindings. Every regenerated H3 sum cast float32 must equal stored H3 bitwise; mismatch stops. No new sensor seed or synthesis parameter.

Street H3 covers all 1920 with alley-only fitted parameters. Already-authorized same-code source-local worker960 reconstruction runs after this freeze, passed with `--worker-raw`. Require ordered frame keys, SHA256, finite shape960×64×128, and bitwise float32 H3 parity against stored worker observations; local960 is independently reconstructed once via R1 and concatenated in original order. No remote fitting. If this cache cannot be produced, raw Street is NOT_RUN_MISSING_WORKER_SOURCE; do not report local960 as all Street or repeat alley outcomes to select a method. Root may await the frozen cache before the single run. Remote source-local extraction is a bounded prerequisite, not additional scientific trials.

## Frozen arms and selection

* B0_EXACT_STORED_H3_UNCENTERED: original stored float32 H3 promoted float64, old public query weights; no template subtraction. Assert all960×6 scores bitwise equal prior R1 alley-scores.npz R0 with identical ordered frame keys; pooled dev AP parity 0.21011599820183707 within 1e-12 also required.
* Each of zero / train_median templates is independently reported. H3 template = per-cell alley train median. Raw128 template = per-raw-cell alley train median, then residual processing; its aggregated median need not equal median H3. Report this explicit template difference, never imply resolution is the sole difference between median branches. Zero branches isolate shift resolution more directly.
* CENTERED_SUM_K1 is named separately from B0. B1_SUM_K4_SCRATCH is causal K4 residual **sum**, exactly the scratch sum convention, not R1's mean. Historical samples remain in same clip, available history at starts, nominal displacement .1m/frame times zone mean forward cosine; this is privileged nominal motion, not measured device ego-motion.
* H3_CONTROL performs fractional shifts on H3. RAW128_PRIMARY shifts original raw128 residuals and variance first, aggregates each eight raw bins into H3, and only then scans. **No raw-bin scan windows.** Both use H3 windows (rows,cols,bins): (1,1,1),(2,1,1),(3,1,1),(1,2,1),(1,3,1),(2,2,1),(1,1,2). All cells in a window must have public query overlap >=tau. Scores are maximum standardized window sums, initial -50 if no admissible window.
* S1_K1: single frame. S2_K4_LITERAL: scratch linear interpolation of variance. S2_K4_CORRECTED: squared-coefficient variance plus induced adjacent-bin covariance. Both freeze before any outcomes; neither may replace the other retrospectively.
* tau candidates .25/.5/.75: independently select per resolution/template/S-family using pooled alley train AP only; tie smaller tau. Report all candidates as disclosed sensitivity, selected status explicit. Never choose from dev/Street. For every arm alarm threshold maximizes alley train F1 with larger-threshold tie using existing `train_threshold`; Street reuses exact threshold. No template selection winner gate.

## Null mathematics and correction

Declared original-bin independent null proxy V = 2*A*ambient + max(template,0), A=8 H3 or 1 raw. Median template contains static scene as well as residual crosstalk; not claimed an unbiased scene-free noise calibration. Template fitting uncertainty and correlated sensor errors are not included. Raw128 reconstruction is a privileged input relative to the stored H3 interface.

For fractional interpolation matrix T, corrected residual is Tr, diagonal variance diag(T diag(V) T^T), adjacent covariance the first off-diagonal. Independent temporal frames sum these matrices. A d-bin scan window variance includes diagonal sum plus twice internal adjacent covariance. Raw-to-H3 aggregation likewise includes within-H3 off-diagonal covariance and retains cross-H3 boundary covariance. Literal scratch instead uses TV and ignores covariance. Signal contributes additional shot variance not represented by a null-only proxy; no universal calibration guarantee.

Fixed-window standardized null may approximate N(0,1); maximum across windows **does not**. Synthetic seed20260925 fixed-window test: 30000 independent Poisson128-Poisson128 /16, |mean|<.03 and |std-1|<.03. Separate descriptive null receipt: 2048 H3 frames with independent Poisson32-Poisson32 cells, tau=.5; record fixed-window and six-query MAX mean/std and .5/.9/.95/.99 quantiles. Do not test MAX as normal or interpret its Z as a fixed-window p-value.

## Reporting and bounds

Every arm reports train/dev pooled AP, AUROC, six-query metrics and train-F1 threshold TP/FP/FN/TN. Street same descriptive metrics and frozen thresholds. Selected arms plus centered sums get paired delta AP vs exact B0 with 2000 clip bootstrap draws seed20260925, 12 dev clips resampled with replacement, percentile95%. The clips share repeated geometry/layouts: these are conditional sensitivity intervals, **not valid independent-layout generalization evidence**. Report evaluable draws.

Artifacts under `artifacts.local/evidence/` contain protocol hash, original seeds/parity, templates, scores, selections and receipts. Stop at deadline, any source/parity error, or invalid numerics; preserve completed evidence and error. No retries with changed seeds/thresholds or expansion. Synthetic unit tests may run before protocol commit; outcome computation may not.

Tests: interpolation matrix variance/covariance, raw-to-H3 covariance aggregation, exact box sums, causal clip isolation, train-only bias/selection, fixed-window null moments, paired bootstrap identity. No protected data fixtures.
