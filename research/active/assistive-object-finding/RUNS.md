# Runs

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | object-finding-pilot-20261015 / EXPLORE; source version delivered with this row | Official MeViSv2 valid_u, 12 sources/16 queries/528 rows; DINO-base + CLIP; fixed source-grouped 3-fold learning; real KLT with detector cache; official StreamingRVOS-1B mask cache | DINO 82/327 visible; CLIP 70/327; comparative learner 81/327; posthoc full pool 85/327 (saved 69, harmed 66). Strong cached method 147/327 visible and absent output 88/105; 96 rows NOT_EVALUABLE. Adaptive vs uniform at 119 calls: +8 visible, +19 absent outputs. | COMPLETED: pipeline usable, no reliable method gain; keep direction but defer topic choice. RefEgo pixels NOT_RUN; Ego4D application submitted, agreement/approval pending. [Results](README.md); payload artifacts.local/work/assistive-object-finding-pilot-20261015/ |

GPU process wall: 321.516 s including smoke, 528/528 full inference rows, peak
allocated memory approximately 2.13 GB; no paid compute. Real CPU KLT wall 7.88 s;
learned fitting approximately 0.375 s. Candidate-pool probe reused saved features.
Data and model payload stayed below the 6 GB network cap. The pilot finished
within its 90-minute wall budget; authorization waiting does not create a new run.
Archives, source receipts, predictions, features and failures retained. Duplicate
download chunks removed with hash/path checks; task-owned model processes and
writer lock released. No full StreamingRVOS runtime or first-person video run.

Delivery checks: evaluator self-test passed (join, missing rows, censoring,
unknown FPS and native mask state); all route Python files compiled; saved
full-pool replay reproduced 85/327; 26 local documentation links and scoped
whitespace/identity/path checks passed. Whole-repository structure check also
passed after concurrent maintenance settled. The new route is included in its
explicit route list and compact-current budget.

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | alternative-audio-functional-parts / EXPLORE; source version delivered with this row | Independent user-requested direction screen. ESC-50 120 real clips, source-disjoint fit40/cal20/eval60; CLAP whole+3 windows. Seven fixed comparisons. Public AGD20K/IIT ZIP range audit. | Audio eval known30/unknown-primary30: margin27 correct accepts/1 unknown accept; global LR27/1; local LR29/3. 480 embeddings, GPU-process wall53.503s including canary. Functional data: 6 real AGD image/heatmap pairs and 3 IIT image/native-pixel-label pairs decoded, 16.37 MB body downloaded. | COMPLETED: audio simple baseline strong; no clear improvement from richer gate. Prefer functional-part feasibility next; no visual-model performance or novelty claim. [Results and reproduction](audio_screen/README.md), [fixed analysis](audio_screen/PLAN.md). Payload artifacts.local/work/assistive-audio-pilot-20261015/ |

The audio metric is dominant-label vocabulary membership, not exhaustive sound
absence or real-world false alarms. FSD50K natural multi-event audio NOT_RUN.
Functional data labels are soft heatmaps (AGD) versus categorical pixels (IIT),
not physical grasp-success truth. No additional models/parameters were tried
after the seven-method score. Download scripts now pin metadata revisions;
original predictions and receipts are preserved. Ego4D agreement is now
user-reported signed; AWS credential delivery remains pending.

Delivery validation: all 6 new Python files compiled in memory, fixed prompts
matched the inference input byte-for-byte, 15 scoped documentation links passed,
and the project structure plus hot-doc index checks passed (458 local links).
The evaluator's three hand examples and independent reconstruction of all 840
decisions, frozen logistic probabilities and AUROC agreed. No full application
build was needed for this research-only change. No task-owned process remains;
models/data/receipts are retained under the payload for authorized reuse.

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | functional-parts-280 / EXPLORE | AGD240 (80cal/160eval), IIT40 (16cal/24eval), 280 unique image SHA; released SelectiveCL, cal-selected ree, six fixed methods and three gates. | AGD M1/M4/M5/M6 point proxy .3620/.3769/.3368/.3697; IIT hits3/4/5/4 out of24. M6 vs M5 saves1/harms2 on IIT. Small AGD regions82: M1 .2823, M6 .2841. Entropy retains125/160 AGD at .4065, but IIT23/24 with only3 hits. | COMPLETE: automatic data/inference/scoring feasible; crop does not beat matched flip, rejection cross-domain weakness remains. No topic or novelty confirmation. [Results](functional_parts/README.md), [frozen plan](functional_parts/PLAN.md). Payload artifacts.local/work/functional-parts-pilot-20261015/. |

GPU-process wall153.564s (conservative158.878s), peak allocation779,306,496B;
model/source download714,079,887B, data46,082,704B. A progress-file sharing lock
interrupted additional views after242 images; preserved failure and completed
rows, retried only38 missing images, and merged by identity. Only bounded I/O
retry changed; inference math stayed fixed. Processes and writer locks released.
Independent checks passed for280 image geometries,4200 finite raw maps,1680
method rows,552 eval gate rows and32 case-level scores. Native metrics matched
released source. AGD heatmap proxies are not part-mask accuracy; IIT native hits
are not grasp success. Its5/24 evaluation native-empty images remain in denominators.

Delivery checks: all5 new Python files compiled in memory; metric/geometry/gate
self-tests,16 scoped local documentation links, staged whitespace, project
structure and hot-doc index (458 links) passed. No application build was needed.
