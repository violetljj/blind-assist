# Runs

2026-10-15 — Functional-part literature/data research and full experiment design
completed, with no new training, dataset bulk download, evaluation or paid compute.
Scope: primary-method overlap, released-data supervision, structural hypotheses,
strong baselines, training scale and conditional cost scenarios. Decision: prioritize
role-conditioned region binding only if native paired supervision and converged
baselines support the mechanism; rejection is not a predetermined innovation.
[Research plan](functional_parts/RESEARCH_PLAN.md). Read-only hardware query found
8GB laptop GPU, not an assumed24GB card. Detailed source notes:
`artifacts.local/work/functional-parts-research-plan-20261015/`.

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

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | functional-parts-multicandidate / EXPLORE successor | User-authorized reuse280 images/splits/ree; threeNMS crops, passive36class hook, GAP semantic support, spatial/semantic/combined selection; nine fixed methods, matched4call controls. | AGD160 C4 .344926 vs TTA .353431/fixed fusion .321765; IIT24 C4 4 vs5/7. C4 vs first candidate saves19/harms37 AGD; candidate oracle .529951. At128/160 ranked coverage C4 .360263 vs TTA .387714; IIT20/24 C4 3hits vs fixed7. | COMPLETE: no stable gain for full method; candidate retention has same-domain diagnostic headroom but selector loses it. Reused Development, not confirmation or topic selection. [Results](functional_parts/MULTICANDIDATE_RESULTS.md), [plan](functional_parts/MULTICANDIDATE_PLAN.md). Payload artifacts.local/work/functional-parts-multicandidate-20261015/. |

New1960 calls, GPU-process152.355s, peak779,306,496B; no download/training/failed
rerun. ScoringCPU32.453s and independent audit15.16s. All280 original baseline
points/qualities unchanged; independently checked2520 method points,3360
candidate/seed points,144 paired comparisons,20 gates,100 matched-coverage
points and72 oracle gaps. Models/process math unchanged by passive hook.
GPU processes and locks released. Preserved source/plan snapshots and all outputs.

Delivery: both new Python files compiled, evaluator self-tests and23 scoped
documentation links passed; project structure, hot-doc index458 links and
scoped whitespace checks passed. No application build needed.

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | functional-parts-learned-ranking / EXPLORE | Native heatmap supervision, fit158/cal79 after pHash screening; fresh200 project-unseen images. Frozen93 features, matched MSE/pairwise MLP with3 seeds, ridge/HGB and fixed-crop controls,4 backbone calls per family. | Ranking .361902 vs MSE .352627/ridge .364588; paired delta .009275,95% interval[-.014708,.033532],23 better/18 worse. Every individual rank seed loses to matched MSE. Candidate oracle144/200 heatmap hits, actual76/200. Rejection no stable advantage. | COMPLETE, no stable improvement or topic selection; do not tune on fresh200. [Results](functional_parts/LEARNED_RANK_RESULTS.md), [frozen plan](functional_parts/LEARNED_RANK_PLAN.md). Payload artifacts.local/work/functional-parts-learned-ranking-20261015/. |

Actual shared inference1843 calls, GPU-process110.374s, peak779,306,496B.
Successful trainingCPU25.625s, fresh feature/predictionCPU7.34375s,
evaluationCPU54.1875s. Download body bounded28,015,877–139,556,973B,
within500MB; acquisition met15min wall deadline. Two failed acquisition
attempts lost CPU counters, so600CPU-second budget compliance is UNVERIFIABLE;
wall time is not a valid multithreaded CPU upper bound. Corrections and failures
retained. One pretraining input-join failure was repaired without model changes.
Independent checks covered5600 scored points,800 ensemble vectors,8 paired
bootstrap comparisons,28-method gates/AURC and36x93 feature entries. No manual
labels or backbone training. Task-owned processes/locks released; source snapshots,
data, weights, predictions, seals and casebook retained for reproduction.

Delivery checks: four new Python files compiled in memory; learner loss/geometry/
schema self-tests,29 scoped documentation links, staged whitespace, project
structure and hot-doc index458 links passed. No application build needed.

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | functional-parts-visual-ranking / EXPLORE | Same fit158/cal79, fresh150 after excluding all460 prior image IDs; frozen DINO/pre-classifier point vs GAP features,193D matched MLP3seeds/600steps plus ridge/HGB, both crop families,4calls. | DINO point .492157/84 of150 vs GAP .422118/66 and scores .433490/70; point-GAP delta .070039 CI[.019425,.120224], all3seed directions positive. Candidate fusion .456105, fixed DINO tree .467660; superiority to these unresolved. Oracle123 hits, selected84. | COMPLETE: positive spatial-visual mechanism signal, no topic/novelty claim or fresh tuning. Next independent-source/part-label validation remains unrun. [Results](functional_parts/VISUAL_RANK_RESULTS.md), [plan](functional_parts/VISUAL_RANK_PLAN.md). Payload artifacts.local/work/functional-parts-visual-ranking-20261015/. |

New150 covers29actions/32objects; RGB checked157, rejected7, cached160. Old fit/cal
byte-identical. Download networkwall386.203s, CPU117.71875s including all threads,
20,828,165B received,22,735,075B conservative bound,18 retried failures counted.
Development237+fresh150 inference3561calls, GPU-process317.357s, CPU713.469s,
peak779,306,496B. Training featureCPU37.78125s, learning/cal49.5s, fresh21.3125s;
evaluation52.765625s. No labels, paid resources or backbone training.
Old7110maps/1659logits/candidate geometry unchanged. Independent scalers,
initializations/batches,1500ensembles,288visual vectors,10200scored points,
13paired comparisons and68-method gates/AURC passed. A failed audit expected
a missing manifest hash field; preserved then corrected to recompute RGB hashes.
That failed-auditCPU is unmeasured; no wall-as-CPU upper-bound claim. All processes
and locks released. Strongest observed control comparison separately labelled
posthoc descriptive; no model or predictions changed after fresh scoring.

Delivery checks:5 Python files compiled, learner/scorer/feature synthetic checks,
34 scoped documentation links, project structure and458 hot-doc links passed.
Scoped whitespace check passed; no application build was needed.

| Date | Run | Inputs / changes | Results | Decision / evidence |
| --- | --- | --- | --- | --- |
| 2026-10-15 | functional-parts-cross-source / EXPLORE | New IIT150 native masks, frozen AGD158/79 models/scalers/projection/gates; hold mapped to grasp5/wrap9 full-image union. Cross-source RGB SHA/pHash screen against665 prior files; no training or target calibration. | DINO point44/150 vs GAP48/scores54/fixed tree42/fusion49/base47. Point-GAP −2.67pp CI[-9.33,4.00]; point-scores −6.67pp CI[-12.67,−.67], exact p=.05248. All3seed directions negative. Oracle81, selected44, missed37. Empty20 retained. | COMPLETE: same-source local-visual signal not reproduced on IIT native grasp masks; no generalization/topic claim or tuning. [Results](functional_parts/CROSS_SOURCE_RESULTS.md), [plan](functional_parts/CROSS_SOURCE_PLAN.md). Payload artifacts.local/work/functional-parts-cross-source-20261015/. |

Metadata-class quotas15each, inspected168/rejected18/cached176; no native GT used
for selection. Download434.859wall_s,22,542,832B received/23,988,206B bound,
CPU126.75s. Inference1422shared calls,GPU-process74.859s,CPU237.844s,
peak779,306,496B. Frozen predictionCPU15.875s, runneraudit6.46875s,
evaluationCPU21.609375s, dataaudit7.484375s, aggregateaudit.453125s.
157training files unchanged,1500ensemble vectors,180visual reconstructions,
10200categorical decisions and8840EDT/KDTree distances independently checked.
68method aggregates/gates/AURC,13paired bootstrap/exact tests and2oracles pass.
Native NOT_EVALUABLE0; empty20 are label absence, not proof of real ungraspability.
All processes/locks released. Source snapshots preserve the pre-download plan
clarification that additionally excludes prior AGD RGB; no outcomes preceded it.

Delivery:3 new Python files compiled, categorical/empty/invalid-label synthetic
checks passed,37 scoped documentation links and458 hot-doc links verified;
project structure and scoped whitespace passed. No application build needed.
