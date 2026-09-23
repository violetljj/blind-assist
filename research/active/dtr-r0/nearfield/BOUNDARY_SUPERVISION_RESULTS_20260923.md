# Dense boundary supervision improves metric precision, with recall cost

2026-09-23, one new fit completed under the
[fixed protocol](BOUNDARY_SUPERVISION_PROTOCOL_20260923.md).
**ACCURACY_COST_TRADEOFF:** both boundary axes improve substantially on train
and layout-held evaluation. False positives decrease, but combined-query recall
falls5.44percentage points, exceeding the predeclared3point limit. The joint
mechanism criterion and full component gate fail. Retain the measured boundary
supervision effect and all costs; this is not an A/LOCAL/App replacement.

## What changed

Same consumed1728simulation images,864train/288dev/576evaluation, same frozen
4864D public features, train normalization, initial geometry-model parameters,
100image permutations, optimizer, positive weight4.1667774 and100epochs.
The sole treatment is training-query allocation and corresponding binary labels.
Per layer/epoch:12coarse anchors,12fine width-slice and12fine horizon-slice queries.
Fixed banks contain36/47/46coordinates; every image receives the same schedule.
No per-image evaluator geometry selects query coordinates.

Each image still receives72distinct queries per epoch and7200total exposures.
Unique queries increase72to258. Total training exposures remain6,220,800.
The positive fraction changes19.35%to21.85% despite unchanged class weight.
Thus resolution, query allocation and unique labels change together; the result
does not isolate sparsity alone. Inference receives public features and requested
query coordinates, never geometry, labels or the training sampling identity.

Baseline is the saved old geometry fit, not an additional control refit.
Both select epoch10 under the same original seen-query dev BCE rule; the new
cutoff is.5454619526863098 versus old.5281916856765747. Both thresholds come
from the unchanged dev FPR<=5% rule, not equal realized evaluation FPR.
New dev recall87.47%,FPR4.978%; old88.38%,4.995%.
All ten new checkpoint-logit arrays are retained for selection audit.

## Boundary accuracy and missing predictions

Every finite interior truth stays in the5cm denominator. Missing crossings
remain failures. Width is full cross-section width, twice lateral gap; horizon
is axial first-contact coordinate. Cases are correlated image/layer queries.

| Cohort / boundary | Old geometry within5cm | Dense supervision within5cm | Gain |
| --- | ---: | ---: | ---: |
| Train width,840truths |124,14.76%|385,45.83%|31.07points|
| Train horizon,560truths |175,31.25%|322,57.50%|26.25points|
| Evaluation width,528truths |43,8.14%|192,36.36%|28.22points|
| Evaluation horizon,352truths |61,17.33%|122,34.66%|17.33points|

Both axes pass the>=10point improvement requirement on both splits. However,
more than60%of evaluation boundaries still miss5cm. This is useful partial
precision evidence, not established precise virtual-body contact geometry.

| Evaluation boundary metric | Old geometry | Dense supervision |
| --- | ---: | ---: |
| Width finite predicted coverage |486/528|484/528|
| Width conditional MAE |18.09cm|12.85cm|
| Horizon finite predicted coverage |294/352|283/352|
| Horizon conditional MAE |21.67cm|19.76cm|
| False width crossings /624right-censored truths |87|75|
| False horizon crossings /800right-censored truths |119|97|

Conditional MAE excludes missing predictions; the reduction does not erase
coverage loss. Both retain zero sampled monotonicity violations. Trained dense
slices are not unseen-query evaluation. Original combined-interpolation query
combinations remain untrained, although their individual coordinate values may
occur on the trained slices. The stored legacy `extrapolation` field no longer
means untrained width values, because the dense width bank includes.24and1.20.

Paired5cm outcomes expose individual losses: width retains24old successes,
loses19and gains168; horizon retains35,loses26and gains87. There is no lossless
boundary improvement claim. In the previously audited returned-near strata,
width succeeds39to175of463cases and horizon50to105of277. Other native-support
strata remain in the independent receipt. These are evaluator-defined descriptive
partitions, not a causal sensor ablation or public ownership classifier.

## Contact-query costs and subgroups

Same27,648combined-interpolation queries on576layout-held images;4482positive
and23166negative queries, not27,648independent trials.

| Metric | Old geometry | Dense supervision |
| --- | ---: | ---: |
| TP / FP / FN / TN |3894 /2196 /588 /20970|3650 /1393 /832 /21773|
| Recall |86.88%|81.44%|
| Precision |63.94%|72.38%|
| FPR |9.479%|6.013%|
| Brier score |.08331|.06216|

Net244fewer TP and803fewer FP. Recall cost fails the<=3point bound; FPR and
right-censored-crossing retention pass. Absolute recall remains>=.8, but
FPR remains>.05 and both boundary accuracies remain<.8, so the original full
component gate also fails. No replacement threshold was selected afterward.

| Family | Old TP / FP / FN | Dense TP / FP / FN |
| --- | ---: | ---: |
| BODY protruding plane |1210 /727 /176|1145 /505 /241|
| BODY suspended solid |1110 /578 /312|1062 /413 /360|
| HEAD hanging plane |797 /516 /31|772 /300 /56|
| HEAD horizontal |777 /375 /69|671 /175 /175|

HEAD-horizontal has the largest net TP loss,106of the total244. All families
reduce FP and lose recall. By original lateral layout relation, INSIDE changes
2039/577/202to1893/445/348, BOUNDARY1273/783/221to1170/506/324,
OUTSIDE582/836/165to587/442/160(TP/FP/FN). OUTSIDE describes the original.6m
layout; wider virtual queries can legitimately be positive.
Truth-changing pair ordering is2973/2988(99.50%); both decisions correct1390/2988
(46.52%). Accurate relative ranking still does not establish metric boundaries.
Original sensor UNKNOWN remains487/576frames; no model success certifies clear
space. These are contact queries, not event-alert timing or hardware metrics.

## Interpretation and stop

The fixed public representation and geometry network can express substantially
better metric boundaries when supervised at finer queries. That weakens an
explanation based solely on immutable feature or sensor incapacity for all
errors. It does not prove the features preserve every required metric detail,
or that coarse labels caused every former error. Precision improved together
with a more conservative contact working point; there is no equal-evaluation-
FPR comparison and no causal isolation of density from query weighting.

The fit ends here. Preserve this exact accuracy/recall tradeoff as a scoped
negative control for the joint upgrade, while retaining its positive boundary
supervision evidence. Further work would require a separately justified
mechanism; no loss, cutoff, query allocation, seed or epoch retry follows.

## Execution and evidence

Four focused schedule/mapping/gate tests pass. Training uses actual
RTX5060Laptop CUDA,9.25s for2700updates; complete scientific run12.81s.
Equivalent cloned initial-step probes measure CPU2.806ms versus CUDA2.554ms.
These are host training timings, not end-device latency. No new UE capture,
paid worker, background service or continuing task-owned compute.

All original input hashes remain unchanged. New train/dev/evaluation predictions
were sealed before evaluation target loading and old-outcome comparison.
This is code-path separation on previously consumed data, not fresh confirmation
or evaluator-process isolation. Same generator and one seed limit generality.

Evidence root: `artifacts.local/evidence/ba-boundary-supervision-20260923-v2`;
retains schedule,mapping,training labels,initial-input hashes,feasibility,
backend probes,fit-start receipt,selected weights,ten dev-logit arrays,prediction
seals and full metrics. The first launch failed before backend probes/fit-start
because cached-feature reuse bypassed old module-path setup. Its receipt remains;
version2 fixes that path only and consumes the sole fit. No failed fit was retried.

Independent audit PASS:222912unique world-space training labels,6,220,800scheduled
label exposures,714816original-query world labels,all ten checkpoint dev losses,
four full old/new train/evaluation summaries,selection threshold and every gate.
It also verifies2304prior native-support rows and paired boundary successes.
No checkpoint inference or training trajectory was rerun; weights and source
use are hash/implementation verified. Seven thousand two hundred synthetic label
expansions and independent schedule parity pass. The final audit RunSpec explicitly
declares inherited observation,visibility,manifest and plan inputs whose hashes
are verified; receipt `artifacts.local/evidence/ba-boundary-supervision-20260923-audit-v2/result.json`.
