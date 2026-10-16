# Cheap own-object consistency on the cached training-user holdout

2026-10-16, EXPLORE / thesis line. Owner: this Codex chat. This successor study
is authorized by the user's request to try low-cost consistency on the secondary
worker while retaining the combined selector as control and reusing caches.
This plan is fixed before producing new candidate metrics.

## Mechanism and fixed inputs

The control is the frozen combined selector: mean of the five own-object
neighbors after dropping the maximum from top six, minus maximum other-object
similarity. The alternative changes only the own-object neighbor pool from at
most 30,000 patches to K in **{2,048, 4,096, 8,192}**. Draw other-object indices
first with the original per-object CPU generator, then draw the same full
own-object permutation; take its nested prefix of min(K, N). Do not generate a
permutation of K instead of N. Other-object samples, coefficient, 15% retention,
stable ranking, 32 prototypes, 15 clustering iterations, prototype initialization,
float32 math and top-8 query pooling remain fixed. Smaller pools change neighbor
statistics and self-inclusion probability; this is a defined approximation,
not an unbiased estimator of the original consistency score.

Reuse the exact manifest, query cache and 278 hash-bound context feature caches
from [the previous study](../negative_train_holdout_20261016/RESULTS.md).
The cohort is the same already-consumed 44 training Development users, 461
videos / 11,064 distinct queries / 42,363 clean-context frames. No images,
backbone inference, new training, test-user inference, boxes or masks are needed.
Enrollment object identities are support inputs. Query targets are attached
only after predictions and used only by the scorer. Preserve all previous
results and the previous run's failed center-consistency gate.

## Development selection, then fixed held-out comparison

1. Screen the three fixed K values on folds 1/2/3 only: **33 users / 328 videos /
   7,872 queries**. One build and prediction per candidate/user is sufficient for
   selection. Reuse previous combined predictions as the development reference.
   Shared other-object similarity work is allowed here to reduce screening cost;
   these timings cannot support a standalone performance claim. Select maximum
   development video-macro accuracy, resolving exact ties by smaller K. Freeze
   the choice, source/config and all 33 user receipt hashes in `choice.json`.
2. Only after that choice is frozen, evaluate fold 0: **11 users / 133 videos /
   3,192 distinct queries / 11,940 context frames / 76 objects**. Build the fresh
   original combined control and the chosen approximation independently three
   times each per user, alternating arm order. Each build predicts the same
   query cache. Timed paths share no selector or clustering computation.

Fold 0 was already inspected in the previous study. This is a reused held-out
group within Development, not fresh or blind confirmation. All-44 summaries
combine selection data with evaluation data and must be labeled accordingly.
No new K, coefficient, subset or pooling adjustment follows held-out results.

## Primary accuracy and timing decision

The primary estimate is candidate minus combined video-macro accuracy, using
each video's arithmetic mean accuracy across its **three separate memory-build
repetitions**. This estimates build variability; it is neither prediction voting
nor an ensemble. Resample paired users as whole clusters (5,000 bootstrap draws,
seed 20261016), retaining all their videos and repetitions together. Repetitions
do not increase the independent denominator beyond 11 users / 3,192 queries.
Every video has 24 queries, so query-micro and video-macro points coincide.
Report each repetition, repeat-0 saved/harmed queries, user-macro accuracy,
per-user results, and argmax/correctness changes across repeats separately.

Retain the previous practical accuracy-loss margin of **0.5 percentage points**
and required warm construction-time reduction of **20%**. The previous fold-0
control was 80.55%, leaving headroom; one query is 0.0313 pp and the chosen loss
margin corresponds to about 16 queries per build. The paired inference unit is
still the user (11 clusters), so a few changed queries cannot establish that
margin. Support replacing the
control only if the fold-0 primary paired interval lower bound exceeds -0.5 pp
and warm construction time falls by at least 20%. A small point difference does
not establish the bound. Use each user's median of repeats 1 and 2 as warm time,
sum these paired user times, and also report repeat-0 and components. The timing
claim concerns memory construction with cached features, not end-to-end/mobile
latency. Selection screening and cache verification costs remain separate.

Numerical center allclose is a diagnostic for this new run, because the previous
study observed center differences with unchanged predictions. Save all repeat
predictions, center diagnostics and same-other-sample evidence. If repetitions
change the practical conclusion, report the result as numerically unsteady.
Input identity, source binding, correct frozen choice, complete query alignment
and scorer separation are required for interpreting any accuracy or time result.

## Resources, evidence and delivery

New-task budget: **1 GPU worker-hour, 1 CPU worker-hour, 64 MiB LAN input**;
zero external downloads and zero feature extraction. Count screen and holdout
worker runtime together, including resume. Use float32 with TF32 disabled,
4 torch CPU threads, 1,024-patch chunks and a 2.5 GiB allocated-VRAM guard.
Inspect live capacity, preserve practical RAM headroom and other users' jobs,
and record any contention. Stop the affected run at budget or concrete integrity
failure; preserve failed receipts and completed user outputs. Fix genuine
implementation defects without retuning methods or erasing observed results.

Primary payloads: `artifacts.local/work/orbit-cheap-consistency-20261016/`.
Worker payloads: `G:/DevWorkspace/BlindAssist/artifacts/work/orbit-cheap-consistency-20261016/`.
Read old worker caches in place; transfer only thin sources, receipts and logs.
Deliver source/input hashes, selection record, paired metrics, timing and resource
receipts, and a Chinese result here. Release task-owned workers after completion,
retain reusable scientific caches, stage exact task-owned paths and push normally
to the default branch under the workspace's existing authorization.
