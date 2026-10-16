# Negative-only support screening: training-user holdout

2026-10-16, EXPLORE / thesis line. Owner: this Codex chat. The user asked to
continue the simplification check on the secondary worker and reuse caches.

## Fixed comparison and decision

Compare the frozen combined support selector (own-object consistency minus
maximum other-object similarity) with negative-only selection (minus maximum
other-object similarity). Both retain 15% of patches, build 32 prototypes per
object, and use top-8 query pooling. Reuse the immutable implementations and
checkpoint from [the earlier study](../independent_20261016/PLAN.md); no parameter
sweep, new training, test-user inference, or box/mask access. Enrollment object
identities are support inputs; query object labels are used only by the scorer.

Use all 44 already-consumed training users and their exact cached queries
(461 videos / 11,064 frames). Preserve the existing four user-disjoint folds
from the prior score-controls receipt. Process folds 1/2/3 first (33 users),
then fold 0 (11 users) as the primary held-out comparison, with no intervening
method changes. Both methods are training-free personal memory construction;
each user's own clean context videos remain their allowed support. This is a
held-out user group within Development, not fresh confirmatory evidence.

Input preparation fixes fold 0 at **133 videos / 3,192 queries / 11,940 context
frames / 76 objects**; all 44 users use 42,363 context frames / 278 objects.
Each query video has exactly 24 cached frames, so video-macro and query-micro
point accuracy coincide here. A 0.5 pp query loss is about 16 fold-0 frames;
the decision nevertheless uses the paired user-cluster interval.

The primary accuracy estimate is negative-only minus combined **video-macro
accuracy**, with paired user-cluster bootstrap (5,000 draws, fixed seed). Report
query-micro and user-macro accuracy, saved/harmed queries, per-user/fold results,
and object-count groups (<=4, 5-7, >=8) as supporting descriptions.

Predeclare a practical loss tolerance of **0.5 percentage points** and required
warm memory-construction time reduction of **20%**. Recommend simplification as
accuracy-preserving only if the fold-0 primary interval lower bound is above
-0.5 pp and the timing requirement is met. A near-zero point estimate alone is
insufficient; report inconclusive evidence when the interval crosses the margin.
Report all-44 Development intervals separately. Do not alter the tolerance,
folds, selector, or query pooling after observing results.

## Cache, timing, and budget

Reuse packed query patch features (float16, 11,064 x 256 x 384), query identities,
labels, existing worker metadata and DINOv2-S/224 checkpoint. Derive context
lists from the existing train plan using the frozen <=1,000-frame clean-context
rule and sorted object order. Transfer only missing context images and the
query feature file. Do not read boxes or masks. Compute normalized context
features once on the worker and persist hash-bound per-user caches for resume.
Both arms match the same cached query tokens in float32. This shared cached
representation is disclosed; an exact reproduction of prior pixel inference
is not assumed.

Measure three separate, alternating real builds per arm/user; report first
build and the median of the remaining two. No shared selection computation in
the timing comparison. Report selection, clustering, total construction,
context extraction, cache hits, query scoring and resource use separately.
Extraction-plus-construction is a cost decomposition, not end-to-end/mobile
latency. Verify matched other-object sampling, repeatable centers and complete
query alignment before interpreting results.

Budget for this task: **2 GPU-hours, 2 CPU worker-hours, 4 GiB LAN input**, zero
external downloads or paid resources. Inspect live capacity and preserve
practical RAM/VRAM headroom. Stop the affected run at its budget or a concrete
data-integrity failure; retain failures and successfully completed user caches.
Fix implementation defects and resume remaining users without retuning methods.

Deliver scripts, source/input hashes, complete user/fold statistics, thin
receipts and a Chinese decision report here. Payloads and logs live under
`artifacts.local/work/orbit-negative-train-holdout-20261016/`; the worker retains
its reusable input/features. Release owned jobs on completion. Stage exact
task-owned paths and deliver directly to the default branch.

## Observed numerical diagnostic

At 21/44 completed users, P174 (development fold 2) matched other-object samples
but failed the predeclared repeated-center allclose check (maximum absolute
center difference 0.00165094). Preserve the main run, source, tolerance and
eligibility result. After its completion, use the retained context/query feature
caches to rebuild only center-inconsistent users for three fixed repetitions per
arm and compare their actual predictions with the original and each other. This
is an implementation-stability diagnostic within the same 2 GPU-hour budget,
with no feature re-extraction, voting, method tuning or replacement of main
predictions. Distinguish center movement from observed classification changes;
the diagnostic cannot retroactively make the original consistency gate pass.

Completed: all44 users and264 main builds, followed by24 cached diagnostic builds
on four center-inconsistent users. The0.5pp accuracy-loss bound is unestablished;
the construction-time criterion is met. See [RESULTS](RESULTS.md) for numerical
limits, costs, retained caches and verified release of owned jobs.
