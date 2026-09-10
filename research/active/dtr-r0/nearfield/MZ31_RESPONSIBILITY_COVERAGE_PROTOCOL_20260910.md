# MZ31: does the original TRAIN pool cover both sensor responsibilities?

2026-09-10 EXPLORE, consumed TRAIN diagnostic, no fit or DEV selection.
MZ30's168 eligible training queries have severe conditional coverage gaps:
BODY_NEAR has118 RGB-correct versus5 ToF-correct examples; HEAD_NEAR and
HEAD_FAR have zero RGB-correct examples. The global120/48 opportunity gate
therefore did not establish per-query responsibility coverage.

Hypothesis: restricting supervision to baseline-positive, geometrically
unsupported disagreements discards already available counterexamples that
could teach branch responsibility. Count every frozen RGB/ToF disagreement
among the exact7562 unique MZ20 TRAIN frames reconstructed by MZ30. Keep all
models, scores, event truths, original TRAIN identities and role assignments.
No DEV outcome is used to form new targets, select cutoffs, or choose examples.

Partition disagreements by baseline sign and original geometric support, for
each of the four body/head near/far queries. Target is the branch whose sign
matches the frozen event truth. Report both target-class counts, unique explicit
sites and groups, source/family distributions, and unknown identity counts for
the original eligible subset and all TRAIN disagreements. Save exact query rows
so potential future supervision can be reproduced without outcome selection.

Fixed coverage criterion: the complete disagreement pool must have at least20
examples and5 explicit sites for each branch-correct class, separately for all
four queries. These are diagnostic coverage minima, not guarantees of adequate
training or transfer. Unknown sequence sites cannot satisfy explicit-site counts.
If any class fails, identify it and do not infer that a larger objective fixes
the gap. If all pass, a separately registered comparison may broaden supervision
while holding the MZ30 runtime correction eligibility and architecture fixed.

One saved-array pass; zero training, new neural inference, protected EVAL,
threshold changes or new task predictions. Verify the subset nesting, exact
168/120/48 original counts, four-way partition closure, mutually exclusive
branch-correct labels, and no non-TRAIN rows. Bind input/code/protocol hashes.
Output: artifacts.local/work/mz31-responsibility-coverage-20260910/run-v1/.
Complete the report, structured diagnostic inheritance and scoped delivery.
