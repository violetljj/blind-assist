# L10M GOR-0 ceiling decomposition

GOR-0 deterministically separates representation expressibility, operator
grammar and strict-acceptance support, and historical generation coverage. It
uses exhaustive truth over all 162 frozen `PolicySpec` states and consumed
candidate trajectories only.

`operator_distance` permits neutral and downhill transitions.
`strict_admissible_distance` requires every retained edge to improve score.
Historical loss of reachability after an initially supported path is attributed
to generation path coverage, not retroactively relabeled as initial operator
non-support.

No model experiment, new operator, fresh cohort, or admission is authorized.

## Terminal result

GOR-0 completed with `GENERATION_COVERAGE_CEILING`.

Representation expressibility passed: every harder landscape has one or two
completion states inside the exhaustive 162-state representation. Operator
grammar support passed: all 162 states are grammar-reachable, and every harder
initial state has both operator distance and strict-admissible distance `5`.
No coupled multi-field jump is required by the oracle.

Historical generation did make completion-directed progress, but never
completed it. Across B4-A and B5-A there were 57 retainable distance-reducing
candidates, including 48 exact oracle edges, yet zero completion candidates in
36 trajectories. Minimum generated strict-distance distributions were:

- B4-A: distance 2 in 9 trajectories, 3 in 5, and 4 in 4;
- B5-A: distance 1 in 1 trajectory, 2 in 5, 3 in 8, 4 in 2, and 5 in 2.

Strict reachability was lost in 4/18 B4-A and 9/18 B5-A trajectories. The B5-A
distance-1 trajectory reached that state only at the final generation, so the
result is a ceiling of the full frozen generation setup and horizon, not proof
of intrinsic model incapacity.

Easy successful final edits used only
`action_selection_turn_threshold` (11 recorded edit occurrences). Hard oracle
opportunities additionally required recovery-transition, fallback-action, and
quality-floor edits. Hard missed-oracle-field counts were recovery transition
86, fallback action 56, quality floor 52, and action threshold 2. These are
opportunity counts, not independent samples or causal effect estimates.

The only routed next question is generation-mechanism development. It is not
automatically authorized for execution; no model call, new operator, fresh
cohort, or admission follows from GOR-0.
