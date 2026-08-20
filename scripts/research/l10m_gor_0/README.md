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
