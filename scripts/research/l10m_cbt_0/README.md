# L10M CBT-0 candidate bottleneck triage

CBT-0 is a zero-experimental-model-call analysis over consumed historical L10M
trajectories. It separates candidate availability, ranking, retention, and
parent-to-descendant transmission. A single non-incumbent candidate never makes
selection identifiable, and candidates from paired arms or future generations
are never treated as available at the current decision.

The prior exploration-policy closure is immutable. CBT-0 can route a later
development question, but it cannot authorize a model experiment, fresh cohort,
operator admission, RPS, Balanced V2, or another exploration heuristic.

## Terminal result

CBT-0 completed with:

`SELECTION_BOTTLENECK_NOT_IDENTIFIABLE_FROM_HISTORICAL_LOGS`

All 384 eligible decisions contained exactly one legal non-incumbent candidate,
so selection regret is undefined rather than zero evidence of good selection.
Neither paired-arm nor future candidates were added to any decision set.

The availability analysis nevertheless established a development routing
diagnostic. B4-A and B5-A each had a zero global-optimum candidate trajectory
rate (`0/18` in each cohort), despite positive candidate availability at
`38/144` and `45/144` decisions respectively. Their substantive normalized
availability rates were `34/144` and `39/144`. Thus incremental candidates were
sometimes produced, but no completion candidate appeared in either harder
cohort.

Every strictly improving chosen candidate was retained: positive retention
regret was `0/384`. Of 91 retained candidates with a later generation, all
became parents and `47/91` had a later productive descendant. Transmission is
secondary descriptive evidence only.

The resulting bottleneck map is:

- generation/operator/representation: development routing diagnostic supported;
- ranking/credit assignment: not identifiable;
- retention/population mechanics: not supported as the bottleneck;
- parent-to-descendant transmission: secondary descriptive only.

No model experiment, ranking development, fresh cohort, or operator admission
is authorized. The exploration-policy route remains closed.
