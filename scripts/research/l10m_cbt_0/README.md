# L10M CBT-0 candidate bottleneck triage

CBT-0 is a zero-experimental-model-call analysis over consumed historical L10M
trajectories. It separates candidate availability, ranking, retention, and
parent-to-descendant transmission. A single non-incumbent candidate never makes
selection identifiable, and candidates from paired arms or future generations
are never treated as available at the current decision.

The prior exploration-policy closure is immutable. CBT-0 can route a later
development question, but it cannot authorize a model experiment, fresh cohort,
operator admission, RPS, Balanced V2, or another exploration heuristic.
