# GOAL-COPILOT-1 public SearchTaskBundle

This bundle is exported by BlindAssist for an external proposal engine. Only
`initial_policy.py` is a candidate search surface. `task_api.py`, `protocol.json`,
public scenario descriptions, this README, checksums, and the manifest are immutable.

The bundle deliberately excludes the BlindAssist evaluator, hidden scenario graphs,
completion truth, safety gate implementation, score computation, and acceptance gate.
An external score is provenance only. A candidate has no BlindAssist status until it is
imported and independently assessed as `ACCEPT`, `REJECT`, or `NOT_EVALUABLE` by the
source repository.

This V0 package authorizes only a deterministic mock roundtrip. It does not authorize a
Sky model search, EvoX run, perception training, device use, or scientific claim.

