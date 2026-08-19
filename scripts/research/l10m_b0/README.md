# L10M-B0: Closed-loop Representation Canary

This package is a deterministic, controlled-evidence mechanics canary for the
BlindAssist last-10m goal-approach task.

It compares only three hand-written arms: reactive, stateful, and stateful with
the evidence-bounded safety shield. It does **not** use Sky/EvoX, RGB, depth,
detectors, or real-device inputs. Its output is a behavioral vector rather than
a composite score, and can support only a policy-mechanics claim.

The evaluator must remain fixed when B1 structured-search experiments are
introduced. Search may modify only policy parameters explicitly admitted by a
future frozen IR; it may never modify truth, unsafe definitions, termination
truth, or hard safety invariants.

Run the frozen synthetic canary from the repository root with:

```text
python -m scripts.research.l10m_b0.run_canary --output artifacts.local/evidence/l10m_b0/result.json
```

The generated JSON is a replayable mechanics receipt. Its cohort is synthetic
controlled evidence and its claim ceiling must remain attached to the result.

B0-A/B localization matrix (a new protocol; it does not mutate B0-V1) runs with:

```text
python -m scripts.research.l10m_b0.scenario_matrix --output artifacts.local/evidence/l10m_b0/b_matrix.json
```

The B0-B matrix is composed of matched counterfactuals for transient
no-progress, true stuck, recovery exit, recovery-plus-arrival, reactive-solvable
preservation, and uncertain progress. Progress is represented as the explicit
three-state contract `POSITIVE_PROGRESS`, `CONFIRMED_NO_PROGRESS`, or
`UNKNOWN_PROGRESS`; only confirmed no-progress may increase stuck evidence.

B0-C is a separate, minimal intervention over the consumed B0-B scenarios. It
keeps all thresholds and evidence contracts fixed and changes only precedence:
confirmed terminal evidence preempts a recovery action, while credible positive
progress exits recovery before normal action selection. Run it with:

```text
python -m scripts.research.l10m_b0.b0c_precedence --output artifacts.local/evidence/l10m_b0/c_matrix.json
```

Its claim ceiling is causal recovery-transition repair on matched synthetic
mechanics only. It does not authorize B1 or Structured Search.
