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

The frozen B0-C observation is not a single-case preservation pass. Both
`recovery_plus_arrival` and `true_stuck` flip because both end in the same
confirmed-arrival terminal truth. Its formal verdict is
`B0_C_TERMINAL_PRECEDENCE_MECHANISM_CONFIRMED_CAUSAL_SELECTIVITY_NOT_IDENTIFIED`:
the code intervention is limited to precedence, while its behavioral effect
must propagate to every trajectory satisfying `ARRIVAL => terminal`. The
consumed B0-C fixtures and results remain unchanged.

B0-D is a new four-case synthetic canary; it does not retrofit B0-C. It
orthogonalizes stuck/recovery history from final arrival truth and checks that
confirmed arrival remains terminal across stuck-evidence counts, recovery
state/attempt number, and the action that the parent policy would have selected.
It also requires no-arrival and UNKNOWN/no-arrival cases not to fabricate
success. Run it with:

```text
python -m scripts.research.l10m_b0.b0d_arrival_orthogonality --output artifacts.local/evidence/l10m_b0/d_matrix.json
```

Its claim ceiling is state-independent terminal semantics on synthetic policy
mechanics only. The B0-D receipt keeps its historical execution boundary:
B1 and Structured Search had not started when B0-D was consumed.

B0 is formally closed without a B0-E. The closure verifies the frozen B0
source identities, B0-C/B0-D result hashes and verdicts, the progress three-state
contract, and the no-fabricated-stuck/no-fabricated-arrival UNKNOWN behavior:

```text
python -m scripts.research.l10m_b0.closure --output artifacts.local/evidence/l10m_b0/closure.json
```

The successor protocol is the small matched B1 searchability experiment in
`scripts/research/l10m_b1`. B1 may not modify any B0 semantic lock.
