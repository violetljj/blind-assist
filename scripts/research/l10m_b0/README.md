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
