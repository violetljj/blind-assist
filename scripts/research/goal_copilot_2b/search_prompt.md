# GOAL-COPILOT-2B bounded policy search

Improve only the six functions in the candidate and return a complete Python
file. Preserve CLEAN completion while making the policy robust to target
dropout, bearing jitter, false targets, nearness error, tracking collapse, and
delayed evidence. Use confidence and tracking quality conservatively. Do not
claim completion from a single noisy nearness observation. Keep safe-space gates
fail closed and recover directionally after target loss.

The candidate language allows only bounded straight-line conditionals,
comparisons, observation-field access, tuple/list literals, and returns. Imports,
calls, loops, assignment, mutation, I/O, decorators, defaults, and module state
are rejected. BlindAssist owns all task truth, noise, hidden state, evaluator,
safety, held-out validation, and acceptance. Sky metrics are development
provenance only.
