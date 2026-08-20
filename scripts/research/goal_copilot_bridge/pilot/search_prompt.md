# GOAL-COPILOT-1 policy search

Improve only the six functions in the candidate. Return a complete Python file.
The policy is deterministic and acts in a hidden-state closed loop using only
the supplied Observation. Valid actions are SCAN_LEFT, SCAN_RIGHT, ALIGN_LEFT,
ALIGN_RIGHT, FORWARD, INTERACT, STOP, and COMPLETE.

Safety is a hard gate: never issue FORWARD, ALIGN_LEFT, or ALIGN_RIGHT when the
corresponding free-space observation is false. Never claim COMPLETE or INTERACT
without evidence of true task completion. Seek full episode completion across
all three task families, including target loss and reacquisition. The candidate
language admits conditionals, comparisons, tuple/list literals, observation
field access, and returns; it rejects imports, calls, loops, assignments,
mutation, I/O, decorators, defaults, and module state.

SkyDiscover search metrics are development guidance only. BlindAssist retains
truth, evaluator, safety, fresh-cohort, winner-lock, and acceptance authority.
