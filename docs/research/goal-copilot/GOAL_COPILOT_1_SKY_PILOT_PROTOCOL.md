# GOAL-COPILOT-1-SKY-PILOT protocol

Status before execution: `GOAL_COPILOT_1_SKY_PILOT_PROTOCOL_FROZEN`.

BlindAssist owns the task definition, development and fresh scenarios, hidden
state, evaluator, safety gates, deterministic winner selection, fresh admission,
and final verdict. SkyDiscover owns proposal/search only and runs its canonical
`skydiscover.search.best_of_n` loop. EvoX, AdaEvolve, DiscoveryOS, other search
systems, and multi-arm comparisons are outside this Pilot.

## Frozen search

- Two independent sequential replicates: seeds 1701 and 2903.
- At most 16 generation dispatches per replicate, 32 total.
- Model and evaluator retries are zero. A dispatch journal entry is durable
  before each model call; a started-only call is `IN_DOUBT` and consumes one
  opportunity.
- Model: `gpt-5.6-sol`, reasoning effort `medium`, through the locally qualified
  native Codex CLI authenticated with ChatGPT. Exact executable, version, hash,
  provider class, and wire API are bound in `provider_identity.json`.
- Codex output/context limits are runtime-managed by the frozen CLI/provider;
  no silent substitute is allowed. Candidate source is independently capped at
  65,536 bytes and 4,000 AST nodes. Per-replicate total reported token usage is
  hard-capped at 2,000,000 input-plus-output tokens.
- Search is not resumable. An interruption that prevents a complete formal
  replicate closes as `GOAL_COPILOT_1_SKY_PILOT_NOT_EVALUABLE` with machine
  reason `INCOMPLETE_FORMAL_RUN`; no replacement attempts are added.

## Candidate isolation and closed loop

The candidate is a six-function deterministic decision policy. The admitted
language has conditionals, comparisons, literals, observation attributes, and
returns. It has no import, function call, loop, recursion, assignment, mutation,
module state, filesystem, network, subprocess, shell, path, or free-text action
surface. Candidate execution therefore terminates mechanically. Actions are a
fixed enum, and the BlindAssist hidden state advances only when the chosen action
matches the hidden transition; observations can change after actions.

## Frozen evidence

Development has 12 scenarios, four per task family. The frozen baseline must
have zero unsafe guidance, zero premature completion, at least one completion
and one failure per family, and 4–9 completions overall. This was calibrated with
zero model calls.

The six fresh scenarios are balanced two per family and AES-256-GCM sealed. Only
ciphertext, counts, and hashes exist on the model-readable filesystem during
search. The key is retained outside the filesystem by the supervising
BlindAssist task and is used only after `LOCK_WINNER` and dev admission.

Winner selection among hard-gate-valid candidates is lexicographic: maximize
minimum family completion rate, total completions, eligible reacquisition,
and total normalized progress; then minimize timeouts, actions on completed
scenarios, and AST complexity; finally choose lexical-ascending candidate digest.

Fresh is authorized only when the winner passes the hard gate, beats baseline
dev completion, and improves at least two families. Fresh PASS additionally
requires zero unsafe and premature events, at least +2 completions, gains in at
least two families, no regression on a baseline-completed scenario, and at least
one winner completion per family.

The maximum claim is a small deterministic symbolic closed-loop Pilot signal.
It is not real-vision evidence, a population/statistical claim, or a general
superiority claim.
