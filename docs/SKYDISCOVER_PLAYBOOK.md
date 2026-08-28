# SkyDiscover playbook

SkyDiscover is BlindAssist's optional offline candidate-search engine. Use it
when a narrow mutable algorithm surface and a trustworthy evaluator already
exist. BlindAssist remains authoritative for inputs, hidden boundaries,
evaluation, route decisions, and claims.

## Decide whether to search

Do not start with a search algorithm. Answer these questions in order:

1. **Information reachability:** Does the public observation contain the
   identity, motion, progress, or arrival signal the candidate needs? If not,
   change the information source or representation; search cannot create it.
2. **Evaluator validity:** Can the evaluator distinguish a meaningful success
   from abstention, unsafe action, shortcut use, or metric gaming? Invalid
   candidates must fail closed before any weighted score is considered.
3. **Narrow candidate surface:** Can one policy block or small set of explicit
   subcontracts change while inputs, evaluator, cohort, and gates stay fixed?
4. **Headroom:** Does a credible baseline leave observable Development
   headroom? All-zero, constant, saturated, or mostly `NOT_EVALUABLE` feedback
   is not a useful search landscape.
5. **Fresh authority:** Is the search cohort Development-only, with protected
   outcomes unavailable to generation and retry decisions?

If any answer is no, fix that layer or close the attempt before spending model
budget.

## Choose the smallest search

| Situation | Default | Why |
| --- | --- | --- |
| First feasibility check or about 10 model calls | `incumbent_only` or direct Codex | Tests whether the model already produces a visible effect |
| Small bounded budget with independent candidates | `best_of_n` | Cheapest useful search-structure baseline |
| Several promising families need to survive | `topk` or `beam_search` | Preserves multiple paths without meta-search |
| Stable evaluator, informative failures, roughly 50--100 useful evaluations, observed plateau | `adaevolve` | Adaptive islands, migration, paradigms, and auditable Pareto search |
| Long budget, stable plateau, and evidence that candidate organization itself must change | `evox` | Evolves the search database as well as candidates, with higher cost and audit risk |
| Multi-file tooling or implementation from a specification | SkyDiscover Synthesize or ordinary Codex task | Separate system-building workflow, not Optimize evidence |

Always retain the direct-model and simple-search baselines when the research
question is whether ADA or EvoX adds value. Compare equal model, evaluator,
call/token/evaluation ceilings, cohort roles, and retry accounting. Current
BlindAssist L10M comparison evidence contains descriptive EvoX signals but did
not establish meaningful incremental superiority over direct Codex or
Best-of-N; therefore neither EvoX nor ADA is the unconditional default.

## Shape the candidate before searching

Prefer explicit subcontracts over one unconstrained `decide()` function:

```python
def update_identity(observation, state): ...
def update_tracking(observation, state): ...
def assess_safety(observation, state): ...
def assess_progress(observation, state): ...
def propose_action(observation, state): ...
def arbitrate(proposals, state): ...
```

Open only the block needed for the current bottleneck. Keep evaluator files,
data access, thresholds owned by the frozen contract, hidden identifiers,
future frames, and final adjudication outside the mutable candidate. Search one
subcontract first; combine blocks only after each has interpretable headroom.

## Build the evaluator contract

The evaluator must expose `evaluate(program_path)` and return numeric metrics
plus optional JSON-compatible `artifacts`. Keep `combined_score` for scalar
compatibility, but return the raw metrics needed for diagnosis and Pareto search.

Typical goal-lock metrics include correct commit, false commit, identity
retention, reacquisition, arrival quality, instruction count, and runtime.
Typical DTR metrics include event recall, false-alert segments, lead time,
dropout recovery, and event F1. Unsafe motion, wrong-target arrival, premature
arrival, evaluator leakage, invalid output, and hidden-data access are hard
invalidities; success must never compensate for them.

Evaluator `artifacts` should describe actionable public failure mechanisms such
as "lost after occlusion" or "false commit on repeated entrance". They must not
contain target IDs, future truth, hidden optimal actions, or diagnostics from
which a protected case can be reconstructed. `UNKNOWN` and `NOT_EVALUABLE` stay
distinct from failure and from known-safe behavior.

For ADA or EvoX Pareto mode, return every configured metric and declare direction
explicitly:

```yaml
search:
  type: adaevolve
  database:
    pareto_objectives: [task_success, false_commit, runtime_ms]
    higher_is_better:
      task_success: true
      false_commit: false
      runtime_ms: false
    fitness_key: combined_score
```

The representative `fitness_key` does not replace the Pareto front. Missing,
boolean, or non-finite objective values are not valid measurements and must not
be allowed to appear favorable.

## Use the isolated bridge

Configure the local tool checkout and Python in ignored `config/local.toml`
using the keys shown in `config/local.example.toml`. Do not install one project
into the other. A route-owned job manifest uses schema
`skydiscover-assist-job-v1`:

```json
{
  "schema": "skydiscover-assist-job-v1",
  "consumer": "blindassist",
  "working_directory": "../../../..",
  "initial_program": "initial_program.py",
  "evaluator": "evaluator.py",
  "config": "config.yaml",
  "output": "../../../../artifacts.local/evidence/<route>/<task>/search",
  "evaluator_timeout_s": 300
}
```

Paths are relative to the manifest. The launcher injects the selected
BlindAssist research Python as the evaluator process, while SkyDiscover remains
in its own environment. The output root must be outside the SkyDiscover
checkout.

Run a zero-model-call import and transport probe first:

```powershell
pwsh -NoProfile -File tools/skydiscover_assist.ps1 check <job.json> -Profile research-dtr-r0
```

Then run the frozen job:

```powershell
pwsh -NoProfile -File tools/skydiscover_assist.ps1 run <job.json> -Profile research-dtr-r0
```

Use `research-l10-r0` for the goal-lock environment. The launcher records the
resolved SkyDiscover commit, SkyDiscover Python, evaluator launcher, and
manifest before dispatch. It does not perform dependency synchronization or
modify the SkyDiscover environment.

## Close out a run

Record the BlindAssist source hash, SkyDiscover commit, initial candidate,
config/evaluator/input hashes, model, call/token/evaluation ceilings, retry and
`in_doubt` accounting, random seed, output root, and selected-candidate hash.
Keep raw checkpoints and search databases under ignored `artifacts.local/`;
promote only the chosen candidate, concise result, and route-changing evidence.

Re-evaluate the frozen selected candidate through the owning BlindAssist route
without generation-time artifacts. A Development win authorizes only its stated
Development mechanism claim. It does not authorize protected-final access,
Android promotion, live navigation, user benefit, or safety claims.

Release task-owned workers, ports, locks, and temporary state. Never clean
SkyDiscover `.runs`, shared caches, environments, or outputs whose ownership is
not established. If resumability matters, retain the exact task checkpoint and
provenance rather than a mutable shared environment.
