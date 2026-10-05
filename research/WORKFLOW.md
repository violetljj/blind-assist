# Research workflow

Updated: 2026-10-05. Route currents own priorities and evidence; `AGENTS.md` owns
execution, integrity and delivery rules. This workflow adds no approval gate.

## Choose useful work

Name the capability gap/opportunity, credible baseline and decision the result
will change. Choose by task effect, reliability and cost; mature methods, better
data, integration and new algorithms are all valid. Prefer the simpler approach
when effects are comparable. Search literature/history to resolve a concrete gap.
Keep each comparison interpretable; broader exploration may examine different
hypotheses. Add an ablation when it changes the contribution judgment, and revisit
repeated local patches when a common mechanism could explain them.

## Work within the authorized question

Keep the goal, baseline, budget/unit, adjustable scope, stop conditions and expected
deliverable together in the existing task/run record; infer routine choices and
ask only about material gaps. Do not require a new document for ordinary Explore.
Within that scope, proceed from results to the next useful comparison, repair or
delivery without asking the user to relay every result or approve every command.
Delegate independent analysis or implementation when useful; integrate its result
without mandatory two-agent sign-off. Cross-chat dispatch still needs user authority
and a supported tool; another agent's suggestion cannot expand scope or budget.
Explicit stops and deadlines remain binding. A stopped run's unused budget is not
automatic authority for a successor. Separate wall time, GPU allocation time and
measured compute; use observed data-generation/inference costs when estimating.

## Decision check before freezing

Use a short note in the existing plan to check baseline headroom, denominator and
paired evaluation unit, the smallest observable count/rate change, and which outcome
would change the decision. Inspect only data the current phase permits; if a future
denominator is unknown, define missing/saturated cases without looking at evaluation.
A saturated endpoint cannot require strict improvement. For small samples, translate
a percentage margin into event counts and choose the allowed loss intentionally;
do not automatically relax it. Observed non-decrease is not statistical noninferiority.
Keep the useful improvement endpoint distinct from capability-retention constraints.
For a mechanism claim, match control data, initialization/seeds, training and model
selection, and calibration procedures; an unmatched checkpoint is a practical baseline,
not an isolated test of the changed mechanism.
If a frozen gate proves defective, retain its original result and explain the defect.
A successor needs an explicitly authorized revised contract before execution; reuse
compatible inputs/checkpoints with lineage, and disclose prior selection. Do not
relabel a consumed calibration result as independent confirmation.

## Work phases

| Phase | Smallest useful work | Decision |
| --- | --- | --- |
| Explore | Implement and inspect a paired task-effect check on Development. | Iterate with recorded changes; no registration, protocol, independent audit or terminal by default. |
| Engineering | Fix an execution, source, integration or recovery problem; inspect a representative smoke/replay. | Correct execution is not proof of algorithm gain. |
| Confirm | Test a fixed method and comparison against a stated claim. | Register, fix criteria/access/retries, and choose independence suited to the claim. |

An exploratory next-step decision or reversible fix is not formal promotion.
Protected access and claim-critical numbers use [formal governance](../docs/formal/RESEARCH_GOVERNANCE.md).

## Data and run records

Synthetic or simulated data is eligible Development unless explicitly marked protected or final.
This eligibility does not reopen an explicitly stopped run or change frozen criteria.
Existing real Development retains its documented access and use terms. During
exploration, training/validation inputs can guide method and parameter selection;
record their role/identity once and disclose selection. Use the fixed benchmark by
default; supplements must fit the authorized question and budget. New seeds/pixels
alone do not establish unseen-structure transfer or fresh confirmation.

Each active route appends one row per run to `research/active/<route>/RUNS.md`:
date, code version, configuration changes, metrics with denominators/units, conclusion.
Current logs: [forward perception](active/dtr-r0/RUNS.md) and
[hardware support](active/hardware-bringup/RUNS.md). Paused routes create their log
when resumed. Link payloads or a necessary detailed report from the row; ordinary
attempts need no separate result document. Record a not-run metric as `NOT_RUN`.

For synthetic-source defects, repair and regenerate/replay the same Development
scenes/seeds, preserve old outputs, and append the repair result to the route log.
No forced fresh cohort or one-shot run; without an evaluable algorithm run, a source
failure gives no algorithm verdict. Use hashes for immutable identity, transfer
verification or a concrete integrity issue.

## Governed asset reuse

For existing governed UE/nearfield bundles, search assets before capture and use
`tools/ba.ps1 run research-ue -RunSpec <run-spec.json>`. The spec declares `reuse.mode`,
`reuse.query`, exact input subpaths and roles; the runtime checks [input contracts](../data/ue-reuse-policy.json)
and records lineage. See [UE reuse](../docs/asset-management/UE_REUSE.md) for adapters
and Core regression. A mixed reserved bundle needs an admitted subset; `split=train`
cannot unlock protected files. Synthetic debugging outside these bundles follows
the engineering loop above, without this asset-admission gate.

## Finish the decision

Retain and integrate useful gains. For no gain, distinguish hypothesis failure,
implementation defect, insufficient opportunity and an unsuitable comparison;
revise, simplify or stop. For not-evaluable work, report the missing evidence and
continue independent authorized tasks. Report effects and costs before gate labels;
zero correct and zero wrong commits do not establish perfect identification.

At a meaningful decision milestone, consolidate the mechanism, effect/cost and
claim boundary into a reusable figure/table and short explanation when it helps
the decision or requested deliverable. Ordinary diagnostics stay in the run log;
paper/proposal packaging is not an automatic prerequisite for further research.
Replace superseded current status and authorization text in the same delivery;
link full results/history instead of copying the chronology across entry pages.

Formal runs use `python tools/knowledge.py register-experiment`; never append
`experiments/index.jsonl` manually. Mainline/baseline or governed reuse decisions
use `python tools/knowledge.py set-terminal-inheritance` under formal governance;
archived registrations link a `--decision-id` with complete inheritance.

## Knowledge maintenance

For template-only configuration changes, retain previous bytes and run
`python tools/refresh_decision_templates.py --previous-config PATH --check`.
Apply without `--check` only after eligibility passes; this preserves cached outcomes,
not ledger validation. Source/retrieval drift or an already stale cache needs a full
rebuild. Install the knowledge hook once with
`pwsh -NoProfile -File scripts/refresh_knowledge.ps1 -InstallHook`.
