# Research workflow

Updated: 2026-09-24

Improve a concrete BlindAssist capability and explain why the improvement works.
Use this page for implementation choices; route currents own active evidence and
priorities. The workflow is a default, not an extra approval or registration gate.

## User direction and proportionate execution

The user decides a new research question, budget expansion, or changed success
criteria. Reuse authorization already given for that question: routine implementation,
mechanical recovery, targeted validation and delivery proceed without new approval.
A failed frozen experiment cannot be reopened by changing its threshold or subset.
This is not a requirement to ask before every command, diagnostic or engineering fix.

For exploration, put question, comparison, decision rule, result and next decision
in one existing record, with necessary input/code identity and UNKNOWN/cost reporting.
Do not create separate protocol, audit and approval documents by default. Register
runs and terminal inheritance through the existing supported commands; batch the
metadata work at a decision-changing terminal, without inventing ledger entries.
Use independent audit when a paper-critical claim or a concrete integrity risk
justifies it. Large assertion counts are mechanical checks, not independent samples.
Use one meaningful falsifier; expand only for an observed defect or evidence gap.

## Effect before novelty

Choose work by expected capability gain, reliability, and observation/runtime cost.
Established methods, better data or integration, incremental changes, and novel
algorithms are all eligible; innovation depth is a means, not an acceptance gate.
Start with the simplest credible approach that can meet the goal. Invest in a new
mechanism when an observed gap or a testable opportunity justifies the complexity.

Compare actual task effects under comparable inputs and budgets. When effects are
comparable, prefer the simpler, more stable, lower-cost option. Keep a useful gain
even if its novelty is modest; revise or stop a novel method that adds no useful
benefit. Explain algorithmic, engineering, or integration contributions accurately
without inflating novelty. Evidence and claim boundaries still apply.

## Start from the decision

Read the project route and the affected current. Name the capability gap or new
opportunity, the current credible baseline, and the decision the next result will
change. Search literature or history when it resolves a mechanism or evidence gap.
Candidate rankings and historical successor suggestions are not an exhaustive
research agenda. Keep a simple comparator when it can challenge added complexity.

Use one explanatory hypothesis. It may require coordinated changes to representation,
observation, state, and decision interfaces. Preserve comparable inputs and metrics;
add an ablation only if it changes the contribution judgment. Revisit a run of local
patches when a simpler common mechanism may explain their gains and failure modes.

## Three kinds of work

| Phase | Smallest useful work | Evidence and recovery |
| --- | --- | --- |
| Explore | Implement a promising hypothesis and inspect a paired task-effect check. | Disclosed consumed/curated Development is allowed. Record changes; keep the original result intact. Repair task-owned mechanical failures with logs and input/output identity. |
| Confirm | Test a fixed method and comparison against a stated claim. | Choose independence appropriate to that claim; fix criteria, access and retries before outcomes. Consumed results cannot become fresh confirmation. |
| Engineering | Resolve an observed execution, cost, integration, or recovery problem. | Use representative task-owned smoke/replay inputs. Measure correctness and cost; successful execution does not prove an algorithm gain. |

These phases do not replace `EXPLORE`, `FINAL`, or `EXTERNAL` access rules.
Protected blind/final access and claim-critical numbers follow
[formal governance](../docs/formal/RESEARCH_GOVERNANCE.md) and the
[formal template](../docs/formal/RESEARCH_PROTOCOL_TEMPLATE.md). External release,
privacy, credentials and real-user claims retain their applicable boundaries.
No workflow change retrospectively authorizes a sealed retry or changes old gates.

## A short experiment brief

For UE/nearfield data work, search existing assets before new capture and run through
`tools/ba.ps1 run research-ue -RunSpec <run-spec.json>`. The existing run spec declares
`reuse.mode`, a capability `reuse.query`, exact input subpaths and their data roles.
The runtime records reuse candidates and checks the [UE input contracts](../data/ue-reuse-policy.json)
before consumption, then registers result lineage. Use the
[UE reuse guide](../docs/asset-management/UE_REUSE.md) for source choices and the built-in
Core regression. A mixed reserved bundle needs a separately admitted subset; a folder
label or `split=train` declaration cannot authorize its protected files. Add scoped
contracts with new adapters; legacy direct scripts are not automatically governed.

Put the following in the existing idea, command, protocol, or owning result; do not
create another mandatory document. Fill in only details needed for the decision.

- Capability/question and explanatory hypothesis; why the proposed change matters.
- Phase, baseline, input/evaluation identity, and necessary coupled changes.
- Task effect plus relevant errors, UNKNOWN/coverage, observation or compute cost.
- What gain would justify keeping/integrating it; what would falsify the hypothesis.
- Decision for gain, no gain, and not evaluable; budget and experiment stop condition.
- Mechanical recovery and resource release when the run needs them.

Diagnostic example:

```powershell
python tools/knowledge.py diagnose --route ten-meter-copilot --phase explore --question "Does a reference verifier retain useful correct commitments?" --objective "Reduce wrong bindings without losing correct coverage or hiding observation cost" --baseline "Unchanged triggered observation" --hypothesis "Missing reference support and contradictory identity need different treatment" --change "Analyze the sealed paired episode outcomes and oracle support opportunity" --metric "correct/wrong/UNKNOWN, correct retention, online views, separate reference setup"
```

The card is a draft. `--phase auto` selects engineering for a runtime diagnosis and
explore otherwise; confirmation is explicit. `--question` also admits a new
opportunity without a known failure. Supplied hypothesis/baseline/change/metric
take precedence over retrieved templates. It does not authorize execution.

## Decide, then finish

- **Gain:** retain the useful mechanism and complete the authorized integration and
  delivery; confirm unchanged behavior when required for the intended claim.
- **No gain:** distinguish a wrong hypothesis, implementation defect, insufficient
  opportunity, and unsuitable comparison. Revise, simplify, or stop accordingly.
- **Not evaluable:** state the missing evidence or runtime prerequisite. Continue
  independent authorized work; missing evidence is not a negative method result.

Report raw effect and costs before interpreting a gate. Zero wrong commits with
zero correct commits is not useful identification or perfect precision. A source
failure does not adjudicate an algorithm. A local positive result need not establish
system-level benefit, natural-distribution performance, or safety.

An experiment stop ends its allocation, not automatically the user's broader task.
Finish remaining authorized validation, documentation, scoped commit/push, and
release of task-owned resources. Do not expand a frozen run or budget to continue.

## Keep the process useful

Current pages hold capability, baseline, bottleneck, next check and outcome decisions;
results/ledgers and exact Git anchors hold history. Replace superseded current prose.
Keep hard evidence, access and concurrent-work protections. Revise local templates,
gates and proposed routes when their scope or decision benefit no longer applies.
Use an existing result/change note to explain this; no routine meta-review is needed.

Assess a process change in real work: did it yield an interpretable next decision,
avoid wasted execution, or expose a meaningful effect/cost tradeoff? Passing a
workflow unit test alone does not establish faster research or better algorithms.

For a template-only configuration change, keep the previous configuration bytes
and run `python tools/refresh_decision_templates.py --previous-config PATH --check`.
Remove `--check` only after eligibility succeeds. This path preserves cached
experiment outcomes and records that ledger validation was not performed; source
or retrieval drift requires a full rebuild. It cannot repair an already stale
cache or certify historical records.
