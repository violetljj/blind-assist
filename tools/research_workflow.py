"""Phase-aware research proposals; these helpers never execute or authorize runs."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


PHASES = ("auto", "explore", "confirm", "engineering")


def prepare_research_proposal(
    plan: dict[str, Any], *, question: str, phase: str = "auto",
    objective: str | None = None, hypothesis: str | None = None,
    baseline: str | None = None, change: str | None = None,
    metric: str | None = None,
) -> dict[str, Any]:
    """Add decision branches and evidence policy without changing historical records."""
    if phase not in PHASES:
        raise ValueError(f"Unknown research phase: {phase}")
    result = deepcopy(plan)
    layer = result["failure_layer"]["id"]
    resolved = (
        "engineering" if phase == "auto" and layer == "runtime_infrastructure"
        else "explore" if phase == "auto" else phase
    )
    for field, value in (("hypothesis", hypothesis), ("baseline", baseline),
                         ("single_change", change), ("primary_metric", metric)):
        if value is not None:
            if not value.strip():
                raise ValueError(f"{field} must not be blank")
            result[field] = value.strip()

    if hypothesis is not None and change is not None:
        result["status"] = "proposal"
    if layer == "unlocalized":
        # An open question does not imply a fault or require one to be reproduced.
        if result["status"] == "localization_needed":
            result["status"] = "hypothesis_needed"
        result["failure_layer"]["name"] = "Open capability question"
        result["hypothesis"] = hypothesis or "Turn the stated question into a falsifiable explanation using a small relevant comparison."
        result["single_change"] = change or "Use one question-focused exploratory check; inspect interfaces if an observed failure requires it."
        result["baseline"] = baseline or "Current behavior and a relevant simple comparator on matched inputs."
        result["cohort"] = "A small relevant Development panel with disclosed input identity and prior consumption."
        result["primary_metric"] = metric or "Paired task effect, errors, UNKNOWN/coverage, and relevant observation or compute cost."
        result["stop_conditions"] = ["Declared budget exhausted or input/evaluation identity cannot be maintained."]
        result["not_evaluable_conditions"] = ["Missing evidence needed to distinguish the proposed hypothesis from the baseline."]
        result["claim_ceiling"] = "Named Development evidence only; no fresh confirmation, general performance, or safety claim."

    if resolved == "explore":
        evidence = (
            "Use a small relevant Development panel, including disclosed consumed or "
            "curated data. Record changes and comparison conditions. Exploration "
            "does not regain fresh-confirmation authority."
        )
        retry = (
            "Repair mechanical failures on task-owned Development inputs; retain logs "
            "and distinguish failed execution from a tested hypothesis. Do not reopen "
            "a protected run through this policy."
        )
        branches = {
            "gain": "Keep the useful mechanism, complete integration, then decide whether "
                    "unchanged-method confirmation is needed for the intended claim.",
            "no_gain": "Inspect the discrepancy and tradeoffs; revise the hypothesis, "
                       "simplify the composition, or stop this candidate.",
            "not_evaluable": "Identify source, implementation, or evaluation-contract "
                             "limitations; do not count missing evidence as method failure.",
        }
    elif resolved == "confirm":
        evidence = (
            "Fix the implementation, comparison, criteria, and access/retry rules before "
            "opening claim-relevant outcomes. Use evidence independent enough for the "
            "declared claim; consumed data remains Development evidence."
        )
        result["cohort"] = (
            "A predeclared, outcome-unopened panel suitable for the intended confirmation "
            "claim; name source overlap, exclusions, denominator, and coverage."
        )
        result["stop_conditions"] = list(dict.fromkeys([
            *result["stop_conditions"],
            "Protected outcome access, method changes, or retries outside the frozen protocol.",
        ]))
        retry = (
            "Follow only the frozen retry/checkpoint rules. Record uncertain external "
            "consumption as in_doubt; new policy does not authorize resuming a sealed run."
        )
        branches = {
            "gain": "Accept only the claim supported by this fixed comparison; complete "
                    "the relevant delivery and retain failures and coverage limits.",
            "no_gain": "Keep the frozen result. Return to a separately labeled exploration "
                       "or stop the candidate; do not repair the confirmation score.",
            "not_evaluable": "Preserve partial results and the reason; resolve engineering "
                             "or source readiness separately under the existing access rules.",
        }
    else:
        evidence = (
            "Use a task-owned replay, fixture, or smoke input without protected outcomes. "
            "Measure execution correctness, runtime, resource cost, and recovery."
        )
        result["cohort"] = "Task-owned representative smoke/replay input with no new protected outcome access."
        result["hypothesis"] = hypothesis or "The observed bottleneck is mechanical and can be corrected without changing the scientific comparison."
        result["baseline"] = baseline or "The current execution path on the same representative input and output contract."
        result["single_change"] = change or "Repair the observed execution or resource bottleneck and verify output equivalence and recovery."
        result["primary_metric"] = metric or "Completion and output validity, runtime/resource cost, and recoverable interruption behavior."
        result["claim_ceiling"] = (
            "Engineering evidence only; successful execution does not establish algorithm "
            "gain, fresh confirmation, user benefit, or safety."
        )
        result["selected_mechanism"] = None
        retry = (
            "Define recoverable errors and checkpoints before longer runs. Resume only "
            "task-owned engineering work with verified input/output identity; release "
            "resources on exit. Protected runs retain their original retry contract."
        )
        branches = {
            "gain": "Resume the next already-authorized research step after the mechanical "
                    "check; any protected run still follows its own frozen contract.",
            "no_gain": "Change the engineering diagnosis using the observed failure; "
                       "do not spend another formal run to debug infrastructure.",
            "not_evaluable": "Record the missing runtime/input prerequisite and complete "
                             "independent work; make no algorithm verdict.",
        }
    result["workflow"] = {
        "phase": resolved,
        "phase_selection": "automatic" if phase == "auto" else "explicit",
        "question": question,
        "practical_objective": objective or question,
        "evidence_policy": evidence,
        "contribution_check": (
            "Test one explanatory hypothesis. Necessary coupled edits are allowed; "
            "use a relevant simple baseline and incumbent when they change the decision. "
            "Add an ablation only when it resolves contribution or an observed ambiguity."
        ),
        "tradeoffs": ["primary task effect", "errors and UNKNOWN/coverage",
                      "observation, latency, or compute cost when relevant"],
        "retry_policy": retry,
        "decision_branches": branches,
    }
    return result
