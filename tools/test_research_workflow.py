from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import unittest

from tools import knowledge
from tools.research_workflow import prepare_research_proposal


def base_plan(layer="identity_binding"):
    return {
        "status": "proposal", "failure_layer": {"id": layer},
        "selected_mechanism": {"id": "old-matcher", "name": "Matcher"},
        "hypothesis": "Improve target commitment.", "baseline": "Fixed policy.",
        "single_change": "New evidence representation.", "cohort": "Development episodes.",
        "primary_metric": "Correct/wrong/UNKNOWN and observation cost.",
        "claim_ceiling": "Named Development only.", "stop_conditions": ["truth leak"],
        "not_evaluable_conditions": ["missing target truth"],
        "guardrails": ["Preserve UNKNOWN and consumed evidence identity."],
        "history_scope_notes": [{"id": "old-failure", "applicability": "unassessed"}],
    }


class ResearchWorkflowTest(unittest.TestCase):
    def test_exploration_accepts_coupled_hypothesis_and_keeps_history(self):
        original = base_plan()
        before = deepcopy(original)
        result = prepare_research_proposal(
            original, question="Can active evidence recover the exact target?",
            objective="Retain correct targets with fewer observations.",
            hypothesis="Observation selection and belief state must share support.",
            baseline="Fixed sweep plus current triggered policy.",
            change="Change observation selection and belief update together.",
            metric="Paired correct/wrong/UNKNOWN and extra views.",
        )
        self.assertEqual("explore", result["workflow"]["phase"])
        self.assertEqual("Change observation selection and belief update together.", result["single_change"])
        self.assertIn("consumed", result["workflow"]["evidence_policy"])
        self.assertEqual({"gain", "no_gain", "not_evaluable"}, set(result["workflow"]["decision_branches"]))
        self.assertEqual(before, original)
        self.assertEqual(before["history_scope_notes"], result["history_scope_notes"])
        self.assertTrue(knowledge._experiment_plan_is_valid(result))

    def test_confirmation_is_explicit_and_retains_access_and_retry_boundaries(self):
        result = prepare_research_proposal(base_plan(), question="Does the fixed method transfer?", phase="confirm")
        self.assertEqual("confirm", result["workflow"]["phase"])
        self.assertIn("outcome-unopened", result["cohort"])
        self.assertIn("in_doubt", result["workflow"]["retry_policy"])
        self.assertIn("truth leak", result["stop_conditions"])
        self.assertTrue(any("frozen protocol" in item for item in result["stop_conditions"]))
        self.assertEqual("Named Development only.", result["claim_ceiling"])

    def test_runtime_routes_to_engineering_without_claiming_algorithm_gain(self):
        result = prepare_research_proposal(base_plan("runtime_infrastructure"), question="Capture process crashes.")
        self.assertEqual("engineering", result["workflow"]["phase"])
        self.assertIsNone(result["selected_mechanism"])
        self.assertIn("output validity", result["primary_metric"])
        self.assertIn("does not establish algorithm", result["claim_ceiling"])
        self.assertIn("Protected runs", result["workflow"]["retry_policy"])

    def test_explicit_engineering_replaces_scientific_template(self):
        result = prepare_research_proposal(base_plan(), question="Benchmark the input join.", phase="engineering")
        self.assertNotEqual("New evidence representation.", result["single_change"])
        self.assertIn("output equivalence", result["single_change"])
        self.assertIsNone(result["selected_mechanism"])

    def test_engineering_diagnosis_can_be_explicitly_scoped_as_exploration(self):
        result = prepare_research_proposal(base_plan("runtime_infrastructure"), question="Does representation reduce compute?", phase="explore")
        self.assertEqual("explore", result["workflow"]["phase"])
        self.assertEqual("explicit", result["workflow"]["phase_selection"])

    def test_question_cli_alias_and_overrides_reach_the_plan(self):
        args = knowledge._build_parser().parse_args([
            "diagnose", "--route", "l10-r0", "--question", "A new capability opportunity",
            "--phase", "explore", "--hypothesis", "Joint representation and policy",
            "--change", "Implement both coupled interfaces", "--metric", "Task gain and cost",
        ])
        plan = knowledge._build_minimum_experiment(
            {"failure_layers": [], "global_guardrails": ["UNKNOWN and consumed evidence remain scoped."]},
            args.route, args.symptom, {"layers": []}, [], [], phase=args.phase,
            hypothesis=args.hypothesis, change=args.change, metric=args.metric,
        )
        self.assertEqual(args.symptom, plan["workflow"]["question"])
        self.assertEqual(args.hypothesis, plan["hypothesis"])
        self.assertEqual(args.change, plan["single_change"])
        self.assertEqual(args.metric, plan["primary_metric"])
        self.assertEqual("proposal", plan["status"])
        self.assertIn("Development", plan["cohort"])
        self.assertIn("Development", plan["claim_ceiling"])
        self.assertNotIn("需要同时修改多个接口", plan["stop_conditions"])

    def test_updated_configuration_is_valid_without_changing_retrieval_schema(self):
        path = Path(__file__).resolve().parents[1] / "research/knowledge/decision/config.json"
        config = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual([], knowledge._validate_decision_config(config))
        self.assertEqual(13, len(config["failure_layers"]))

    def test_open_question_does_not_require_an_existing_reproducible_fault(self):
        for hypothesis in (None, "Joint observation and belief may reduce cost"):
            with self.subTest(hypothesis=hypothesis):
                plan = knowledge._build_minimum_experiment(
                    {"failure_layers": [], "global_guardrails": ["Preserve evidence scope."]},
                    "l10-r0", "Can we improve a new capability?", {"layers": []}, [], [],
                    hypothesis=hypothesis,
                )
                self.assertEqual("hypothesis_needed", plan["status"])
                self.assertNotIn("故障", plan["hypothesis"])
                self.assertNotIn("需要同时修改多个接口", plan["stop_conditions"])
                self.assertNotIn("故障不可复现", plan["not_evaluable_conditions"])
                self.assertTrue(knowledge._experiment_plan_is_valid(plan))

    def test_invalid_phase_and_blank_override_fail_before_execution(self):
        with self.assertRaises(ValueError):
            prepare_research_proposal(base_plan(), question="Question", phase="run-final")
        with self.assertRaises(ValueError):
            prepare_research_proposal(base_plan(), question="Question", change=" ")


if __name__ == "__main__":
    unittest.main()
