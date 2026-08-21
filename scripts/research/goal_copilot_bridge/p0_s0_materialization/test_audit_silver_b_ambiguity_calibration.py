from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p0_s0_materialization import audit_silver_b_ambiguity_calibration as audit
from scripts.research.goal_copilot_bridge.p0_s0_materialization import run_silver_b_brain_baseline as runner


def episode(episode_id: str, resolution: str, parent: str) -> dict:
    return {
        "episode_id": episode_id,
        "evaluator_episode": {
            "goal_reference_resolution": resolution,
            "goal_spec": {"target_name": parent},
        },
    }


def report(policy_id: str, decisions: list[tuple[str, str]], correct: set[str]) -> dict:
    provider = {
        "executable": "codex.exe", "executable_sha256": "abc", "cli_version": "codex-cli 1.0.0",
        "model": "terra", "reasoning_effort": "medium",
    }
    return {
        "policy_id": policy_id,
        "cohort_report_sha256": "cohort-sha",
        "report_sha256": policy_id,
        "provider": provider,
        "raw_decisions": [{"episode_id": key, "action": action} for key, action in decisions],
        "frozen_evaluator": {"episodes": [{
            "episode_id": key,
            "brain_selection": {"top1_correct_given_available": True if key in correct else None},
            "end_to_end": {"outcome": "CORRECT_GROUNDING" if key in correct else "UNJUSTIFIED_GROUNDING"},
        } for key, _action in decisions]},
    }


class CalibrationAuditTest(unittest.TestCase):
    def test_reports_micro_macro_retention_and_refusal(self) -> None:
        cohort = {
            "report_sha256": "cohort-sha",
            "episodes": [
                episode("a1", "AMBIGUOUS", "venue-a"), episode("a2", "AMBIGUOUS", "venue-a"),
                episode("b1", "AMBIGUOUS", "venue-b"), episode("u1", "UNIQUE", "venue-u"),
            ],
        }
        baseline = report(runner.POLICY_ID, [("a1", "SELECT"), ("a2", "SELECT"), ("b1", "ABSTAIN"), ("u1", "SELECT")], {"u1"})
        candidate = report(runner.CALIBRATION_POLICY_ID, [("a1", "AMBIGUOUS"), ("a2", "SELECT"), ("b1", "AMBIGUOUS"), ("u1", "SELECT")], {"u1"})
        result = audit.audit(cohort, baseline, candidate)
        self.assertEqual(2 / 3, result["metrics"]["unsupported_commit_rate"]["baseline"]["value"])
        self.assertEqual(1 / 3, result["metrics"]["unsupported_commit_rate"]["candidate"]["value"])
        self.assertEqual(0.5, result["metrics"]["venue_parent_macro_unsupported_commit_rate"]["baseline"])
        self.assertEqual(0.25, result["metrics"]["venue_parent_macro_unsupported_commit_rate"]["candidate"])
        self.assertEqual(1.0, result["metrics"]["baseline_correct_grounding_retention"]["value"])
        self.assertEqual(0.0, result["metrics"]["unnecessary_unique_refusal_rate"]["value"])

    def test_rejects_provider_drift(self) -> None:
        cohort = {"report_sha256": "cohort-sha", "episodes": [episode("a1", "AMBIGUOUS", "venue-a")]}
        baseline = report(runner.POLICY_ID, [("a1", "SELECT")], set())
        candidate = report(runner.CALIBRATION_POLICY_ID, [("a1", "AMBIGUOUS")], set())
        candidate["provider"]["model"] = "different"
        with self.assertRaisesRegex(audit.CalibrationAuditError, "provider drift"):
            audit.audit(cohort, baseline, candidate)


if __name__ == "__main__":
    unittest.main()
