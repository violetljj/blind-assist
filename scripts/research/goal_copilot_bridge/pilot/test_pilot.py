from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

from scripts.research.goal_copilot_bridge.pilot.evaluator import (
    CandidateContractError, DEV_SCENARIOS, evaluate_scenarios, load_candidate,
)


class PilotTest(unittest.TestCase):
    def test_dev_shape_and_baseline_calibration_gate(self) -> None:
        scenarios = json.loads(DEV_SCENARIOS.read_text())["scenarios"]
        self.assertEqual(12, len(scenarios))
        self.assertEqual({4}, {
            sum(item["task_family"] == family for item in scenarios)
            for family in {item["task_family"] for item in scenarios}
        })
        result = evaluate_scenarios(HERE / "initial_policy.py", DEV_SCENARIOS)
        metrics = result["metrics"]
        self.assertTrue(metrics["hard_gate_pass"])
        self.assertEqual(0, metrics["unsafe_guidance"])
        self.assertEqual(0, metrics["premature_completion"])
        self.assertGreaterEqual(metrics["completion_count"], 4)
        self.assertLessEqual(metrics["completion_count"], 9)
        for count in metrics["family_completion_counts"].values():
            self.assertGreaterEqual(count, 1)
            self.assertLessEqual(count, 3)

    def test_fresh_manifest_is_encrypted_and_balanced_without_truth_access(self) -> None:
        manifest = json.loads((HERE / "fresh_cohort_manifest.json").read_text())
        envelope = json.loads((HERE / "fresh_scenarios.enc.json").read_text())
        self.assertEqual("SEALED_ENCRYPTED_NOT_EXPOSED_TO_SEARCH", manifest["status"])
        self.assertEqual(6, manifest["scenario_count"])
        self.assertEqual({2}, set(manifest["family_counts"].values()))
        self.assertIn("ciphertext_b64", envelope)
        self.assertNotIn("scenarios", envelope)

    def test_action_changes_hidden_state(self) -> None:
        scenario = json.loads(DEV_SCENARIOS.read_text())["scenarios"][0]
        self.assertEqual("SCAN_LEFT", scenario["steps"][0]["expected_action"])
        result = evaluate_scenarios(HERE / "initial_policy.py", DEV_SCENARIOS)
        outcome = next(item for item in result["outcomes"] if item["scenario_id"] == scenario["id"])
        self.assertTrue(outcome["goal_completion"])
        self.assertEqual(4, outcome["total_actions"])

    def test_malicious_and_unbounded_surfaces_fail_closed(self) -> None:
        attacks = {
            "import": "import os\n",
            "call": "x = open('x')\n",
            "loop": "while True:\n    pass\n",
            "dunder": "x = ().__class__\n",
            "subprocess": "import subprocess\nsubprocess.run('x')\n",
        }
        baseline = (HERE / "initial_policy.py").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            for name, attack in attacks.items():
                path = Path(tmp) / f"{name}.py"
                path.write_text(attack + baseline)
                with self.subTest(name=name), self.assertRaises(CandidateContractError):
                    load_candidate(path)

    def test_replay_is_byte_deterministic(self) -> None:
        first = evaluate_scenarios(HERE / "initial_policy.py", DEV_SCENARIOS)
        second = evaluate_scenarios(HERE / "initial_policy.py", DEV_SCENARIOS)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
