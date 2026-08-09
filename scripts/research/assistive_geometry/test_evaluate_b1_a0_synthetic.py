from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.research.assistive_geometry.evaluate_b1_a0_synthetic import (
    EXPECTED_SEEDS,
    SYNTHETIC_ROLE,
    TERMINALS,
    run_package,
    sha256_file,
)
from scripts.research.assistive_geometry.run_b1_a0_evaluation_dry_run import (
    build_fixture,
    run_dry_run,
    synthetic_observations,
    write_json,
    write_jsonl,
)


def evaluation_protocol() -> dict:
    return {
        "schema": "blindassist_assistive_geometry_b1_a0_evaluation_dry_run_protocol_v1",
        "protocol_id": "TEST_A0_EVALUATION_DRY_RUN_PROTOCOL",
        "checkpoint_integrity": {
            "expected_optimizer_steps_by_epoch": {"5": 1499, "10": 2999, "15": 4499, "20": 6000},
        },
        "aggregation": {
            "required_seeds": [17, 29, 43],
            "selected_seed": None,
            "statistics": ["mean", "sample_std", "median", "min", "max"],
        },
        "metric_gates": {
            "known_coverage_min": 0.90,
            "clearance_mae_m_max": 0.20,
            "false_clear_all_known_max": 0.08,
            "false_block_given_clear_max": 0.02,
            "temporal_clearance_delta_mae_m_max": 0.15,
            "worst_parent_false_clear_all_known_max": 0.12,
        },
        "firewalls": {"dry_run_data_role": SYNTHETIC_ROLE, "development_outcome_access": False, "confirmation_outcome_access": False},
    }


class A0SyntheticEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.protocol_path = self.root / "evaluation_protocol.json"
        write_json(self.protocol_path, evaluation_protocol())

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_full_dry_run_exercises_pass_and_failure_terminals(self) -> None:
        result = run_dry_run(self.protocol_path, self.root / "dry-run")
        self.assertEqual("PASS", result["status"])
        self.assertEqual(7, result["scenario_count"])
        self.assertTrue(all(item["match"] for item in result["scenarios"]))
        for item in result["scenarios"]:
            if item["name"] != "pass":
                self.assertTrue(Path(item["failure_log_path"]).is_file())

        pass_result = json.loads((self.root / "dry-run" / "scenarios" / "pass" / "evaluation_result.json").read_text(encoding="utf-8"))
        self.assertEqual(12, pass_result["checkpoint_integrity"]["checkpoint_count"])
        self.assertEqual([17, 29, 43], pass_result["aggregate"]["seed_order"])
        self.assertIsNone(pass_result["aggregate"]["selected_seed"])
        self.assertTrue(pass_result["aggregate"]["best_seed_selection_forbidden"])
        self.assertFalse(pass_result["development_content_opened"])
        self.assertFalse(pass_result["confirmation_content_opened"])
        for seed_metrics in pass_result["seed_metrics"]:
            self.assertEqual(9, len(seed_metrics["by_grid"]))
            self.assertEqual(1, seed_metrics["pooled"]["unknown_truth_excluded_count"])

    def test_tampered_checkpoint_fails_before_metrics(self) -> None:
        package_path = build_fixture(self.root / "fixture", sha256_file(self.protocol_path), "pass")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        result_path = Path(package["seed_runs"][0]["train_result_path"])
        train_result = json.loads(result_path.read_text(encoding="utf-8"))
        checkpoint_path = Path(train_result["checkpoints"][0]["path"])
        with checkpoint_path.open("ab") as stream:
            stream.write(b"tamper")
        result = run_package(package_path, self.protocol_path, self.root / "tampered-output", expected_data_role=SYNTHETIC_ROLE)
        self.assertEqual(TERMINALS["CHECKPOINT"], result["terminal"])
        self.assertEqual("CHECKPOINT_BYTE_COUNT_MISMATCH", result["code"])

    def test_missing_seed_set_is_rejected_in_frozen_order_gate(self) -> None:
        package_path = build_fixture(self.root / "fixture", sha256_file(self.protocol_path), "pass")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        package["seed_runs"].pop()
        write_json(package_path, package)
        result = run_package(package_path, self.protocol_path, self.root / "missing-seed-output", expected_data_role=SYNTHETIC_ROLE)
        self.assertEqual(TERMINALS["SEEDS"], result["terminal"])
        self.assertEqual("SEED_SET_OR_ORDER_INVALID", result["code"])

    def test_unknown_truth_is_not_a_negative(self) -> None:
        package_path = build_fixture(self.root / "fixture", sha256_file(self.protocol_path), "pass")
        result = run_package(package_path, self.protocol_path, self.root / "pass-output", expected_data_role=SYNTHETIC_ROLE)
        for metrics in result["seed_metrics"]:
            self.assertEqual(1, metrics["pooled"]["unknown_truth_excluded_count"])
            self.assertEqual(71, metrics["pooled"]["truth_known_count"])
            self.assertEqual(0.0, metrics["pooled"]["false_clear_all_known"])

    def test_dry_run_protocol_rejects_development_role_before_evaluation(self) -> None:
        package_path = build_fixture(self.root / "fixture", sha256_file(self.protocol_path), "pass")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        package["data_role"] = "DEVELOPMENT_SELECTION"
        write_json(package_path, package)
        result = run_package(package_path, self.protocol_path, self.root / "development-blocked", expected_data_role="DEVELOPMENT_SELECTION")
        self.assertEqual(TERMINALS["DATA_ROLE"], result["terminal"])
        self.assertEqual("DEVELOPMENT_EVALUATION_NOT_ACTIVATED", result["code"])

    def test_false_block_task_gate_fails_closed(self) -> None:
        package_path = build_fixture(self.root / "fixture", sha256_file(self.protocol_path), "pass")
        package = json.loads(package_path.read_text(encoding="utf-8"))
        for seed in EXPECTED_SEEDS[:2]:
            observations_path = Path(next(item["observations_path"] for item in package["seed_runs"] if item["seed"] == seed))
            rows = synthetic_observations(seed)
            changed = False
            for row in rows:
                for band in row["bands"]:
                    for cell in band["cells"]:
                        if not changed and cell["truth_state"] == "CLEAR_OBSERVED":
                            cell["predicted_state"] = "OCCUPIED_OBSERVED"
                            changed = True
            write_jsonl(observations_path, rows)
        result = run_package(package_path, self.protocol_path, self.root / "task-fail-output", expected_data_role=SYNTHETIC_ROLE)
        self.assertEqual(TERMINALS["TASK"], result["terminal"])
        self.assertEqual("AGGREGATE_TASK_GATE_FAILED", result["code"])
        self.assertTrue((self.root / "task-fail-output" / "failure.log").is_file())


if __name__ == "__main__":
    unittest.main()
