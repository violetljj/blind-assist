from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
PROTOCOL_PATH = (
    REPO_ROOT
    / "docs/research/hftf/"
    "HFTF_STAGE_B_SPLIT_SOURCE_VALIDATION_R4_2026-08-01.json"
)
sys.path.insert(0, str(SCRIPT_DIR))

from run_r4_analytic_terrain import (  # noqa: E402
    TERRAIN_SUPPORTED,
    UNKNOWN,
    _expand_cases,
    _load_json,
    run,
)


class R4AnalyticTerrainTest(unittest.TestCase):
    def _protocol_fixture(
        self,
        root: Path,
        mutate: Callable[[dict[str, Any]], None] | None = None,
    ) -> Path:
        protocol = _load_json(PROTOCOL_PATH)
        if mutate is not None:
            mutate(protocol)
        parent_name = str(protocol["parent_result_path"])
        source_parent = PROTOCOL_PATH.parent / parent_name
        (root / parent_name).write_bytes(source_parent.read_bytes())
        target = root / PROTOCOL_PATH.name
        target.write_text(
            json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return target

    def test_exact_42_case_expansion_and_profile_rotation(self) -> None:
        protocol = _load_json(PROTOCOL_PATH)
        cases = _expand_cases(protocol["terrain_source_role"])
        self.assertEqual(42, len(cases))
        counts: dict[str, int] = {}
        for case in cases:
            counts[case["family"]] = counts.get(case["family"], 0) + 1
        self.assertEqual(
            {
                item["name"]: item["count"]
                for item in protocol["terrain_source_role"][
                    "scenario_families"
                ]
            },
            counts,
        )
        rises = [
            case
            for case in cases
            if case["family"] == "hazardous_rise"
        ]
        self.assertEqual([0, 1, 2, 0, 1, 2], [
            case["profile_index"] for case in rises
        ])

    def test_full_frozen_expected_pass(self) -> None:
        report = run(PROTOCOL_PATH)
        self.assertEqual(TERRAIN_SUPPORTED, report["terminal"])
        self.assertEqual(42, report["scenario_count"])
        self.assertEqual(36, report["known_scenario_count"])
        self.assertEqual(6, report["unknown_scenario_count"])
        self.assertTrue(report["all_terrain_gates_passed"])
        self.assertTrue(
            all(item["passed"] for item in report["ordered_gates"])
        )
        metrics = report["candidate"]["metrics"]
        for name in ("precision", "recall", "f1", "specificity"):
            self.assertEqual(1.0, metrics[name])
        self.assertGreaterEqual(
            report["candidate_f1_delta_over_best_baseline"], 0.15
        )
        self.assertFalse(report["stage_c_protocol_freeze_authorized"])
        self.assertFalse(report["stage_c_execution_authorized"])
        self.assertFalse(report["student_training_authorized"])

    def test_unknown_firewall_abstains_instead_of_safe(self) -> None:
        report = run(PROTOCOL_PATH)
        unknown_cases = [
            case for case in report["cases"] if case["truth"] == UNKNOWN
        ]
        self.assertEqual(6, len(unknown_cases))
        self.assertTrue(
            all(case["candidate"] == UNKNOWN for case in unknown_cases)
        )
        metrics = report["candidate"]["metrics"]
        self.assertEqual(1.0, metrics["unknown_abstention_rate"])
        self.assertEqual(0, metrics["unknown_to_safe_violations"])

    def test_protocol_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._protocol_fixture(
                Path(temp_dir),
                lambda protocol: protocol.__setitem__(
                    "status", "OUTCOME_OPEN"
                ),
            )
            with self.assertRaisesRegex(ValueError, "not frozen"):
                run(path)

    def test_parent_hash_drift_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = self._protocol_fixture(
                Path(temp_dir),
                lambda protocol: protocol.__setitem__(
                    "parent_result_sha256", "0" * 64
                ),
            )
            with self.assertRaisesRegex(ValueError, "parent result hash"):
                run(path)

    def test_truth_mismatch_fails_closed_before_sampling(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            def mutate(protocol: dict[str, Any]) -> None:
                families = protocol["terrain_source_role"][
                    "scenario_families"
                ]
                families[0]["truth"] = "RISK"

            path = self._protocol_fixture(Path(temp_dir), mutate)
            with self.assertRaisesRegex(ValueError, "Protocol truth mismatch"):
                run(path)


if __name__ == "__main__":
    unittest.main()
