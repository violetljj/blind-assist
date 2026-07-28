from __future__ import annotations

import importlib.util
import copy
import json
import tempfile
import unittest
from pathlib import Path


REPO = Path(__file__).resolve().parents[4]
MODULE_PATH = (
    REPO
    / "scripts"
    / "research"
    / "egomotion_compensated_looming"
    / "rgb_segment_confirmation_r2"
    / "validate_repair_contract.py"
)
CONTRACT_PATH = (
    REPO
    / "docs"
    / "research"
    / "rcle"
    / "RCLE_RGB_SEGMENT_CONFIRMATION_R2_TRANSPORT_REPAIR_CONTRACT_2026-07-28.json"
)


def load_module():
    spec = importlib.util.spec_from_file_location("validate_repair_contract", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RepairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_module()
        self.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))

    def validate_mutation(self, mutate) -> dict:
        changed = copy.deepcopy(self.contract)
        mutate(changed)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mutated.json"
            path.write_text(
                json.dumps(changed, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            return self.module.validate(REPO, path)

    def test_frozen_contract_passes(self) -> None:
        result = self.module.validate(REPO, CONTRACT_PATH)
        self.assertEqual(result["decision"], "PASS")
        self.assertEqual(result["pass_count"], result["check_count"])

    def test_network_and_downstream_authority_are_false(self) -> None:
        result = self.module.validate(REPO, CONTRACT_PATH)
        passed = {
            row["check_id"] for row in result["checks"] if row["status"] == "PASS"
        }
        self.assertIn("authority_false:real_source_network_access", passed)
        self.assertIn("authority_false:rgb_algorithm_execution", passed)
        self.assertIn("authority_false:android", passed)

    def test_terminal_mutation_fails(self) -> None:
        result = self.validate_mutation(
            lambda value: value.__setitem__(
                "current_terminal", "DESIGN_REVIEW_PASS_EXECUTION_NOT_AUTHORIZED"
            )
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_segment_window_mutation_fails(self) -> None:
        result = self.validate_mutation(
            lambda value: value["scope_lock"]["segments"][1].__setitem__(
                "half_open_window_s",
                ["1634201323.0", "1634201334.0"],
            )
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_dlr_cap_mutation_fails(self) -> None:
        result = self.validate_mutation(
            lambda value: value["r1_diagnosis"]["dlr"][
                "minimum_no_retry_full_member_hard_cap"
            ].__setitem__("hard_cap_if_later_activated", 9_999_999_999)
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_openloris_cap_permission_mutation_fails(self) -> None:
        result = self.validate_mutation(
            lambda value: value["r1_diagnosis"]["openloris"][
                "frozen_transport_limits"
            ].__setitem__("cap_increase_allowed", True)
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_added_true_authority_fails(self) -> None:
        result = self.validate_mutation(
            lambda value: value["execution_authority"].__setitem__(
                "unreviewed_extra_authority", True
            )
        )
        self.assertEqual(result["decision"], "FAIL")

    def test_removed_activation_separation_fails(self) -> None:
        def mutate(value) -> None:
            value["future_activation_requirements"]["dlr"] = [
                row
                for row in value["future_activation_requirements"]["dlr"]
                if "index claim and a later exact-window extraction claim" not in row
            ]

        result = self.validate_mutation(mutate)
        self.assertEqual(result["decision"], "FAIL")


if __name__ == "__main__":
    unittest.main()
