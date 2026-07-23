from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import exploratory_profiles_r2_l1 as r1
from r2_l1x_l2p import (
    LEGAL_TERMINAL_STATES,
    build_context,
    load_and_verify_prereg,
    load_json,
    validate_preoutput_freeze,
)


class R2L1XL2PContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo = Path(__file__).resolve().parents[3]
        cls.config_path = (
            cls.repo / "configs/ustrf_route_target_r2_l1x_l2p_prereg_r1.json"
        )
        cls.config = load_json(cls.config_path)

    def mutated_path(self, mutator) -> Path:
        payload = copy.deepcopy(self.config)
        mutator(payload)
        temporary = tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".json",
            delete=False,
        )
        json.dump(payload, temporary, ensure_ascii=False)
        temporary.close()
        self.addCleanup(lambda: Path(temporary.name).unlink(missing_ok=True))
        return Path(temporary.name)

    def assert_input_blocked(self, mutator, pattern: str) -> None:
        path = self.mutated_path(mutator)
        with self.assertRaisesRegex(r1.InputBlocked, pattern):
            load_and_verify_prereg(self.repo, path)

    def test_frozen_prereg_and_parent_bindings_validate(self) -> None:
        prereg, context = build_context(self.repo, self.config_path)
        self.assertEqual(prereg["stage"], "R2-L1X-L2P")
        self.assertEqual(set(prereg["terminal_states"]), LEGAL_TERMINAL_STATES)
        self.assertEqual(context["parent_summary"]["guard_attempt_count"], 3)
        self.assertFalse(
            context["parent_summary"]["old_attempts_count_toward_r2"]
        )
        self.assertEqual(len(context["groups"]), 41)
        self.assertEqual(sum(len(rows) for _, rows in context["groups"]), 62229)
        self.assertEqual(len(context["resets"]), 15)

    def test_l2_l3_contracts_are_valid_before_output(self) -> None:
        result = validate_preoutput_freeze(self.repo, self.config)
        self.assertEqual(result["decision"], "VALID_L2_L3_PREREG_R1")
        self.assertFalse(result["new_data_or_candidate_execution"])
        self.assertFalse(result["l3_executable"])
        self.assertIsNone(result["l3_candidate_id"])

    def test_six_gib_guard_cannot_be_lowered(self) -> None:
        self.assert_input_blocked(
            lambda value: value["execution_recovery"]["resource_guards"].__setitem__(
                "minimum_system_available_physical_memory_bytes", 1
            ),
            "six_gib_memory_guard_drift",
        )

    def test_old_attempts_cannot_count_toward_r2(self) -> None:
        self.assert_input_blocked(
            lambda value: value["execution_recovery"].__setitem__(
                "parent_r1_attempts_count_toward_this_stage", True
            ),
            "old_r1_attempts_reused",
        )

    def test_attempt_namespace_is_versioned(self) -> None:
        self.assert_input_blocked(
            lambda value: value["execution_recovery"].__setitem__(
                "attempt_namespace", "r2-l1e"
            ),
            "recovery_attempt_namespace_drift",
        )

    def test_retry_budget_cannot_expand(self) -> None:
        self.assert_input_blocked(
            lambda value: value["execution_recovery"].__setitem__(
                "bounded_retries", 3
            ),
            "recovery_attempt_budget_drift",
        )

    def test_authority_cannot_open(self) -> None:
        self.assert_input_blocked(
            lambda value: value["authority"].__setitem__(
                "candidate_selection", True
            ),
            "authority_opened",
        )

    def test_candidate_order_cannot_change(self) -> None:
        self.assert_input_blocked(
            lambda value: value["l1_profile"]["candidate_order"].reverse(),
            "candidate_order_drift",
        )

    def test_l2_binding_replacement_is_rejected(self) -> None:
        self.assert_input_blocked(
            lambda value: value["preoutput_frozen_contracts"][
                "l2_prereg"
            ].__setitem__("sha256", "0" * 64),
            "l2_prereg_sha256_mismatch",
        )

    def test_l3_binding_replacement_is_rejected(self) -> None:
        self.assert_input_blocked(
            lambda value: value["preoutput_frozen_contracts"][
                "l3_lockbox_template"
            ].__setitem__("sha256", "0" * 64),
            "l3_lockbox_template_sha256_mismatch",
        )

    def test_old_terminal_replacement_is_rejected(self) -> None:
        self.assert_input_blocked(
            lambda value: value["immutable_r1_parent"]["bindings"][
                "terminal_receipt"
            ].__setitem__("sha256", "0" * 64),
            "immutable_r1_terminal_receipt_sha256_mismatch",
        )

    def test_terminal_roster_has_no_fourth_state(self) -> None:
        self.assert_input_blocked(
            lambda value: value["terminal_states"].append("PARTIAL_SUCCESS"),
            "terminal_state_contract_drift",
        )


if __name__ == "__main__":
    unittest.main()

