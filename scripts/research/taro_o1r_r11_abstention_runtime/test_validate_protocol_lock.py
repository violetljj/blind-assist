from __future__ import annotations

import copy
import unittest
from pathlib import Path

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r11_abstention_runtime import validate_protocol_lock as validator


ROOT = Path(__file__).resolve().parents[3]


def reseal(value: dict) -> dict:
    value = copy.deepcopy(value)
    value.pop("content_sha256", None)
    value["content_sha256"] = adapter.canonical_sha256(value)
    return value


class ProtocolLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = validator._read_json(ROOT / validator.PROTOCOL_RELATIVE)

    def test_protocol_and_exact_fresh_pool_recompute(self) -> None:
        record = validator.validate_protocol(self.protocol, repo_root=ROOT)
        self.assertEqual(record["scientific_outcome"], "NOT_RUN")
        self.assertFalse(record["execution_authority"]["data_use_authorized"])

    def test_formation_or_identity_exception_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["formation"]["role"] = "SEALED_UNSEEN"
        with self.assertRaisesRegex(validator.ProtocolLockError, "formation role"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["formation"]["identity_specific_exception_allowed"] = True
        with self.assertRaises(validator.ProtocolLockError):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)

    def test_margin_rule_or_r9_authority_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["candidate"]["stronger_cells_any"][0]["minimum_connected_confidence2_pixels"] = 8
        with self.assertRaisesRegex(validator.ProtocolLockError, "margin rule"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["predecessors"]["r9_selector"]["query_label_or_abstention_authority"] = True
        with self.assertRaisesRegex(validator.ProtocolLockError, "R9 selector"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)

    def test_r10_full_pool_exclusion_and_roster_are_frozen(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["predecessors"]["r10"]["all_32_source_pool_parents_excluded_from_r11"] = False
        with self.assertRaisesRegex(validator.ProtocolLockError, "full-pool exclusion"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["fresh_frontdoor"]["selected_parent_count"] = 8
        with self.assertRaisesRegex(validator.ProtocolLockError, "fresh roster"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)

    def test_cluster_gate_unknown_or_terminal_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["dual_class_evaluability"]["minimum_physical_frames_with_definite_clear"] = 4
        with self.assertRaisesRegex(validator.ProtocolLockError, "evaluability"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["confirmation_gates"]["unknown_is_negative"] = True
        with self.assertRaisesRegex(validator.ProtocolLockError, "confirmation gate"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["terminal_precedence"][0] = "WILD_LAB_RESEARCH_FACTOR_CONFIRMATION_PASS"
        with self.assertRaisesRegex(validator.ProtocolLockError, "terminal precedence"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)

    def test_execution_or_binding_mutation_fails_closed(self) -> None:
        mutated = copy.deepcopy(self.protocol)
        mutated["execution_authority"]["network"] = True
        with self.assertRaisesRegex(validator.ProtocolLockError, "execution"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)
        mutated = copy.deepcopy(self.protocol)
        mutated["bindings"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(validator.ProtocolLockError, "binding hash"):
            validator.validate_protocol(reseal(mutated), repo_root=ROOT, recompute_pool=False)


if __name__ == "__main__":
    unittest.main()
