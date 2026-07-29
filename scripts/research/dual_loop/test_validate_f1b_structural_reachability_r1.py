from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_f1b_structural_reachability_r1.py")
SPEC = importlib.util.spec_from_file_location(
    "validate_f1b_structural_reachability_r1", MODULE_PATH
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
FORMAL_SPEC_PATH = (
    PROJECT_ROOT
    / "docs/research/dual-loop/F1B_STRUCTURAL_REACHABILITY_PROTOCOL_REPAIR_R1_SPEC_2026-07-30.json"
)


def formal_spec() -> dict:
    return json.loads(FORMAL_SPEC_PATH.read_text(encoding="utf-8"))


class StructuralReachabilityR1Test(unittest.TestCase):
    def test_derives_complete_fresh_state_table_and_zero_endpoint_reachability(self) -> None:
        spec = formal_spec()
        subject.require_exact_semantics(spec)
        rows = subject.derive_fresh_state_table(spec)
        proofs, history_equal = subject.derive_endpoint_proofs(spec, rows)
        self.assertEqual(19, len(rows))
        self.assertTrue(history_equal)
        self.assertTrue(all(not proof["reachable"] for proof in proofs.values()))
        side = next(row for row in rows if row["state_id"] == "NEAR_MEDIUM_LEFT")
        self.assertTrue(side["a_deliverable_possible"])
        self.assertEqual(2, side["a_confirmation_frames"])
        self.assertFalse(side["geometry_attributable"])
        self.assertFalse(side["b_can_advance_delivery"])
        center = next(row for row in rows if row["state_id"] == "NEAR_HIGH_CENTER")
        self.assertEqual(1, center["a_confirmation_frames"])
        self.assertTrue(center["geometry_attributable"])
        self.assertFalse(center["b_can_advance_delivery"])

    def test_planner_pair_mutation_fails(self) -> None:
        spec = formal_spec()
        spec["production_semantics"]["planner_alertable_pairs"] = ["MID_LOW"]
        with self.assertRaisesRegex(ValueError, "production semantics drifted"):
            subject.require_exact_semantics(spec)

    def test_side_near_direction_or_confirmation_mutation_fails(self) -> None:
        spec = formal_spec()
        spec["production_semantics"]["near_level_by_direction"]["LEFT"] = "HIGH"
        spec["production_semantics"]["medium_confirm_frames"] = 99
        with self.assertRaisesRegex(ValueError, "production semantics drifted"):
            subject.require_exact_semantics(spec)

    def test_permitted_action_contradiction_fails(self) -> None:
        spec = formal_spec()
        spec["admissible_fusion"]["only_permitted_action"] = "PROMOTE_TO_NEAR"
        spec["admissible_fusion"]["may_change_proximity"] = True
        with self.assertRaisesRegex(ValueError, "admissible fusion drifted"):
            subject.require_exact_semantics(spec)

    def test_detection_source_mutation_fails(self) -> None:
        spec = formal_spec()
        spec["production_semantics"]["detection_source_remains_object_detector"] = False
        with self.assertRaisesRegex(ValueError, "production semantics drifted"):
            subject.require_exact_semantics(spec)

    def test_cooldown_or_fatigue_mutation_fails(self) -> None:
        spec = formal_spec()
        spec["production_semantics"]["near_cooldown_ms"] = 0
        spec["production_semantics"]["fatigue_window_ms"] = 0
        with self.assertRaisesRegex(ValueError, "production semantics drifted"):
            subject.require_exact_semantics(spec)

    def test_secondary_endpoint_omission_fails(self) -> None:
        spec = formal_spec()
        spec["required_endpoint_proofs"].remove("RISK_CONTINUITY")
        with self.assertRaisesRegex(ValueError, "secondary endpoint"):
            subject.require_exact_semantics(spec)

    def test_state_coverage_mutation_fails(self) -> None:
        spec = formal_spec()
        spec["required_fresh_state_ids"].remove("MID_MEDIUM_TEMPORAL_CENTER")
        subject.require_exact_semantics(spec)
        with self.assertRaisesRegex(ValueError, "fresh-state coverage drifted"):
            subject.derive_fresh_state_table(spec)

    def test_decision_access_remains_declaration_and_consumption_fails(self) -> None:
        spec = formal_spec()
        spec["decision_output_declaration"]["machine_verified"] = True
        with self.assertRaisesRegex(ValueError, "protocol declaration"):
            subject.require_exact_semantics(spec)
        spec = formal_spec()
        spec["decision_output_declaration"]["decision_sessions_consumed"] = 1
        with self.assertRaisesRegex(ValueError, "declared consumed"):
            subject.require_exact_semantics(spec)


if __name__ == "__main__":
    unittest.main()
