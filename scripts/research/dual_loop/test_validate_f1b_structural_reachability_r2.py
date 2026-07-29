from __future__ import annotations

import copy
import importlib.util
import json
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_f1b_structural_reachability_r2.py")
SPEC = importlib.util.spec_from_file_location(
    "validate_f1b_structural_reachability_r2", MODULE_PATH
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
R1_SPEC_PATH = (
    PROJECT_ROOT
    / "docs/research/dual-loop/F1B_STRUCTURAL_REACHABILITY_PROTOCOL_REPAIR_R1_SPEC_2026-07-30.json"
)
R2_SPEC_PATH = (
    PROJECT_ROOT
    / "docs/research/dual-loop/F1B_STRUCTURAL_REACHABILITY_PROTOCOL_REPAIR_R2_SPEC_2026-07-30.json"
)


def specs() -> tuple[dict, dict]:
    return (
        json.loads(R1_SPEC_PATH.read_text(encoding="utf-8")),
        json.loads(R2_SPEC_PATH.read_text(encoding="utf-8")),
    )


class StructuralReachabilityR2Test(unittest.TestCase):
    def test_corrected_table_has_no_internal_or_delivery_action(self) -> None:
        r1, r2 = specs()
        subject.require_exact_r2_repair(r2)
        effective = subject.effective_semantics(r1, r2)
        rows = subject.derive_state_table(effective)
        proofs, transition_equal, history_equal = subject.derive_proofs(
            effective, rows
        )
        self.assertEqual(19, len(rows))
        self.assertTrue(transition_equal)
        self.assertTrue(history_equal)
        self.assertTrue(all(not proof["reachable"] for proof in proofs.values()))
        self.assertFalse(any(row["fusion_action_reachable"] for row in rows))

    def test_center_mid_medium_cannot_change_internal_state(self) -> None:
        r1, r2 = specs()
        rows = subject.derive_state_table(subject.effective_semantics(r1, r2))
        center_mid = next(
            row
            for row in rows
            if row["state_id"] == "MID_MEDIUM_TEMPORAL_CENTER"
        )
        self.assertTrue(center_mid["geometry_attributable"])
        self.assertFalse(center_mid["a_deliverable_possible"])
        self.assertFalse(center_mid["fusion_action_reachable"])

    def test_side_near_temporal_remains_medium(self) -> None:
        r1, r2 = specs()
        rows = subject.derive_state_table(subject.effective_semantics(r1, r2))
        for state_id in (
            "NEAR_MEDIUM_TEMPORAL_LEFT",
            "NEAR_MEDIUM_TEMPORAL_RIGHT",
        ):
            row = next(row for row in rows if row["state_id"] == state_id)
            self.assertEqual("MEDIUM", row["level"])
            self.assertEqual(2, row["a_confirmation_frames"])
            self.assertFalse(row["geometry_attributable"])

    def test_removing_planner_precondition_breaks_history_equivalence(self) -> None:
        r1, r2 = specs()
        effective = subject.effective_semantics(r1, r2)
        effective["admissible_fusion"][
            "confirmation_substitution_requires_planner_eligible_pair"
        ] = False
        rows = subject.derive_state_table(effective)
        _, transition_equal, history_equal = subject.derive_proofs(effective, rows)
        center_mid = next(
            row
            for row in rows
            if row["state_id"] == "MID_MEDIUM_TEMPORAL_CENTER"
        )
        self.assertTrue(center_mid["fusion_action_reachable"])
        self.assertFalse(transition_equal)
        self.assertFalse(history_equal)

    def test_side_attribution_requires_empirical_ab(self) -> None:
        r1, r2 = specs()
        effective = subject.effective_semantics(r1, r2)
        effective["admissible_fusion"]["geometry_attributable_directions"] = [
            "CENTER",
            "LEFT",
        ]
        effective["geometry_information_semantics"][
            "left_right_attribution_forbidden"
        ] = False
        rows = subject.derive_state_table(effective)
        proofs, transition_equal, history_equal = subject.derive_proofs(
            effective, rows
        )
        left = next(row for row in rows if row["state_id"] == "NEAR_MEDIUM_LEFT")
        self.assertTrue(left["fusion_action_reachable"])
        self.assertTrue(left["b_can_advance_delivery"])
        self.assertTrue(proofs["EARLY_RESPONSE"]["reachable"])
        self.assertFalse(transition_equal)
        self.assertFalse(history_equal)

    def test_repair_fact_mutation_fails(self) -> None:
        _, r2 = specs()
        r2["repairs"][
            "confirmation_substitution_requires_planner_eligible_pair"
        ] = False
        with self.assertRaisesRegex(ValueError, "R2 repair semantics drifted"):
            subject.require_exact_r2_repair(r2)

    def test_inherited_binding_declaration_mutation_fails(self) -> None:
        _, r2 = specs()
        r2["inherited_bindings"][
            "contract_prerequisites_and_13_implementation_identities"
        ] = "PARTIAL"
        with self.assertRaisesRegex(ValueError, "inherited binding contract drifted"):
            subject.require_exact_r2_repair(r2)

    def test_exact_implementation_path_set_is_frozen(self) -> None:
        r1, _ = specs()
        subject.require_inherited_identity_sets(r1)
        r1["implementation_identities"] = [
            item
            for item in r1["implementation_identities"]
            if item["path"]
            != "core/assist/src/main/java/com/linnan/blindassist/risk/TemporalRiskTracker.kt"
        ]
        with self.assertRaisesRegex(
            ValueError, "implementation identity set drifted"
        ):
            subject.require_inherited_identity_sets(r1)

    def test_state_coverage_mutation_fails(self) -> None:
        r1, r2 = specs()
        r2["required_fresh_state_ids"].remove("NEAR_MEDIUM_TEMPORAL_LEFT")
        effective = subject.effective_semantics(r1, r2)
        with self.assertRaisesRegex(ValueError, "fresh-state coverage drifted"):
            subject.derive_state_table(effective)


if __name__ == "__main__":
    unittest.main()
