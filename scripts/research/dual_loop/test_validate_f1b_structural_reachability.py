from __future__ import annotations

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("validate_f1b_structural_reachability.py")
SPEC = importlib.util.spec_from_file_location("validate_f1b_structural_reachability", MODULE_PATH)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def minimal_spec() -> dict:
    return {
        "decision_output_access": {
            "yolo_executed_on_decision_sessions": False,
            "sparse_lk_executed_on_decision_sessions": False,
            "ab_output_viewed_on_decision_sessions": False,
        },
        "frozen_semantics": {
            "geometry_has_target_identity": False,
            "geometry_has_left_center_right_region": False,
            "geometry_has_approach_direction": False,
            "geometry_has_radial_expansion": False,
            "geometry_has_ttc": False,
            "only_defensible_geometry_region": "CENTER_CORRIDOR_GLOBAL",
            "center_near_level": "HIGH",
            "center_critical_level": "HIGH",
            "side_near_level": "MEDIUM",
            "medium_confirm_frames": 2,
            "high_confirm_frames": 1,
        },
        "maximally_thin_admissible_fusion": {
            "may_change_label": False,
            "may_change_direction": False,
            "may_change_proximity": False,
            "may_change_risk_level": False,
            "may_create_alert_from_no_semantic_candidate": False,
            "may_hold_missing_semantic_candidate": False,
            "may_bypass_production_cooldown_or_fatigue": False,
            "region_attribution_rule": (
                "CENTER only; LEFT and RIGHT abstain because the existing vector is global "
                "center-corridor evidence"
            ),
            "abstention_falls_back_to_a": True,
        },
        "structural_truth_table": [
            {"semantic_state": state, "b_permitted_difference": False}
            for state in (
                "NO_CANDIDATE",
                "FAR_NONE",
                "MID_LOW",
                "NEAR_HIGH",
                "CRITICAL_HIGH",
                "NEAR_MEDIUM",
            )
        ],
        "dominance_stop_rule": {
            "lead_upper_bound_frames": 0,
            "terminal": "NO_INCREMENT",
            "science_protocol_status": "VALID",
            "decision_execution_required": False,
            "f1c_authorized": False,
            "paper_claim_stops": True,
        },
    }


class StructuralReachabilityTest(unittest.TestCase):
    def test_no_admissible_row_stops_without_decision_execution(self) -> None:
        result = subject.analyze_reachability(minimal_spec())
        self.assertEqual("NO_INCREMENT", result["terminal"])
        self.assertEqual(0, result["lead_upper_bound_frames"])
        self.assertEqual([], result["reachable_b_advance_rows"])

    def test_any_permitted_advance_requires_empirical_ab(self) -> None:
        spec = minimal_spec()
        spec["structural_truth_table"][-1]["b_permitted_difference"] = True
        spec["dominance_stop_rule"]["lead_upper_bound_frames"] = 1
        spec["dominance_stop_rule"]["terminal"] = "REQUIRES_EMPIRICAL_AB"
        spec["dominance_stop_rule"]["decision_execution_required"] = True
        spec["dominance_stop_rule"]["f1c_authorized"] = False
        spec["dominance_stop_rule"]["paper_claim_stops"] = False
        result = subject.analyze_reachability(spec)
        self.assertEqual("REQUIRES_EMPIRICAL_AB", result["terminal"])
        self.assertEqual(1, result["lead_upper_bound_frames"])

    def test_forbidden_risk_promotion_is_rejected(self) -> None:
        spec = minimal_spec()
        spec["maximally_thin_admissible_fusion"]["may_change_proximity"] = True
        with self.assertRaisesRegex(ValueError, "exceeds the frozen thin boundary"):
            subject.analyze_reachability(spec)

    def test_decision_output_access_is_rejected(self) -> None:
        spec = minimal_spec()
        spec["decision_output_access"]["yolo_executed_on_decision_sessions"] = True
        with self.assertRaisesRegex(ValueError, "unconsumed"):
            subject.analyze_reachability(spec)

    def test_bound_file_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "bound.txt"
            source.write_text("actual", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "hash drift"):
                subject.validate_bound_file(
                    root,
                    {"path": "bound.txt", "sha256": "0" * 64},
                )


if __name__ == "__main__":
    unittest.main()
