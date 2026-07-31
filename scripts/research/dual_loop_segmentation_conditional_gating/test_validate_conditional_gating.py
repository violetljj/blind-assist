"""Focused tests for the independent conditional-gating validator."""

from __future__ import annotations

import unittest

from scripts.research.dual_loop_segmentation_conditional_gating.validate_conditional_gating import (
    _validate_component_decisions,
)


class ComponentDecisionValidationTest(unittest.TestCase):
    def test_reject_and_is_not_silently_changed_to_positive_and(self) -> None:
        config = {
            "candidate_order": ["CLASS_CONDITIONED_MULTI_NEGATIVE"],
            "forbidden_candidate_inputs": [
                "truth_mask",
                "false_activation",
                "session_id",
            ],
            "input_contract": {"expected_component_count": 1},
        }
        row = {
            "candidate_id": "CLASS_CONDITIONED_MULTI_NEGATIVE",
            "component_id": "c",
            "predicted_class": "obstacle",
            "raw_area_pixels": 4,
            "raw_pixels": 4,
            "causal_supported_pixels": 1,
            "noncausal_pixels": 3,
            "kept_pixels": 1,
            "rejected_pixels": 3,
            "low_confidence": True,
            "small_fragment": True,
            "intersects_upper_head_band": False,
            "action": "PARTIAL",
            "post_fragment_count": 1,
            "gate_input_fields": [
                "predicted_class",
                "raw_component_mask",
                "same_class_raw_history_masks",
                "top1_confidence_median",
                "raw_area_pixels",
                "upper_head_band_geometry",
                "frozen_thresholds",
            ],
        }
        checks, outcomes = _validate_component_decisions([row], config)
        self.assertGreater(checks, 0)
        self.assertEqual(outcomes["CLASS_CONDITIONED_MULTI_NEGATIVE"]["partially_retained"], 1)

    def test_truth_field_in_callable_contract_is_rejected(self) -> None:
        config = {
            "candidate_order": ["CLASS_CONDITIONED_MULTI_NEGATIVE"],
            "forbidden_candidate_inputs": ["truth_mask"],
            "input_contract": {"expected_component_count": 1},
        }
        row = {
            "candidate_id": "CLASS_CONDITIONED_MULTI_NEGATIVE",
            "component_id": "c",
            "predicted_class": "boundary_step_curb",
            "raw_area_pixels": 1,
            "raw_pixels": 1,
            "causal_supported_pixels": 0,
            "noncausal_pixels": 1,
            "kept_pixels": 1,
            "rejected_pixels": 0,
            "low_confidence": False,
            "small_fragment": True,
            "intersects_upper_head_band": False,
            "action": "KEEP",
            "post_fragment_count": 1,
            "gate_input_fields": ["truth_mask"],
        }
        with self.assertRaisesRegex(ValueError, "leaked forbidden"):
            _validate_component_decisions([row], config)


if __name__ == "__main__":
    unittest.main()
