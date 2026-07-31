"""Synthetic tests for the independent R0.1 shadow validator."""

from __future__ import annotations

import unittest

from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    SHADOW_CLASS_TEMPORAL_ID,
    SHADOW_MULTI_NEGATIVE_ID,
)
from scripts.research.dual_loop_segmentation_conditional_gating.validate_shadow_ablation import (
    _dominates,
    _expected_rejected,
    _material_assessment,
)


def component_row(
    candidate_id: str,
    predicted_class: str,
    *,
    raw: int = 4,
    noncausal: int = 3,
    low: bool = True,
    small: bool = True,
    upper: bool = False,
) -> dict:
    return {
        "candidate_id": candidate_id,
        "predicted_class": predicted_class,
        "raw_pixels": raw,
        "noncausal_pixels": noncausal,
        "low_confidence": low,
        "small_fragment": small,
        "intersects_upper_head_band": upper,
    }


def decision(
    *,
    fp: bool,
    overall: bool,
    session: bool,
    boundary: bool,
    obstacle: bool,
) -> dict:
    return {
        "checks": {
            "false_positive_reduction": {"passed": fp},
            "overall_recall_retention": {"passed": overall},
            "minimum_session_recall_retention": {"passed": session},
            "boundary_step_curb_recall_retention": {"passed": boundary},
            "obstacle_recall_retention": {"passed": obstacle},
        }
    }


class IndependentPredicateTest(unittest.TestCase):
    def test_class_temporal_obstacle_rejects_every_noncausal_pixel(self) -> None:
        row = component_row(
            SHADOW_CLASS_TEMPORAL_ID,
            "obstacle",
            low=False,
            small=False,
        )
        self.assertEqual(_expected_rejected(row), 3)

    def test_class_temporal_boundary_only_rejects_low_small_component(self) -> None:
        row = component_row(
            SHADOW_CLASS_TEMPORAL_ID,
            "boundary_step_curb",
        )
        self.assertEqual(_expected_rejected(row), 4)
        row["low_confidence"] = False
        self.assertEqual(_expected_rejected(row), 0)

    def test_multi_negative_requires_all_negative_evidence(self) -> None:
        row = component_row(SHADOW_MULTI_NEGATIVE_ID, "boundary_step_curb")
        self.assertEqual(_expected_rejected(row), 3)
        row["low_confidence"] = False
        self.assertEqual(_expected_rejected(row), 0)


class IndependentInterpretationTest(unittest.TestCase):
    def test_dominance_does_not_accept_equality(self) -> None:
        point = {"false_positive_reduction": 0.2, "recall_retention": 0.9}
        self.assertFalse(_dominates(point, dict(point)))

    def test_material_rule_cannot_ignore_minimum_session_failure(self) -> None:
        result = _material_assessment(
            decision=decision(
                fp=True,
                overall=True,
                session=False,
                boundary=True,
                obstacle=True,
            ),
            point={"false_positive_reduction": 0.31, "recall_retention": 0.91},
            references={
                "r": {
                    "false_positive_reduction": 0.20,
                    "recall_retention": 0.90,
                }
            },
        )
        self.assertFalse(result["MATERIAL"])
        self.assertTrue(result["H_min"])

    def test_frontier_increment_is_evaluated_against_fixed_reference(self) -> None:
        result = _material_assessment(
            decision=decision(
                fp=False,
                overall=True,
                session=True,
                boundary=True,
                obstacle=True,
            ),
            point={"false_positive_reduction": 0.25, "recall_retention": 0.95},
            references={
                "r": {
                    "false_positive_reduction": 0.20,
                    "recall_retention": 0.90,
                }
            },
        )
        self.assertTrue(result["N_frontier_and_dominates_reference"])
        self.assertTrue(result["MATERIAL"])
        self.assertFalse(result["may_rewrite_primary"])


if __name__ == "__main__":
    unittest.main()
