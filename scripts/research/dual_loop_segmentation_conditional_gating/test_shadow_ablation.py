"""Synthetic tests for the post-primary R0.1 shadow ablation."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    SHADOW_CLASS_TEMPORAL_ID,
    SHADOW_MULTI_NEGATIVE_ID,
    apply_frozen_candidate_to_component,
)
from scripts.research.dual_loop_segmentation_conditional_gating.shadow_ablation import (
    _strict_pareto_dominates_point,
    _validate_shadow_config_shape,
    assess_cross_session_heterogeneity,
    assess_shadow_material,
)


def apply_shadow(
    candidate_id: str,
    predicted_class: str,
    component: np.ndarray,
    causal: np.ndarray,
    confidence: float | None,
    upper: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    return apply_frozen_candidate_to_component(
        candidate_id=candidate_id,
        predicted_class=predicted_class,
        component_mask=component,
        same_class_causal_mask=causal,
        confidence_median=confidence,
        raw_area_pixels=int(component.sum()),
        upper_band_mask=np.zeros_like(component) if upper is None else upper,
        confidence_minimum=0.65,
        small_fragment_max_area_pixels=63,
    )


def decision(
    *,
    fp_pass: bool,
    overall_pass: bool = True,
    session_pass: bool = True,
    boundary_pass: bool = True,
    obstacle_pass: bool = True,
    fp_value: float = 0.31,
    recall_value: float = 0.91,
) -> dict:
    return {
        "checks": {
            "false_positive_reduction": {
                "passed": fp_pass,
                "value": fp_value,
            },
            "overall_recall_retention": {
                "passed": overall_pass,
                "value": recall_value,
            },
            "minimum_session_recall_retention": {
                "passed": session_pass,
                "value": 0.81 if session_pass else 0.79,
            },
            "boundary_step_curb_recall_retention": {
                "passed": boundary_pass,
                "value": 0.81 if boundary_pass else 0.79,
            },
            "obstacle_recall_retention": {
                "passed": obstacle_pass,
                "value": 0.81 if obstacle_pass else 0.79,
            },
        }
    }


class ShadowPredicateTest(unittest.TestCase):
    def test_class_temporal_obstacle_keeps_only_same_class_causal_pixels(self) -> None:
        component = np.array([[True, True, True]])
        causal = np.array([[False, True, False]])
        kept, evidence = apply_shadow(
            SHADOW_CLASS_TEMPORAL_ID,
            "obstacle",
            component,
            causal,
            confidence=0.99,
        )
        np.testing.assert_array_equal(kept, causal)
        self.assertEqual(evidence["rejected_pixels"], 2)

    def test_class_temporal_boundary_matches_primary_fragment_protection(self) -> None:
        component = np.array([[True, True]])
        kept, _ = apply_shadow(
            SHADOW_CLASS_TEMPORAL_ID,
            "boundary_step_curb",
            component,
            np.zeros_like(component),
            confidence=0.64,
        )
        self.assertFalse(kept.any())
        kept_at_boundary, _ = apply_shadow(
            SHADOW_CLASS_TEMPORAL_ID,
            "boundary_step_curb",
            component,
            np.zeros_like(component),
            confidence=0.65,
        )
        np.testing.assert_array_equal(kept_at_boundary, component)

    def test_multi_negative_applies_same_pixel_rule_to_both_classes(self) -> None:
        component = np.array([[True, True, True]])
        causal = np.array([[False, True, False]])
        for class_name in ("obstacle", "boundary_step_curb"):
            with self.subTest(class_name=class_name):
                kept, _ = apply_shadow(
                    SHADOW_MULTI_NEGATIVE_ID,
                    class_name,
                    component,
                    causal,
                    confidence=0.64,
                )
                np.testing.assert_array_equal(kept, causal)

    def test_multi_negative_missing_confidence_keeps_component(self) -> None:
        component = np.array([[True, True]])
        kept, evidence = apply_shadow(
            SHADOW_MULTI_NEGATIVE_ID,
            "boundary_step_curb",
            component,
            np.zeros_like(component),
            confidence=None,
        )
        np.testing.assert_array_equal(kept, component)
        self.assertFalse(evidence["confidence_known"])


class ShadowConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.config = json.loads(
            (
                repo_root
                / "configs"
                / "dual_loop_segmentation_conditional_gating_r0_1"
                / "shadow.json"
            ).read_text(encoding="utf-8")
        )

    def test_frozen_shadow_config_is_accepted(self) -> None:
        _validate_shadow_config_shape(self.config)

    def test_shadow_reordering_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.config))
        mutated["shadow_candidate_order"].reverse()
        with self.assertRaisesRegex(ValueError, "exact two frozen arms"):
            _validate_shadow_config_shape(mutated)

    def test_shadow_cannot_gain_selection_authority(self) -> None:
        mutated = json.loads(json.dumps(self.config))
        mutated["material_rule"]["mutual_shadow_selection"] = True
        with self.assertRaisesRegex(ValueError, "authority drifted"):
            _validate_shadow_config_shape(mutated)


class ShadowInterpretationTest(unittest.TestCase):
    def test_material_requires_recall_guards_and_fp_or_fixed_frontier_increment(self) -> None:
        references = {
            "r": {
                "false_positive_reduction": 0.20,
                "recall_retention": 0.90,
            }
        }
        full = assess_shadow_material(
            shadow_decision=decision(fp_pass=True),
            reference_points=references,
        )
        self.assertTrue(full["MATERIAL"])
        low_session = assess_shadow_material(
            shadow_decision=decision(fp_pass=True, session_pass=False),
            reference_points=references,
        )
        self.assertFalse(low_session["MATERIAL"])

    def test_strict_pareto_requires_weak_both_and_strict_one(self) -> None:
        point = {"false_positive_reduction": 0.2, "recall_retention": 0.9}
        self.assertFalse(_strict_pareto_dominates_point(point, dict(point)))
        self.assertTrue(
            _strict_pareto_dominates_point(
                {"false_positive_reduction": 0.21, "recall_retention": 0.9},
                point,
            )
        )

    def test_cross_session_inversion_is_heterogeneous_not_selection(self) -> None:
        reports = {
            SHADOW_CLASS_TEMPORAL_ID: {
                "by_session_id": {
                    "a": {
                        "comparison_to_baseline": {
                            "false_positive_reduction": 0.3,
                            "recall_retention": 0.9,
                        }
                    },
                    "b": {
                        "comparison_to_baseline": {
                            "false_positive_reduction": 0.1,
                            "recall_retention": 0.8,
                        }
                    },
                }
            },
            SHADOW_MULTI_NEGATIVE_ID: {
                "by_session_id": {
                    "a": {
                        "comparison_to_baseline": {
                            "false_positive_reduction": 0.2,
                            "recall_retention": 0.8,
                        }
                    },
                    "b": {
                        "comparison_to_baseline": {
                            "false_positive_reduction": 0.2,
                            "recall_retention": 0.9,
                        }
                    },
                }
            },
        }
        result = assess_cross_session_heterogeneity(reports=reports)
        self.assertTrue(result["H_cross"])
        self.assertFalse(result["may_select_shadow"])


if __name__ == "__main__":
    unittest.main()
