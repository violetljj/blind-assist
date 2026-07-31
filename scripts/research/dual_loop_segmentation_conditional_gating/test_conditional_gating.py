"""Unit tests for the frozen conditional-gating protocol."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    _candidate_definition_hash,
    _validate_membership,
    _validate_static_config,
    apply_frozen_candidate_to_component,
    causal_two_of_three,
    decode_packed_mask,
    encode_packed_mask,
    upper_head_band,
)


CANDIDATE_ID = "CLASS_CONDITIONED_MULTI_NEGATIVE"


def apply_candidate(
    *,
    predicted_class: str,
    component_mask: np.ndarray,
    causal_mask: np.ndarray,
    confidence: float | None,
    raw_area: int | None = None,
    upper_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    return apply_frozen_candidate_to_component(
        candidate_id=CANDIDATE_ID,
        predicted_class=predicted_class,
        component_mask=component_mask,
        same_class_causal_mask=causal_mask,
        confidence_median=confidence,
        raw_area_pixels=int(component_mask.sum()) if raw_area is None else raw_area,
        upper_band_mask=(
            np.zeros_like(component_mask)
            if upper_mask is None
            else upper_mask
        ),
        confidence_minimum=0.65,
        small_fragment_max_area_pixels=63,
    )


class PackedMaskTest(unittest.TestCase):
    def test_round_trip_preserves_non_byte_aligned_mask(self) -> None:
        source = np.zeros((3, 5), dtype=bool)
        source[0, 1] = True
        source[2, 4] = True
        np.testing.assert_array_equal(
            decode_packed_mask(encode_packed_mask(source), source.shape),
            source,
        )


class TemporalContractTest(unittest.TestCase):
    def test_first_observation_has_no_support(self) -> None:
        current = np.array([[True, False, True]])
        np.testing.assert_array_equal(
            causal_two_of_three(current, None, None),
            np.zeros_like(current),
        )

    def test_second_observation_uses_only_raw_previous(self) -> None:
        current = np.array([[True, True, False]])
        previous = np.array([[False, True, True]])
        np.testing.assert_array_equal(
            causal_two_of_three(current, previous, None),
            np.array([[False, True, False]]),
        )

    def test_cross_class_history_cannot_rescue_obstacle(self) -> None:
        current_obstacle = np.array([[True, False]])
        previous_boundary = np.array([[True, False]])
        obstacle_support = causal_two_of_three(current_obstacle, None, None)
        union_support = causal_two_of_three(
            current_obstacle, previous_boundary, None
        )
        self.assertFalse(obstacle_support.any())
        self.assertTrue(union_support.any())


class GeometryContractTest(unittest.TestCase):
    def test_upper_band_uses_ceil_and_excludes_boundary_row(self) -> None:
        mask = upper_head_band((10, 4), 0.35)
        self.assertTrue(mask[3, 0])
        self.assertFalse(mask[4, 0])
        self.assertEqual(int(mask.sum()), 16)


class CandidatePredicateTest(unittest.TestCase):
    def test_obstacle_rejects_only_noncausal_pixels_under_all_negative_evidence(self) -> None:
        component = np.array([[True, True, True, True]])
        causal = np.array([[False, True, False, False]])
        kept, evidence = apply_candidate(
            predicted_class="obstacle",
            component_mask=component,
            causal_mask=causal,
            confidence=0.64,
        )
        np.testing.assert_array_equal(kept, causal)
        self.assertEqual(evidence["action"], "PARTIAL")
        self.assertEqual(evidence["rejected_pixels"], 3)

    def test_obstacle_high_confidence_keeps_noncausal_pixels(self) -> None:
        component = np.array([[True, True, True]])
        kept, evidence = apply_candidate(
            predicted_class="obstacle",
            component_mask=component,
            causal_mask=np.zeros_like(component),
            confidence=0.65,
        )
        np.testing.assert_array_equal(kept, component)
        self.assertFalse(evidence["low_confidence"])

    def test_obstacle_nonproxy_keeps_noncausal_pixels(self) -> None:
        component = np.ones((8, 8), dtype=bool)
        kept, evidence = apply_candidate(
            predicted_class="obstacle",
            component_mask=component,
            causal_mask=np.zeros_like(component),
            confidence=0.1,
        )
        np.testing.assert_array_equal(kept, component)
        self.assertFalse(evidence["small_fragment"])
        self.assertFalse(evidence["intersects_upper_head_band"])

    def test_upper_intersection_is_broadcast_from_raw_component(self) -> None:
        component = np.array([[True, True, True, True]])
        upper = np.array([[True, False, False, False]])
        causal = np.array([[False, True, False, False]])
        kept, evidence = apply_candidate(
            predicted_class="obstacle",
            component_mask=component,
            causal_mask=causal,
            confidence=0.2,
            upper_mask=upper,
        )
        np.testing.assert_array_equal(kept, causal)
        self.assertTrue(evidence["intersects_upper_head_band"])

    def test_boundary_low_confidence_small_fragment_is_wholly_rejected(self) -> None:
        component = np.array([[True, True, True]])
        kept, evidence = apply_candidate(
            predicted_class="boundary_step_curb",
            component_mask=component,
            causal_mask=np.array([[False, True, False]]),
            confidence=0.64,
        )
        self.assertFalse(kept.any())
        self.assertEqual(evidence["action"], "REJECT")

    def test_boundary_area_64_is_protected(self) -> None:
        component = np.ones((8, 8), dtype=bool)
        kept, evidence = apply_candidate(
            predicted_class="boundary_step_curb",
            component_mask=component,
            causal_mask=np.zeros_like(component),
            confidence=0.1,
        )
        np.testing.assert_array_equal(kept, component)
        self.assertFalse(evidence["small_fragment"])

    def test_missing_confidence_is_not_negative_evidence(self) -> None:
        component = np.array([[True, True]])
        kept, evidence = apply_candidate(
            predicted_class="obstacle",
            component_mask=component,
            causal_mask=np.zeros_like(component),
            confidence=None,
        )
        np.testing.assert_array_equal(kept, component)
        self.assertFalse(evidence["confidence_known"])

    def test_callable_signature_has_no_truth_or_session_input(self) -> None:
        parameters = set(
            inspect.signature(apply_frozen_candidate_to_component).parameters
        )
        forbidden = {
            "truth_mask",
            "residual_truth",
            "false_activation",
            "mechanism_tags",
            "dominant_truth_class",
            "session_id",
            "scene_bucket",
            "role",
        }
        self.assertFalse(parameters & forbidden)


class ConfigContractTest(unittest.TestCase):
    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        self.config = json.loads(
            (
                repo_root
                / "configs"
                / "dual_loop_segmentation_conditional_gating_r0"
                / "default.json"
            ).read_text(encoding="utf-8")
        )

    def test_frozen_config_is_accepted(self) -> None:
        _validate_static_config(self.config)
        self.assertEqual(
            self.config["candidate_order"],
            [CANDIDATE_ID],
        )
        self.assertEqual(len(_candidate_definition_hash(self.config)), 64)

    def test_threshold_mutation_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.config))
        mutated["thresholds"]["confidence_minimum"] = 0.651
        with self.assertRaisesRegex(ValueError, "thresholds drifted"):
            _validate_static_config(mutated)

    def test_candidate_addition_is_rejected(self) -> None:
        mutated = json.loads(json.dumps(self.config))
        mutated["candidate_order"].append("RESCUE_CANDIDATE")
        with self.assertRaisesRegex(ValueError, "exact single frozen candidate"):
            _validate_static_config(mutated)


class MembershipTest(unittest.TestCase):
    def test_duplicate_view_row_is_rejected(self) -> None:
        config = {
            "input_contract": {
                "expected_frame_count": 2,
                "expected_component_count": 0,
                "expected_session_frame_counts": {"session": 2},
                "expected_role_session_counts": {"dev": 1},
            }
        }
        frame = {
            "view_row_id": "dev:one",
            "session_id": "session",
            "rehearsal_role": "dev",
        }
        with self.assertRaisesRegex(ValueError, "duplicate view_row_id"):
            _validate_membership(config, [frame, dict(frame)], [])


if __name__ == "__main__":
    unittest.main()
