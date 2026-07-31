from __future__ import annotations

import copy
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    Component,
)

from .benchmark import RuntimeFeatureExtractor
from .core import (
    FEATURE_NAMES,
    FrameComponent,
    read_json,
    sigmoid_scores,
    validate_static_config,
)
from .evaluate import select_threshold, session_class_balanced_weights


class FrozenContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[3]
        cls.config = read_json(
            cls.repo_root
            / "configs"
            / "dual_loop_segmentation_learned_component_validator_r0"
            / "default.json"
        )

    def test_static_config_fixes_exact_feature_allowlist(self) -> None:
        validate_static_config(self.config)
        self.assertEqual(
            self.config["feature_contract"]["feature_names"],
            list(FEATURE_NAMES),
        )
        mutated = copy.deepcopy(self.config)
        mutated["feature_contract"]["feature_names"][-1] = "terminal_outcome"
        with self.assertRaises(ValueError):
            validate_static_config(mutated)

    def test_threshold_selection_uses_frozen_lexicographic_rule(self) -> None:
        def candidate(
            threshold: float,
            margin: float,
            session_retention: float,
            fp_reduction: float,
        ) -> dict[str, object]:
            return {
                "threshold": threshold,
                "minimum_normalized_gate_margin": margin,
                "values": {
                    "minimum_session_recall_retention": session_retention,
                    "fp_pixel_reduction": fp_reduction,
                },
            }

        selected = select_threshold(
            [
                candidate(0.60, -0.1, 0.80, 0.4),
                candidate(0.55, -0.1, 0.85, 0.3),
                candidate(0.50, -0.1, 0.85, 0.3),
                candidate(0.45, -0.2, 0.99, 0.9),
            ]
        )
        self.assertEqual(selected["threshold"], 0.50)

    def test_sample_weights_are_exact_product_and_mean_normalized(self) -> None:
        labels = np.asarray([0, 1, 1, 0, 0, 1], dtype=np.int64)
        sessions = ["a", "a", "b", "b", "b", "c"]
        observed = session_class_balanced_weights(labels, sessions)
        total = len(labels)
        session_counts = {"a": 2, "b": 3, "c": 1}
        class_counts = {0: 3, 1: 3}
        expected = np.asarray(
            [
                (total / (3 * session_counts[session]))
                * (total / (2 * class_counts[int(label)]))
                for label, session in zip(labels, sessions)
            ],
            dtype=np.float64,
        )
        expected /= np.mean(expected)
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)
        self.assertAlmostEqual(float(np.mean(observed)), 1.0)

    def test_pure_numpy_sigmoid_is_bounded_and_deterministic(self) -> None:
        matrix = np.asarray([[0.0, 1.0], [2.0, -1.0]], dtype=np.float64)
        scores = sigmoid_scores(
            matrix,
            mean=np.asarray([0.5, 0.0]),
            scale=np.asarray([2.0, 0.5]),
            coefficients=np.asarray([1.5, -0.25]),
            intercept=0.1,
        )
        repeated = sigmoid_scores(
            matrix,
            mean=np.asarray([0.5, 0.0]),
            scale=np.asarray([2.0, 0.5]),
            coefficients=np.asarray([1.5, -0.25]),
            intercept=0.1,
        )
        np.testing.assert_array_equal(scores, repeated)
        self.assertTrue(np.all((scores > 0.0) & (scores < 1.0)))


class RuntimeCausalityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[3]
        cls.config = read_json(
            repo_root
            / "configs"
            / "dual_loop_segmentation_learned_component_validator_r0"
            / "default.json"
        )
        cls.config = copy.deepcopy(cls.config)
        cls.config["analysis_shape"] = [8, 8]

    def _context(
        self,
        *,
        session: str,
        frame: int,
        x: int,
        component_id: str,
    ) -> SimpleNamespace:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:4, x : x + 2] = True
        empty = np.zeros((8, 8), dtype=bool)
        component = Component(
            index=0,
            mask=mask,
            area=4,
            bbox=(x, 2, x + 2, 4),
        )
        table_row = {
            "features": {
                "top1_confidence_median": 0.8,
                "top1_confidence_missing": 0.0,
                "top1_top2_margin_median": 0.3,
                "top1_top2_margin_missing": 0.0,
                "nearest_yolo_union_bbox_gap_fraction": 2.0 / np.hypot(8, 8),
                "nearest_yolo_union_bbox_gap_missing": 0.0,
            }
        }
        item = FrameComponent(
            component_id=component_id,
            predicted_class="obstacle",
            component=component,
            table_row=table_row,
        )
        return SimpleNamespace(
            view_row_id=f"{session}:{frame}",
            session_id=session,
            sequence_id="sequence",
            components=[item],
            raw_class_masks={
                "boundary_step_curb": empty,
                "obstacle": mask,
            },
        )

    def test_future_append_does_not_change_prefix_features(self) -> None:
        first = self._context(session="a", frame=0, x=2, component_id="a0")
        second = self._context(session="a", frame=1, x=2, component_id="a1")
        future = self._context(session="a", frame=2, x=5, component_id="a2")

        prefix = RuntimeFeatureExtractor(self.config)
        _, first_prefix = prefix.extract(first)
        _, second_prefix = prefix.extract(second)

        extended = RuntimeFeatureExtractor(self.config)
        _, first_extended = extended.extract(first)
        _, second_extended = extended.extract(second)
        extended.extract(future)

        np.testing.assert_array_equal(first_prefix, first_extended)
        np.testing.assert_array_equal(second_prefix, second_extended)
        index = list(FEATURE_NAMES).index(
            "causal_previous_component_iou_missing"
        )
        self.assertEqual(first_prefix[0, index], 1.0)
        self.assertEqual(second_prefix[0, index], 0.0)

    def test_session_boundary_resets_temporal_state(self) -> None:
        extractor = RuntimeFeatureExtractor(self.config)
        extractor.extract(
            self._context(session="a", frame=0, x=2, component_id="a0")
        )
        _, next_session = extractor.extract(
            self._context(session="b", frame=0, x=2, component_id="b0")
        )
        missing_index = list(FEATURE_NAMES).index(
            "causal_previous_component_iou_missing"
        )
        age_index = list(FEATURE_NAMES).index(
            "causal_same_footprint_age_5"
        )
        self.assertEqual(next_session[0, missing_index], 1.0)
        self.assertEqual(next_session[0, age_index], 1.0)


if __name__ == "__main__":
    unittest.main()
