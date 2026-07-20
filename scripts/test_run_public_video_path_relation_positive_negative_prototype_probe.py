#!/usr/bin/env python3
"""Pure tests for source-isolated positive/negative prototypes."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import run_public_video_path_relation_positive_negative_prototype_probe as subject


class PositiveNegativePrototypeProbeTest(unittest.TestCase):
    def test_unit_rejects_zero_vector(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-degenerate"):
            subject.unit(np.zeros(3))

    def test_prototype_direction_points_from_negative_to_positive(self) -> None:
        values = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1]])
        labels = np.asarray([1, 1, 0, 0])
        direction = subject.prototype_direction(values, labels)
        self.assertGreater(direction[0], 1.9)

    def test_leave_one_source_out_is_perfect_for_separable_sources(self) -> None:
        values = np.asarray([
            [1.0, 0.1], [1.0, -0.1], [-1.0, 0.1], [-1.0, -0.1],
            [0.9, 0.2], [-0.9, -0.2],
        ])
        labels = np.asarray([1, 1, 0, 0, 1, 0])
        sources = ["a", "b", "a", "b", "c", "c"]
        result = subject.leave_one_source_out(values, labels, [f"s{i}" for i in range(6)], sources)
        self.assertEqual(result["metrics"]["balanced_accuracy"], 1.0)

    def test_parent_source_holdout_removes_all_descendants(self) -> None:
        values = np.asarray([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0], [-0.9, -0.1], [1.0, 0.2], [-1.0, -0.2]])
        labels = np.asarray([1, 1, 0, 0, 1, 0])
        sources = ["parent", "parent", "parent", "parent", "other", "third"]
        result = subject.leave_one_source_out(values, labels, [f"s{i}" for i in range(6)], sources)
        parent_fold = next(row for row in result["folds"] if row["held_out_source_id"] == "parent")
        self.assertEqual(len(parent_fold["held_out_sample_ids"]), 4)

    def test_nuisance_projection_removes_only_parallel_component(self) -> None:
        delta = np.asarray([2.0, 3.0])
        nuisance = np.asarray([1.0, 0.0])
        actual = subject.project_out_nuisance(delta, nuisance)
        np.testing.assert_allclose(actual, [0.0, 3.0])

    def test_synthetic_pair_inherits_parent_source(self) -> None:
        class Teacher:
            def extract(self, images, *, batch_size):
                return np.asarray([[float(image.mean()), float(image[:, :, 0].mean()) + index] for index, image in enumerate(images)])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            clear = np.full((8, 8, 3), 20, dtype=np.uint8)
            risk = clear.copy()
            risk[:, :, 0] = 80
            cv2.imwrite(str(root / "clear.png"), clear)
            cv2.imwrite(str(root / "risk.png"), risk)
            manifest = [
                {"image_path": "clear.png", "attributes": {"counterfactual_pair_id": "p", "risk_state": "clear"}, "source": {"parent_source_id": "parent-a"}},
                {"image_path": "risk.png", "attributes": {"counterfactual_pair_id": "p", "risk_state": "risk"}, "source": {"parent_source_id": "parent-a"}},
            ]
            samples, mirrors = subject.synthetic_samples(Teacher(), root, manifest, batch_size=2)
            self.assertEqual(samples[0]["source_id"], "parent-a")
            self.assertEqual(mirrors[0]["source_id"], "parent-a")

    def test_spatial_grid_vector_preserves_patch_location(self) -> None:
        tokens = np.zeros((17, 2), dtype=np.float64)
        tokens[1 + 3 * 4 + 0] = [1.0, 0.0]
        left = subject.dino.spatial_grid_vector(tokens, output_side=2)
        tokens[1 + 3 * 4 + 0] = 0.0
        tokens[1 + 3 * 4 + 3] = [1.0, 0.0]
        right = subject.dino.spatial_grid_vector(tokens, output_side=2)
        self.assertFalse(np.allclose(left, right))


if __name__ == "__main__":
    unittest.main()
