import unittest

import numpy as np
import torch

from scripts.research.assistive_geometry.run_ag_obstacle_selective_interaction_head_canary import (
    FEATURE_NAMES,
    balanced_training_matrix,
    build_observable_tensor,
    fit_linear_head,
    nested_leave_one_parent_out,
    predict_linear_head,
)


class ObstacleSelectiveInteractionHeadCanaryTest(unittest.TestCase):
    def test_observable_tensor_has_frozen_finite_feature_order(self) -> None:
        shape = (1, 1, 3, 4)
        output = {
            "obstacle_logits": torch.full(shape, 2.0),
            "support_logits": torch.full(shape, -1.0),
            "boundary_logits": torch.full(shape, 0.5),
        }
        depth = torch.arange(12, dtype=torch.float32).reshape(shape) / 10.0 + 0.5
        tensor = build_observable_tensor(output, depth)
        self.assertEqual(tuple(tensor.shape), (3, 4, len(FEATURE_NAMES)))
        self.assertTrue(torch.isfinite(tensor).all())
        self.assertTrue(torch.allclose(tensor[..., 6], tensor[..., 0] * tensor[..., 1]))

    def test_training_sample_is_parent_and_class_balanced(self) -> None:
        parents = {}
        for index in range(3):
            truth = np.asarray([False] * 80 + [True] * 48)
            features = np.repeat(np.arange(len(truth))[:, None], len(FEATURE_NAMES), axis=1)
            parents[f"p{index}"] = {
                "features": features.astype(np.float32),
                "truth": truth,
            }
        x, y, receipt = balanced_training_matrix(parents)
        self.assertEqual(receipt["per_parent_class"], 48)
        self.assertEqual(x.shape[0], 3 * 2 * 48)
        self.assertEqual(int(np.sum(y == 0.0)), int(np.sum(y == 1.0)))
        self.assertTrue(all(row == {"negative": 48, "positive": 48} for row in receipt["by_parent"].values()))

    def test_nested_parent_holdout_passes_separable_fixture(self) -> None:
        parents = {}
        for parent_index in range(6):
            negative = np.full((64, len(FEATURE_NAMES)), -2.0, dtype=np.float32)
            positive = np.full((64, len(FEATURE_NAMES)), 2.0, dtype=np.float32)
            negative[:, 1:] = 0.1 * parent_index
            positive[:, 1:] = 0.1 * parent_index
            parents[f"p{parent_index}"] = {
                "features": np.concatenate((negative, positive)),
                "truth": np.asarray([False] * 64 + [True] * 64),
            }
        folds = nested_leave_one_parent_out(parents)
        self.assertEqual(len(folds), 6)
        self.assertTrue(all(row["threshold_pair_valid"] for row in folds))
        for fold in folds:
            self.assertNotIn(fold["held_parent"], fold["outer_fit_parents"])
            self.assertLessEqual(fold["held_metrics"]["false_negative_rate"], 0.01)
            self.assertLessEqual(fold["held_metrics"]["false_positive_rate"], 0.05)
            for inner in fold["inner_out_of_parent_receipts"]:
                self.assertNotIn(inner["calibration_parent"], inner["fit_parents"])

    def test_linear_head_learns_obstacle_feature_direction(self) -> None:
        parents = {}
        for index in range(2):
            values = np.linspace(-3.0, 3.0, 128, dtype=np.float32)
            features = np.zeros((128, len(FEATURE_NAMES)), dtype=np.float32)
            features[:, 0] = values
            parents[f"p{index}"] = {
                "features": features,
                "truth": values > 0.0,
            }
        model, _ = fit_linear_head(parents)
        scores = predict_linear_head(model, parents["p0"]["features"])
        self.assertGreater(float(scores[-1]), 0.9)
        self.assertLess(float(scores[0]), 0.1)


if __name__ == "__main__":
    unittest.main()
