from __future__ import annotations

import unittest

import numpy as np
import torch

from scripts.research.riskseg_r0_training.train import (
    IGNORE_LABEL,
    OhemCrossEntropy,
    RisksegDataset,
    Row,
    make_boundary,
    metrics_from_confusion,
    recipe_json,
)


class TrainingContractTest(unittest.TestCase):
    def test_recipe_keeps_single_architecture_and_forbidden_rescues(self) -> None:
        recipe = recipe_json()
        self.assertEqual("official_PIDNet-S", recipe["architecture"])
        self.assertEqual(
            [
                "walkable",
                "blocking_obstacle",
                "boundary_level_change",
                "unknown_nonwalkable",
            ],
            recipe["class_order"],
        )
        self.assertIn("fp_sampler", recipe["forbidden"])
        self.assertFalse(recipe["loss"]["class_balance"])

    def test_boundary_is_generated_at_class_transition(self) -> None:
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[:, 16:] = 1
        boundary = make_boundary(mask)
        self.assertGreater(boundary.sum(), 0)
        self.assertGreater(boundary[:, 14:19].sum(), 0)

    def test_metrics_keep_four_class_order(self) -> None:
        confusion = np.diag([10, 8, 6, 4]).astype(np.int64)
        metrics = metrics_from_confusion(confusion)
        self.assertEqual(1.0, metrics["mean_iou"])
        self.assertEqual(
            {
                "walkable",
                "blocking_obstacle",
                "boundary_level_change",
                "unknown_nonwalkable",
            },
            set(metrics["per_class_iou"]),
        )

    def test_ohem_rejects_all_ignore_target(self) -> None:
        loss = OhemCrossEntropy()
        score = torch.zeros(1, 4, 8, 8)
        target = torch.full((1, 8, 8), IGNORE_LABEL, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "no valid pixels"):
            loss.ohem(score, target)


if __name__ == "__main__":
    unittest.main()
