import unittest

import numpy as np
import torch

import run_public_video_marker_relation_distance_aux_ablation as subject


class MarkerRelationDistanceAuxAblationTest(unittest.TestCase):
    def test_matched_initialization_is_exact(self) -> None:
        torch.manual_seed(7)
        initial = subject.RelationHead(2, 3).state_dict()
        x = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
        y = np.asarray([0.0, 1.0])
        d = np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        w = np.asarray([0.5, 0.5])
        spec = {"hidden_dimension": 3, "steps": 1, "learning_rate": 0.01, "weight_decay": 0.0}
        baseline, _ = subject.train_model(x, y, d, w, initial, spec, 0.0)
        treatment, _ = subject.train_model(x, y, d, w, initial, spec, 0.0)
        for left, right in zip(baseline.parameters(), treatment.parameters()):
            torch.testing.assert_close(left, right)

    def test_distance_loss_changes_shared_model(self) -> None:
        torch.manual_seed(8)
        initial = subject.RelationHead(2, 3).state_dict()
        x = np.asarray([[-1.0, 0.0], [1.0, 0.0]])
        y = np.asarray([0.0, 1.0])
        d = np.asarray([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
        w = np.asarray([0.5, 0.5])
        spec = {"hidden_dimension": 3, "steps": 3, "learning_rate": 0.01, "weight_decay": 0.0}
        baseline, _ = subject.train_model(x, y, d, w, initial, spec, 0.0)
        treatment, _ = subject.train_model(x, y, d, w, initial, spec, 0.2)
        self.assertFalse(torch.equal(baseline.shared.weight, treatment.shared.weight))


if __name__ == "__main__":
    unittest.main()
