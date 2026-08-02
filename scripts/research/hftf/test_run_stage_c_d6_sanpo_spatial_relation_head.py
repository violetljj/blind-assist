import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_spatial_relation_head import (
    infer_spatial_matrices,
    top_coefficients,
)


class _Dataset(torch.utils.data.Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return torch.zeros(3, 8, 12), 0, index


class _Model(torch.nn.Module):
    pass


class SanpoSpatialRelationHeadTest(unittest.TestCase):
    def test_top_coefficients_orders_absolute_magnitude(self):
        rows = top_coefficients(
            ["a", "b", "c"],
            np.asarray([0.1, -2.0, 1.0]),
            count=2,
        )
        self.assertEqual(["b", "c"], [row["feature"] for row in rows])

    def test_spatial_feature_shape_contract(self):
        # Patch only the imported feature extractor at its module boundary.
        import run_stage_c_d6_sanpo_spatial_relation_head as module

        original = module.single_frame_spatial_features
        module.single_frame_spatial_features = (
            lambda model, frames: torch.ones(
                len(frames),
                128,
                4,
                7,
            )
        )
        try:
            matrices, names = infer_spatial_matrices(
                _Model(),
                _Dataset(),
                {"events": [{"frames": [{}, {}]}]},
                batch_size=2,
            )
        finally:
            module.single_frame_spatial_features = original
        self.assertEqual((2, 2304), matrices[0].shape)
        self.assertEqual(2304, len(names))
        self.assertTrue(np.allclose(matrices[0], 1.0))


if __name__ == "__main__":
    unittest.main()
