import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grail_paired_orientation_r1cp import (  # noqa: E402
    consensus_index,
    masked_group_crop,
    orientation_matrix,
    paired_mode_bases,
    projected_basis,
)
from collect_grail_paired_orientation_r1cp import _admit  # noqa: E402


class PairedOrientationMechanicsTest(unittest.TestCase):
    def test_identity_projects_canonical_right_and_up(self):
        basis = projected_basis(orientation_matrix(0, 0, 0))
        self.assertTrue(basis["evaluable"])
        np.testing.assert_allclose(basis["right"], [1, 0], atol=1e-7)
        np.testing.assert_allclose(basis["down"], [0, 1], atol=1e-7)

    def test_reference_modes_share_one_relative_transform(self):
        modes = paired_mode_bases(0, 0, 0, 2, 30, 0, 0)
        self.assertEqual(len(modes), 2)
        for mode in modes:
            self.assertTrue(mode["reference"]["evaluable"])
            self.assertTrue(mode["query"]["evaluable"])

    def test_consensus_abstains_on_mode_disagreement(self):
        self.assertEqual(consensus_index([3, 3, 3]), 3)
        self.assertIsNone(consensus_index([3, 4]))
        self.assertIsNone(consensus_index([]))

    def test_masked_crop_whitens_background(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mask = np.zeros((10, 10), dtype=np.uint8)
            mask[3:7, 4:8] = 255
            Image.fromarray(mask).save(root / "mask.png")
            image = np.zeros((10, 10, 3), dtype=np.uint8)
            crop = np.asarray(masked_group_crop(
                image, [{"mask_image": "mask.png"}], [0], root, padding_fraction=0.25
            ))
            self.assertTrue(np.any(np.all(crop == 255, axis=-1)))
            self.assertTrue(np.any(np.all(crop == 0, axis=-1)))

    def test_admission_has_exact_frozen_strata(self):
        rows = [
            {"sample_id": f"d-{index}", "same_type_visible_candidates": 2} for index in range(50)
        ] + [
            {"sample_id": f"c-{index}", "same_type_visible_candidates": 1} for index in range(40)
        ]
        admitted = _admit(rows, "salt")
        self.assertEqual(len(admitted), 78)
        self.assertEqual(sum(row["same_type_visible_candidates"] >= 2 for row in admitted), 43)


if __name__ == "__main__":
    unittest.main()
