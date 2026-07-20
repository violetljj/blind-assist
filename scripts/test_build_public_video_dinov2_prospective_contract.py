import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_public_video_dinov2_prospective_contract as contract


class DinoV2ProspectiveContractTest(unittest.TestCase):
    def test_direction_hash_is_dtype_and_order_stable(self):
        vector = np.array([1.0, -2.0, 3.5], dtype=np.float64)
        first = contract.direction_sha256(vector)
        second = contract.direction_sha256(vector.astype(">f8"))
        self.assertEqual(first, second)
        self.assertEqual(64, len(first))

    def test_direction_hash_rejects_nonfinite(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            contract.direction_sha256(np.array([np.nan]))


if __name__ == "__main__":
    unittest.main()
