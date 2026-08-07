#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from scripts.research.hftf.deployment.depthart.build_dav2_selective_w8a16_qdq_a5s_r2 import quantize_weight


class SelectiveW8A16QdqTest(unittest.TestCase):
    def test_per_output_channel_reconstruction(self) -> None:
        weight = np.array([[-1.0, -2.0], [1.0, 2.0]], dtype=np.float32)
        quantized, scale = quantize_weight(weight)
        self.assertEqual(quantized.dtype, np.int8)
        self.assertEqual(scale.shape, (2,))
        reconstructed = quantized.astype(np.float32) * scale[None, :]
        np.testing.assert_allclose(reconstructed, weight, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

