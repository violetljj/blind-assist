#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_ag_st_unified_student_icl_boundary import (
    ICL_INTRINSICS_OUTPUT,
    load_icl_bound_rgb,
)


class UnifiedStudentIclBoundaryTest(unittest.TestCase):
    def test_output_intrinsics_match_flip_downsample_contract(self) -> None:
        expected = np.asarray(
            [[481.2 / 4.0, 0.0, (319.5 - 2.0) / 4.0],
             [0.0, 480.0 / 4.0, (239.5 - 2.0) / 4.0],
             [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(ICL_INTRINSICS_OUTPUT, expected)

    def test_bound_rgb_is_vertical_flip_then_exact_stride_four(self) -> None:
        raw = np.zeros((480, 640, 3), dtype=np.uint8)
        raw[..., 0] = np.arange(480, dtype=np.uint16)[:, None] % 256
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rgb.png"
            Image.fromarray(raw).save(path)
            import hashlib

            row = {
                "rgb_path": str(path),
                "rgb_sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
            }
            output = load_icl_bound_rgb(row)
        self.assertEqual((120, 160, 3), output.shape)
        np.testing.assert_array_equal(output, np.flipud(raw)[2::4, 2::4])


if __name__ == "__main__":
    unittest.main()
