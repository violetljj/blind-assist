#!/usr/bin/env python3
from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path


try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency-free test runtime
    np = None

if np is not None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import build_bonn_static_map_geometry_r0 as subject


@unittest.skipIf(np is None, "numpy not installed in dependency-free test runtime")
class BonnStaticMapGeometryTest(unittest.TestCase):
    def test_hash_selection_is_deterministic(self) -> None:
        points = np.asarray(
            [[0.001 * index, -0.002 * index, 1.0] for index in range(1000)],
            dtype=np.float64,
        )
        first = subject.deterministic_keep_mask(points)
        second = subject.deterministic_keep_mask(points.copy())
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(first.sum()), 0)
        self.assertLess(int(first.sum()), len(points))

    def test_complete_line_blocks_preserves_partial_lines(self) -> None:
        stream = io.BytesIO(b"1 2 3 4 5 6 7\n8 9 10 11 12 13 14")
        blocks = list(subject.complete_line_blocks(stream, chunk_bytes=9))
        self.assertEqual(
            b"".join(blocks),
            b"1 2 3 4 5 6 7\n8 9 10 11 12 13 14",
        )


if __name__ == "__main__":
    unittest.main()
