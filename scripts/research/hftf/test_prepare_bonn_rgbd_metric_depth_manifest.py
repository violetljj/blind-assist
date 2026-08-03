#!/usr/bin/env python3

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_bonn_rgbd_metric_depth_manifest import (
    associate_nearest,
    normalize_depth_image,
    read_tum_index,
    sample_timestamp_pairs,
)


class BonnRgbdMetricDepthManifestTest(unittest.TestCase):
    def test_normalizes_only_singleton_channel_depth(self) -> None:
        depth = np.zeros((2, 3, 1), dtype=np.uint16)
        normalized = normalize_depth_image(depth, Path("depth.png"))
        self.assertEqual(normalized.shape, (2, 3))
        with self.assertRaises(ValueError):
            normalize_depth_image(
                np.zeros((2, 3, 3), dtype=np.uint16), Path("depth.png")
            )

    def test_reads_tum_index_and_ignores_comments(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index = Path(temporary) / "rgb.txt"
            index.write_text(
                "# timestamp path\n1.000 rgb/a.png\n\n1.100 rgb/b.png\n",
                encoding="utf-8",
            )
            self.assertEqual(
                read_tum_index(index),
                [(1.0, Path("rgb/a.png")), (1.1, Path("rgb/b.png"))],
            )

    def test_associates_nearest_depth_with_bounded_delta(self) -> None:
        rgb = [(1.0, Path("r0")), (1.1, Path("r1")), (1.2, Path("r2"))]
        depth = [(1.01, Path("d0")), (1.09, Path("d1")), (1.25, Path("d2"))]
        pairs = associate_nearest(rgb, depth, 0.02)
        self.assertEqual([pair[3] for pair in pairs], [Path("d0"), Path("d1")])

    def test_samples_by_timestamp_with_start_and_duration(self) -> None:
        pairs = [
            (10.0 + index * 0.05, Path(f"r{index}"), 10.0, Path("d"))
            for index in range(20)
        ]
        selected = sample_timestamp_pairs(
            pairs, start_s=0.2, duration_s=0.5, target_fps=5.0
        )
        self.assertEqual([round(row[0], 2) for row in selected], [10.2, 10.4, 10.6])


if __name__ == "__main__":
    unittest.main()
