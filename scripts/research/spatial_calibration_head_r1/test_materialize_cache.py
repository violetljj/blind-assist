#!/usr/bin/env python3

import tempfile
import unittest
from pathlib import Path

from materialize_cache import (
    matched_frame_stems,
    nearest_intrinsics,
    read_intrinsics,
    sample_150,
    timestamp_from_stem,
)


class ARKitScenesMaterializationPrimitiveTest(unittest.TestCase):
    def test_timestamp_and_sampling_are_deterministic(self) -> None:
        stems = [f"41048097_{index / 60:.3f}" for index in range(200)]
        selected = sample_150(stems)
        self.assertEqual(len(selected), 150)
        self.assertEqual(len(set(selected)), 150)
        self.assertEqual(timestamp_from_stem(selected[0]), 0.0)
        self.assertAlmostEqual(timestamp_from_stem(selected[-1]), 199 / 60, places=3)

    def test_matching_requires_all_three_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("lowres_wide", "lowres_depth", "confidence"):
                (root / name).mkdir()
            for stem in ("1_0.000", "1_0.017"):
                for name in ("lowres_wide", "lowres_depth", "confidence"):
                    (root / name / f"{stem}.png").touch()
            (root / "lowres_wide" / "1_0.033.png").touch()
            self.assertEqual(matched_frame_stems(root), ["1_0.000", "1_0.017"])

    def test_intrinsics_nearest_tolerance_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "1_2.001.pincam"
            path.write_text("256 192 200 201 128 96\n", encoding="utf-8")
            self.assertEqual(nearest_intrinsics(2.0, [(2.001, path)]), path)
            self.assertEqual(read_intrinsics(path, 256, 192), [200.0, 201.0, 128.0, 96.0])
            with self.assertRaises(ValueError):
                nearest_intrinsics(2.0, [(2.01, path)])


if __name__ == "__main__":
    unittest.main()
