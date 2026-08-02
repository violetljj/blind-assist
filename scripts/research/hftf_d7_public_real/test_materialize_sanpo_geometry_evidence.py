from __future__ import annotations

import gzip
import tempfile
import unittest
from pathlib import Path

from materialize_sanpo_geometry_evidence import (
    build_parser,
    _finite_stats,
    _read_depth,
    _sample_indices,
    _write_depth_pgm,
)


class MaterializeSanpoGeometryEvidenceTest(unittest.TestCase):
    def _numpy(self):
        try:
            import numpy as np
        except ModuleNotFoundError:
            self.skipTest("numpy is required for SANPO float16 geometry tests")
        return np

    def test_sample_indices_are_bounded_and_evenly_spaced(self) -> None:
        self.assertEqual(_sample_indices(0), [])
        self.assertEqual(_sample_indices(1), [0])
        self.assertEqual(_sample_indices(10, limit=4), [0, 3, 6, 9])

    def test_relative_nominal_phase_contract_is_explicitly_opt_in(self) -> None:
        args = build_parser().parse_args([
            "--output-root", "F:\\out",
            "--batch-id", "batch",
            "--run-id", "run",
            "--ffmpeg-path", "F:\\ffmpeg.exe",
            "--relative-nominal-phase-contract",
        ])
        self.assertTrue(args.relative_nominal_phase_contract)

    def test_read_depth_validates_sanpo_header_and_shape(self) -> None:
        np = self._numpy()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.float16.gz"
            values = np.concatenate(
                [np.asarray([2, 3], dtype="<f2"), np.asarray([1, 2, 3, 4, 5, 6], dtype="<f2")]
            )
            with gzip.open(path, "wb") as handle:
                handle.write(values.tobytes())
            array = _read_depth(path, np)
            self.assertEqual(array.shape, (2, 3))
            self.assertEqual(float(array[1, 2]), 6.0)

    def test_stats_and_preview_keep_invalid_depth_explicit(self) -> None:
        np = self._numpy()

        array = np.asarray([[1.0, 2.0], [0.0, np.nan]], dtype="<f2")
        stats = _finite_stats(array, np)
        self.assertEqual(stats["valid_positive_finite_pixel_count"], 2)
        self.assertEqual(stats["pixel_count"], 4)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "depth.pgm"
            _write_depth_pgm(path, array, np, display_min_m=0.5, display_max_m=4.0)
            self.assertTrue(path.is_file())
            self.assertTrue(path.read_bytes().startswith(b"P5\n"))


if __name__ == "__main__":
    unittest.main()
