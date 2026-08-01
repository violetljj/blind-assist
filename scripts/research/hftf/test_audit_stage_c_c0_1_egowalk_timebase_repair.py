from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_stage_c_c0_1_egowalk_timebase_repair import (
    _parquet_timeline_metrics,
)


class StageCC01EgoWalkTimebaseRepairTest(unittest.TestCase):
    def test_parquet_timestamp_defines_five_hz_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pose.parquet"
            pl.DataFrame(
                {
                    "frame": list(range(5)),
                    "timestamp": [1000, 1200, 1400, 1600, 1800],
                }
            ).write_parquet(path)
            metrics = _parquet_timeline_metrics(path)
            self.assertTrue(metrics["frame_zero_contiguous"])
            self.assertTrue(metrics["timestamps_strictly_increasing"])
            self.assertEqual(200.0, metrics["median_timestamp_delta_ms"])
            self.assertEqual(5.0, metrics["effective_rate_hz"])

    def test_non_contiguous_frame_is_visible(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "pose.parquet"
            pl.DataFrame(
                {
                    "frame": [0, 1, 3],
                    "timestamp": [1000, 1200, 1400],
                }
            ).write_parquet(path)
            self.assertFalse(
                _parquet_timeline_metrics(path)["frame_zero_contiguous"]
            )


if __name__ == "__main__":
    unittest.main()
