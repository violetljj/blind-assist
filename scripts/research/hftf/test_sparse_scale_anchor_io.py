#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from sparse_scale_anchor_io import SCHEMA, ScaleAnchorStream, load_scale_anchors


class SparseScaleAnchorIoTest(unittest.TestCase):
    def test_stream_releases_only_causally_available_anchors(self) -> None:
        rows = [
            {
                "schema": SCHEMA,
                "sequence_id": "s",
                "timestamp_ns": timestamp,
                "scale": 0.5,
                "pair_count": 3,
                "median_abs_ratio_residual": 0.01,
                "source": "tof",
            }
            for timestamp in (10, 20)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            stream = ScaleAnchorStream(load_scale_anchors(path))
        self.assertEqual(stream.take_available("s", 9), [])
        self.assertEqual([item.timestamp_ns for item in stream.take_available("s", 20)], [10, 20])
        self.assertEqual(stream.take_available("s", 30), [])

    def test_loader_rejects_nonincreasing_or_invalid_scale(self) -> None:
        rows = [
            {
                "schema": SCHEMA,
                "sequence_id": "s",
                "timestamp_ns": 10,
                "scale": scale,
                "pair_count": 3,
                "median_abs_ratio_residual": 0.0,
                "source": "tof",
            }
            for scale in (1.0, -1.0)
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchors.jsonl"
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaises(ValueError):
                load_scale_anchors(path)


if __name__ == "__main__":
    unittest.main()
