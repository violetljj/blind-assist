#!/usr/bin/env python3

import unittest

from materialize_sparse_scale_anchor_replay import materialize


def field(value: float) -> dict:
    return {
        "status": "VALID",
        "bands": {
            band: {"clearance_m": value}
            for band in ("left", "center", "right")
        },
    }


class MaterializeSparseScaleAnchorReplayTest(unittest.TestCase):
    def test_anchor_uses_manifest_clock_not_frame_filename_clock(self) -> None:
        frames = [
            {
                "sequence_id": "s",
                "timestamp": 1000.0 + index,
                "frame_path": f"C:/frames/{index}.png",
                "candidate": field(2.0),
                "sensor": field(1.0),
            }
            for index in range(30)
        ]
        binding = {
            f"C:\\frames\\{index}.png": index * 10
            for index in range(30)
        }
        rows = materialize({"frames": frames}, binding)
        self.assertEqual(rows[0]["timestamp_ns"], 90)
        self.assertEqual(rows[0]["scale"], 0.5)


if __name__ == "__main__":
    unittest.main()
