from __future__ import annotations

import unittest

from materialize_sanpo_candidates import (
    _required_runs,
    _window_starts,
)


class MaterializeSanpoCandidatesTest(unittest.TestCase):
    def test_required_runs_intersect_rgb_and_depth(self) -> None:
        self.assertEqual(
            _required_runs(
                {
                    "rgb": [0, 1, 2, 3, 5, 6],
                    "depth": [0, 1, 2, 3, 4, 5],
                },
                required=("rgb", "depth"),
            ),
            [{"start": 0, "end": 3, "count": 4}, {"start": 5, "end": 5, "count": 1}],
        )

    def test_window_starts_are_bounded_and_non_overlapping_by_stride(self) -> None:
        self.assertEqual(
            _window_starts({"start": 0, "end": 149, "count": 150}, window_frames=60, stride_frames=30),
            [0, 30, 60, 90],
        )
        self.assertEqual(
            _window_starts({"start": 0, "end": 59, "count": 60}, window_frames=60, stride_frames=30),
            [0],
        )
        self.assertEqual(
            _window_starts({"start": 0, "end": 58, "count": 59}, window_frames=60, stride_frames=30),
            [],
        )


if __name__ == "__main__":
    unittest.main()
