from __future__ import annotations

import unittest

from augment_egowalk_depth_evidence import _sample_indices


class EgoWalkDepthEvidenceTest(unittest.TestCase):
    def test_sample_indices_follow_pose_ordinals_and_clamp(self) -> None:
        candidate = {"candidate_id": "c", "start_frame_index": 100, "end_frame_index": 119}
        self.assertEqual(_sample_indices(candidate, 530), [96, 100, 110, 119, 123])

        edge = {"candidate_id": "edge", "start_frame_index": 0, "end_frame_index": 19}
        self.assertEqual(_sample_indices(edge, 20), [0, 10, 19])


if __name__ == "__main__":
    unittest.main()
