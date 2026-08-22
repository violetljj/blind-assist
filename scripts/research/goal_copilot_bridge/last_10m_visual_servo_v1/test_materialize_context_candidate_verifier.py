from __future__ import annotations

import unittest

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_context_candidate_verifier import VIEW_SIZE, dual_view


class MaterializeContextCandidateVerifierTest(unittest.TestCase):
    def test_dual_view_geometry(self) -> None:
        image = Image.new("RGB", (100, 80), "white")
        output = dual_view(image, [20, 10, 60, 70])
        self.assertEqual(output.size, (VIEW_SIZE * 2, VIEW_SIZE))


if __name__ == "__main__":
    unittest.main()
