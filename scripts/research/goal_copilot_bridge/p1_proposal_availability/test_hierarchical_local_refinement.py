from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_hierarchical_local_refinement import map_local_box


class HierarchicalLocalRefinementTest(unittest.TestCase):
    def test_maps_crop_local_box_back_to_full_frame(self) -> None:
        self.assertEqual([110.0, 220.0, 150.0, 280.0], map_local_box([10, 20, 50, 80], [100, 200, 300, 400]))

    def test_rejects_degenerate_local_box(self) -> None:
        with self.assertRaises(ValueError):
            map_local_box([10, 20, 10, 80], [100, 200, 300, 400])


if __name__ == "__main__":
    unittest.main()
