from __future__ import annotations

import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.run_hierarchical_functional_context import (
    CONTEXT_SCALE,
    expand_box,
    hierarchical_candidates,
)


class HierarchicalFunctionalContextTest(unittest.TestCase):
    def test_fixed_context_expansion_clips_to_frame(self) -> None:
        self.assertEqual([0.0, 0.0, 25.0, 25.0], expand_box([0, 0, 20, 20], scale=CONTEXT_SCALE, width=100, height=100))

    def test_semantic_supported_functional_candidate_is_ranked_first(self) -> None:
        semantic = [{"bbox_xyxy": [40, 40, 80, 80], "score": 0.1}]
        functional = [
            {"bbox_xyxy": [0, 0, 10, 10], "score": 0.9, "label": "door"},
            {"bbox_xyxy": [50, 50, 60, 60], "score": 0.2, "label": "entrance"},
        ]
        rows = hierarchical_candidates(semantic, functional, width=100, height=100)
        self.assertEqual(2, rows[0]["functional_provider_rank"])
        self.assertTrue(rows[0]["semantic_supported"])
        self.assertEqual(1, rows[0]["rank"])
        self.assertFalse(rows[1]["semantic_supported"])


if __name__ == "__main__":
    unittest.main()
