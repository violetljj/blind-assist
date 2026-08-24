from __future__ import annotations

import inspect
import unittest

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    semantic_anchor_graph_and_belief_v2_1_real_ocr as transfer,
)


class SemanticAnchorGraphBeliefV21RealOcrTest(unittest.TestCase):
    def test_cohort_covers_requested_transfer_mechanisms(self) -> None:
        cohort = transfer.build_cohort()
        episodes = {item.episode_id for item in cohort}
        self.assertEqual(
            episodes,
            {
                "adjacent_301_302_320",
                "directory_binding",
                "suffix_302a",
                "target_absent",
                "blur_observability",
                "directory_absence_burst",
            },
        )
        sign_numbers = {number for item in cohort for _, _, number in item.signs}
        self.assertTrue({"301", "302", "320", "302A"}.issubset(sign_numbers))

    def test_rapidocr_polygon_and_confidence_survive_prefix_adaptation(self) -> None:
        raw = [{"text": "30", "confidence": 0.73, "polygon_px": [[640, 140], [700, 140], [700, 180], [640, 180]]}]
        token = transfer._tokens_from_ocr(raw)[0]
        self.assertEqual(token.text, "30?")
        self.assertEqual(token.confidence, 0.73)
        self.assertAlmostEqual(token.box.x0, 0.5)
        self.assertAlmostEqual(token.box.height, 40 / transfer.IMAGE_HEIGHT)

    def test_transfer_imports_the_frozen_v2_implementation(self) -> None:
        source = inspect.getsourcefile(transfer.graph_candidate_scores)
        self.assertIsNotNone(source)
        self.assertTrue(str(source).endswith("semantic_anchor_graph_and_belief_v2.py"))


if __name__ == "__main__":
    unittest.main()
