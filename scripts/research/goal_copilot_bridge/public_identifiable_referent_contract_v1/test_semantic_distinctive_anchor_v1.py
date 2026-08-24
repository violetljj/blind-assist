from __future__ import annotations

import unittest

import cv2
import numpy as np

from .semantic_distinctive_anchor_v1 import (
    _arm_metrics,
    _paste_placard,
    decide_unique_anchor,
    detect_aruco_ids,
    make_aruco_marker,
    normalize_text,
    patch_evidence,
    text_anchor_present,
)


class SemanticDistinctiveAnchorV1Test(unittest.TestCase):
    def test_text_normalization_and_exact_substring(self) -> None:
        self.assertEqual(normalize_text("Starbucks Coffee"), "STARBUCKSCOFFEE")
        self.assertTrue(text_anchor_present(["STARBUCKS COFFEE"], "coffee"))
        self.assertFalse(text_anchor_present(["STARBUCKS"], "coffee"))

    def test_unique_anchor_policy_abstains_on_zero_or_multiple(self) -> None:
        self.assertEqual(decide_unique_anchor({"A": True, "B": False})["selected_candidate"], "A")
        self.assertEqual(decide_unique_anchor({"A": False, "B": False})["decision"], "ABSTAIN")
        self.assertEqual(decide_unique_anchor({"A": True, "B": True})["decision"], "ABSTAIN")

    def test_aruco_exact_id(self) -> None:
        marker = make_aruco_marker(17, 180)
        canvas = np.full((360, 480, 3), 220, dtype=np.uint8)
        canvas[80:260, 140:320] = marker
        self.assertEqual(detect_aruco_ids(canvas), [17])

    def test_distinctive_patch_requires_geometric_support(self) -> None:
        template = np.full((160, 240, 3), 245, dtype=np.uint8)
        cv2.putText(template, "HOUSE BAKE", (12, 92), cv2.FONT_HERSHEY_DUPLEX, 1.15, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.circle(template, (205, 45), 20, (20, 120, 220), 4)
        candidate = _paste_placard(np.full((480, 640, 3), 190, dtype=np.uint8), template)
        blank = np.full((480, 640, 3), 190, dtype=np.uint8)
        self.assertTrue(patch_evidence(template, candidate)["present"])
        self.assertFalse(patch_evidence(template, blank)["present"])

    def test_reacquisition_counts_fresh_semantic_lock(self) -> None:
        rows = [
            {"target_id": "t", "step_index": 0, "target_present": True, "target_slot": "A", "semantic": {"decision": "LOCK", "selected_candidate": "A", "rank1_candidate": "A"}},
            {"target_id": "t", "step_index": 1, "target_present": False, "target_slot": None, "semantic": {"decision": "ABSTAIN", "selected_candidate": None, "rank1_candidate": None}},
            {"target_id": "t", "step_index": 2, "target_present": True, "target_slot": "B", "semantic": {"decision": "LOCK", "selected_candidate": "B", "rank1_candidate": "B"}},
        ]
        metrics = _arm_metrics(rows, "semantic")
        self.assertEqual(metrics["target_top1"], 2)
        self.assertEqual(metrics["wrong_target_locks"], 0)
        self.assertEqual(metrics["reacquisition"], 1)


if __name__ == "__main__":
    unittest.main()
