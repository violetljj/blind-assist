from __future__ import annotations

import unittest

import cv2
import numpy as np

from scripts.research.goal_copilot_bridge.public_identifiable_referent_contract_v1 import (
    active_distinctive_evidence_acquisition_v0 as subject,
)


class ActiveDistinctiveEvidenceAcquisitionV0Test(unittest.TestCase):
    @staticmethod
    def _pattern() -> np.ndarray:
        image = np.full((420, 560, 3), 235, dtype=np.uint8)
        cv2.rectangle(image, (60, 55), (500, 365), (30, 30, 30), 6)
        cv2.putText(image, "BLIND ASSIST 27", (85, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.35, (10, 80, 190), 4)
        for index in range(12):
            center = (95 + (index % 6) * 72, 245 + (index // 6) * 70)
            cv2.circle(image, center, 16 + index % 4, (20 * index, 180 - 8 * index, 35 + 12 * index), -1)
            cv2.line(image, center, (center[0] + 25, center[1] - 22), (0, 0, 0), 3)
        return image

    def test_stable_anchor_prefers_warped_target(self) -> None:
        target = self._pattern()
        sweep = subject._reference_sweep(target)
        bank = subject.build_anchor_bank(sweep)
        warped = subject._scaled_on_canvas(target, 0.82)
        distractor = np.full_like(target, 210)
        cv2.putText(distractor, "OTHER STORE", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5)
        target_score = subject.score_candidate(bank, warped)
        distractor_score = subject.score_candidate(bank, distractor)
        decision = subject.decide_active({"A": target_score, "B": distractor_score})
        self.assertGreaterEqual(bank.stable_anchor_count, subject.MIN_STABLE_ANCHORS)
        self.assertGreater(target_score["score"], distractor_score["score"])
        self.assertEqual(decision["selected_candidate"], "A")

    def test_metrics_keep_lost_step_out_of_top1_denominator(self) -> None:
        rows = [
            {
                "target_id": "x", "scenario": "S", "step_index": 0, "time_seconds": 0.0,
                "target_present": True, "target_slot": "A",
                "passive": {"decision": "LOCK", "selected_candidate": "B", "rank1_candidate": "B"},
                "active": {"decision": "LOCK", "selected_candidate": "A", "rank1_candidate": "A"},
            },
            {
                "target_id": "x", "scenario": "S", "step_index": 1, "time_seconds": 0.9,
                "target_present": False, "target_slot": None,
                "passive": {"decision": "LOCK", "selected_candidate": "A", "rank1_candidate": "A"},
                "active": {"decision": "ABSTAIN", "selected_candidate": None, "rank1_candidate": "B"},
            },
            {
                "target_id": "x", "scenario": "S", "step_index": 2, "time_seconds": 1.8,
                "target_present": True, "target_slot": "B",
                "passive": {"decision": "LOCK", "selected_candidate": "B", "rank1_candidate": "B"},
                "active": {"decision": "LOCK", "selected_candidate": "B", "rank1_candidate": "B"},
            },
        ]
        passive = subject._arm_metrics(rows, "passive")
        active = subject._arm_metrics(rows, "active")
        self.assertEqual(passive["target_present_decisions"], 2)
        self.assertEqual(passive["wrong_target_locks"], 2)
        self.assertEqual(active["wrong_target_locks"], 0)
        self.assertEqual(active["reacquisition"], 1)


if __name__ == "__main__":
    unittest.main()
