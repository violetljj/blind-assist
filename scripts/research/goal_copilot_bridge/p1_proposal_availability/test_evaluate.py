from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.research.goal_copilot_bridge.p1_proposal_availability.evaluate import evaluate


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


class EvaluateTest(unittest.TestCase):
    def test_recall_at_k_and_top1_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_path = root / "public.json"
            private_path = root / "private.json"
            prediction_path = root / "prediction.json"
            write(public_path, {"protocol_id": "p", "cases": [{"case_id": "a"}, {"case_id": "b"}]})
            public_hash = hashlib.sha256(public_path.read_bytes()).hexdigest()
            write(private_path, {
                "public_input_sha256": public_hash,
                "primary_correct_iou_threshold": 0.1,
                "diagnostic_correct_iou_thresholds": [0.3, 0.5],
                "recall_at_k": [1, 3, 5, 10],
                "claim_ceiling": "diagnostic",
                "cases": [
                    {"case_id": "a", "target_bbox_xyxy": [0, 0, 10, 10], "target_shortest_side_px": 10, "target_visibility_ratio": 1, "diagnostic_target_metadata": {}},
                    {"case_id": "b", "target_bbox_xyxy": [0, 0, 10, 10], "target_shortest_side_px": 10, "target_visibility_ratio": 1, "diagnostic_target_metadata": {}},
                ],
            })
            write(prediction_path, {
                "public_input_sha256": public_hash,
                "private_truth_access": False,
                "provider": {"name": "fixture"},
                "cases": [
                    {"case_id": "a", "latency_ms": 1, "candidates": [{"rank": 1, "bbox_xyxy": [20, 20, 30, 30]}, {"rank": 2, "bbox_xyxy": [0, 0, 10, 10]}]},
                    {"case_id": "b", "latency_ms": 2, "candidates": []},
                ],
            })
            result = evaluate(public_path, private_path, prediction_path)
            self.assertEqual(0.0, result["recall"]["0.1"]["recall_at_1"])
            self.assertEqual(0.5, result["recall"]["0.1"]["recall_at_3"])
            self.assertEqual("P1_PA0_TOP1_COLLAPSE_SIGNAL_ON_FAILURE_COHORT", result["terminal"])
            self.assertEqual("NOT_EVALUABLE_PROVIDER_INTERFACE", result["raw_retention_attribution"])

    def test_rejects_private_truth_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            public_path = root / "public.json"
            private_path = root / "private.json"
            prediction_path = root / "prediction.json"
            write(public_path, {"protocol_id": "p", "cases": []})
            public_hash = hashlib.sha256(public_path.read_bytes()).hexdigest()
            write(private_path, {"public_input_sha256": public_hash})
            write(prediction_path, {"public_input_sha256": public_hash, "private_truth_access": True})
            with self.assertRaisesRegex(ValueError, "zero private truth access"):
                evaluate(public_path, private_path, prediction_path)


if __name__ == "__main__":
    unittest.main()
