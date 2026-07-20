import hashlib
import importlib.util
import json
import tempfile
from pathlib import Path
import unittest


SCRIPT = Path(__file__).with_name("analyze_revel_detector_failures.py")
SPEC = importlib.util.spec_from_file_location("revel_failures", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


class RevelDetectorFailureAnalysisTest(unittest.TestCase):
    def test_validates_totals_and_groups_small_misses(self):
        with tempfile.TemporaryDirectory() as temporary:
            details = Path(temporary) / "details.jsonl"
            records = [
                {"selected_index": 0, "image_name": "100.jpg", "source_timestamp_ns": 100, "ground_truth": [{"normalized_area": .01, "stratum": "small", "matched_at_fixed_score": False}], "predictions_over_score_floor": [], "fixed_score_counts": {"tp": 0, "fp": 0, "fn": 1}},
                {"selected_index": 1, "image_name": "200.jpg", "source_timestamp_ns": 200, "ground_truth": [{"normalized_area": .04, "stratum": "medium", "matched_at_fixed_score": True}], "predictions_over_score_floor": [{"score": .8}], "fixed_score_counts": {"tp": 1, "fp": 0, "fn": 0}},
            ]
            details.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            benchmark = {
                "dataset": {"evaluated_frames": 2},
                "details_receipt": {"frame_records": 2, "sha256": hashlib.sha256(details.read_bytes()).hexdigest()},
                "fixed_score_metrics": {"tp": 1, "fp": 0, "fn": 1},
                "recall_by_normalized_box_area": {"small": {"ground_truth": 1, "matched": 0}, "medium": {"ground_truth": 1, "matched": 1}, "large": {"ground_truth": 0, "matched": 0}},
            }
            report = MODULE.analyze(benchmark, details)
            self.assertEqual(1, report["recall_by_area"]["small"]["missed"])
            self.assertEqual(1, len(report["small_miss_segments"]))
            self.assertEqual(1, report["frames_with_no_prediction_over_score_floor"])

    def test_rejects_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            details = Path(temporary) / "details.jsonl"
            details.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                MODULE.analyze({"dataset": {"evaluated_frames": 1}, "details_receipt": {"frame_records": 1, "sha256": "wrong"}}, details)


if __name__ == "__main__":
    unittest.main()
