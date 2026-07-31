from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from .prepare_burned_revel import prepare_input, prepare_truth


class PrepareBurnedRevelTests(unittest.TestCase):
    def test_input_and_truth_late_join_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_root = root / "images"
            image_root.mkdir()
            replay_rows = []
            for index in range(3):
                image = np.zeros((24, 32, 3), dtype=np.uint8)
                image[:, index * 3 : index * 3 + 2] = 255
                image_path = image_root / f"{index:03d}.jpg"
                self.assertTrue(cv2.imwrite(str(image_path), image))
                replay_rows.append({
                    "captured_at_ns": 1_000_000_000 + index * 50_000_000,
                    "history_reset": index == 0,
                    "image_relative_path": image_path.name,
                    "roi_xywh_normalized": [0.5, 0.5, 0.25, 0.5],
                    "source_frame_id": f"src:{index}",
                    "source_frame_index": index,
                    "target_id": "track-000",
                    "track_epoch": "track-000:epoch-0001",
                })
            replay_path = root / "replay.jsonl"
            replay_path.write_text("\n".join(json.dumps(row) for row in replay_rows) + "\n", encoding="utf-8")
            input_path = root / "input.jsonl"
            input_receipt = root / "input-receipt.json"
            receipt = prepare_input(replay_path, image_root, input_path, input_receipt, source_id="src", session_id="sess", sequence_id="seq")
            self.assertEqual(receipt["input_row_count"], 2)
            self.assertFalse(receipt["truth_read"])
            rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(rows[0]["previous_frame_shape"], [24, 32])
            self.assertEqual(rows[0]["previous_dynamic_bboxes"], [])
            source_truth = root / "truth-source.jsonl"
            truth_rows = []
            for index in range(3):
                truth_rows.append({
                    "event_id": "event-0001",
                    "source_frame_index": index,
                    "target_id": "track-000",
                    "truth_available": True,
                    "truth_state": "approaching" if index < 2 else "quasi_static",
                    "unique_roi_available": True,
                })
            source_truth.write_text("\n".join(json.dumps(row) for row in truth_rows) + "\n", encoding="utf-8")
            truth_path = root / "truth-late.jsonl"
            truth_receipt_path = root / "truth-receipt.json"
            truth_receipt = prepare_truth(input_path, source_truth, truth_path, truth_receipt_path)
            self.assertEqual(truth_receipt["truth_late_row_count"], 2)
            self.assertFalse(truth_receipt["producer_output_read"])
            truth_output = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(truth_output[0]["truth_state"], "approach")
            self.assertEqual(truth_output[0]["parent_event_id"], "event-0001")


if __name__ == "__main__":
    unittest.main()
