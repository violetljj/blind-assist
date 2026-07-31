from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from .evaluate import evaluate
from .common import sha256_file
from .produce import run


class PipelineTests(unittest.TestCase):
    def test_truth_late_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = np.random.default_rng(3).integers(0, 256, size=(160, 160), dtype=np.uint8)
            image[45:105, 50:90] = 128
            image_path = root / "frame.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            image_sha256 = sha256_file(image_path)
            input_path = root / "input.jsonl"
            rows = []
            truth_rows = []
            for index in range(3):
                row = {
                    "source_id": "SYNTH",
                    "session_id": "S1",
                    "sequence_id": "Q",
                    "parent_event_id": "E1",
                    "previous_source_frame_id": f"f{index}",
                    "current_source_frame_id": f"f{index + 1}",
                    "previous_frame_index": index,
                    "current_frame_index": index + 1,
                    "previous_image": str(image_path),
                    "current_image": str(image_path),
                    "previous_image_sha256": image_sha256,
                    "current_image_sha256": image_sha256,
                    "previous_frame_shape": [160, 160],
                    "current_frame_shape": [160, 160],
                    "captured_at_ns_previous": index * 50_000_000,
                    "captured_at_ns_current": (index + 1) * 50_000_000,
                    "target_id": "T1",
                    "track_epoch": 0,
                    "previous_bbox": [50, 45, 90, 105],
                    "current_bbox": [50, 45, 90, 105],
                    "previous_dynamic_bboxes": [],
                    "current_dynamic_bboxes": [],
                }
                rows.append(row)
                truth_rows.append({**{key: row[key] for key in ("source_id", "session_id", "sequence_id", "previous_source_frame_id", "current_source_frame_id", "target_id", "track_epoch")}, "parent_event_id": "E1", "truth_eligible": True, "truth_state": "quasi-static"})
            input_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            truth_path = root / "truth.jsonl"
            truth_path.write_text("".join(json.dumps(row) + "\n" for row in truth_rows), encoding="utf-8")
            producer_output = root / "producer.jsonl"
            receipt_path = root / "receipt.json"
            evaluation_path = root / "evaluation.json"
            receipt = run(input_path, producer_output, receipt_path, Path(__file__).resolve().parents[3])
            self.assertFalse(receipt["truth_read"])
            result = evaluate(producer_output, receipt_path, truth_path, evaluation_path, Path(__file__).resolve().parents[3])
            self.assertEqual(result["status"], "VALID")
            self.assertFalse(result["development_gate_passed"])
            self.assertEqual(result["truth_read_by_producer"], False)


if __name__ == "__main__":
    unittest.main()
