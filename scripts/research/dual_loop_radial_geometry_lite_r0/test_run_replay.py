from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))
RUNNER = importlib.import_module("run_replay")


class RunReplayTest(unittest.TestCase):
    def test_two_frame_fixture_emits_complete_two_arm_ledger(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-producer-") as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            frame = np.zeros((160, 160), dtype=np.uint8)
            for y in range(45, 116, 14):
                for x in range(45, 116, 14):
                    cv2.rectangle(frame, (x - 2, y - 2), (x + 2, y + 2), 255, -1)
            enlarged = cv2.warpAffine(
                frame,
                cv2.getRotationMatrix2D((80.0, 80.0), 0.0, 1.08),
                (160, 160),
            )
            cv2.imwrite(str(images / "f0.png"), frame)
            cv2.imwrite(str(images / "f1.png"), enlarged)
            replay_input = root / "replay_input.jsonl"
            rows = [
                {
                    "source_frame_id": "f0",
                    "captured_at_ns": 0,
                    "image_relative_path": "f0.png",
                    "target_id": "track-000",
                    "track_epoch": "track-000:epoch-0001",
                    "history_reset": True,
                    "roi_xywh_normalized": [0.5, 0.5, 0.6, 0.6],
                    "region": "CENTER",
                },
                {
                    "source_frame_id": "f1",
                    "captured_at_ns": 50_000_000,
                    "image_relative_path": "f1.png",
                    "target_id": "track-000",
                    "track_epoch": "track-000:epoch-0001",
                    "history_reset": False,
                    "roi_xywh_normalized": [0.5, 0.5, 0.648, 0.648],
                    "region": "CENTER",
                },
            ]
            replay_input.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "producer.jsonl"
            receipt = RUNNER.run(
                replay_input,
                images,
                output,
                RUNNER._sha256(replay_input),
            )
            output_rows = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(receipt["output_rows"], 4)
            self.assertFalse(receipt["truth_joined"])
            self.assertEqual(
                [row["abstention_reason"] for row in output_rows[:2]],
                ["INSUFFICIENT_HISTORY", "INSUFFICIENT_HISTORY"],
            )
            self.assertTrue(all(row["abstention_reason"] is None for row in output_rows[2:]))
            self.assertTrue(all(row["signed_approach_rate_per_s"] > 0 for row in output_rows[2:]))

    def test_truth_named_input_path_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            RUNNER._assert_producer_path(Path("artifact") / "truth.jsonl", "input")

    def test_output_cannot_overwrite_truth_or_input_freeze(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-producer-") as directory:
            root = Path(directory)
            input_freeze = root / "input-freeze"
            input_freeze.mkdir()
            replay_input = input_freeze / "replay_input.jsonl"
            replay_input.write_text("", encoding="utf-8")
            images = root / "images"
            images.mkdir()
            with self.assertRaisesRegex(ValueError, "forbidden"):
                RUNNER.run(
                    replay_input,
                    images,
                    root / "truth.jsonl",
                    RUNNER._sha256(replay_input),
                )
            with self.assertRaisesRegex(ValueError, "input-freeze"):
                RUNNER.run(
                    replay_input,
                    images,
                    input_freeze / "producer.jsonl",
                    RUNNER._sha256(replay_input),
                )

    def test_activation_hash_mismatch_stops_before_decode(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-producer-") as directory:
            root = Path(directory)
            replay_input = root / "replay_input.jsonl"
            replay_input.write_text("", encoding="utf-8")
            images = root / "images"
            images.mkdir()
            with self.assertRaisesRegex(ValueError, "activation identity"):
                RUNNER.run(replay_input, images, root / "output.jsonl", "0" * 64)


if __name__ == "__main__":
    unittest.main()
