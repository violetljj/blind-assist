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


def _write_image(path: Path, height: int, width: int) -> None:
    image = np.zeros((height, width), dtype=np.uint8)
    for y in range(30, height - 20, 14):
        for x in range(30, width - 20, 14):
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), 255, -1)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError("fixture image write failed")


class RunReplayTest(unittest.TestCase):
    def test_shape_change_is_published_as_two_arm_abstention(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-r1-producer-") as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            _write_image(images / "f0.png", 160, 160)
            _write_image(images / "f1.png", 158, 160)
            _write_image(images / "f2.png", 158, 160)
            replay = root / "replay_input.jsonl"
            rows = []
            for frame in range(3):
                rows.append({
                    "source_frame_id": f"f{frame}",
                    "captured_at_ns": frame * 40_000_000,
                    "image_relative_path": f"f{frame}.png",
                    "target_id": "track-000",
                    "track_epoch": "track-000:epoch-0001",
                    "history_reset": frame == 0,
                    "roi_xywh_normalized": [0.5, 0.5, 0.6, 0.6],
                    "region": "CENTER",
                })
            replay.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            output = root / "producer.jsonl"
            receipt = RUNNER.run(
                replay,
                images,
                output,
                RUNNER._sha256(replay),
            )
            produced = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(receipt["shape_change_opportunities"], 1)
            self.assertEqual(receipt["shape_change_arm_rows"], 2)
            self.assertEqual(
                [row["abstention_reason"] for row in produced[2:4]],
                ["FRAME_SHAPE_CHANGE", "FRAME_SHAPE_CHANGE"],
            )
            self.assertNotIn(
                "FRAME_SHAPE_CHANGE",
                [row["abstention_reason"] for row in produced[4:6]],
            )
            progress = json.loads(
                (root / "producer.jsonl.progress.json").read_text(encoding="utf-8")
            )
            self.assertEqual(progress["status"], "completed")

    def test_decode_failure_publishes_only_failure_terminal(self) -> None:
        with tempfile.TemporaryDirectory(prefix="radial-r1-producer-") as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            replay = root / "replay_input.jsonl"
            replay.write_text(json.dumps({
                "source_frame_id": "f0",
                "captured_at_ns": 0,
                "image_relative_path": "missing.png",
                "target_id": "track-000",
                "track_epoch": "track-000:epoch-0001",
                "history_reset": True,
                "roi_xywh_normalized": [0.5, 0.5, 0.6, 0.6],
                "region": "CENTER",
            }) + "\n", encoding="utf-8")
            output = root / "producer.jsonl"
            with self.assertRaisesRegex(ValueError, "cannot decode"):
                RUNNER.run(
                    replay,
                    images,
                    output,
                    RUNNER._sha256(replay),
                )
            self.assertFalse(output.exists())
            self.assertFalse(
                (root / "producer.jsonl.receipt.json").exists()
            )
            failure = json.loads(
                (root / "producer.jsonl.failure.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failure["status"], "PRODUCER_FAILED")

    def test_formal_mode_cannot_be_truncated(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be truncated"):
            RUNNER.run(
                Path("replay_input.jsonl"),
                Path("images"),
                Path("producer.jsonl"),
                "0" * 64,
                mode="formal",
                max_rows=1,
            )

    def test_truth_named_output_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "forbidden"):
            RUNNER._assert_producer_path(Path("truth-output.jsonl"), "output")

    def test_formal_mode_requires_implementation_lock_binding(self) -> None:
        with self.assertRaisesRegex(ValueError, "implementation-lock"):
            RUNNER.run(
                Path("replay_input.jsonl"),
                Path("images"),
                Path("producer.jsonl"),
                "0" * 64,
                mode="formal",
            )


if __name__ == "__main__":
    unittest.main()
