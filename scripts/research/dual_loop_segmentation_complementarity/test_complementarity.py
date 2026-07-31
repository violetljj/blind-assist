from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .complementarity import (
    ComplementarityInputError,
    box_union_mask,
    load_manifest,
    mask_iou,
    pair_inputs,
)


class ComplementarityContractTests(unittest.TestCase):
    def test_box_union_mask_clips_normalizes_and_unions(self) -> None:
        mask = box_union_mask(
            [
                {
                    "left": -2.0,
                    "top": 0.0,
                    "right": 4.0,
                    "bottom": 4.0,
                    "frame_width": 10.0,
                    "frame_height": 10.0,
                },
                {
                    "left": 5.0,
                    "top": 5.0,
                    "right": 10.0,
                    "bottom": 10.0,
                    "frame_width": 10.0,
                    "frame_height": 10.0,
                },
            ],
            source_width=10,
            source_height=10,
            analysis_width=10,
            analysis_height=10,
        )
        self.assertTrue(mask[:4, :4].all())
        self.assertTrue(mask[5:, 5:].all())
        self.assertFalse(mask[4, 4])
        self.assertEqual(int(mask.sum()), 16 + 25)

    def test_mask_iou_empty_pair_is_stable(self) -> None:
        empty = np.zeros((3, 3), dtype=bool)
        self.assertIsNone(mask_iou(None, empty))
        self.assertEqual(mask_iou(empty, empty), 1.0)
        other = empty.copy()
        other[0, 0] = True
        self.assertEqual(mask_iou(empty, other), 0.0)

    def test_pair_inputs_rejects_timestamp_drift(self) -> None:
        manifest = [
            {
                "source_id": "s",
                "frame_id": 0,
                "image_sha256": "a",
                "source_capture_timestamp_ns": 0,
                "image_path": Path("x"),
                "width": 1,
                "height": 1,
            }
        ]
        trace = {
            ("s", 0, "a"): {
                "source_id": "s",
                "frame_id": 0,
                "image_sha256": "a",
                "source_capture_timestamp_ns": 1,
                "detections": [],
            }
        }
        with self.assertRaisesRegex(ComplementarityInputError, "timestamp mismatch"):
            pair_inputs(manifest, trace)

    def test_load_manifest_verifies_image_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "frame.bin"
            image_path.write_bytes(b"image")
            manifest_path = root / "manifest.jsonl"
            row = {
                "source_id": "s",
                "frame_id": 0,
                "source_capture_timestamp_ns": 0,
                "image_path": "frame.bin",
                "image_sha256": "0" * 64,
                "width": 1,
                "height": 1,
            }
            manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ComplementarityInputError, "image hash mismatch"):
                load_manifest(manifest_path, root)


if __name__ == "__main__":
    unittest.main()
