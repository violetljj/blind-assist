from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).with_name("audit_source_shapes.py")
SPEC = importlib.util.spec_from_file_location("audit_source_shapes", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def row(
    frame: int,
    target: str,
    image: str,
    *,
    epoch: str = "epoch-1",
    reset: bool = False,
) -> dict:
    return {
        "source_frame_id": f"f{frame}",
        "source_frame_index": frame,
        "captured_at_ns": frame * 40_000_000,
        "target_id": target,
        "track_epoch": epoch,
        "history_reset": reset,
        "region": "CENTER",
        "image_relative_path": image,
    }


class SourceShapeAuditTest(unittest.TestCase):
    def test_immediate_target_pair_mismatches_preserve_denominator(self) -> None:
        rows = [
            row(0, "track-000", "a.jpg", reset=True),
            row(0, "track-001", "a.jpg", reset=True),
            row(1, "track-000", "b.jpg"),
            row(1, "track-001", "b.jpg"),
            row(2, "track-000", "c.jpg"),
            row(2, "track-001", "c.jpg"),
        ]
        result = MODULE.summarize(
            rows,
            {
                "a.jpg": (260, 346),
                "b.jpg": (258, 346),
                "c.jpg": (258, 346),
            },
        )
        self.assertEqual(result["same_target_same_epoch_pair_count"], 4)
        self.assertEqual(result["same_target_immediate_pair_count"], 4)
        self.assertEqual(result["shape_mismatch_pair_count"], 2)
        self.assertEqual(result["expected_common_shape_abstention_arm_rows"], 4)
        self.assertEqual(result["shape_mismatch_by_target"], {
            "track-000": 1,
            "track-001": 1,
        })

    def test_epoch_change_and_reset_do_not_create_pairs(self) -> None:
        rows = [
            row(0, "track-000", "a.jpg", reset=True),
            row(1, "track-000", "b.jpg", epoch="epoch-2", reset=True),
        ]
        result = MODULE.summarize(
            rows,
            {"a.jpg": (260, 346), "b.jpg": (258, 346)},
        )
        self.assertEqual(result["same_target_same_epoch_pair_count"], 0)
        self.assertEqual(result["same_target_immediate_pair_count"], 1)
        self.assertEqual(result["shape_mismatch_pair_count"], 0)


if __name__ == "__main__":
    unittest.main()
