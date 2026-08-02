import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_stage_c_d6_sanpo_yolo_hftf_fusion_head import (
    YOLO_FEATURE_NAMES,
    load_yolo_rows,
    yolo_feature_matrices,
)


def _row(index, source, **overrides):
    row = {
        "arm": "A_CURRENT_YOLO_ONLY",
        "parent_event_id": "event-1",
        "frame_index": index,
        "source_frame_index": source,
        "detection_count": index,
        "actual_alert": False,
        "raw_risk_level": "NONE",
        "stable_risk_level": "NONE",
        "risk_direction": "NONE",
    }
    row.update(overrides)
    return row


class SanpoYoloHftfFusionHeadTest(unittest.TestCase):
    def test_causal_window_uses_current_and_previous_frame(self):
        events = [
            {
                "parent_event_id": "event-1",
                "frames": [
                    {"source_frame_index": 10},
                    {"source_frame_index": 11},
                ],
            }
        ]
        rows = {
            ("event-1", 0): _row(
                0,
                10,
                detection_count=4,
                actual_alert=True,
                raw_risk_level="HIGH",
                stable_risk_level="MEDIUM",
                risk_direction="LEFT",
            ),
            ("event-1", 1): _row(
                1,
                11,
                detection_count=2,
                risk_direction="CENTER",
            ),
        }
        matrix = yolo_feature_matrices(events, rows)[0]
        self.assertEqual((2, len(YOLO_FEATURE_NAMES)), matrix.shape)
        self.assertEqual(4.0, matrix[1, 0])
        self.assertEqual(1.0, matrix[1, 1])
        self.assertEqual(1.0, matrix[1, 2])
        self.assertEqual(2.0 / 3.0, matrix[1, 3])
        self.assertEqual([1.0, 1.0, 0.0], matrix[1, 4:].tolist())

    def test_trace_loader_filters_to_yolo_arm(self):
        rows = [
            _row(0, 10),
            {
                **_row(0, 10),
                "arm": "B_LEARNED_SEGMENTATION_ONLY",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_text(
                "\n".join(
                    __import__("json").dumps(row) for row in rows
                )
                + "\n",
                encoding="utf-8",
            )
            loaded = load_yolo_rows(path)
        self.assertEqual([("event-1", 0)], list(loaded))

    def test_extra_trace_coverage_is_rejected(self):
        events = [
            {
                "parent_event_id": "event-1",
                "frames": [{"source_frame_index": 10}],
            }
        ]
        rows = {
            ("event-1", 0): _row(0, 10),
            ("event-1", 1): _row(1, 11),
        }
        with self.assertRaisesRegex(ValueError, "extra rows"):
            yolo_feature_matrices(events, rows)


if __name__ == "__main__":
    unittest.main()
