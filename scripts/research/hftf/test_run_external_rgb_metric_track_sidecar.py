import json
import sys
import tempfile
import unittest
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_external_rgb_metric_track_sidecar import (
    append_contiguous_history,
    d44_predict,
    load_manifest,
    relative_position,
    validate_intrinsics,
)


class ExternalRgbMetricTrackSidecarTest(unittest.TestCase):
    def test_relative_position_uses_calibrated_pinhole_geometry(self) -> None:
        value = relative_position([10, 20, 30, 40], [100, 100, 20, 30], 2.0)
        np.testing.assert_allclose(value, [2.0, 0.0, 0.0])

    def test_d44_recovers_constant_velocity(self) -> None:
        history = []
        for index in range(7):
            history.append(
                {
                    "timestamp_ns": index * 100_000_000,
                    "depth_m": 3.0 - 0.1 * index,
                    "torso_roi_xyxy_px": [10, 20, 30, 40],
                    "intrinsics_fx_fy_cx_cy": [100, 100, 20, 30],
                }
            )
        prediction = d44_predict(history, 1_600_000_000)
        np.testing.assert_allclose(prediction, [1.4, 0.0, 0.0], atol=1e-9)

    def test_manifest_preserves_sequence_blocks_with_reset_timestamps(self) -> None:
        rows = [
            {
                "sequence_id": "first",
                "frame_path": "first.png",
                "timestamp_ns": 10,
            },
            {
                "sequence_id": "second",
                "frame_path": "second.png",
                "timestamp_ns": 0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "".join(f"{json.dumps(row)}\n" for row in rows),
                encoding="utf-8",
            )
            loaded = load_manifest(path)
        self.assertEqual([row["sequence_id"] for row in loaded], ["first", "second"])

    def test_manifest_rejects_nonincreasing_timestamp_within_sequence(self) -> None:
        rows = [
            {"sequence_id": "same", "frame_path": "a.png", "timestamp_ns": 1},
            {"sequence_id": "same", "frame_path": "b.png", "timestamp_ns": 1},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "".join(f"{json.dumps(row)}\n" for row in rows),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "timestamps must increase"):
                load_manifest(path)

    def test_track_history_resets_after_processed_frame_gap(self) -> None:
        history: deque[dict[str, int]] = deque(maxlen=7)
        append_contiguous_history(history, {"frame_index": 0})
        append_contiguous_history(history, {"frame_index": 1})
        append_contiguous_history(history, {"frame_index": 3})
        self.assertEqual(list(history), [{"frame_index": 3}])

    def test_intrinsics_must_match_a_valid_frame_domain(self) -> None:
        validate_intrinsics([1000, 1000, 640, 360], (720, 1280, 3))
        with self.assertRaisesRegex(ValueError, "focal lengths"):
            validate_intrinsics([0, 1000, 640, 360], (720, 1280, 3))
        with self.assertRaisesRegex(ValueError, "principal point"):
            validate_intrinsics([1000, 1000, 2000, 360], (720, 1280, 3))


if __name__ == "__main__":
    unittest.main()
