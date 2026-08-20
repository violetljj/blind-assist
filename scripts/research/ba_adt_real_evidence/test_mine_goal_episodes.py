from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mine_goal_episodes import Thresholds, mine
from run_rgb_observer import choose_target, flow_bbox, iou
from evaluate_rgb_observations import metrics
from build_offline_demo import copilot_state


def csv_bytes(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class EpisodeMinerTest(unittest.TestCase):
    def test_sparse_flow_propagates_bbox_translation(self):
        import cv2
        import numpy as np

        first = np.zeros((80, 80), dtype=np.uint8)
        for point in ((22, 22), (30, 22), (22, 30), (30, 30)):
            cv2.circle(first, point, 2, 255, -1)
        matrix = np.float32([[1, 0, 4], [0, 1, 3]])
        second = cv2.warpAffine(first, matrix, (80, 80))
        points = np.float32([[22, 22], [30, 22], [22, 30], [30, 30]]).reshape(-1, 1, 2)
        bbox, moved = flow_bbox(first, second, points, [18, 18, 34, 34], 80, 80)
        self.assertIsNotNone(moved)
        self.assertAlmostEqual(bbox[0], 22, delta=0.6)
        self.assertAlmostEqual(bbox[1], 21, delta=0.6)

    def test_offline_copilot_state_keeps_short_dropout_uncertain(self):
        self.assertEqual(copilot_state(False, False, 99, 99, None), "SEARCHING")
        self.assertEqual(copilot_state(False, True, 8, 99, None), "UNCERTAIN")
        self.assertEqual(copilot_state(False, True, 31, 99, None), "LOST")
        self.assertEqual(copilot_state(True, True, 0, 2, None), "REACQUIRED")
        self.assertEqual(copilot_state(True, True, 0, 20, 0.01), "APPROACHING")

    def test_evaluator_reports_reacquisition_and_quality_separation(self):
        rows = []
        for index in range(50):
            gt_visible = index < 15 or index >= 25
            localized = gt_visible and index not in {25, 26}
            rows.append({
                "gt_visible": gt_visible,
                "predicted_visible": localized,
                "localized": localized,
                "iou": 0.8 if localized else 0.0,
                "bearing_error_normalized": 0.02 if localized else None,
                "predicted_scale": 0.01 * index if localized else None,
                "truth_scale": 0.01 * index if localized else None,
                "observation_quality": 0.9 if localized else 0.1,
            })
        result = metrics(rows)
        self.assertEqual(result["eligible_reacquisition_count"], 1)
        self.assertEqual(result["reacquisition_delay_frames"], [2])
        self.assertGreater(result["observation_quality_mean_when_localized"], result["observation_quality_mean_when_missed"])

    def test_iou_association_prefers_temporal_match(self):
        previous = [0.0, 0.0, 10.0, 10.0]
        candidates = [
            {"bbox_xyxy": [0.0, 0.0, 9.0, 9.0], "confidence": 0.6},
            {"bbox_xyxy": [50.0, 50.0, 60.0, 60.0], "confidence": 0.95},
        ]
        selected, overlap = choose_target(candidates, previous)
        self.assertEqual(selected, candidates[0])
        self.assertGreater(overlap, 0.8)
        self.assertAlmostEqual(iou(previous, previous), 1.0)

    def test_mines_search_track_loss_reacquire_and_approach(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "fixture.zip"
            times = [index * 1_000_000_000 for index in range(40)]
            visible = set(range(6, 19)) | set(range(26, 40))
            bbox_rows = [{"stream_id": "214-1", "object_uid": "7", "timestamp[ns]": timestamp, "x_min[pixel]": 100, "x_max[pixel]": 200, "y_min[pixel]": 100, "y_max[pixel]": 200, "visibility_ratio[%]": 0.9 if index in visible else 0.0} for index, timestamp in enumerate(times)]
            trajectory_rows = [{"tracking_timestamp_us": timestamp // 1000, "tx_world_device": index * 0.1, "ty_world_device": 0, "tz_world_device": 0} for index, timestamp in enumerate(times)]
            scene_rows = [{"object_uid": "7", "timestamp[ns]": -1, "t_wo_x[m]": 5, "t_wo_y[m]": 0, "t_wo_z[m]": 0}]
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("instances.json", json.dumps({"7": {"instance_name": "chair"}}))
                zf.writestr("2d_bounding_box.csv", csv_bytes(list(bbox_rows[0]), bbox_rows))
                zf.writestr("aria_trajectory.csv", csv_bytes(list(trajectory_rows[0]), trajectory_rows))
                zf.writestr("scene_objects.csv", csv_bytes(list(scene_rows[0]), scene_rows))
            result = mine(archive, Thresholds(min_track_frames=10))
            self.assertEqual(result["candidate_count"], 1)
            self.assertEqual(result["candidates"][0]["target_name"], "chair")
            self.assertEqual(result["candidates"][0]["phases"], ["SEARCH", "ACQUIRE", "TRACK", "LOST", "REACQUIRE", "APPROACH"])
            self.assertEqual(result["rgb_estimator_access_count"], 0)

    def test_rejects_archive_without_required_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("instances.json", "{}")
            with self.assertRaisesRegex(ValueError, "missing"):
                mine(archive, Thresholds())


if __name__ == "__main__":
    unittest.main()
