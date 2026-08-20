from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from mine_goal_episodes import Thresholds, mine


def csv_bytes(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class EpisodeMinerTest(unittest.TestCase):
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
