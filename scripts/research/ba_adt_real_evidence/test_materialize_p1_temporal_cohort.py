from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from materialize_p1_temporal_cohort import SourceSpec, materialize


def csv_bytes(fieldnames, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def make_source(root: Path, sequence_id: str, uid_base: int) -> SourceSpec:
    archive = root / f"{sequence_id}.zip"
    video = root / f"{sequence_id}.mp4"
    video.write_bytes(b"fixture-video-not-decoded")
    frame_times = [index * 33_333_000 for index in range(180)]
    trajectory = [
        {
            "tracking_timestamp_us": timestamp // 1000,
            "tx_world_device": 0,
            "ty_world_device": 0,
            "tz_world_device": 0,
        }
        for timestamp in frame_times
    ]
    instances = {}
    bbox_rows = []
    for target_index in range(6):
        uid = str(uid_base + target_index)
        category_uid = 999 if target_index >= 4 else uid_base + target_index
        instances[uid] = {
            "instance_id": int(uid),
            "instance_name": f"Object_{uid}",
            "prototype_name": f"Prototype_{category_uid}",
            "category": f"category_{category_uid}",
            "category_uid": category_uid,
            "instance_type": "object",
        }
        if target_index == 0:
            visible = set(range(0, 25)) | set(range(100, 140))
            annotated = set(range(180))
        elif target_index == 1:
            visible = set(range(0, 35)) | set(range(43, 100))
            annotated = set(range(180))
        elif target_index == 2:
            visible = set(range(0, 35)) | set(range(55, 110))
            annotated = visible
        else:
            visible = set(range(10, 160))
            annotated = visible
        for index, timestamp in enumerate(frame_times):
            if index not in annotated:
                continue
            bbox_rows.append(
                {
                    "stream_id": "214-1",
                    "object_uid": uid,
                    "timestamp[ns]": timestamp,
                    "x_min[pixel]": 10 + target_index * 20,
                    "x_max[pixel]": 25 + target_index * 20,
                    "y_min[pixel]": 20,
                    "y_max[pixel]": 40,
                    "visibility_ratio[%]": 0.9 if index in visible else 0.0,
                }
            )
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("instances.json", json.dumps(instances))
        zf.writestr("aria_trajectory.csv", csv_bytes(list(trajectory[0]), trajectory))
        zf.writestr("2d_bounding_box.csv", csv_bytes(list(bbox_rows[0]), bbox_rows))
    return SourceSpec(sequence_id, archive, video)


class TemporalCohortMaterializerTest(unittest.TestCase):
    def test_materializes_bounded_gt_only_cohort_and_safety_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = [make_source(root, "seq-a", 100), make_source(root, "seq-b", 200)]
            output = root / "cohort"
            summary = materialize(sources, output, episode_budget=12, probe_video=False)

            self.assertEqual(summary["terminal"], "P1_TEMPORAL_DEVELOPMENT_COHORT_READY")
            self.assertEqual(summary["episodes"], 12)
            self.assertEqual(summary["physical_targets"], 12)
            self.assertIn("REACQUISITION", summary["temporal_mode_counts"])
            manifest = json.loads((output / "p1_d0_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["model_detector_tracker_call_counts"], {"model": 0, "detector": 0, "tracker": 0})
            self.assertEqual(len(manifest["safety_cases"]), 2)
            self.assertTrue(all(case["handoff"] == "NO_REFERENT" for case in manifest["safety_cases"]))
            episode = json.loads(next((output / "episodes").glob("*.json")).read_text(encoding="utf-8"))
            self.assertTrue(episode["physical_target_id"].startswith("adt:"))
            self.assertTrue(all("target_bbox_xyxy" in frame for frame in episode["frames"]))

    def test_rejects_budget_outside_frozen_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = make_source(root, "seq-a", 100)
            with self.assertRaisesRegex(ValueError, "episode budget"):
                materialize([source], root / "cohort", episode_budget=19, probe_video=False)


if __name__ == "__main__":
    unittest.main()
