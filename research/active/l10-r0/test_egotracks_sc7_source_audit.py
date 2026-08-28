import json
import tempfile
import unittest
from pathlib import Path

from egotracks_sc7_source_audit import audit


def boxes(frames: list[int]) -> list[dict[str, int]]:
    return [{"frame_number": frame, "x": 1, "y": 1, "width": 10, "height": 20} for frame in frames]


def query(title: str, frames: list[int], valid: bool = True) -> dict:
    return {
        "is_valid": valid,
        "errors": [],
        "warnings": [],
        "object_title": title,
        "visual_crop": {"frame_number": 0, "x": 1, "y": 1, "width": 10, "height": 20},
        "lt_track": boxes(frames),
        "visual_clip": [],
    }


class EgoTracksSourceAuditTest(unittest.TestCase):
    def test_selects_only_building_door_with_reacquisition_gap(self) -> None:
        eligible_frames = list(range(20)) + list(range(30, 45))
        payload = {
            "version": "fixture",
            "videos": [
                {
                    "video_uid": "video-a",
                    "clips": [
                        {
                            "clip_uid": "clip-a",
                            "annotations": [
                                {
                                    "query_sets": {
                                        "1": query("main entrance door", eligible_frames),
                                        "2": query("cabinet door", eligible_frames),
                                        "3": query("door", list(range(35))),
                                        "4": query("door", eligible_frames, valid=False),
                                    }
                                }
                            ],
                        }
                    ],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "egotracks_fixture.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = audit(path, "fixture", 12)
        self.assertEqual(4, result["query_set_count"])
        self.assertEqual(3, result["building_door_query_set_count"])
        self.assertEqual(1, result["eligible_track_count"])
        self.assertEqual(1, result["cohort_count"])
        selected = result["cohort"][0]
        self.assertEqual("main entrance door", selected["object_title"])
        self.assertEqual(10, selected["max_gap_frames"])
        self.assertEqual(30, selected["first_reentry_frame"])

    def test_rejects_missing_videos_array(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "videos array"):
                audit(path, "fixture", 12)


if __name__ == "__main__":
    unittest.main()
