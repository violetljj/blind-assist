from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

from real_observation_adapter import (
    BBox,
    CameraCalibration,
    CausalPersonTracker,
    CausalPoseLookup,
    Detection,
    EXPECTED_SCENE_COUNTS,
    MANIFEST_SCHEMA,
    PoseSample,
    project_bbox_bottom_center,
    validate_frozen_manifest,
)


def frozen_manifest_value() -> dict[str, object]:
    camera = {
        "image_width_px": 640,
        "image_height_px": 480,
        "fx_px": 400.0,
        "fy_px": 400.0,
        "cx_px": 320.0,
        "cy_px": 240.0,
        "camera_height_m": 1.5,
        "pitch_down_rad": 0.05,
    }
    episodes: list[dict[str, object]] = []
    for scene_type in EXPECTED_SCENE_COUNTS:
        for index in range(4):
            episode_id = f"{scene_type}-{index:02d}"
            episodes.append(
                {
                    "episode_id": episode_id,
                    "scene_type": scene_type,
                    "video_path": f"videos/{episode_id}.mp4",
                    "pose_jsonl_path": f"poses/{episode_id}.jsonl",
                    "camera": dict(camera),
                }
            )
    return {"schema_version": MANIFEST_SCHEMA, "frozen": True, "episodes": episodes}


class RealObservationAdapterTests(unittest.TestCase):
    def test_ground_projection_and_horizon_rejection(self) -> None:
        camera = CameraCalibration(640, 480, 400.0, 400.0, 320.0, 240.0, 1.5, 0.0)
        projected = project_bbox_bottom_center(
            BBox(350.0, 100.0, 370.0, 440.0), camera
        )
        self.assertIsNotNone(projected)
        assert projected is not None
        self.assertAlmostEqual(projected.forward_m, 3.0)
        self.assertAlmostEqual(projected.left_m, -0.3)
        self.assertIsNone(
            project_bbox_bottom_center(BBox(300.0, 100.0, 340.0, 200.0), camera)
        )

    def test_pose_lookup_is_causal_and_limited_blocks_old_tracking(self) -> None:
        samples = [
            PoseSample(1.0, "TRACKING", 1.0, 2.0, 0.10, 0.70),
            PoseSample(1.4, "LIMITED"),
            PoseSample(2.0, "TRACKING", 2.0, 3.0, 0.20, 0.90),
        ]
        lookup = CausalPoseLookup(samples, maximum_age_s=1.0)
        self.assertEqual(lookup.resolve(0.9).input_health, "NO_CAUSAL_POSE")
        causal = lookup.resolve(1.3)
        self.assertIsNotNone(causal.sample)
        assert causal.sample is not None
        self.assertEqual(
            (causal.sample.body_yaw_rad, causal.sample.sensor_yaw_rad),
            (0.10, 0.70),
        )
        self.assertEqual(lookup.resolve(1.5).input_health, "LATEST_POSE_LIMITED")
        self.assertEqual(lookup.resolve(2.0).sample, samples[2])
        stale = CausalPoseLookup(samples[:1], maximum_age_s=0.4).resolve(1.5)
        self.assertEqual(stale.input_health, "STALE_TRACKING_POSE")

    def test_tracker_preserves_id_for_iou_or_footpoint_match(self) -> None:
        tracker = CausalPersonTracker(
            minimum_iou=0.20,
            maximum_footpoint_distance_px=25.0,
            maximum_track_age_s=0.50,
        )
        first = tracker.update(
            [Detection(BBox(10.0, 10.0, 30.0, 60.0), 0.9)], time_s=0.0
        )
        overlap = tracker.update(
            [Detection(BBox(12.0, 11.0, 32.0, 61.0), 0.8)], time_s=0.1
        )
        footpoint = tracker.update(
            [Detection(BBox(25.0, 42.0, 35.0, 62.0), 0.8)], time_s=0.2
        )
        distant = tracker.update(
            [Detection(BBox(200.0, 10.0, 230.0, 60.0), 0.9)], time_s=0.3
        )
        self.assertEqual(first[0].track_id, overlap[0].track_id)
        self.assertEqual(first[0].track_id, footpoint[0].track_id)
        self.assertNotEqual(first[0].track_id, distant[0].track_id)

    def test_manifest_is_frozen_balanced_truth_blind_24(self) -> None:
        path = Path.cwd() / "not-written-manifest.json"

        def validate(value: dict[str, object]):
            return validate_frozen_manifest(value, manifest_path=path, require_files=False)

        self.assertEqual(len(validate(frozen_manifest_value()).episodes), 24)
        invalid: list[dict[str, object]] = []
        not_frozen = frozen_manifest_value()
        not_frozen["frozen"] = False
        invalid.append(not_frozen)
        wrong_size = frozen_manifest_value()
        assert isinstance(wrong_size["episodes"], list)
        wrong_size["episodes"].pop()
        invalid.append(wrong_size)
        unbalanced = frozen_manifest_value()
        assert isinstance(unbalanced["episodes"], list)
        assert isinstance(unbalanced["episodes"][0], dict)
        unbalanced["episodes"][0]["scene_type"] = "oncoming"
        invalid.append(unbalanced)
        duplicate = frozen_manifest_value()
        assert isinstance(duplicate["episodes"], list)
        assert isinstance(duplicate["episodes"][0], dict)
        assert isinstance(duplicate["episodes"][1], dict)
        duplicate["episodes"][1]["episode_id"] = duplicate["episodes"][0]["episode_id"]
        invalid.append(duplicate)
        with_truth = deepcopy(frozen_manifest_value())
        assert isinstance(with_truth["episodes"], list)
        assert isinstance(with_truth["episodes"][0], dict)
        with_truth["episodes"][0]["truth"] = {"forbidden": True}
        invalid.append(with_truth)
        for value in invalid:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    validate(value)


if __name__ == "__main__":
    unittest.main()
