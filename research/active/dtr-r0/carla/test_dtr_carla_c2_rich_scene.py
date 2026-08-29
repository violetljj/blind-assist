from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dtr_carla_c2_rich_scene as subject  # noqa: E402


PROTOCOL_PATH = Path(__file__).with_name("dtr_carla_c2_rich_scene_protocol.json")


class RichSceneProtocolTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))

    def test_protocol_validates(self) -> None:
        subject.validate_protocol(self.protocol)

    def test_formal_resolution_and_intrinsics_are_720p(self) -> None:
        self.assertEqual([1280, 720], self.protocol["capture"]["resolution"])
        matrix = subject.camera_intrinsics(1280, 720, 90.0)
        self.assertAlmostEqual(640.0, matrix[0][0], places=9)
        self.assertAlmostEqual(640.0, matrix[1][1], places=9)
        self.assertEqual([640.0, 360.0], [matrix[0][2], matrix[1][2]])

    def test_three_layouts_have_material_asset_counts(self) -> None:
        counts = {
            key: len(subject.materialize_layout_assets(self.protocol, key))
            for key in self.protocol["layouts"]
        }
        self.assertEqual(
            {"layout_01": 28, "layout_02": 32, "layout_03": 30}, counts
        )

    def test_pack_has_at_least_sixty_distinct_preferred_blueprints(self) -> None:
        values = {self.protocol["wearer"]["blueprint_candidates"][0]}
        for layout_id in self.protocol["layouts"]:
            values.update(
                asset["blueprint_candidates"][0]
                for asset in subject.materialize_layout_assets(
                    self.protocol, layout_id
                )
            )
        self.assertEqual(74, len(values))

    def test_occlusion_twin_is_identical_through_two_seconds(self) -> None:
        first, second = self.protocol["scenarios"][:2]
        sample_s = self.protocol["environment"]["sample_seconds"]
        self.assertTrue(
            subject.trajectory_prefix_equal(
                self.protocol["trajectory_library"][first["wearer_trajectory"]],
                self.protocol["trajectory_library"][second["wearer_trajectory"]],
                end_s=2.0,
                sample_s=sample_s,
            )
        )
        for key in ("target_primary", "moving_occluder"):
            self.assertTrue(
                subject.trajectory_prefix_equal(
                    self.protocol["trajectory_library"][
                        first["asset_trajectories"][key]
                    ],
                    self.protocol["trajectory_library"][
                        second["asset_trajectories"][key]
                    ],
                    end_s=2.0,
                    sample_s=sample_s,
                )
            )

    def test_occlusion_contract_uses_normalized_fractions(self) -> None:
        contract = self.protocol["occlusion_contracts"][0]
        self.assertGreater(contract["minimum_trackable_pixel_fraction"], 0.0)
        self.assertLess(contract["minimum_trackable_pixel_fraction"], 1.0)
        self.assertEqual(0.0, contract["complete_occlusion_pixel_fraction"])
        serialized = json.dumps(contract, sort_keys=True).lower()
        self.assertNotIn("minimum_visible_pixels", serialized)
        self.assertNotIn("pixel_count_threshold", serialized)

    def test_model_record_allowlist_rejects_actor_oracle(self) -> None:
        record = {
            "schema_version": "dtr-c2-model-observation-v2",
            "episode_id": "ep_01",
            "sample_index": 0,
            "world_frame": 100,
            "time_s": 0.0,
            "timestamp_s": 0.0,
            "wearable_rgb": {
                "path": "x",
                "bytes": 1,
                "sha256": "A",
                "source_world_frame": 100,
            },
            "metric_depth": {
                "path": "y",
                "bytes": 1,
                "sha256": "B",
                "source_world_frame": 170,
            },
            "camera": {"K": subject.camera_intrinsics(1280, 720, 90.0)},
            "wearer_pose_current": {"x": 0.0, "y": 0.0, "z": 1.0},
            "navigation": {
                "navigation_session_id": "session_pair_01",
                "issued_plan": {
                    "authority": "VALID",
                    "path": "plans/ep_01.json",
                    "receipt_sha256": "C",
                },
            },
            "frame_alignment": {
                "authority": "DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED",
                "reference_modality": "wearable_rgb",
                "receipt_path": "rgbd_alignment_receipt.json",
                "receipt_sha256": "D",
                "depth_minus_wearable_source_world_frame_offset": 70,
            },
        }
        subject.validate_model_record(record)
        contaminated = copy.deepcopy(record)
        contaminated["actors"] = {"target": {"actor_id": 7}}
        with self.assertRaises(ValueError):
            subject.validate_model_record(contaminated)

    def test_contiguous_runs_preserve_discrete_occlusion_duration(self) -> None:
        self.assertEqual(
            [[1, 2, 3], [8, 9], [12]],
            subject.contiguous_runs([3, 2, 1, 8, 9, 12]),
        )
        self.assertAlmostEqual(0.4, len(list(range(22, 30))) * 0.05)

    def test_rgbd_alignment_receipt_preserves_both_source_frames(self) -> None:
        def row(sample: int, source_frame: int) -> dict:
            return {
                "sample_index": sample,
                "time_s": sample * 0.05,
                "world_frame": source_frame,
                "camera_transform": {
                    "x": 1.0,
                    "y": 2.0,
                    "z": 3.0,
                    "pitch": -5.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                },
                "wearer_transform": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.8,
                    "pitch": 0.0,
                    "yaw": 0.0,
                    "roll": 0.0,
                },
            }

        receipt = subject.build_rgbd_alignment_receipt(
            {"ep_01": [row(0, 100), row(1, 101)]},
            {"ep_01": [row(0, 170), row(1, 171)]},
        )
        episode = receipt["episodes"][0]
        self.assertEqual(70, episode["depth_minus_wearable_source_world_frame_offset"])
        self.assertEqual(100, episode["wearable_source_world_frame_first"])
        self.assertEqual(170, episode["depth_source_world_frame_first"])
        self.assertEqual("DETERMINISTIC_REPLAY_ALIGNMENT_VERIFIED", receipt["authority"])
        self.assertEqual(64, len(receipt["receipt_sha256"]))

    def test_layout_receipts_change_when_assets_change(self) -> None:
        original = subject.layout_receipt(self.protocol, "layout_01")
        changed = copy.deepcopy(self.protocol)
        changed["layouts"]["layout_01"]["assets"][0]["role"] = "changed"
        mutated = subject.layout_receipt(changed, "layout_01")
        self.assertNotEqual(original["receipt_sha256"], mutated["receipt_sha256"])

    def test_contact_union_scores_all_declared_polygons(self) -> None:
        polygons = {
            "near": [[0.2, -0.2], [0.4, -0.2], [0.4, 0.2], [0.2, 0.2]],
            "far": [[3.0, -0.2], [3.2, -0.2], [3.2, 0.2], [3.0, 0.2]],
        }
        contact, distance, responsible = subject.contact_union(
            (0.0, 0.0), polygons, wearer_radius_m=0.45
        )
        self.assertTrue(contact)
        self.assertLess(distance, 0.45)
        self.assertEqual(["near"], responsible)

    def test_depth_codec_is_explicit_and_metric(self) -> None:
        codec = self.protocol["capture"]["camera_calibration"]["depth_codec"]
        self.assertEqual(1000.0, codec["maximum_depth_m"])
        self.assertIn("16777215", codec["formula"])
        self.assertIn("meters=", codec["formula"])

    def test_plan_receipt_converts_layout_waypoints_to_world(self) -> None:
        scenario = self.protocol["scenarios"][0]
        receipt = subject.build_plan_receipt(scenario["issued_plan"])
        anchor = self.protocol["layouts"][scenario["layout_id"]]["anchor"]
        world = subject.plan_waypoints_world(receipt, anchor)
        self.assertEqual(len(receipt["time_parameterized_waypoints"]), len(world))
        first = receipt["time_parameterized_waypoints"][0]
        expected_x = (
            anchor["center_xy_m"][0]
            + anchor["forward_xy"][0] * first["forward_m"]
            + anchor["right_xy"][0] * first["right_m"]
        )
        self.assertAlmostEqual(expected_x, world[0]["x_m"], places=6)


if __name__ == "__main__":
    unittest.main()
