from __future__ import annotations

import argparse
import copy
import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
PROTOCOL_PATH = HERE / "dtr_carla_c0_protocol.json"


def load_module(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DTR = load_module("dtr_carla_c0_under_test", HERE / "dtr_carla_c0.py")
with mock.patch.dict(sys.modules, {"carla": types.ModuleType("carla")}):
    CAPTURE = load_module(
        "capture_dtr_carla_c0_under_test", HERE / "capture_dtr_carla_c0.py"
    )

PROTOCOL = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))


class ProtocolContractTest(unittest.TestCase):
    def test_frozen_matrix_is_six_unique_pairs_and_twelve_episodes(self) -> None:
        scenarios = PROTOCOL["scenarios"]
        contracts = PROTOCOL["twin_contracts"]
        scenario_ids = [item["id"] for item in scenarios]
        families = [item["family"] for item in scenarios]
        paired_ids = [item[key] for item in contracts for key in ("a", "b")]

        self.assertEqual(len(scenarios), 12)
        self.assertEqual(len(set(scenario_ids)), 12)
        self.assertEqual(len(contracts), 6)
        self.assertEqual(len({item["family"] for item in contracts}), 6)
        self.assertEqual(len(set(families)), 6)
        self.assertTrue(all(families.count(family) == 2 for family in set(families)))
        self.assertCountEqual(paired_ids, scenario_ids)
        self.assertEqual(len(set(paired_ids)), 12)

        by_id = {item["id"]: item for item in scenarios}
        for contract in contracts:
            self.assertEqual(by_id[contract["a"]]["twin"], "a")
            self.assertEqual(by_id[contract["b"]]["twin"], "b")
            self.assertEqual(by_id[contract["a"]]["family"], contract["family"])
            self.assertEqual(by_id[contract["b"]]["family"], contract["family"])

        report = DTR.validate_pair_protocol(PROTOCOL)
        self.assertTrue(report["all_pass"])
        self.assertEqual(len(report["checks"]), 6)

    def test_each_pair_changes_only_its_declared_protocol_field(self) -> None:
        expected = {
            "same_geometry_different_motion": (
                "target",
                "target_velocity_after_1s",
            ),
            "same_motion_different_route": (
                "ego",
                "ego_route_velocity_after_2s",
            ),
            "same_motion_different_ttc": (
                "target",
                "target_motion_start_time",
            ),
            "same_scene_different_visibility": (
                "camera_yaw_offsets",
                "sensor_yaw_only_between_2.4s_and_4.6s",
            ),
            "same_target_visible_then_occluded": (
                "occluder",
                "physical_occluder_presence",
            ),
            "same_background_static_then_dynamic": (
                "target",
                "off_route_target_velocity_after_1s",
            ),
        }
        by_id = {item["id"]: item for item in PROTOCOL["scenarios"]}
        metadata = {"id", "twin", "expected_critical"}

        for contract in PROTOCOL["twin_contracts"]:
            family = contract["family"]
            left, right = by_id[contract["a"]], by_id[contract["b"]]
            differing_fields = {
                key
                for key in set(left) | set(right)
                if key not in metadata and left.get(key) != right.get(key)
            }
            self.assertEqual(differing_fields, {expected[family][0]}, family)
            self.assertEqual(contract["allowed_difference"], expected[family][1])

        contaminated = copy.deepcopy(PROTOCOL)
        by_bad_id = {item["id"]: item for item in contaminated["scenarios"]}
        by_bad_id["motion_hold"]["camera_yaw_offsets"] = [
            {"start_s": 0.0, "yaw_degrees": 5.0}
        ]
        with self.assertRaisesRegex(
            ValueError, "same_geometry_different_motion:camera_schedule"
        ):
            DTR.validate_pair_protocol(contaminated)

    def test_nonzero_boundary_pairs_have_equal_pre_intervention_state(self) -> None:
        by_id = {item["id"]: item for item in PROTOCOL["scenarios"]}
        for contract in PROTOCOL["twin_contracts"]:
            boundary = float(contract["identical_before_s"])
            if boundary <= 0.0:
                continue
            left, right = by_id[contract["a"]], by_id[contract["b"]]
            for time_s in (0.0, boundary - 1e-6):
                for actor in ("ego", "target"):
                    np.testing.assert_allclose(
                        CAPTURE.trajectory_position(left[actor], time_s),
                        CAPTURE.trajectory_position(right[actor], time_s),
                        atol=1e-9,
                    )
                    np.testing.assert_allclose(
                        CAPTURE.trajectory_velocity(left[actor], time_s),
                        CAPTURE.trajectory_velocity(right[actor], time_s),
                        atol=1e-9,
                    )
                self.assertAlmostEqual(
                    CAPTURE.scheduled_camera_yaw(left, time_s),
                    CAPTURE.scheduled_camera_yaw(right, time_s),
                )


class TrajectoryAndTruthBoundaryTest(unittest.TestCase):
    def test_flow_visualization_uses_the_supported_carla_helper(self) -> None:
        capture_source = inspect.getsource(CAPTURE.main)
        self.assertIn("utility.save_flow_visualization(image, payload_path)", capture_source)
        self.assertNotIn("get_color_coded_flow().save_to_disk", capture_source)

    def test_piecewise_trajectory_switches_velocity_at_exact_boundary(self) -> None:
        trajectory = {
            "start_forward_m": 1.0,
            "start_right_m": 2.0,
            "segments": [
                {
                    "start_s": 1.0,
                    "velocity_forward_mps": 0.0,
                    "velocity_right_mps": -4.0,
                },
                {
                    "start_s": 0.0,
                    "velocity_forward_mps": 2.0,
                    "velocity_right_mps": 0.0,
                },
            ],
        }
        np.testing.assert_allclose(CAPTURE.trajectory_position(trajectory, 0.0), [1, 2])
        np.testing.assert_allclose(CAPTURE.trajectory_position(trajectory, 1.0), [3, 2])
        np.testing.assert_allclose(CAPTURE.trajectory_velocity(trajectory, 1.0), [0, -4])
        np.testing.assert_allclose(CAPTURE.trajectory_position(trajectory, 1.5), [3, 0])

        invalid = copy.deepcopy(trajectory)
        invalid["segments"] = invalid["segments"][:1]
        with self.assertRaisesRegex(ValueError, "start at t=0"):
            CAPTURE.trajectory_position(invalid, 1.0)

    def test_future_truth_and_event_edges_are_inclusive_and_bounded(self) -> None:
        capture_source = inspect.getsource(CAPTURE.main)
        self.assertIn("records[index:]", capture_source)
        self.assertIn("<= horizon_s + 1e-9", capture_source)
        self.assertIn('record["truth"]["future_contact_within_horizon"]', capture_source)

        def row(time_s: float, current: bool, future: bool) -> dict[str, object]:
            return {
                "time_s": time_s,
                "truth": {
                    "current_contact": current,
                    "future_contact_within_horizon": future,
                },
            }

        rows = [
            row(0.0, False, True),  # Contact at 3.0 s is on the horizon boundary.
            row(2.95, False, True),
            row(3.0, True, True),
            row(3.05, True, True),
            row(3.10, False, False),
        ]
        self.assertEqual(
            DTR.event_truth(rows),
            {
                "critical_event": True,
                "warning_start_s": 0.0,
                "event_start_s": 3.0,
                "event_end_s": 3.05,
                "exit_time_s": 3.10,
            },
        )
        self.assertEqual(
            DTR.event_truth([row(0.0, False, False)]),
            {
                "critical_event": False,
                "warning_start_s": None,
                "event_start_s": None,
                "event_end_s": None,
                "exit_time_s": None,
            },
        )


class DepthContractTest(unittest.TestCase):
    @staticmethod
    def encoded_depth(depth_m: float, *, width: int = 9, height: int = 9) -> np.ndarray:
        raw = round(depth_m / 1000.0 * 16777215.0)
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:, :, 0] = (raw >> 16) & 0xFF  # B in an image decoded by OpenCV.
        image[:, :, 1] = (raw >> 8) & 0xFF
        image[:, :, 2] = raw & 0xFF  # CARLA's least-significant R channel.
        return image

    def test_carla_bgr_depth_decode_and_center_projection(self) -> None:
        image = self.encoded_depth(10.0)
        decoded = DTR.decode_depth_m(image, 4, 4)
        self.assertIsNotNone(decoded)
        assert decoded is not None
        self.assertAlmostEqual(decoded, 10.0, places=3)
        self.assertIsNone(DTR.decode_depth_m(np.zeros((9, 9), dtype=np.uint8), 4, 4))
        self.assertIsNone(DTR.decode_depth_m(np.zeros((9, 9, 3), dtype=np.uint8), 4, 4))

        observation = DTR.depth_project(
            DTR.real.BBox(3.0, 0.0, 5.0, 5.0),
            image,
            {"x": 1.0, "y": 2.0, "pitch": 0.0, "yaw": 0.0},
            DTR.EgoPose(
                x_m=1.0,
                y_m=-2.0,
                body_yaw_rad=0.0,
                sensor_yaw_rad=0.0,
            ),
            fov_degrees=90.0,
            radius_m=0.3,
        )
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertAlmostEqual(observation.forward_m, decoded, places=6)
        self.assertAlmostEqual(observation.left_m, 0.0, places=6)
        self.assertEqual(observation.radius_m, 0.3)


class ObservationAuthorityContractTest(unittest.TestCase):
    def test_predict_is_truth_blind_with_teacher_and_current_only_roles(self) -> None:
        predict_source = inspect.getsource(DTR.predict)
        for forbidden in (
            '"truth.json"',
            '"current_contact"',
            '"future_contact_within_horizon"',
            '"target_instance"',
        ):
            self.assertNotIn(forbidden, predict_source)
        self.assertIn('root / "teacher-index.json"', predict_source)
        self.assertIn('teacher_episode["flow_teacher_path"]', predict_source)
        self.assertIn('oracle["target_x_m"]', predict_source)
        self.assertIn('oracle["target_y_m"]', predict_source)

        width, height = (int(value) for value in PROTOCOL["camera"]["resolution"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            flow_path = root / "teacher-flow.npz"
            np.savez_compressed(
                flow_path,
                flow_xy=np.zeros((1, height, width, 2), dtype=np.float32),
            )
            protocol_hash = DTR.sha256_file(PROTOCOL_PATH)
            frame = {
                "sample_index": 0,
                "time_s": 0.0,
                "rgb_path": str(root / "not-read-by-fake-rgb.png"),
                "depth_path": str(root / "not-read-by-fake-depth.png"),
                "camera_transform": {
                    "x": 0.0,
                    "y": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
                "ego_pose": {
                    "x_m": 0.0,
                    "y_m": 0.0,
                    "body_yaw_rad": 0.0,
                    "sensor_yaw_rad": 0.0,
                },
            }
            observation_index = {
                "schema_version": DTR.OBSERVATION_SCHEMA,
                "protocol_sha256": protocol_hash,
                "episodes": [
                    {
                        "episode_id": "contract-episode",
                        "frames": [frame],
                    }
                ],
            }
            oracle_index = {
                "schema_version": DTR.ORACLE_SCHEMA,
                "protocol_sha256": protocol_hash,
                "episodes": [
                    {
                        "episode_id": "contract-episode",
                        "role": "PRIVILEGED_CURRENT_STATE_ONLY",
                        "uses_realized_future": False,
                        "frames": [
                            {
                                "sample_index": 0,
                                "time_s": 0.0,
                                "track_id": "oracle-target",
                                "target_x_m": 5.0,
                                "target_y_m": 0.0,
                                "radius_m": 0.3,
                            }
                        ],
                    }
                ],
            }
            teacher_index = {
                "schema_version": DTR.TEACHER_SCHEMA,
                "protocol_sha256": protocol_hash,
                "episodes": [
                    {
                        "episode_id": "contract-episode",
                        "role": (
                            "TEACHER_ONLY_CARLA_FLOW_AND_"
                            "EVALUATOR_ONLY_INSTANCE_VISIBILITY"
                        ),
                        "flow_teacher_path": str(flow_path),
                        "frames": [{"sample_index": 0, "time_s": 0.0}],
                    }
                ],
            }
            (root / "observation-index.json").write_text(
                json.dumps(observation_index), encoding="utf-8"
            )
            (root / "oracle-current-index.json").write_text(
                json.dumps(oracle_index), encoding="utf-8"
            )
            (root / "teacher-index.json").write_text(
                json.dumps(teacher_index), encoding="utf-8"
            )
            self.assertNotIn("flow_teacher_path", observation_index["episodes"][0])
            self.assertNotIn("truth", frame)
            # Prediction must succeed even though future truth is unreadable.
            (root / "truth.json").write_text("{invalid-json", encoding="utf-8")
            model_path = root / "unused-model.bin"
            model_path.write_bytes(b"unit-test")

            cv2_stub = types.ModuleType("cv2")
            cv2_stub.IMREAD_COLOR = 1
            cv2_stub.IMREAD_UNCHANGED = -1
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width, 3), dtype=np.uint8)
            cv2_stub.imread = lambda _path, flag: rgb if flag == 1 else depth
            detector = mock.Mock()
            detector.detect.return_value = []
            args = argparse.Namespace(
                root=root,
                protocol=PROTOCOL_PATH,
                model=model_path,
                confidence_threshold=0.25,
            )
            with (
                mock.patch.dict(sys.modules, {"cv2": cv2_stub}),
                mock.patch.object(
                    DTR.real,
                    "UltralyticsPersonDetector",
                    return_value=detector,
                ),
            ):
                DTR.predict(args)

            predictions = json.loads(
                (root / "predictions.json").read_text(encoding="utf-8")
            )
            self.assertFalse(predictions["truth_accessed"])
            self.assertFalse(predictions["uses_future_frames"])
            self.assertEqual(
                predictions["arm_roles"]["O2T_RGB_DEPTH_CARLA_FLOW"],
                "CARLA_FLOW_TEACHER_NOT_DEPLOYMENT",
            )
            self.assertEqual(
                predictions["arm_roles"]["O3_PRIVILEGED_CURRENT_STATE"],
                "PRIVILEGED_CURRENT_STATE_ORACLE",
            )
            self.assertEqual(set(predictions["predictions"]), set(DTR.ARMS))
            self.assertTrue(
                all(
                    "contract-episode" in predictions["predictions"][arm]
                    for arm in DTR.ARMS
                )
            )


if __name__ == "__main__":
    unittest.main()
