#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_fresh_metric_snapshot_source_r0 import sha256, validate


REPO_ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = REPO_ROOT / "docs/research/hftf/HFTF_FRESH_METRIC_SNAPSHOT_LAYERED_INTRUSION_R0_PROTOCOL_2026-08-05.json"


class FreshSnapshotSourceValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        calibration = self.root / "calibration.json"
        calibration.write_text(json.dumps({
            "schema": "blindassist_hftf_camera_calibration_v1",
            "device": "SM-S9280",
            "intrinsics_fx_fy_cx_cy": [500.0, 500.0, 320.0, 240.0],
            "reprojection_error_px": 0.25,
            "sealed_before_collection": True,
        }) + "\n", encoding="utf-8")
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        sessions = []
        timestamp = 1_000_000_000
        for session_id in protocol["cohort"]["session_ids"]:
            if session_id.startswith("clear_all_"):
                scenario = "CLEAR_ALL"; layers = {"foot": False, "body": False, "head": False}; cells = []
            elif session_id.startswith("foot_only_"):
                scenario = "FOOT_ONLY"; layers = {"foot": True, "body": False, "head": False}; cells = [{"direction_deg": 0, "distance_m": 1.0, "height": "foot"}]
            elif session_id.startswith("body_only_"):
                scenario = "BODY_ONLY"; layers = {"foot": False, "body": True, "head": False}; cells = [{"direction_deg": 0, "distance_m": 1.5, "height": "body"}]
            elif session_id.startswith("head_only_"):
                scenario = "HEAD_ONLY"; layers = {"foot": False, "body": False, "head": True}; cells = [{"direction_deg": 0, "distance_m": 2.0, "height": "head"}]
            elif session_id.startswith("multi_layer_"):
                scenario = "MULTI_LAYER"; layers = {"foot": True, "body": True, "head": False}; cells = [{"direction_deg": 0, "distance_m": 1.0, "height": "foot"}, {"direction_deg": 0, "distance_m": 1.5, "height": "body"}]
            else:
                scenario = "LEFT_RIGHT_HEIGHT_COMPETITION"; layers = {"foot": True, "body": False, "head": True}; cells = [{"direction_deg": -25, "distance_m": 1.0, "height": "foot"}, {"direction_deg": 25, "distance_m": 1.5, "height": "head"}]
            truth = self.root / f"{session_id}-truth.json"
            truth.write_text(json.dumps({
                "schema": "blindassist_hftf_physical_intrusion_truth_v1",
                "session_id": session_id,
                "qnn_used_for_truth": False,
                "supports_outside_evaluated_envelopes": True,
                "measurement_tools": ["rigid_ruler_or_tape", "laser_distance_meter", "level", "fiducial_pose_board"],
                "distance_error_m": 0.01,
                "height_lateral_error_m": 0.005,
                "obstacle_prisms": [] if scenario == "CLEAR_ALL" else [{"x_min_m": -0.1, "x_max_m": 0.1, "y_min_m": 0.9, "y_max_m": 1.1, "z_min_m": 0.1, "z_max_m": 0.2}],
                "expected_intrusion_by_height": layers,
                "truth_cells": cells,
            }) + "\n", encoding="utf-8")
            frames = []
            sealed = timestamp - 1000
            for index in range(10):
                rgb = self.root / f"{session_id}-{index}.rgb"
                depth = self.root / f"{session_id}-{index}.depth"
                rgb.write_bytes(f"rgb-{session_id}-{index}".encode())
                depth.write_bytes(f"depth-{session_id}-{index}".encode())
                capture = timestamp + index * 100_000_000
                frames.append({
                    "frame_index": index,
                    "capture_timestamp_ns": capture,
                    "depth_capture_timestamp_ns": capture,
                    "depth_completed_timestamp_ns": capture + 90_000_000,
                    "rgb_shape": [480, 640],
                    "depth_shape": [259, 343],
                    "depth_source": "CAMERAX_SAME_FRAME_QNN_METRIC_DEPTH",
                    "rgb": {"path": rgb.name, "sha256": sha256(rgb)},
                    "depth": {"path": depth.name, "sha256": sha256(depth)},
                })
            sessions.append({
                "session_id": session_id,
                "parent_capture_id": f"parent-{session_id}",
                "scenario_class": scenario,
                "lens_height_m": 1.55,
                "pitch_error_deg": 0.5,
                "roll_error_deg": -0.5,
                "yaw_error_deg": 0.25,
                "truth_sealed_before_qnn_output": True,
                "truth_sealed_timestamp_ns": sealed,
                "physical_truth": {"path": truth.name, "sha256": sha256(truth)},
                "expected_intrusion_by_height": layers,
                "truth_cells": cells,
                "frames": frames,
            })
            timestamp += 2_000_000_000
        self.manifest = {
            "schema": "blindassist_hftf_fresh_metric_snapshot_source_package_v1",
            "protocol_sha256": sha256(PROTOCOL),
            "package_role": "FORMAL_DECISION_COHORT",
            "device": "SM-S9280",
            "camera_calibration": {
                "path": calibration.name,
                "sha256": sha256(calibration),
                "sealed_before_collection": True,
            },
            "sessions": sessions,
        }
        self.manifest_path = self.root / "manifest.json"
        self._write()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self) -> None:
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def test_complete_package_is_admitted(self) -> None:
        report = validate(PROTOCOL, self.root, self.manifest_path)
        self.assertEqual(18, report["session_count"])
        self.assertEqual(180, report["snapshot_count"])

    def test_missing_session_is_rejected(self) -> None:
        self.manifest["sessions"].pop(); self._write()
        with self.assertRaisesRegex(ValueError, "roster"):
            validate(PROTOCOL, self.root, self.manifest_path)

    def test_truth_after_qnn_is_rejected(self) -> None:
        session = self.manifest["sessions"][0]
        session["truth_sealed_timestamp_ns"] = session["frames"][0]["depth_completed_timestamp_ns"]
        self._write()
        with self.assertRaisesRegex(ValueError, "truth not sealed"):
            validate(PROTOCOL, self.root, self.manifest_path)

    def test_path_escape_is_rejected(self) -> None:
        self.manifest["camera_calibration"]["path"] = "../outside.json"; self._write()
        with self.assertRaisesRegex(ValueError, "escapes"):
            validate(PROTOCOL, self.root, self.manifest_path)

    def test_hash_mismatch_is_rejected(self) -> None:
        self.manifest["sessions"][0]["frames"][0]["rgb"]["sha256"] = "0" * 64; self._write()
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate(PROTOCOL, self.root, self.manifest_path)

    def test_outcome_key_is_rejected(self) -> None:
        self.manifest["sessions"][0]["arm_outputs"] = {}; self._write()
        with self.assertRaisesRegex(ValueError, "forbidden outcome key"):
            validate(PROTOCOL, self.root, self.manifest_path)


if __name__ == "__main__":
    unittest.main()
