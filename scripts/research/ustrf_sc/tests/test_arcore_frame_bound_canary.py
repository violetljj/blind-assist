#!/usr/bin/env python3
"""Pure tempfile contract tests for the ARCore frame-bound host validator."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from research.ustrf_sc.validate_ustrf_sc_arcore_frame_bound_canary import (  # noqa: E402
    AUDIT_SCHEMA,
    ContractError,
    validate,
)


SAFE_BOUNDARY = {
    "benchmark_only": True,
    "app_runtime_involved": False,
    "navigation_output_issued": False,
    "training_authority": False,
    "production_authorized": False,
    "human_truth": False,
}
ZERO_SHA = "0" * 64


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(value, ensure_ascii=False, sort_keys=True) for value in values) + "\n",
        encoding="utf-8",
    )


def _pose(*, z: float = 0.0) -> dict[str, object]:
    return {
        "translation_m": [0.0, 0.0, z],
        "rotation_quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        "matrix_4x4_column_major": [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, z, 1.0,
        ],
    }


def _intrinsics() -> dict[str, object]:
    return {
        "width_px": 1920,
        "height_px": 1440,
        "focal_x_px": 1200.0,
        "focal_y_px": 1200.0,
        "principal_x_px": 960.0,
        "principal_y_px": 720.0,
    }


def _image(*, timestamp_ns: int, planes: int, salt: str) -> dict[str, object]:
    return {
        "available": True,
        "timestamp_ns": timestamp_ns,
        "width_px": 1920,
        "height_px": 1440,
        "plane_count": planes,
        "content_sha256": hashlib.sha256(salt.encode("utf-8")).hexdigest(),
    }


class CanaryFixture:
    """A fully valid, hash-bound 100-frame benchmark-only capture."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.raw_path = self.root / "raw_frames.jsonl"
        self.summary_path = self.root / "summary.json"
        self.device_path = self.root / "device_receipt.json"
        self.run_id = "arcore-frame-bound-test-run"
        anchor_created_timestamp = 1_000_000_000
        self.rows = [self._row(index, anchor_created_timestamp) for index in range(100)]
        _write_jsonl(self.raw_path, self.rows)
        self.device = {
            "schema": "blindassist_ustrf_arcore_frame_bound_device_receipt_v1",
            "run_id": self.run_id,
            "device": {
                "model": "SM-S9280",
                "android_sdk_int": 35,
                "build_fingerprint": "test/device/fingerprint",
            },
            "arcore": {
                "availability": "SUPPORTED_INSTALLED",
                "sdk_dependency_version": "1.33.0",
            },
            "capture_package": "com.linnan.blindassist.ustrfbenchmark",
            "session_ownership": "EXCLUSIVE_SINGLE_SESSION",
            "autonomous_capture": True,
            "user_motion_instruction": False,
            "evidence_boundary": copy.deepcopy(SAFE_BOUNDARY),
        }
        _write_json(self.device_path, self.device)
        self.summary = {
            "schema": "blindassist_ustrf_arcore_frame_bound_canary_summary_v1",
            "run_id": self.run_id,
            "raw_frames_file": "raw_frames.jsonl",
            "device_receipt_file": "device_receipt.json",
            "raw_frames_sha256": _sha(self.raw_path),
            "device_receipt_sha256": _sha(self.device_path),
            "capture_completed": True,
            "capture_error": None,
            "raw_frame_row_count": 100,
            "session_update_attempt_count": 100,
            "frame_attempts_requested": 100,
            "depth_mode_automatic_supported": True,
            "evidence_boundary": copy.deepcopy(SAFE_BOUNDARY),
        }
        _write_json(self.summary_path, self.summary)

    def _row(self, index: int, anchor_created_timestamp: int) -> dict[str, object]:
        frame_timestamp = 1_000_000_000 + index * 33_333_333
        camera_timestamp = 5_000_000_000 + index * 33_333_333
        identity_matrix = _pose()["matrix_4x4_column_major"]
        return {
            "schema": "blindassist_ustrf_arcore_single_frame_observation_v1",
            "run_id": self.run_id,
            "frame_index": index,
            "session_update_index": index,
            "evidence_boundary": copy.deepcopy(SAFE_BOUNDARY),
            "frame_timestamp_ns": frame_timestamp,
            "android_camera_timestamp_ns": camera_timestamp,
            "camera_image": _image(timestamp_ns=frame_timestamp, planes=3, salt=f"camera:{index}"),
            "raw_depth_image": _image(timestamp_ns=frame_timestamp, planes=1, salt=f"depth:{index}"),
            "raw_confidence_image": _image(timestamp_ns=frame_timestamp, planes=1, salt=f"confidence:{index}"),
            "tracking_state": "TRACKING",
            "intrinsics": {
                "image": _intrinsics(),
                "texture": _intrinsics(),
            },
            "transforms": {
                "world_from_camera": _pose(),
                "camera_view_matrix": list(identity_matrix),
                "camera_projection_matrix": list(identity_matrix),
                "world_from_android_sensor": _pose(),
            },
            "anchor": {
                "available": True,
                "anchor_id": "persistent-anchor-01",
                "created_frame_index": 0,
                "created_frame_timestamp_ns": anchor_created_timestamp,
                "reference_mode": "INTER_FRAME_STABLE",
                "tracking_state": "TRACKING",
                "world_from_anchor": _pose(z=-1.0),
                "anchor_from_camera": _pose(z=1.0),
            },
        }

    def rewrite_raw_and_rebind_summary(self) -> None:
        _write_jsonl(self.raw_path, self.rows)
        self.summary["raw_frames_sha256"] = _sha(self.raw_path)
        _write_json(self.summary_path, self.summary)

    def rewrite_summary(self) -> None:
        _write_json(self.summary_path, self.summary)

    def run(self) -> dict[str, object]:
        return validate(self.raw_path, self.summary_path, self.device_path)


class ArcoreFrameBoundCanaryContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory()
        self.fixture = CanaryFixture(Path(self._temporary.name))

    def tearDown(self) -> None:
        self._temporary.cleanup()

    def assert_gate_closed(self, report: dict[str, object]) -> None:
        self.assertEqual(AUDIT_SCHEMA, report["schema"])
        self.assertFalse(report["gate_open"])
        self.assertEqual("FREEZE_FRAME_BOUND_METRIC_GEOMETRY", report["verdict"])

    def test_complete_100_row_fixture_opens_benchmark_only_gate(self) -> None:
        report = self.fixture.run()
        self.assertTrue(report["gate_open"])
        self.assertEqual("PASS_BENCHMARK_ONLY", report["verdict"])
        self.assertEqual(100, report["recomputed_metrics"]["valid_pair_count"])
        self.assertEqual(SAFE_BOUNDARY, report["evidence_boundary"])
        self.assertEqual(_sha(self.fixture.raw_path), report["input_bindings"]["raw_frames_sha256"])
        self.assertEqual(_sha(self.fixture.summary_path), report["input_bindings"]["summary_sha256"])

    def test_summary_declared_raw_hash_tamper_fails_closed(self) -> None:
        self.fixture.summary["raw_frames_sha256"] = ZERO_SHA
        self.fixture.rewrite_summary()
        with self.assertRaisesRegex(ContractError, "raw frame SHA-256 mismatch"):
            self.fixture.run()

    def test_raw_content_tamper_without_summary_rebind_fails_closed(self) -> None:
        self.fixture.rows[0]["tracking_state"] = "PAUSED"
        _write_jsonl(self.fixture.raw_path, self.fixture.rows)
        with self.assertRaisesRegex(ContractError, "raw frame SHA-256 mismatch"):
            self.fixture.run()

    def test_duplicate_android_camera_timestamp_closes_gate(self) -> None:
        self.fixture.rows[1]["android_camera_timestamp_ns"] = self.fixture.rows[0][
            "android_camera_timestamp_ns"
        ]
        self.fixture.rewrite_raw_and_rebind_summary()
        report = self.fixture.run()
        self.assert_gate_closed(report)
        self.assertEqual(1, report["recomputed_metrics"]["duplicate_android_camera_timestamp_count"])
        self.assertFalse(report["checks"]["duplicate_android_camera_timestamp_count_eq_0"])

    def test_one_missing_raw_depth_frame_closes_100_pair_gate(self) -> None:
        self.fixture.rows[37]["raw_depth_image"] = {"available": False}
        self.fixture.rewrite_raw_and_rebind_summary()
        report = self.fixture.run()
        self.assert_gate_closed(report)
        self.assertEqual(1, report["recomputed_metrics"]["missing_raw_depth_count"])
        self.assertEqual(99, report["recomputed_metrics"]["raw_depth_confidence_pair_count"])
        self.assertFalse(report["checks"]["raw_depth_confidence_pair_count_gte_100"])

    def test_per_frame_anchor_ids_close_persistent_anchor_gate(self) -> None:
        for index, row in enumerate(self.fixture.rows):
            row["anchor"]["anchor_id"] = f"anchor-{index:03d}"
        self.fixture.rewrite_raw_and_rebind_summary()
        report = self.fixture.run()
        self.assert_gate_closed(report)
        self.assertEqual(100, report["recomputed_metrics"]["anchor_id_count"])
        self.assertFalse(report["checks"]["one_persistent_anchor"])

    def test_non_inter_frame_stable_reference_mode_closes_gate(self) -> None:
        for row in self.fixture.rows:
            row["anchor"]["reference_mode"] = "RAW_CAMERA_POSE"
        self.fixture.rewrite_raw_and_rebind_summary()
        report = self.fixture.run()
        self.assert_gate_closed(report)
        self.assertEqual(["RAW_CAMERA_POSE"], report["recomputed_metrics"]["anchor_reference_modes"])
        self.assertFalse(report["checks"]["anchor_reference_mode_inter_frame_stable"])

    def test_unsafe_summary_evidence_boundary_fails_closed(self) -> None:
        self.fixture.summary["evidence_boundary"]["production_authorized"] = True
        self.fixture.rewrite_summary()
        with self.assertRaisesRegex(ContractError, "summary evidence boundary is unsafe"):
            self.fixture.run()

    def test_unsafe_raw_row_evidence_boundary_fails_closed(self) -> None:
        self.fixture.rows[12]["evidence_boundary"]["human_truth"] = True
        self.fixture.rewrite_raw_and_rebind_summary()
        with self.assertRaisesRegex(ContractError, "row 12 evidence boundary is unsafe"):
            self.fixture.run()


if __name__ == "__main__":
    unittest.main()
