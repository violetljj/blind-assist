from __future__ import annotations

import copy
from decimal import Decimal
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.research.egomotion_compensated_looming.rgb_segment_confirmation_r1 import (
    opaque_transport as transport_module,
)
from scripts.research.egomotion_compensated_looming.rgb_segment_confirmation_r1.build_preaccess_lock import (
    build,
    enforce_open_pairing,
)
from scripts.research.egomotion_compensated_looming.rgb_segment_confirmation_r1.opaque_transport import (
    BudgetExceeded,
    BudgetedRemoteRange,
    ByteBudget,
    RosbagImageCollector,
    sha256_file,
    validate_activation,
    validate_runtime_lock,
)
from scripts.research.egomotion_compensated_looming.rgb_segment_confirmation_r1.validate_preaccess_lock import (
    validate,
)


REPO = Path(__file__).resolve().parents[4]


class PreaccessLockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate = build(REPO)

    def test_canonical_lock_passes(self) -> None:
        self.assertEqual([], validate(REPO, self.candidate))

    def test_rgb_execution_escalation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["execution_authority"]["rgb_algorithm"] = True
        self.assertIn("PRE_REVIEW_AUTHORITY", validate(REPO, candidate))

    def test_member_identity_mutation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["segments"][0]["frame_inventory"][0]["rgb_member_crc32"] = "00000000"
        self.assertIn("OPEN_MEMBER_IDENTITY", validate(REPO, candidate))

    def test_source_role_ceiling_mutation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["claim_ceiling"]["source_role_confounding"] = False
        errors = validate(REPO, candidate)
        self.assertIn("SOURCE_ROLE_CONFOUNDING_REMOVED", errors)

    def test_dlr_unreviewed_inventory_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["segments"][1]["rgb_frame_inventory"] = [{"forged": True}]
        self.assertIn("DLR_UNREVIEWED_RGB_INVENTORY", validate(REPO, candidate))

    def test_pair_tolerance_is_enforced(self) -> None:
        rows = [
            {
                "geometry_timestamp_s": "1.000",
                "rgb_timestamp_s": "1.010",
                "rgb_member_path": "color/1.010.png",
                "rgb_minus_geometry_delta_s": "0.010",
            }
        ]
        with self.assertRaisesRegex(ValueError, "OPENLORIS_PAIR_TOLERANCE"):
            enforce_open_pairing(rows)

    def test_nonmonotonic_pairing_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["segments"][0]["frame_inventory"][1][
            "rgb_timestamp_s"
        ] = candidate["segments"][0]["frame_inventory"][0]["rgb_timestamp_s"]
        self.assertIn(
            "OPEN_RGB_ORDER_OR_UNIQUENESS", validate(REPO, candidate)
        )

    def test_guard_mutation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["segments"][0]["guard_before"]["path"] = "forged.png"
        self.assertIn("OPEN_GUARD_BEFORE", validate(REPO, candidate))

    def test_transport_budget_mutation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["transport_identity_authority"][
            "openloris_max_remote_bytes"
        ] += 1
        self.assertIn("OPEN_BUDGET", validate(REPO, candidate))

    def test_frozen_binding_mutation_fails(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["algorithm_bindings"][0]["sha256"] = "0" * 64
        self.assertIn("FROZEN_BINDING_SET_OR_DIGEST", validate(REPO, candidate))

    def test_dlr_guard_is_pending(self) -> None:
        candidate = copy.deepcopy(self.candidate)
        candidate["segments"][1]["closed_rgb_guard_frame_count"] = 2
        self.assertIn("DLR_CLOSED_GUARD_PREACCESS", validate(REPO, candidate))

    def test_byte_budget_stops_before_overrun(self) -> None:
        budget = ByteBudget(10)
        budget.reserve(7)
        with self.assertRaises(BudgetExceeded):
            budget.reserve(4)
        self.assertEqual(7, budget.consumed)

    def test_range_retry_cannot_physically_overrun_budget(self) -> None:
        class FailingResponse:
            status = 206
            headers = {"Content-Range": "bytes 0-3/10"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                raise OSError("mid-stream failure")

        with tempfile.TemporaryDirectory() as directory:
            remote = BudgetedRemoteRange(
                "https://invalid.example/object",
                10,
                6,
                Path(directory) / "progress.json",
            )
            with patch(
                "urllib.request.urlopen",
                return_value=FailingResponse(),
            ) as opener:
                with self.assertRaises(BudgetExceeded):
                    remote.read(4)
            self.assertEqual(1, opener.call_count)
            self.assertEqual(5, remote.budget.consumed)
            self.assertEqual("RETRYABLE_FAILURE", remote.requests[0]["status"])
            progress = json.loads(
                (Path(directory) / "progress.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                {
                    "phase",
                    "completed_units",
                    "total_units",
                    "throughput",
                    "eta_seconds",
                    "last_progress_at",
                    "status",
                }.issubset(progress)
            )

    def test_range_contract_binds_remote_object_length(self) -> None:
        class DriftedResponse:
            status = 206
            headers = {"Content-Range": "bytes 0-3/11"}

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self, size):
                return b"abcd"

        with tempfile.TemporaryDirectory() as directory:
            remote = BudgetedRemoteRange(
                "https://invalid.example/object",
                10,
                20,
                Path(directory) / "progress.json",
            )
            with patch(
                "urllib.request.urlopen",
                return_value=DriftedResponse(),
            ):
                with self.assertRaisesRegex(RuntimeError, "RANGE_CONTRACT"):
                    remote.read(4)
            self.assertEqual(12, remote.budget.consumed)
            self.assertEqual(3, len(remote.requests))

    def test_activation_requires_bound_independent_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            evidence = repo / "artifacts.local" / "evidence" / "r1"
            evidence.mkdir(parents=True)
            candidate_path = repo / "docs" / "candidate.json"
            candidate_path.parent.mkdir(parents=True)
            candidate_path.write_text("{}\n", encoding="utf-8")
            lock_path = evidence / "lock.json"
            lock = {
                "execution_authority": {
                    "opaque_identity_extraction": False,
                    "rgb_algorithm": False,
                    "android": False,
                }
            }
            lock_path.write_text(json.dumps(lock) + "\n", encoding="utf-8")
            validation_path = evidence / "validation.json"
            validation_path.write_text(
                json.dumps(
                    {
                        "decision": "PREACCESS_LOCK_MACHINE_VALIDATION_PASS",
                        "errors": [],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            runtime_path = evidence / "runtime.json"
            base_executable = Path(
                getattr(sys, "_base_executable", sys.executable)
            ).resolve()
            runtime_path.write_text(
                json.dumps(
                    {
                        "python": {
                            "version": ".".join(
                                map(str, sys.version_info[:3])
                            ),
                            "venv_executable": str(
                                Path(sys.executable).resolve()
                            ),
                            "venv_executable_sha256": sha256_file(
                                Path(sys.executable).resolve()
                            ),
                            "base_executable_sha256": sha256_file(
                                base_executable
                            ),
                        },
                        "wheel_cache": "artifacts.local/evidence/r1/wheels",
                        "distributions": {},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            transport_path = Path(
                __import__(
                    "scripts.research.egomotion_compensated_looming."
                    "rgb_segment_confirmation_r1.opaque_transport",
                    fromlist=["__file__"],
                ).__file__
            ).resolve()
            bindings = {
                "preaccess_lock": {
                    "path": "artifacts.local/evidence/r1/lock.json",
                    "sha256": sha256_file(lock_path),
                },
                "candidate_lock": {
                    "path": "docs/candidate.json",
                    "sha256": sha256_file(candidate_path),
                },
                "opaque_transport": {
                    "path": (
                        "scripts/research/egomotion_compensated_looming/"
                        "rgb_segment_confirmation_r1/opaque_transport.py"
                    ),
                    "sha256": sha256_file(transport_path),
                },
                "runtime_lock": {
                    "path": "artifacts.local/evidence/r1/runtime.json",
                    "sha256": sha256_file(runtime_path),
                },
            }
            review_path = evidence / "review.json"
            review = {
                "decision": "PREACCESS_IDENTITY_LOCK_INDEPENDENT_REVIEW_PASS",
                "bindings": {
                    **bindings,
                    "machine_validation": {
                        "path": (
                            "artifacts.local/evidence/r1/validation.json"
                        ),
                        "sha256": sha256_file(validation_path),
                    },
                },
            }
            review_path.write_text(
                json.dumps(review) + "\n", encoding="utf-8"
            )
            signature = {
                "decision": "OPAQUE_IDENTITY_EXTRACTION_ACTIVATED",
                "bindings": {
                    **bindings,
                    "independent_review": {
                        "path": "artifacts.local/evidence/r1/review.json",
                        "sha256": sha256_file(review_path),
                    },
                },
                "authority": {
                    "opaque_identity_extraction": True,
                    "rgb_algorithm": False,
                    "android": False,
                },
            }
            signature_path = evidence / "signature.json"
            signature_path.write_text(
                json.dumps(signature) + "\n", encoding="utf-8"
            )
            with patch.object(
                transport_module, "validate_runtime_lock", return_value={}
            ):
                self.assertEqual(
                    signature,
                    validate_activation(lock_path, lock, signature_path),
                )
                signature["bindings"].pop("independent_review")
                signature_path.write_text(
                    json.dumps(signature) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, "REVIEW_BINDING"):
                    validate_activation(lock_path, lock, signature_path)

    def test_runtime_executable_hash_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            runtime_path = repo / "runtime.json"
            base_executable = Path(
                getattr(sys, "_base_executable", sys.executable)
            ).resolve()
            runtime_path.write_text(
                json.dumps(
                    {
                        "python": {
                            "version": ".".join(
                                map(str, sys.version_info[:3])
                            ),
                            "venv_executable": str(
                                Path(sys.executable).resolve()
                            ),
                            "venv_executable_sha256": "0" * 64,
                            "base_executable_sha256": sha256_file(
                                base_executable
                            ),
                        },
                        "wheel_cache": "wheels",
                        "distributions": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError, "RUNTIME_EXECUTABLE_SHA256"
            ):
                validate_runtime_lock(repo, runtime_path)

    def test_runtime_shadow_path_cannot_override_py7zr(self) -> None:
        runtime_path = (
            REPO
            / "artifacts.local/evidence/rcle_rgb_segment_confirmation_r1/"
            "opaque_transport_runtime_lock.v1.json"
        )
        runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
        expected_executable = Path(
            runtime["python"]["venv_executable"]
        ).resolve()
        if Path(sys.executable).resolve() != expected_executable:
            self.skipTest("requires the frozen opaque-transport venv")
        with tempfile.TemporaryDirectory() as directory:
            shadow = Path(directory) / "py7zr.py"
            shadow.write_text("__version__ = '1.0.0'\n", encoding="utf-8")
            previous_path = list(sys.path)
            previous_module = sys.modules.pop("py7zr", None)
            try:
                sys.path.insert(0, directory)
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("PYTHONPATH", None)
                    validate_runtime_lock(REPO, runtime_path)
                loaded = Path(sys.modules["py7zr"].__file__).resolve()
                self.assertNotEqual(shadow.resolve(), loaded)
            finally:
                sys.path[:] = previous_path
                sys.modules.pop("py7zr", None)
                if previous_module is not None:
                    sys.modules["py7zr"] = previous_module

    def test_cli_routes_success_to_success_receipt_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            signature = root / "signature.json"
            success = root / "success.json"
            failure = root / "failure.json"
            progress = root / "progress.json"
            lock.write_text("{}\n", encoding="utf-8")
            signature.write_text("{}\n", encoding="utf-8")
            arguments = [
                "opaque_transport.py",
                "--lock",
                str(lock),
                "--signature",
                str(signature),
                "--source",
                "openloris",
                "--output",
                str(root / "output"),
                "--claim",
                str(root / "claim.json"),
                "--receipt",
                str(success),
                "--failure-receipt",
                str(failure),
                "--progress",
                str(progress),
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(transport_module, "validate_activation"),
                patch.object(
                    transport_module,
                    "openloris_extract",
                    return_value={
                        "decision": "OPENLORIS_OPAQUE_IDENTITY_EXTRACTED",
                        "remote_bytes": 4,
                        "budget": 10,
                    },
                ),
                self.assertRaisesRegex(SystemExit, "0"),
            ):
                transport_module.main()
            self.assertTrue(success.is_file())
            self.assertFalse(failure.exists())
            self.assertEqual(
                "OPENLORIS_OPAQUE_IDENTITY_EXTRACTED",
                json.loads(success.read_text(encoding="utf-8"))["decision"],
            )

    def test_cli_routes_activation_failure_to_failure_receipt_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = root / "lock.json"
            signature = root / "signature.json"
            success = root / "success.json"
            failure = root / "failure.json"
            progress = root / "progress.json"
            lock.write_text("{}\n", encoding="utf-8")
            signature.write_text("{}\n", encoding="utf-8")
            arguments = [
                "opaque_transport.py",
                "--lock",
                str(lock),
                "--signature",
                str(signature),
                "--source",
                "openloris",
                "--output",
                str(root / "output"),
                "--claim",
                str(root / "claim.json"),
                "--receipt",
                str(success),
                "--failure-receipt",
                str(failure),
                "--progress",
                str(progress),
            ]
            with (
                patch("sys.argv", arguments),
                patch.object(
                    transport_module,
                    "validate_activation",
                    side_effect=ValueError("activation drift"),
                ),
                self.assertRaisesRegex(SystemExit, "1"),
            ):
                transport_module.main()
            self.assertFalse(success.exists())
            self.assertTrue(failure.is_file())
            self.assertTrue(progress.is_file())
            self.assertEqual(
                "INVALID_IDENTITY_EXTRACTION_CLOSE_ATTEMPT",
                json.loads(failure.read_text(encoding="utf-8"))["decision"],
            )

    def test_rosbag_stream_collector_closes_exact_guards(self) -> None:
        collector = RosbagImageCollector(2_000_000_000, 4_000_000_000)
        connection = self._record(
            {
                "op": b"\x07",
                "conn": struct.pack("<I", 1),
                "topic": b"/camera/color/image_raw",
            },
            self._fields(
                {
                    "type": b"sensor_msgs/Image",
                    "md5sum": b"unused",
                    "message_definition": b"unused",
                }
            ),
        )
        inner = b"".join(
            self._record(
                {
                    "op": b"\x02",
                    "conn": struct.pack("<I", 1),
                    "time": struct.pack("<II", second, 0),
                },
                self._image(second, bytes([second]) * 3),
            )
            for second in (1, 2, 3, 4)
        )
        chunk = self._record(
            {
                "op": b"\x05",
                "compression": b"none",
                "size": struct.pack("<I", len(inner)),
            },
            inner,
        )
        collector.feed(b"#ROSBAG V2.0\n" + connection + chunk)
        self.assertTrue(collector.complete)
        self.assertEqual(1_000_000_000, collector.before["header_timestamp_ns"])
        self.assertEqual(2, len(collector.selected))
        self.assertEqual(4_000_000_000, collector.after["header_timestamp_ns"])

    def test_rosbag_aligned_depth_connection_is_excluded(self) -> None:
        collector = RosbagImageCollector(2_000_000_000, 4_000_000_000)
        connection = self._record(
            {
                "op": b"\x07",
                "conn": struct.pack("<I", 1),
                "topic": b"/camera/aligned_depth_to_color/image_raw",
            },
            self._fields({"type": b"sensor_msgs/Image"}),
        )
        collector.feed(b"#ROSBAG V2.0\n" + connection)
        self.assertIsNone(collector.color_connection)

    @staticmethod
    def _fields(values: dict[str, bytes]) -> bytes:
        result = b""
        for key, value in values.items():
            field = key.encode("ascii") + b"=" + value
            result += struct.pack("<I", len(field)) + field
        return result

    @classmethod
    def _record(cls, header: dict[str, bytes], data: bytes) -> bytes:
        fields = cls._fields(header)
        return (
            struct.pack("<I", len(fields))
            + fields
            + struct.pack("<I", len(data))
            + data
        )

    @staticmethod
    def _ros_string(value: str) -> bytes:
        encoded = value.encode("utf-8")
        return struct.pack("<I", len(encoded)) + encoded

    @classmethod
    def _image(cls, second: int, payload: bytes) -> bytes:
        return (
            struct.pack("<I", 0)
            + struct.pack("<II", second, 0)
            + cls._ros_string("camera_color_optical_frame")
            + struct.pack("<II", 1, 1)
            + cls._ros_string("rgb8")
            + b"\x00"
            + struct.pack("<II", 3, len(payload))
            + payload
        )


if __name__ == "__main__":
    unittest.main()
