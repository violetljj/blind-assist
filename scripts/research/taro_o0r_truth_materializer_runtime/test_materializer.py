#!/usr/bin/env python3
"""Synthetic-only adversarial tests for the TARO O0R truth materializer."""

from __future__ import annotations

import copy
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import numpy as np

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o0r_source_adapter_runtime.test_source_adapter import (
    source_receipt,
    synthetic_faro_depth,
)
from scripts.research.taro_o0r_truth_materializer_runtime import materializer as runtime
from scripts.research.taro_o0r_truth_materializer_runtime import run_head_preflight as head_runner
from scripts.research.taro_o0r_truth_materializer_runtime import run_truth_only as truth_runner
from scripts.research.taro_o0r_truth_materializer_runtime.run_truth_only import (
    PreparedParent,
    materialize_prepared_parents,
    validate_execution_lock,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
PREFLIGHT_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_TRUTH_ONLY_PREFLIGHT_R1_LOCK_2026-08-10.json"
AUTHORIZATION_PATH = REPO_ROOT / "docs/research/taro/TARO_O0R_ARKITSCENES_DATA_USE_AUTHORIZATION_R1_RECEIPT_2026-08-10.json"


def reseal(value: dict[str, object]) -> dict[str, object]:
    output = copy.deepcopy(value)
    output.pop("content_sha256", None)
    output["content_sha256"] = runtime.canonical_sha256(output)
    return output


def modality_inventory(tokens: dict[str, str]) -> dict[str, dict[str, runtime.MemberBinding]]:
    return {
        role: {
            token: runtime.MemberBinding(
                role=role,
                timestamp_token=token,
                source_member_path=f"{role}/1_{token}.png",
                canonical_member_path=f"{role}/{token}.png",
                bytes=1,
                crc32="00000000",
            )
        }
        for role, token in tokens.items()
    }


def trajectory_rows(left: str = "0.900000000", right: str = "1.100000000") -> list[dict[str, object]]:
    return [
        {"timestamp_token": left, "rotation_vector": [0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
        {"timestamp_token": right, "rotation_vector": [0.0, 0.0, 0.0], "translation": [0.0, 0.0, 0.0]},
    ]


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str, headers: dict[str, str], status: int = 200) -> None:
        super().__init__(payload)
        self._url = url
        self.headers = headers
        self.status = status

    def geturl(self) -> str:
        return self._url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FakeOpener:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response

    def open(self, request: object, timeout: float) -> FakeResponse:
        return self.response


class AuthorizationAndHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = runtime.load_json(PREFLIGHT_PATH)
        self.authorization = runtime.load_json(AUTHORIZATION_PATH)

    def test_user_authorization_binds_exact_24_parent_72_asset_plan(self) -> None:
        rows = runtime.validate_authorization(
            self.preflight,
            self.authorization,
            preflight_sha256=runtime.sha256_file(PREFLIGHT_PATH),
        )
        self.assertEqual(len(rows), 72)
        self.assertEqual(runtime.canonical_sha256(rows), self.preflight["asset_plan"]["expanded_requests_sha256"])
        mutated = copy.deepcopy(self.authorization)
        mutated["interpreted_scope"]["selected_video_ids"][0] = "00000000"
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime.validate_authorization(self.preflight, mutated, preflight_sha256=runtime.sha256_file(PREFLIGHT_PATH))
        self.assertEqual(caught.exception.code, "AUTHORIZATION_ROSTER_DRIFT")

    def test_head_receipt_retries_and_reads_zero_body_bytes(self) -> None:
        attempts: dict[str, int] = {}

        def fake_head(row: dict[str, str], timeout: float) -> dict[str, object]:
            attempts[row["url"]] = attempts.get(row["url"], 0) + 1
            if attempts[row["url"]] == 1:
                raise runtime.MaterializerError("HEAD_TRANSPORT_ERROR", "synthetic retry")
            return {"http_status": 200, "content_length_bytes": 10, "etag": "E", "last_modified": "L", "redirect_chain": [], "transport_errors": []}

        receipt = runtime.build_head_receipt(
            self.preflight,
            preflight_sha256=runtime.sha256_file(PREFLIGHT_PATH),
            authorization_sha256=runtime.sha256_file(AUTHORIZATION_PATH),
            head_fn=fake_head,
        )
        self.assertEqual(receipt["available_asset_count"], 72)
        self.assertEqual(receipt["response_body_bytes_read"], 0)
        self.assertTrue(all(row["attempt_count"] == 2 for row in receipt["assets"]))
        runtime.validate_head_receipt(self.preflight, runtime.sha256_file(AUTHORIZATION_PATH), receipt)

    def test_availability_successor_retires_failed_parent_before_source_access(self) -> None:
        rows = runtime.expanded_asset_plan(self.preflight)
        self.assertFalse(any(row["video_id"] == "47333152" for row in rows))
        replacement = [row for row in rows if row["video_id"] == "47204786"]
        self.assertEqual(len(replacement), 3)
        self.assertTrue(all(row["role"] == "ADAPTER_FIT" for row in replacement))

    def test_safe_join_preserves_trusted_artifacts_local_junction(self) -> None:
        artifacts = REPO_ROOT / "artifacts.local"
        self.assertTrue(artifacts.is_dir())
        relative = "artifacts.local/evidence/taro/synthetic-path-never-created/result.json"
        output = runtime.safe_join(REPO_ROOT, relative)
        self.assertEqual(output, REPO_ROOT.absolute() / Path(relative))
        self.assertTrue(output.parent.resolve(strict=False).is_relative_to(artifacts.resolve()))


class DownloadAndArchiveTests(unittest.TestCase):
    def test_download_binds_head_length_validators_hash_crc_and_no_range(self) -> None:
        payload = b"bound-source-payload"
        url = "https://docs-assets.developer.apple.com/example.zip"
        row = {"asset": "upsampling.zip", "url": url, "relative_path": "source/example.zip"}
        head = {"url": url, "http_status": 200, "content_length_bytes": len(payload), "etag": "ET", "last_modified": "LM"}
        response = FakeResponse(payload, url, {"Content-Length": str(len(payload)), "ETag": "ET", "Last-Modified": "LM", "Content-Encoding": "identity"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            receipt = runtime.download_bound_asset(row, head, source_root=root, opener=FakeOpener(response))
            self.assertEqual(receipt["sha256"], runtime.sha256_bytes(payload))
            self.assertEqual(receipt["crc32"], runtime.crc32_bytes(payload))
            self.assertEqual((root / "source/example.zip").read_bytes(), payload)

    def test_download_rejects_head_get_validator_drift_without_partial(self) -> None:
        payload = b"payload"
        url = "https://docs-assets.developer.apple.com/example.zip"
        row = {"asset": "upsampling.zip", "url": url, "relative_path": "example.zip"}
        head = {"url": url, "http_status": 200, "content_length_bytes": len(payload), "etag": "HEAD", "last_modified": None}
        response = FakeResponse(payload, url, {"Content-Length": str(len(payload)), "ETag": "GET"})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "source"
            with self.assertRaises(runtime.MaterializerError) as caught:
                runtime.download_bound_asset(row, head, source_root=root, opener=FakeOpener(response))
            self.assertEqual(caught.exception.code, "DOWNLOAD_VALIDATOR_DRIFT")
            self.assertFalse((root / "example.zip").exists())
            self.assertFalse((root / "example.zip.partial").exists())

    def _write_upsampling(self, path: Path, *, compression: int = zipfile.ZIP_DEFLATED) -> None:
        with zipfile.ZipFile(path, "w", compression=compression) as bundle:
            for directory in ("wide", "highres_depth", "lowres_depth", "confidence"):
                bundle.writestr(f"{directory}/123_1.000000000.png", b"x")

    def test_archive_index_preserves_official_to_canonical_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "up.zip"
            self._write_upsampling(path)
            inventory = runtime.index_upsampling_archive(path, "123")
            binding = inventory["color"]["1.000000000"]
            self.assertEqual(binding.source_member_path, "wide/123_1.000000000.png")
            self.assertEqual(binding.canonical_member_path, "color/1.000000000.png")

    def test_archive_rejects_unsafe_symlink_encryption_compression_and_budget(self) -> None:
        unsafe = zipfile.ZipInfo("../escape.png")
        with self.assertRaises(runtime.MaterializerError):
            runtime._validate_zip_info(unsafe)
        symlink = zipfile.ZipInfo("wide/123_1.000000000.png")
        symlink.create_system = 3
        symlink.external_attr = 0o120777 << 16
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime._validate_zip_info(symlink)
        self.assertEqual(caught.exception.code, "ZIP_SYMLINK_FORBIDDEN")
        encrypted = zipfile.ZipInfo("wide/123_1.000000000.png")
        encrypted.flag_bits |= 0x1
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime._validate_zip_info(encrypted)
        self.assertEqual(caught.exception.code, "ZIP_ENCRYPTION_FORBIDDEN")
        unsupported = zipfile.ZipInfo("wide/123_1.000000000.png")
        unsupported.compress_type = zipfile.ZIP_BZIP2
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime._validate_zip_info(unsupported)
        self.assertEqual(caught.exception.code, "ZIP_COMPRESSION_UNSUPPORTED")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "up.zip"
            self._write_upsampling(path)
            with self.assertRaises(runtime.MaterializerError) as caught:
                runtime.index_upsampling_archive(path, "123", maximum_uncompressed_bytes=3)
            self.assertEqual(caught.exception.code, "ZIP_MATERIALIZED_BUDGET_EXCEEDED")

    def test_read_member_rechecks_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "one.zip"
            with zipfile.ZipFile(path, "w") as bundle:
                bundle.writestr("wide/123_1.000000000.png", b"payload")
            binding = runtime.MemberBinding("color", "1.000000000", "wide/123_1.000000000.png", "color/1.000000000.png", 7, "00000000")
            with zipfile.ZipFile(path) as bundle, self.assertRaises(runtime.MaterializerError) as caught:
                runtime._read_zip_member(bundle, binding)
            self.assertEqual(caught.exception.code, "ZIP_MEMBER_CRC_DRIFT")


class ExactFrameAndProvenanceTests(unittest.TestCase):
    def test_same_nanosecond_alias_is_rejected_before_frame_selection(self) -> None:
        upsampling = modality_inventory({"color": "1.0", "highres_depth": "1.000000000", "lowres_depth": "1.0", "confidence": "1.0"})
        intrinsics = {"1.0": runtime.MemberBinding("intrinsics", "1.0", "intrinsics/1_1.0.pincam", "intrinsics/1.0.pincam", 1, "00000000")}
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime.exact_frame_plan("1", upsampling, intrinsics, trajectory_rows())
        self.assertEqual(caught.exception.code, "SAME_NS_TIMESTAMP_ALIAS")

    def test_frame_plan_uses_all_exact_frames_and_records_pose_rejection(self) -> None:
        tokens = {role: "1.000000000" for role in ("color", "highres_depth", "lowres_depth", "confidence")}
        upsampling = modality_inventory(tokens)
        intrinsics = {"1.000000000": runtime.MemberBinding("intrinsics", "1.000000000", "intrinsics/1_1.000000000.pincam", "intrinsics/1.000000000.pincam", 1, "00000000")}
        admitted = runtime.exact_frame_plan("1", upsampling, intrinsics, trajectory_rows())
        self.assertEqual(admitted["exact_timestamp_tokens"], ["1.000000000"])
        self.assertFalse(admitted["truth_or_model_fields_read"])
        rejected = runtime.exact_frame_plan("1", upsampling, intrinsics, trajectory_rows("0.800000000", "1.200000000"))
        self.assertEqual(rejected["exact_timestamp_tokens"], [])
        self.assertEqual(rejected["pose_rejected"][0]["reason_code"], "POSE_BRACKET_GAP_EXCEEDED")

    def test_source_envelope_binds_original_and_canonical_member_paths(self) -> None:
        receipt = copy.deepcopy(source_receipt())
        container_sha = "A" * 64
        token = receipt["sensor_timestamp"]["decimal_token"]
        video_id = receipt["session_id"]
        for role in runtime.DECODED_ROLES:
            receipt["asset_bindings"][role]["container_id"] = f"sha256:{container_sha}"
        receipt = reseal(receipt)
        members: dict[str, dict[str, object]] = {}
        for role in runtime.DECODED_ROLES:
            asset = receipt["asset_bindings"][role]
            decoded = receipt["decoded_payload_bindings"][role]
            source_path = "lowres_wide.traj" if role == "trajectory" else f"official/{role}/{video_id}_{token}.bin"
            members[role] = {
                "source_container_sha256": container_sha,
                "source_member_path": source_path,
                "source_member_bytes": asset["bytes"],
                "source_member_sha256": asset["sha256"],
                "source_member_crc32": asset["crc32"],
                "canonical_member_path": asset["member_path"],
                "decoded_content_sha256": decoded["decoded_content_sha256"],
            }
        envelope = runtime.seal_record(
            {
                "schema": runtime.BOUND_SOURCE_FRAME_SCHEMA,
                "source_role": receipt["source_role"],
                "visit_id": receipt["site_id"],
                "video_id": video_id,
                "timestamp_token": token,
                "physical_frame_id": receipt["physical_frame_id"],
                "source_frame_receipt_sha256": receipt["content_sha256"],
                "members": members,
                "mapping_rule": "OFFICIAL_VIDEO_ID_TIMESTAMP_TO_EXPLICIT_TIMESTAMP_ONLY_CANONICAL_MEMBER",
                "silent_rename_allowed": False,
            }
        )
        runtime.validate_bound_source_frame_envelope(envelope, receipt)
        mutated = copy.deepcopy(envelope)
        mutated["members"]["color"]["source_member_path"] = f"official/color/{token}.bin"
        mutated = runtime.seal_record({key: value for key, value in mutated.items() if key != "content_sha256"})
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime.validate_bound_source_frame_envelope(mutated, receipt)
        self.assertEqual(caught.exception.code, "SOURCE_ENVELOPE_ORIGINAL_PATH")


class QueryLookupAndPersistenceTests(unittest.TestCase):
    def test_each_query_derives_its_own_bound_confidence_and_range(self) -> None:
        receipt = copy.deepcopy(source_receipt())
        faro = synthetic_faro_depth(True)
        confidence = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        confidence[:, adapter.APPLE_SHAPE_HW[1] // 2 :] = 2
        receipt["decoded_payload_bindings"]["confidence"]["decoded_content_sha256"] = adapter.canonical_sha256(confidence)
        receipt = reseal(receipt)
        matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
        geometry = adapter.derive_faro_geometry(faro, matrix, receipt["gravity_up_camera_xyz"], receipt)
        queries = adapter.build_query_receipts(receipt, geometry)
        lookups = [runtime.derive_query_uncertainty_lookup(faro, confidence, receipt, query) for query in queries]
        self.assertEqual(len(lookups), 9)
        self.assertEqual([lookup["query_id"] for lookup in lookups], [query["query_id"] for query in queries])
        self.assertEqual({lookup["confidence_value"] for lookup in lookups}, {0, 2})
        self.assertTrue(all(lookup["support_count"] >= adapter.MINIMUM_QUERY_SUPPORT_POINTS for lookup in lookups))
        self.assertTrue(all(lookup["caller_scalar_allowed"] is False for lookup in lookups))
        self.assertEqual(len({lookup["content_sha256"] for lookup in lookups}), 9)

    def test_query_lookup_rejects_unbound_confidence(self) -> None:
        receipt = source_receipt()
        faro = synthetic_faro_depth(True)
        confidence = np.zeros(adapter.APPLE_SHAPE_HW, dtype=np.uint8)
        matrix = np.asarray(receipt["intrinsics_highres"]["matrix_3x3"], dtype=np.float64)
        geometry = adapter.derive_faro_geometry(faro, matrix, receipt["gravity_up_camera_xyz"], receipt)
        query = adapter.build_query_receipts(receipt, geometry)[0]
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime.derive_query_uncertainty_lookup(faro, confidence, receipt, query)
        self.assertEqual(caught.exception.code, "DECODED_PAYLOAD_CONTENT_MISMATCH")

    def test_content_addressed_round_trip_normalizes_every_array_kind(self) -> None:
        value = {
            "float": np.asarray([1.1234567890126, -0.0], dtype=np.float32),
            "bool": np.asarray([[True, False]], dtype=np.bool_),
            "signed": np.asarray([1, -2], dtype=np.int32),
            "unsigned": np.asarray([3], dtype=np.uint16),
            "empty": np.empty((0, 2), dtype=np.float64),
        }
        package, blobs = runtime.package_content_addressed_artifact(value)
        hydrated = runtime.hydrate_content_addressed_artifact(package, lambda path: blobs[path])
        self.assertEqual(runtime.canonical_sha256(hydrated), runtime.canonical_sha256(value))
        self.assertEqual(hydrated["bool"].dtype, np.dtype(np.bool_))
        self.assertEqual(hydrated["signed"].dtype, np.dtype("<i8"))
        self.assertEqual(hydrated["unsigned"].dtype, np.dtype("<u8"))

    def test_blob_corruption_and_hash_only_persistence_fail_closed(self) -> None:
        value = {"array": np.asarray([1.0, 2.0], dtype=np.float64)}
        package, blobs = runtime.package_content_addressed_artifact(value)
        path = next(iter(blobs))
        corrupted = dict(blobs)
        corrupted[path] = blobs[path] + b"x"
        with self.assertRaises(runtime.MaterializerError) as caught:
            runtime.hydrate_content_addressed_artifact(package, lambda item: corrupted[item])
        self.assertEqual(caught.exception.code, "ARRAY_BLOB_GZIP_HASH_MISMATCH")
        hash_only = copy.deepcopy(package)
        hash_only["payload_with_array_refs"] = hash_only["payload_with_array_refs"]["array"]["canonical_array_receipt"]
        hash_only = runtime.seal_record({key: value for key, value in hash_only.items() if key != "content_sha256"})
        with self.assertRaises(runtime.MaterializerError):
            runtime.hydrate_content_addressed_artifact(hash_only, lambda item: blobs[item])

    def test_uncertainty_model_cells_are_persisted_as_reloadable_arrays(self) -> None:
        class Cell:
            target = "scale_log_abs_residual"
            parent_id = adapter.ADAPTER_FIT_ROSTER[0][0]
            confidence = 2
            range_bin = 1
            values = np.asarray([0.1, 0.2], dtype=np.float64)

        class Model:
            content_sha256 = "D" * 64
            fit_parent_ids = tuple(visit for visit, _ in adapter.ADAPTER_FIT_ROSTER)
            source_receipt_sha256s = ("E" * 64,)
            source_evidence_sha256 = "F" * 64
            support_frame_observations = 1
            observation_counts = {target: 1 for target in adapter.UNCERTAINTY_TARGETS}
            cells = (Cell(),)

            def _assert_integrity(self) -> None:
                return None

        artifact = runtime.uncertainty_model_artifact(Model())
        package, blobs = runtime.package_content_addressed_artifact(artifact)
        hydrated = runtime.hydrate_content_addressed_artifact(package, lambda path: blobs[path])
        self.assertEqual(runtime.canonical_sha256(hydrated), runtime.canonical_sha256(artifact))
        self.assertEqual(hydrated["cells"][0]["values"].tolist(), [0.1, 0.2])

    def test_atomic_writer_consumes_root_deduplicates_blobs_and_forbids_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "evidence"
            writer = runtime.AtomicEvidenceWriter(root, 1_000_000)
            writer.activate({"schema": "synthetic.execution.v1"})
            first = writer.write_content_addressed_artifact("one.json.gz", {"a": np.asarray([1, 2], dtype=np.int32)})
            second = writer.write_content_addressed_artifact("two.json.gz", {"a": np.asarray([1, 2], dtype=np.int64)})
            self.assertTrue(first["reload_verified"] and second["reload_verified"])
            self.assertTrue(second["blob_files"][0]["deduplicated"])
            with self.assertRaises(runtime.MaterializerError) as caught:
                writer.write_json("one.json.gz", {"overwrite": True})
            self.assertEqual(caught.exception.code, "EVIDENCE_OVERWRITE_FORBIDDEN")
            with self.assertRaises(runtime.MaterializerError):
                runtime.AtomicEvidenceWriter(root, 10).activate({"schema": "again"})

    def test_atomic_writer_enforces_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            writer = runtime.AtomicEvidenceWriter(Path(temporary) / "evidence", 20)
            with self.assertRaises(runtime.MaterializerError) as caught:
                writer.activate({"schema": "this-receipt-is-too-large"})
            self.assertEqual(caught.exception.code, "EVIDENCE_BUDGET_EXCEEDED")


class GateAndPhaseTests(unittest.TestCase):
    def test_truth_gate_enforces_exact_parent_rosters_and_denominators(self) -> None:
        fit_ids = [visit for visit, _ in adapter.ADAPTER_FIT_ROSTER]
        summaries = [
            {"parent_id": visit, "video_id": video, "exact_timestamp_frames": 12, "source_eligible_frames": 12, "admitted_frames": 12, "clear_frames": 6, "occupied_frames": 6}
            for visit, video in adapter.O0R_EVAL_CANDIDATE_ROSTER
        ]
        passed = runtime.evaluate_truth_only_gates(fit_ids, summaries)
        self.assertTrue(passed["passed"])
        mutated = copy.deepcopy(summaries)
        mutated[0]["admitted_frames"] = 11
        failed = runtime.evaluate_truth_only_gates(fit_ids, mutated)
        self.assertFalse(failed["passed"])
        self.assertTrue(any(code.startswith("PARENT_TRUTH_GATE") for code in failed["failure_codes"]))

    def test_materializer_seals_uncertainty_before_any_eval_decode(self) -> None:
        events: list[str] = []
        prepared: list[PreparedParent] = []
        for role, roster in (("ADAPTER_FIT", adapter.ADAPTER_FIT_ROSTER), ("O0R_EVAL_CANDIDATE", adapter.O0R_EVAL_CANDIDATE_ROSTER)):
            for visit, video in roster:
                tokens = [f"1.{index:09d}" for index in range(1 if role == "ADAPTER_FIT" else 12)]
                prepared.append(
                    PreparedParent(
                        parent={"role": role, "visit_id": visit, "video_id": video, "official_fold": "Training"},
                        upsampling_archive=Path("unused-up.zip"),
                        intrinsics_archive=Path("unused-k.zip"),
                        trajectory_path=Path("unused.traj"),
                        upsampling_inventory={},
                        intrinsics_inventory={},
                        trajectory_rows=[],
                        container_receipts={},
                        frame_plan={"exact_timestamp_tokens": tokens},
                        materialized_bytes=0,
                    )
                )

        def fake_decode(**kwargs: object) -> dict[str, object]:
            parent = kwargs["parent"]
            assert isinstance(parent, dict)
            events.append(str(parent["role"]))
            return {
                "source_frame_receipt": {
                    "source_role": parent["role"],
                    "site_id": parent["visit_id"],
                    "session_id": parent["video_id"],
                    "intrinsics_highres": {"matrix_3x3": np.eye(3).tolist()},
                    "gravity_up_camera_xyz": [0.0, 1.0, 0.0],
                    "content_sha256": "A" * 64,
                },
                "bound_source_frame_envelope": {"schema": "synthetic.envelope.v1"},
                "highres_faro_depth_mm": np.ones((1, 1), dtype=np.uint16),
                "apple_depth_mm": np.ones((1, 1), dtype=np.uint16),
                "confidence": np.ones((1, 1), dtype=np.uint8),
            }

        class FakeModel:
            content_sha256 = "B" * 64
            fit_parent_ids = tuple(visit for visit, _ in adapter.ADAPTER_FIT_ROSTER)
            source_receipt_sha256s = tuple(f"{index:064X}" for index in range(1, 9))
            source_evidence_sha256 = "C" * 64
            support_frame_observations = 8
            observation_counts = {target: 1 for target in adapter.UNCERTAINTY_TARGETS}
            cells: tuple[object, ...] = ()

            def _assert_integrity(self) -> None:
                return None

        def fake_fit(frames: object) -> FakeModel:
            list(frames)
            events.append("UNCERTAINTY_SEALED")
            return FakeModel()

        def fake_eval(frame: object, model: object, *, geometry: object) -> dict[str, object]:
            return {
                "query_bundle": {
                    "complete_factor_query_truth": True,
                    "state_counts": {"CLEAR_OBSERVED": 1, "OCCUPIED_OBSERVED": 1, "UNKNOWN": 7},
                },
                "array": np.asarray([1.0, -0.0], dtype=np.float64),
            }

        with tempfile.TemporaryDirectory() as temporary:
            writer = runtime.AtomicEvidenceWriter(Path(temporary) / "evidence", 20_000_000)
            writer.activate({"schema": "synthetic.execution.v1"})
            gates, phase = materialize_prepared_parents(
                prepared,
                writer=writer,
                started=runtime.time.monotonic(),
                wall_seconds=60.0,
                peak_rss_bytes=10**12,
                decode_fn=fake_decode,
                fit_fn=fake_fit,
                geometry_fn=lambda *args: object(),
                eval_fn=fake_eval,
            )
        seal_index = events.index("UNCERTAINTY_SEALED")
        self.assertTrue(all(event == "ADAPTER_FIT" for event in events[:seal_index]))
        self.assertTrue(all(event == "O0R_EVAL_CANDIDATE" for event in events[seal_index + 1 :]))
        self.assertEqual(phase["eval_decode_count_before_uncertainty_seal"], 0)
        self.assertTrue(gates["passed"])

    def test_truth_runner_rejects_invalid_lock_before_creating_roots(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "lock.json"
            lock_path.write_text('{"schema":"wrong"}', encoding="utf-8")
            with self.assertRaises(runtime.MaterializerError) as caught:
                validate_execution_lock(lock_path, REPO_ROOT)
            self.assertEqual(caught.exception.code, "EXECUTION_LOCK_SCHEMA_DRIFT")

    def test_head_unexpected_transport_failure_is_consumed_after_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            lock_path = root / "head-lock.json"
            lock_path.write_text("{}", encoding="utf-8")
            preflight_path = root / "docs/preflight.json"
            authorization_path = root / "docs/authorization.json"
            preflight_path.parent.mkdir(parents=True)
            preflight_path.write_text("{}", encoding="utf-8")
            authorization_path.write_text("{}", encoding="utf-8")
            budget = {
                "maximum_compressed_source_bytes": 100,
                "head_timeout_seconds": 1,
                "head_retries": 1,
                "head_workers": 1,
                "head_response_body_bytes": 0,
            }
            lock = {
                "required_environment": {},
                "resource_budget": budget,
                "argv": ["synthetic-head-runner"],
                "_verified_bindings": {"SYNTHETIC": {"sha256": "A" * 64}},
            }
            preflight = {"required_environment": {}, "resource_budget": budget}

            def invoke_unexpected_head(_preflight: object, **kwargs: object) -> dict[str, object]:
                head_fn = kwargs["head_fn"]
                assert callable(head_fn)
                return head_fn({"url": "https://example.invalid/asset"}, 1.0)

            def unexpected_head(_row: object, _timeout: float) -> dict[str, object]:
                raise RuntimeError("synthetic unexpected HEAD failure")

            with (
                mock.patch.object(head_runner, "REPO_ROOT", root),
                mock.patch.object(head_runner, "HEAD_OUTPUT_ROOT", "head"),
                mock.patch.object(head_runner, "HEAD_RECEIPT_PATH", "head/head-receipt.json"),
                mock.patch.object(head_runner, "EXPECTED_ABSENT_ROOTS", ["source", "truth"]),
                mock.patch.object(
                    head_runner,
                    "EXPECTED_HEAD_BINDINGS",
                    {
                        "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/preflight.json",
                        "DATA_USE_AUTHORIZATION": "docs/authorization.json",
                    },
                ),
                mock.patch.object(head_runner, "validate_head_execution_lock", return_value=lock),
                mock.patch.object(head_runner, "load_json", side_effect=[preflight, {}]),
                mock.patch.object(head_runner, "validate_authorization", return_value=[]),
                mock.patch.object(head_runner, "build_head_receipt", side_effect=invoke_unexpected_head),
            ):
                with self.assertRaises(runtime.MaterializerError) as caught:
                    head_runner.execute_head_preflight(lock_path, head_fn=unexpected_head)
            self.assertEqual(caught.exception.code, "HEAD_EXECUTION_ONE_SHOT_CONSUMED")
            self.assertTrue((root / "head/start-receipt.json").is_file())
            failure = json.loads((root / "head/failure.json").read_text(encoding="utf-8"))
            self.assertTrue(failure["head_one_shot_consumed"])
            self.assertEqual(failure["failure_code"], "RuntimeError")

    def test_head_write_failure_is_consumed_after_root_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            lock_path = root / "head-lock.json"
            lock_path.write_text("{}", encoding="utf-8")
            preflight_path = root / "docs/preflight.json"
            authorization_path = root / "docs/authorization.json"
            preflight_path.parent.mkdir(parents=True)
            preflight_path.write_text("{}", encoding="utf-8")
            authorization_path.write_text("{}", encoding="utf-8")
            budget = {
                "maximum_compressed_source_bytes": 100,
                "head_timeout_seconds": 1,
                "head_retries": 1,
                "head_workers": 1,
                "head_response_body_bytes": 0,
            }
            lock = {
                "required_environment": {},
                "resource_budget": budget,
                "argv": ["synthetic-head-runner"],
                "_verified_bindings": {"SYNTHETIC": {"sha256": "A" * 64}},
            }
            preflight = {"required_environment": {}, "resource_budget": budget}
            receipt = {
                "terminal": "TARO_O0R_ASSET_HEADERS_NOT_AVAILABLE_NO_REPLACEMENT",
                "asset_count": 72,
                "available_asset_count": 0,
                "response_body_bytes_read": 0,
            }
            original_write = head_runner._write_exclusive

            def fail_head_receipt(path: Path, payload: bytes) -> dict[str, object]:
                if path.name == "head-receipt.json":
                    raise OSError("synthetic receipt write failure")
                return original_write(path, payload)

            with (
                mock.patch.object(head_runner, "REPO_ROOT", root),
                mock.patch.object(head_runner, "HEAD_OUTPUT_ROOT", "head"),
                mock.patch.object(head_runner, "HEAD_RECEIPT_PATH", "head/head-receipt.json"),
                mock.patch.object(head_runner, "EXPECTED_ABSENT_ROOTS", ["source", "truth"]),
                mock.patch.object(
                    head_runner,
                    "EXPECTED_HEAD_BINDINGS",
                    {
                        "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/preflight.json",
                        "DATA_USE_AUTHORIZATION": "docs/authorization.json",
                    },
                ),
                mock.patch.object(head_runner, "validate_head_execution_lock", return_value=lock),
                mock.patch.object(head_runner, "load_json", side_effect=[preflight, {}]),
                mock.patch.object(head_runner, "validate_authorization", return_value=[]),
                mock.patch.object(head_runner, "build_head_receipt", return_value=receipt),
                mock.patch.object(head_runner, "_write_exclusive", side_effect=fail_head_receipt),
            ):
                with self.assertRaises(runtime.MaterializerError) as caught:
                    head_runner.execute_head_preflight(lock_path)
            self.assertEqual(caught.exception.code, "HEAD_EXECUTION_ONE_SHOT_CONSUMED")
            failure = json.loads((root / "head/failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["failure_code"], "OSError")
            self.assertTrue((root / "head/manifest.json").is_file())

    def test_truth_root_creation_window_reports_consumed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            lock_path = root / "truth-lock.json"
            lock_path.write_text("{}", encoding="utf-8")
            paths = {
                "TRUTH_ONLY_PREFLIGHT_LOCK": "docs/preflight.json",
                "DATA_USE_AUTHORIZATION": "docs/authorization.json",
                "HEAD_RECEIPT": "head/head-receipt.json",
            }
            for relative in paths.values():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}", encoding="utf-8")
            budget = {
                "maximum_scientific_evidence_bytes": 1_000_000,
                "truth_only_materialization_wall_seconds": 60,
                "truth_only_peak_rss_bytes": 10**9,
            }
            bindings = {
                role: {"path": relative, "sha256": chr(65 + index) * 64}
                for index, (role, relative) in enumerate(paths.items())
            }
            lock = {
                "required_environment": {},
                "resource_budget": budget,
                "argv": ["synthetic-truth-runner"],
                "_verified_bindings": bindings,
            }
            preflight = {"required_environment": {}, "resource_budget": budget}
            roots = {
                "SOURCE": "source",
                "WORK": "work",
                "TRUTH_EVIDENCE": "truth",
                "O0R_EVIDENCE_SEALED": "factor",
            }

            def fail_source_root(path: Path) -> None:
                raise OSError(f"synthetic mkdir failure: {path.name}")

            def load_fixture(path: Path) -> dict[str, object]:
                if path.name == "preflight.json":
                    return preflight
                return {}

            with (
                mock.patch.object(truth_runner, "REPO_ROOT", root),
                mock.patch.object(truth_runner, "EXPECTED_ROOTS", roots),
                mock.patch.object(truth_runner, "validate_execution_lock", return_value=lock),
                mock.patch.object(truth_runner, "load_json", side_effect=load_fixture),
                mock.patch.object(truth_runner, "validate_authorization", return_value=[]),
                mock.patch.object(truth_runner, "validate_head_receipt", return_value={}),
                mock.patch.object(truth_runner, "expanded_asset_plan", return_value=[]),
            ):
                result = truth_runner.execute_truth_only(lock_path, root_mkdir_fn=fail_source_root)
            self.assertFalse(result["passed"])
            self.assertTrue(result["one_shot_consumed"])
            self.assertEqual(result["failure_code"], "OSError")
            self.assertTrue((root / "truth/execution-receipt.json").is_file())
            self.assertTrue((root / "truth/failure.json").is_file())
            self.assertFalse((root / "source").exists())


if __name__ == "__main__":
    unittest.main()
