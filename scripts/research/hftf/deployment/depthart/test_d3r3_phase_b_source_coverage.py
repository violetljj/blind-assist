from __future__ import annotations

import ast
import hashlib
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest import mock

from scripts.research.hftf.deployment.depthart import (
    census_depthart_task_preserving_d3r3_phase_b_source_coverage as producer,
)
from scripts.research.hftf.deployment.depthart import (
    validate_depthart_task_preserving_d3r3_phase_b_source_coverage as validator,
)


def write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


class SourceCoverageTests(unittest.TestCase):
    @staticmethod
    def selected_fixture() -> list[dict[str, object]]:
        return [
            {
                "selection_order": index,
                "pool_order": index + 10,
                "visit_id": str(100000 + index),
                "video_id": str(20000000 + index),
                "fold": "Training",
                "selected_frame_stems": [f"{20000000 + index}_{frame}.000" for frame in range(300)],
            }
            for index in range(1, 33)
        ]

    def test_asset_checkpoint_plan_is_exact_continuous_64(self) -> None:
        selected = self.selected_fixture()
        plan = producer.request_plan(selected)
        self.assertEqual(list(range(1, 65)), [row["request_order"] for row in plan])
        self.assertEqual(
            ["lowres_depth.zip", "confidence.zip"] * 32,
            [row["asset"] for row in plan],
        )

    def test_request_plan_digest_uses_frozen_slash_encoding(self) -> None:
        row = {
            "selection_order": 1,
            "pool_order": 5,
            "visit_id": "422389",
            "video_id": "42447264",
            "asset": "lowres_depth.zip",
            "url": "https://example.invalid/Training/42447264/lowres_depth.zip",
        }
        expected = __import__("hashlib").sha256(
            ("1/5/422389/42447264/lowres_depth.zip/https://example.invalid/Training/42447264/lowres_depth.zip\n").encode("ascii")
        ).hexdigest().upper()
        self.assertEqual(expected, producer.request_plan_sha256([row]))

    def test_producer_has_no_pixel_or_truth_dependency(self) -> None:
        source = Path(producer.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            str(node.module)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertFalse(any(name == "numpy" or name.startswith("PIL") for name in imports))
        self.assertNotIn("truth_reader", source)
        self.assertNotIn(".testzip(", source)

    def test_archive_coverage_preserves_exact_order(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth.zip"
            write_zip(path, {"frames/v_1.000.png": b"a", "frames/v_1.033.png": b"b", "frames/other.png": b"c"})
            selected = ["v_1.000", "v_1.016", "v_1.033"]
            result = producer.archive_coverage(path, selected)
            self.assertEqual(2, result["selected_present_count"])
            self.assertEqual(["v_1.016"], result["selected_missing_stems"])
            self.assertEqual(1, result["selected_extra_member_count"])

    def test_paired_missing_is_union_without_substitution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            stems = [f"v_{index:.3f}" for index in range(300)]
            write_zip(source / "lowres_depth.zip", {f"d/{stem}.png": b"d" for stem in stems if stem != stems[2]})
            write_zip(source / "confidence.zip", {f"c/{stem}.png": b"c" for stem in stems if stem != stems[3]})
            depth = producer.archive_coverage(source / "lowres_depth.zip", stems)
            confidence = producer.archive_coverage(source / "confidence.zip", stems)
            result = producer.combine_identity_coverage(
                {"selected_frame_stems": stems}, depth, confidence
            )
            self.assertEqual(298, result["paired_exact_present_count"])
            self.assertEqual([stems[2], stems[3]], result["paired_exact_missing_stems"])
            self.assertFalse(result["neighbor_substitution_used"])
            self.assertFalse(result["source_truth_derived"])

    def test_independent_replay_matches_producer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.zip"
            write_zip(path, {"frames/v_1.000.png": b"a", "frames/v_1.033.png": b"b"})
            selected = ["v_1.000", "v_1.016", "v_1.033"]
            self.assertEqual(
                producer.archive_coverage(path, selected),
                validator.independent_archive_coverage(path, selected),
            )

    def test_directory_census_never_opens_member_payload_or_runs_crc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "asset.zip"
            write_zip(path, {"frames/v_1.000.png": b"payload"})
            with mock.patch.object(zipfile.ZipFile, "open", side_effect=AssertionError("member payload opened")), mock.patch.object(
                zipfile.ZipFile, "testzip", side_effect=AssertionError("CRC probe ran")
            ):
                result = producer.archive_coverage(path, ["v_1.000"])
                replay = validator.independent_archive_coverage(path, ["v_1.000"])
            self.assertEqual(0, result["archive_member_payload_bytes_read"])
            self.assertFalse(result["zip_crc_verified"])
            self.assertEqual(result, replay)

    def test_unsafe_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "unsafe.zip"
            write_zip(path, {"../v_1.000.png": b"a"})
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                producer.archive_coverage(path, ["v_1.000"])
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                validator.independent_archive_coverage(path, ["v_1.000"])

    def test_duplicate_stem_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.zip"
            write_zip(path, {"a/v_1.000.png": b"a", "b/v_1.000.png": b"b"})
            with self.assertRaisesRegex(ValueError, "duplicate ZIP frame stem"):
                producer.archive_coverage(path, ["v_1.000"])
            with self.assertRaisesRegex(ValueError, "duplicate ZIP frame stem"):
                validator.independent_archive_coverage(path, ["v_1.000"])

    def test_activation_refuses_scientific_authority(self) -> None:
        activation = {
            "schema": producer.ACTIVATION_SCHEMA,
            "activation_id": "fixture",
            "activated_at": "2026-08-12T00:00:00+08:00",
            "activated_by": "user",
            "status": "D3R3_PHASE_B_SOURCE_MEMBER_COVERAGE_CENSUS_ACTIVATED",
            "authorization_verbatim": "授权 census",
            "authorization_context": "fixture",
            "protocol": {},
            "protocol_sha256": "placeholder",
            "bindings": {},
            "request_scope": {},
            "execution_policy": dict(producer.ACTIVATION_EXECUTION_POLICY),
            "authority": {
                "exact64_get": True,
                "archive_container_read": True,
                "archive_member_name_read": True,
                "zip_directory_parse": True,
                "range_get": False,
                "redirect_following": False,
                "archive_member_payload_read": False,
                "zip_crc_verification": False,
                "pixel_decode": True,
                "source_truth": False,
                "phase_b_selection": False,
                "phase_c_rgb": False,
                "model_output": False,
                "role_assignment": False,
                "training": False,
                "development_outcome": False,
                "r2_access": "NONE",
                "performance": False,
                "android_default": False,
                "production": False,
                "safety": False,
            },
            "forbidden": list(producer.ACTIVATION_FORBIDDEN),
            "next_action": producer.ACTIVATION_NEXT_ACTION,
        }
        with tempfile.TemporaryDirectory() as temporary:
            protocol = Path(temporary) / "protocol.json"
            protocol.write_text("{}\n", encoding="utf-8")
            activation["protocol_sha256"] = producer.sha256_file(protocol)
            activation["protocol"] = {
                "path": str(protocol.resolve()),
                "bytes": protocol.stat().st_size,
                "sha256": producer.sha256_file(protocol),
            }
            with self.assertRaisesRegex(ValueError, "authority widened: pixel_decode"):
                producer.validate_activation(activation, protocol)
            activation["authority"]["pixel_decode"] = False
            activation["authority"]["model_output"] = True
            with self.assertRaisesRegex(ValueError, "authority widened: model_output"):
                producer.validate_activation(activation, protocol)
            with self.assertRaisesRegex(ValueError, "authority widened: model_output"):
                validator.validate_activation(activation, protocol)
            activation["authority"]["model_output"] = False
            activation["extra_claim"] = "widened"
            with self.assertRaisesRegex(ValueError, "top-level field set drift"):
                producer.validate_activation(activation, protocol)
            with self.assertRaisesRegex(ValueError, "top-level field set drift"):
                validator.validate_activation(activation, protocol)

    def test_resume_accepts_exact_one_asset_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            selected = self.selected_fixture()
            planned = producer.request_plan(selected)[0]
            attempt = {"schema": producer.ATTEMPT_SCHEMA, "token": "fixture"}
            producer.write_json_exclusive(root / "attempt.json", attempt)
            source = root / "source" / "Training" / str(planned["video_id"]) / str(planned["asset"])
            write_zip(source, {f"frames/{stem}.png": b"x" for stem in planned["selected_frame_stems"]})
            head = {
                "content_length_bytes": source.stat().st_size,
                "etag": '"fixture"',
                "last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
            }
            receipt = {
                "asset": planned["asset"],
                "url": "https://example.invalid/asset.zip",
                "path": str(source.resolve()),
                "bytes": source.stat().st_size,
                "sha256": producer.sha256_file(source),
                "attempts": 1,
                "attempt_history": [{
                    "attempt": 1, "method": "GET", "http_status": 200,
                    "error": None, "error_type": None, "retry_class": None,
                    "expected_body_bytes": source.stat().st_size,
                    "response_body_bytes_read": source.stat().st_size,
                }],
                "response_http_status": 200,
                "response_final_url": "https://example.invalid/asset.zip",
                "response_content_length_bytes": source.stat().st_size,
                "response_etag": head["etag"],
                "response_last_modified": head["last_modified"],
                "frozen_head_content_length_bytes": source.stat().st_size,
                "frozen_head_etag": head["etag"],
                "frozen_head_last_modified": head["last_modified"],
                "range_request_used": False,
                "redirect_followed": False,
                "transport_response_body_bytes_read_total": source.stat().st_size,
            }
            head["url"] = receipt["url"]
            checkpoint = {
                "schema": producer.CHECKPOINT_SCHEMA,
                "attempt_sha256": hashlib.sha256(producer.json_bytes(attempt)).hexdigest().upper(),
                **{key: planned[key] for key in ("request_order", "selection_order", "pool_order", "visit_id", "video_id", "fold", "asset")},
                "selected_frame_plan_sha256": producer.selected_stem_sha256(planned["selected_frame_stems"]),
                "source_asset": receipt,
                **producer.archive_coverage(source, planned["selected_frame_stems"]),
                "transport_response_body_bytes_read": source.stat().st_size,
                "scientific_terminal": None,
                "pixel_decode": False,
                "source_truth": False,
                "selection_evaluated": False,
                "range_get_used": False,
                "redirect_followed": False,
                "role_assigned": False,
                "training": False,
                "development_outcome_read": False,
                "r2_access": "NONE",
            }
            producer.write_sealed_json(producer.checkpoint_path(root / "receipts", planned), checkpoint)
            completed = producer.validate_resume_inventory(
                output_root=root,
                selected=selected,
                attempt=attempt,
                heads={(str(planned["video_id"]), str(planned["asset"])): head},
            )
            self.assertEqual(1, len(completed))

    def test_resume_rejects_sparse_checkpoint_and_extra_source_fold(self) -> None:
        selected = self.selected_fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sparse"
            (root / "receipts").mkdir(parents=True)
            (root / "attempt.json").write_text("{}\n", encoding="utf-8")
            third = producer.request_plan(selected)[2]
            (root / "receipts" / producer.checkpoint_path(root / "receipts", third).name).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "checkpoint prefix drift"):
                producer.validate_resume_inventory(output_root=root, selected=selected, attempt={}, heads={})
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "extra-fold"
            (root / "source" / "RGB").mkdir(parents=True)
            (root / "attempt.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extra source fold"):
                producer.validate_resume_inventory(output_root=root, selected=selected, attempt={}, heads={})

    def test_exhausted_download_keeps_nonresumable_marker(self) -> None:
        selected = self.selected_fixture()
        planned = producer.request_plan(selected)[0]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            temporary_root = root / "_temporary_downloads" / "asset"
            head = {
                "asset": planned["asset"],
                "url": "https://example.invalid/asset.zip",
                "content_length_bytes": 1,
                "etag": '"fixture"',
                "last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
            }
            with self.assertRaises(producer.DownloadFailure) as failure:
                producer.download_file(
                    head,
                    root / "source" / "Training" / str(planned["video_id"]) / str(planned["asset"]),
                    temporary_root,
                    opener=mock.Mock(side_effect=urllib.error.URLError("offline")),
                )
            self.assertEqual(3, len(failure.exception.history))
            self.assertTrue((root / "_temporary_downloads").exists())
            with self.assertRaisesRegex(ValueError, "partial downloads make attempt non-resumable"):
                producer.validate_resume_inventory(output_root=root, selected=selected, attempt={}, heads={})

    def test_short_body_retries_from_zero_and_counts_all_bytes(self) -> None:
        class Response:
            def __init__(self, payload: bytes):
                self.status = 200
                self.headers = {
                    "Content-Length": "4",
                    "ETag": '"fixture"',
                    "Last-Modified": "Wed, 17 Nov 2021 00:00:00 GMT",
                }
                self.payload = payload
                self.offset = 0

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def geturl(self):
                return "https://example.invalid/asset.zip"

            def read(self, size):
                if self.offset:
                    return b""
                self.offset = len(self.payload)
                return self.payload

        calls = []

        def opener(request, timeout):
            calls.append(request.headers.get("Accept-encoding"))
            return Response(b"ab" if len(calls) == 1 else b"abcd")

        row = {
            "asset": "lowres_depth.zip",
            "url": "https://example.invalid/asset.zip",
            "content_length_bytes": 4,
            "etag": '"fixture"',
            "last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = producer.download_file(
                row, root / "source" / "asset.zip", root / "temp", opener=opener
            )
        self.assertEqual(2, receipt["attempts"])
        self.assertEqual("TRANSIENT_BODY_SHORT_READ", receipt["attempt_history"][0]["retry_class"])
        self.assertEqual([2, 4], [event["response_body_bytes_read"] for event in receipt["attempt_history"]])
        self.assertEqual(6, receipt["transport_response_body_bytes_read_total"])
        self.assertEqual(["identity", "identity"], calls)
        planned = producer.request_plan(self.selected_fixture())[0]
        planned["asset"] = row["asset"]
        validator.validate_transport_receipt(receipt, planned, row)

    def test_validator_rejects_terminal_http_retry_history(self) -> None:
        planned = producer.request_plan(self.selected_fixture())[0]
        head = {
            "url": "https://example.invalid/asset.zip",
            "content_length_bytes": 1,
            "etag": '"fixture"',
            "last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
        }
        receipt = {
            "asset": planned["asset"], "url": head["url"], "bytes": 1,
            "sha256": "A" * 64, "path": "unused", "attempts": 2,
            "attempt_history": [
                {"attempt": 1, "method": "GET", "http_status": 404, "error": "HTTPError: 404", "error_type": "HTTPError", "retry_class": "TERMINAL", "expected_body_bytes": 1, "response_body_bytes_read": 0},
                {"attempt": 2, "method": "GET", "http_status": 200, "error": None, "error_type": None, "retry_class": None, "expected_body_bytes": 1, "response_body_bytes_read": 1},
            ],
            "response_http_status": 200, "response_final_url": head["url"],
            "response_content_length_bytes": 1, "response_etag": head["etag"],
            "response_last_modified": head["last_modified"],
            "frozen_head_content_length_bytes": 1, "frozen_head_etag": head["etag"],
            "frozen_head_last_modified": head["last_modified"],
            "range_request_used": False, "redirect_followed": False,
            "transport_response_body_bytes_read_total": 1,
        }
        with self.assertRaisesRegex(ValueError, "terminal attempt was retried"):
            validator.validate_transport_receipt(receipt, planned, head)
        receipt["attempt_history"][0] = {
            "attempt": 1, "method": "GET", "http_status": None,
            "error": "ValueError: fake", "error_type": "ValueError",
            "retry_class": "TRANSIENT_TRANSPORT",
            "expected_body_bytes": 1, "response_body_bytes_read": 0,
        }
        with self.assertRaisesRegex(ValueError, "transport retry error type drift"):
            validator.validate_transport_receipt(receipt, planned, head)


if __name__ == "__main__":
    unittest.main()
