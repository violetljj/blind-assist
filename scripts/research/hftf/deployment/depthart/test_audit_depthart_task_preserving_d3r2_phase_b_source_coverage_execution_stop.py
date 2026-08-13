from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.research.hftf.deployment.depthart import (
    audit_depthart_task_preserving_d3r2_phase_b_source_coverage_execution_stop as audit,
)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_sealed(path: Path, value: dict) -> None:
    write_json(path, value)
    payload = path.read_bytes()
    write_json(
        path.with_suffix(".sha256.json"),
        {"bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest().upper()},
    )


class ExecutionStopAuditTests(unittest.TestCase):
    def fixture(self, root: Path) -> tuple[dict, dict]:
        assets = ("lowres_depth.zip", "confidence.zip")
        plan = []
        heads = []
        for request_order, asset in enumerate(assets, start=1):
            selection_order = request_order
            visit_id = str(422388 + request_order)
            video_id = str(42447263 + request_order)
            row = {
                "request_order": request_order,
                "selection_order": selection_order,
                "pool_order": 4 + request_order,
                "visit_id": visit_id,
                "video_id": video_id,
                "fold": "Training",
                "asset": asset,
            }
            plan.append(row)
            heads.append(
                row
                | {
                    "url": f"https://example.invalid/Training/{video_id}/{asset}",
                    "content_length_bytes": 7,
                    "etag": '"fixture"',
                    "last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
                }
            )
        attempt = {
            "schema": audit.ATTEMPT_SCHEMA,
            "output_root": str(root.resolve()),
            "bindings": {
                "protocol_sha256": audit.EXPECTED_PROTOCOL_SHA256,
                "activation_sha256": "A" * 64,
            },
            "scientific_terminal": None,
            "selection_evaluated": False,
            "policy": {
                "max_attempts": 3,
                "range_get": False,
                "redirect_following": False,
                "pixel_decode": False,
                "source_truth": False,
                "selection": False,
            },
            "asset_plan": plan,
        }
        write_json(root / "attempt.json", attempt)
        attempt_sha = hashlib.sha256((root / "attempt.json").read_bytes()).hexdigest().upper()
        source = root / "source/Training/42447264/lowres_depth.zip"
        source.parent.mkdir(parents=True)
        source.write_bytes(b"fixture")
        receipt = {
            "asset": "lowres_depth.zip",
            "url": heads[0]["url"],
            "path": str(source.resolve()),
            "bytes": 7,
            "sha256": "B" * 64,
            "attempts": 1,
            "attempt_history": [{"attempt": 1, "method": "GET", "http_status": 200, "error": None, "error_type": None, "retry_class": None}],
            "response_http_status": 200,
            "response_final_url": heads[0]["url"],
            "response_content_length_bytes": 7,
            "response_etag": '"fixture"',
            "response_last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
            "frozen_head_content_length_bytes": 7,
            "frozen_head_etag": '"fixture"',
            "frozen_head_last_modified": "Wed, 17 Nov 2021 00:00:00 GMT",
            "range_request_used": False,
            "redirect_followed": False,
        }
        checkpoint = {
            "schema": audit.CHECKPOINT_SCHEMA,
            "attempt_sha256": attempt_sha,
            **plan[0],
            "scientific_terminal": None,
            "selection_evaluated": False,
            "archive_member_payload_bytes_read": 0,
            "zip_crc_verified": False,
            "pixel_decode": False,
            "source_truth": False,
            "role_assigned": False,
            "training": False,
            "development_outcome_read": False,
            "r2_access": "NONE",
            "source_asset": receipt,
            "selected_missing_count": 0,
        }
        write_sealed(root / "receipts/001-42447264-lowres_depth.json", checkpoint)
        failure = {
            "schema": audit.FAILURE_SCHEMA,
            "attempt_sha256": attempt_sha,
            **plan[1],
            "failure_stage": "DOWNLOAD",
            "error_type": "DownloadFailure",
            "source_body_retained": False,
            "scientific_terminal": None,
            "selection_evaluated": False,
            "selected_phase_b": None,
            "next_gate": None,
            "archive_member_payload_bytes_read": 0,
            "pixel_decode": False,
            "source_truth": False,
            "attempt_history": [{"attempt": 1, "method": "GET", "http_status": 200, "error": "ValueError: download length mismatch", "error_type": "ValueError", "retry_class": "TERMINAL"}],
        }
        (root / "source/Training/42447265").mkdir(parents=True)
        write_sealed(root / "receipts/failures/002-42447265-confidence.json", failure)
        (root / "_temporary_downloads/marker/confidence").mkdir(parents=True)
        return {"output_root": str(root)}, {"assets": heads}

    def validate(self, root: Path, protocol: dict, heads: dict) -> dict:
        return audit.validate_attempt_root(
            root=root,
            protocol=protocol,
            activation_sha256="A" * 64,
            head_machine=heads,
            expected_plan_count=2,
            expected_completed=1,
            expected_failure_order=2,
            expected_failure_selection_order=2,
            expected_failure_pool_order=6,
            expected_failure_visit_id="422390",
            expected_failure_video_id="42447265",
            expected_failure_asset="confidence.zip",
        )

    def test_valid_fixture_never_reads_source_body(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            protocol, heads = self.fixture(root)
            source_root = (root / "source").resolve()
            original = Path.read_bytes

            def guarded(path: Path) -> bytes:
                if path.resolve().is_relative_to(source_root):
                    raise AssertionError("source body read")
                return original(path)

            with mock.patch.object(Path, "read_bytes", guarded):
                result = self.validate(root, protocol, heads)
            self.assertEqual(1, result["checkpoint_count"])
            self.assertEqual(1, result["retained_source_asset_count"])

    def test_missing_nonresumable_marker_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            protocol, heads = self.fixture(root)
            (root / "_temporary_downloads/marker/confidence").rmdir()
            (root / "_temporary_downloads/marker").rmdir()
            (root / "_temporary_downloads").rmdir()
            with self.assertRaisesRegex(ValueError, "attempt root inventory drift"):
                self.validate(root, protocol, heads)

    def test_source_length_or_manifest_tamper_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            protocol, heads = self.fixture(root)
            (root / "source/Training/42447264/lowres_depth.zip").write_bytes(b"bad")
            with self.assertRaisesRegex(ValueError, "source length drift"):
                self.validate(root, protocol, heads)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "root"
            root.mkdir()
            protocol, heads = self.fixture(root)
            write_json(root / "manifest.json", {})
            with self.assertRaisesRegex(ValueError, "attempt root inventory drift"):
                self.validate(root, protocol, heads)


if __name__ == "__main__":
    unittest.main()
