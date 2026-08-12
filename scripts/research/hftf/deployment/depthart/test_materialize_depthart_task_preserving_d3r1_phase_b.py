from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path

from scripts.research.hftf.deployment.depthart import materialize_depthart_task_preserving_d3r1_phase_b as subject


class FakeResponse:
    def __init__(self, body: bytes, url: str, *, status: int = 200, etag: str = '"etag"', last_modified: str = "date") -> None:
        self.body = io.BytesIO(body)
        self.status = status
        self.url = url
        self.headers = {
            "Content-Length": str(len(body)),
            "ETag": etag,
            "Last-Modified": last_modified,
        }

    def geturl(self) -> str:
        return self.url

    def read(self, size: int) -> bytes:
        return self.body.read(size)

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        return None


def head_row(body: bytes, url: str = "https://example.test/depth.zip") -> dict[str, object]:
    return {
        "asset": "lowres_depth.zip",
        "url": url,
        "content_length_bytes": len(body),
        "etag": '"etag"',
        "last_modified": "date",
    }


def passing_counts() -> dict[str, object]:
    counts = subject.empty_counts()
    counts["valid_band_clearances"] = 450
    for key in counts["known_by_grid"]:
        counts["clear_by_grid"][key] = 30
        counts["occupied_by_grid"][key] = 100
        counts["known_by_grid"][key] = 130
    counts["clear_cells"] = 270
    counts["occupied_cells"] = 900
    counts["known_cells"] = 1170
    return counts


THRESHOLDS = {
    "minimum_truth_known_cells": 1100,
    "minimum_truth_clear_cells": 270,
    "minimum_truth_occupied_cells": 900,
    "minimum_truth_clear_cells_per_band_horizon": 30,
    "minimum_truth_occupied_cells_per_band_horizon": 100,
    "minimum_valid_band_clearances": 450,
    "model_output_access": False,
    "source": "fixture",
}


class D3R1PhaseBUnitTest(unittest.TestCase):
    def test_clear_and_occupied_grid_gates_are_independent(self) -> None:
        counts = passing_counts()
        passed, failures = subject.qualifies(counts, THRESHOLDS)
        self.assertTrue(passed, failures)
        counts["clear_by_grid"]["left@1.0m"] = 29
        counts["occupied_by_grid"]["left@1.0m"] = 101
        counts["clear_cells"] = 269
        counts["occupied_cells"] = 901
        passed, failures = subject.qualifies(counts, THRESHOLDS)
        self.assertFalse(passed)
        self.assertIn("left@1.0m_clear=29<30", failures)

    def test_unknown_is_excluded_from_truth_summary(self) -> None:
        truth = {
            "bands": {
                "left": {"clearance_m": 1.0, "occupied_by_horizon": {"1.0": False, "1.5": None, "2.0": True}},
                "center": {"clearance_m": None, "occupied_by_horizon": {"1.0": None, "1.5": None, "2.0": None}},
            }
        }
        counts = subject.summarize_truth(truth)
        self.assertEqual((counts["known_cells"], counts["clear_cells"], counts["occupied_cells"]), (2, 1, 1))
        self.assertEqual(counts["valid_band_clearances"], 1)
        subject.validate_count_identities(counts)

    def test_finalize_processes_all_32_and_never_publishes_partial_lock(self) -> None:
        rows = [
            {
                "selection_order": index,
                "pool_order": index + 10,
                "visit_id": str(1000 + index),
                "video_id": str(2000 + index),
                "source_truth_support_qualified": index <= 15,
                "selected_frame_plan_sha256": f"{index:064X}",
            }
            for index in range(1, 33)
        ]
        passed, selected = subject.finalize_selection(rows)
        self.assertFalse(passed)
        self.assertEqual(selected, [])
        rows[15]["source_truth_support_qualified"] = True
        passed, selected = subject.finalize_selection(rows)
        self.assertTrue(passed)
        self.assertEqual([row["phase_a_selection_order"] for row in selected], list(range(1, 17)))
        with self.assertRaisesRegex(ValueError, "all exact 32"):
            subject.finalize_selection(rows[:31])

    def test_frame_plan_digest_uses_exact_frozen_stems(self) -> None:
        rows = [
            {
                "selection_order": 1,
                "pool_order": 5,
                "visit_id": "1",
                "video_id": "2",
                "selected_frame_stems": ["2_1.000", "2_1.001"],
            }
        ]
        expected = hashlib.sha256(b"1:5:1:2:2_1.000\n1:5:1:2:2_1.001\n").hexdigest().upper()
        self.assertEqual(subject.frame_plan_sha256(rows), expected)

    def test_safe_member_map_rejects_unsafe_and_duplicate_stems(self) -> None:
        for members, message in (
            ({"../evil.png": b"x"}, "unsafe ZIP member"),
            ({"a/frame.png": b"x", "b/frame.png": b"y"}, "duplicate ZIP frame stem"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                archive_path = Path(temporary) / "test.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    for name, payload in members.items():
                        archive.writestr(name, payload)
                with zipfile.ZipFile(archive_path) as archive, self.assertRaisesRegex(ValueError, message):
                    subject.safe_member_map(archive, ".png")

    def test_download_binds_headers_and_writes_exclusively(self) -> None:
        body = b"payload"
        row = head_row(body)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = subject.download_file(
                row,
                root / "source" / "lowres_depth.zip",
                root / "temporary",
                opener=lambda request, timeout: FakeResponse(body, request.full_url),
            )
            self.assertEqual(result["sha256"], hashlib.sha256(body).hexdigest().upper())
            self.assertFalse(result["range_request_used"])
            with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
                subject.download_file(row, root / "source" / "lowres_depth.zip", root / "other")

    def test_transient_retry_recovers_but_redirect_and_header_drift_do_not_retry(self) -> None:
        body = b"payload"
        row = head_row(body)
        with tempfile.TemporaryDirectory() as temporary:
            calls: list[int] = []

            def transient(request: object, timeout: int) -> FakeResponse:
                calls.append(1)
                if len(calls) == 1:
                    raise urllib.error.HTTPError(str(row["url"]), 500, "server", {}, None)
                return FakeResponse(body, str(row["url"]))

            result = subject.download_file(row, Path(temporary) / "ok.zip", Path(temporary) / "tmp1", opener=transient)
            self.assertEqual(result["attempts"], 2)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result["attempt_history"][0]["retry_class"], "TRANSIENT_HTTP")
            self.assertIsNone(result["attempt_history"][1]["retry_class"])

        for opener, expected_calls in (
            (lambda request, timeout: (_ for _ in ()).throw(urllib.error.HTTPError(str(row["url"]), 302, "redirect", {}, None)), 1),
            (lambda request, timeout: FakeResponse(body, str(row["url"]), etag='"drift"'), 1),
        ):
            with tempfile.TemporaryDirectory() as temporary:
                calls = []

                def counted(request: object, timeout: int, wrapped=opener):
                    calls.append(1)
                    return wrapped(request, timeout)

                with self.assertRaises(OSError):
                    subject.download_file(row, Path(temporary) / "bad.zip", Path(temporary) / "tmp", opener=counted)
                self.assertEqual(len(calls), expected_calls)

    def test_resume_rejects_sparse_checkpoint_prefix(self) -> None:
        selected = [
            {"selection_order": index, "pool_order": index, "visit_id": str(index), "video_id": str(100 + index), "fold": "Training"}
            for index in range(1, 33)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipts = root / "receipts"
            receipts.mkdir()
            (receipts / "002-102.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "continuous prefix"):
                subject.validate_resume_inventory(output_root=root, selected=selected, attempt={})


if __name__ == "__main__":
    unittest.main()
