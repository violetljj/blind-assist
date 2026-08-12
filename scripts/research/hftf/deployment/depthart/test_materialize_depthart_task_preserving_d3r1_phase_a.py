import hashlib
import io
import tempfile
import unittest
import urllib.error
import zipfile
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from scripts.research.hftf.deployment.depthart.materialize_depthart_task_preserving_d3r1_phase_a import (
    continuous_window,
    download_file,
    finalize_phase_a_selection,
    first_qualified,
    first_continuous_window,
    parse_intrinsics,
    read_checkpoint,
    split_continuous_portrait_runs,
    timestamp_from_stem,
    validate_intrinsics_archive,
    validate_resume_inventory,
    write_checkpoint,
)


class D3R1PhaseAMaterializerTest(unittest.TestCase):
    def test_continuous_window_resets_on_gap(self) -> None:
        stems = ["v_1.0", "v_1.1", "v_2.0", "v_2.1", "v_2.2"]
        self.assertEqual(["v_2.0", "v_2.1", "v_2.2"], continuous_window(stems, 3, 0.5))

    def test_continuous_window_fails_when_short(self) -> None:
        with self.assertRaisesRegex(ValueError, "fewer than 3"):
            continuous_window(["v_1.0", "v_1.1"], 3, 0.5)

    def test_first_qualified_uses_frozen_pool_order(self) -> None:
        processed = [
            {"pool_order": 1, "eligible": False},
            {"pool_order": 2, "eligible": True},
            {"pool_order": 3, "eligible": True},
            {"pool_order": 4, "eligible": True},
        ]
        self.assertEqual([2, 3], [row["pool_order"] for row in first_qualified(processed, 2)])

    def test_failed_pool_does_not_publish_a_partial_selection(self) -> None:
        processed = [
            {"pool_order": index + 1, "eligible": index < 31}
            for index in range(127)
        ]
        passed, selected = finalize_phase_a_selection(processed, 127, 32)
        self.assertFalse(passed)
        self.assertEqual([], selected)

        passed, selected = finalize_phase_a_selection(
            [{"pool_order": index + 1, "eligible": True} for index in range(127)],
            127,
            32,
        )
        self.assertTrue(passed)
        self.assertEqual(list(range(1, 33)), [row["pool_order"] for row in selected])

    def test_full_127_pool_is_required_after_target_is_reached(self) -> None:
        early = [{"pool_order": index + 1, "eligible": index < 32} for index in range(32)]
        passed, selected = finalize_phase_a_selection(early, 127, 32)
        self.assertFalse(passed)
        self.assertEqual([], selected)

    def test_selection_boundary_31_32_33(self) -> None:
        for eligible_count, expected_pass in ((31, False), (32, True), (33, True)):
            processed = [
                {"pool_order": index + 1, "eligible": index < eligible_count}
                for index in range(127)
            ]
            passed, selected = finalize_phase_a_selection(processed, 127, 32)
            self.assertEqual(expected_pass, passed)
            self.assertEqual(32 if expected_pass else 0, len(selected))

    def test_nonportrait_and_gap_break_runs(self) -> None:
        classified = [
            {"stem": "v_1.0", "timestamp": 1.0, "portrait": True},
            {"stem": "v_1.1", "timestamp": 1.1, "portrait": False},
            {"stem": "v_1.2", "timestamp": 1.2, "portrait": True},
            {"stem": "v_1.6", "timestamp": 1.6, "portrait": True},
            {"stem": "v_1.6b", "timestamp": 1.6, "portrait": True},
        ]
        runs = split_continuous_portrait_runs(classified, 0.5)
        self.assertEqual([["v_1.0"], ["v_1.2", "v_1.6"], ["v_1.6b"]], [[row["stem"] for row in run] for run in runs])
        self.assertEqual(["v_1.2", "v_1.6"], first_continuous_window(runs, 2))

        boundary = split_continuous_portrait_runs(
            [
                {"stem": "v_2.0", "timestamp": 2.0, "portrait": True},
                {"stem": "v_2.5", "timestamp": 2.5, "portrait": True},
            ],
            0.5,
        )
        self.assertEqual(1, len(boundary))

    def test_intrinsics_archive_all_payloads_are_validated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "intrinsics.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("lowres_wide_intrinsics/v_1.0.pincam", "256 192 200 200 128 96")
                bundle.writestr("lowres_wide_intrinsics/v_1.1.pincam", "256 192 200 200 128 96")
            members, dimensions = validate_intrinsics_archive(archive)
            self.assertEqual({"v_1.0", "v_1.1"}, set(members))
            self.assertEqual({"256x192": 2}, dimensions)
            pincam = root / "v_1.0.pincam"
            pincam.write_text("256 192 200 200 128 96", encoding="utf-8")
            self.assertEqual((256, 192), parse_intrinsics(pincam)[:2])

    def test_invalid_intrinsics_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pincam = Path(directory) / "bad.pincam"
            pincam.write_text("256 192 -1 200 128 96", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid intrinsics"):
                parse_intrinsics(pincam)

    def test_malformed_unselected_pincam_fails_archive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "intrinsics.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("lowres_wide_intrinsics/v_1.0.pincam", "256 192 200 200 128 96")
                bundle.writestr("lowres_wide_intrinsics/v_1.1.pincam", "malformed")
            with self.assertRaisesRegex(ValueError, "six fields"):
                validate_intrinsics_archive(archive)

    def test_unsafe_or_duplicate_intrinsics_member_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            unsafe = Path(directory) / "unsafe.zip"
            with zipfile.ZipFile(unsafe, "w") as bundle:
                bundle.writestr("../v_1.0.pincam", "256 192 200 200 128 96")
            with self.assertRaisesRegex(ValueError, "unsafe ZIP member"):
                validate_intrinsics_archive(unsafe)

            duplicate = Path(directory) / "duplicate.zip"
            with zipfile.ZipFile(duplicate, "w") as bundle:
                bundle.writestr("a/v_1.0.pincam", "256 192 200 200 128 96")
                bundle.writestr("b/v_1.0.pincam", "256 192 200 200 128 96")
            with self.assertRaisesRegex(ValueError, "duplicate intrinsics stem"):
                validate_intrinsics_archive(duplicate)

    def test_nonfinite_timestamp_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite timestamp"):
            timestamp_from_stem("v_nan")

    def test_checkpoint_sidecar_tamper_blocks_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            source.write_bytes(b"source")
            source_entry = {
                "path": str(source),
                "bytes": source.stat().st_size,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest().upper(),
            }
            trajectory_entry = {"asset": "lowres_wide.traj", **source_entry}
            intrinsics_entry = {"asset": "lowres_wide_intrinsics.zip", **source_entry}
            checkpoint = root / "checkpoint.json"
            identity = {
                "pool_order": 1,
                "visit_id": "visit",
                "video_id": "video",
                "fold": "Training",
            }
            value = {
                "schema": "blindassist_depthart_task_preserving_d3r1_phase_a_identity_checkpoint_v1",
                "attempt_sha256": "ATTEMPT",
                **identity,
                "eligible": False,
                "source_assets": [trajectory_entry, intrinsics_entry],
                "trajectory": source_entry,
                "rgb_depth_confidence_read": False,
                "truth_or_model_output_read": False,
            }
            write_checkpoint(checkpoint, value)
            self.assertEqual(
                "video",
                read_checkpoint(checkpoint, identity, "ATTEMPT")["video_id"],
            )
            checkpoint.write_bytes(checkpoint.read_bytes() + b" ")
            with self.assertRaisesRegex(ValueError, "checkpoint bytes drift"):
                read_checkpoint(checkpoint, identity, "ATTEMPT")

    def test_resume_inventory_rejects_sparse_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = root / "receipts"
            receipts.mkdir()
            pool = [
                {"pool_order": index, "visit_id": f"v{index}", "video_id": f"s{index}", "fold": "Training"}
                for index in range(1, 4)
            ]
            (receipts / "002-s2.json").write_text("{}", encoding="utf-8")
            (receipts / "002-s2.sha256.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "continuous prefix"):
                validate_resume_inventory(root, pool, "ATTEMPT")

    def test_get_headers_must_match_frozen_head(self) -> None:
        payload = b"abc"

        class Response(io.BytesIO):
            status = 200
            headers = {
                "Content-Length": "3",
                "ETag": '"etag"',
                "Last-Modified": "date",
            }

            def geturl(self):
                return row["url"]

        row = {
            "url": "https://example.invalid/asset",
            "content_length_bytes": 3,
            "etag": '"etag"',
            "last_modified": "date",
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.build_opener"
        ):
            opener = unittest.mock.MagicMock()
            opener.open.return_value = Response(payload)
            with patch("urllib.request.build_opener", return_value=opener):
                receipt = download_file(row, Path(directory) / "asset", retries=1)
            self.assertEqual(hashlib.sha256(payload).hexdigest().upper(), receipt["sha256"])

        drift = {**row, "etag": '"different"'}
        with tempfile.TemporaryDirectory() as directory, patch(
            "urllib.request.build_opener"
        ):
            opener = unittest.mock.MagicMock()
            opener.open.return_value = Response(payload)
            with patch("urllib.request.build_opener", return_value=opener):
                with self.assertRaisesRegex(OSError, "GET ETag differs"):
                    download_file(drift, Path(directory) / "asset", retries=1)

    def test_redirect_is_not_followed_or_retried(self) -> None:
        row = {
            "url": "https://example.invalid/asset",
            "content_length_bytes": 3,
            "etag": '"etag"',
            "last_modified": "date",
        }
        calls = []

        def opener(request, timeout):
            calls.append(request.get_method())
            headers = Message()
            headers["Location"] = "https://redirect.invalid/body"
            raise urllib.error.HTTPError(row["url"], 302, "Found", headers, io.BytesIO())

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, "HTTP Error 302"):
                download_file(row, Path(directory) / "asset", retries=3, opener=opener)
        self.assertEqual(["GET"], calls)

    def test_transient_500_retries_but_final_url_drift_stops(self) -> None:
        payload = b"abc"
        row = {
            "url": "https://example.invalid/asset",
            "content_length_bytes": 3,
            "etag": '"etag"',
            "last_modified": "date",
        }

        class GoodResponse(io.BytesIO):
            status = 200
            headers = {"Content-Length": "3", "ETag": '"etag"', "Last-Modified": "date"}

            def geturl(self):
                return row["url"]

        calls = []

        def recovering(request, timeout):
            calls.append(request.get_method())
            if len(calls) == 1:
                raise urllib.error.HTTPError(row["url"], 500, "Server Error", Message(), io.BytesIO())
            return GoodResponse(payload)

        with tempfile.TemporaryDirectory() as directory:
            receipt = download_file(row, Path(directory) / "asset", retries=3, opener=recovering)
        self.assertEqual(2, receipt["attempts"])
        self.assertEqual(["GET", "GET"], calls)

        class RedirectedResponse(GoodResponse):
            def geturl(self):
                return "https://other.invalid/asset"

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(OSError, "final URL drift"):
                download_file(
                    row,
                    Path(directory) / "asset",
                    retries=3,
                    opener=lambda request, timeout: RedirectedResponse(payload),
                )

    def test_existing_download_is_rejected_before_transport(self) -> None:
        row = {
            "url": "https://example.invalid/asset",
            "content_length_bytes": 3,
            "etag": '"etag"',
            "last_modified": "date",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "asset"
            output.write_bytes(b"frozen")
            with self.assertRaisesRegex(ValueError, "overwrite forbidden"):
                download_file(
                    row,
                    output,
                    retries=3,
                    opener=lambda request, timeout: self.fail("transport was called"),
                )


if __name__ == "__main__":
    unittest.main()
