from __future__ import annotations

import binascii
import copy
import hashlib
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from scripts.research.taro_o0r_source_adapter_runtime import source_adapter as adapter
from scripts.research.taro_o1r_r11_abstention_runtime import fresh_pool
from scripts.research.taro_o1r_r11_abstention_runtime import run_pool_download as runner


ROOT = Path(__file__).resolve().parents[3]


def reseal(value: dict[str, object]) -> dict[str, object]:
    clone = copy.deepcopy(value)
    clone.pop("content_sha256", None)
    clone["content_sha256"] = adapter.canonical_sha256(clone)
    return clone


class PoolDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = fresh_pool.build_pool(ROOT)

    def test_expanded_download_plan_is_exact_and_unique(self) -> None:
        rows = runner.expanded_download_plan(self.plan)
        self.assertEqual(len(rows), 144)
        self.assertEqual(len({row["relative_path"] for row in rows}), 144)
        self.assertTrue(all(row["relative_path"].startswith(("upsampling/Training/", "raw/Training/")) for row in rows))

    def test_head_admission_is_exact_and_media_unopened(self) -> None:
        head = runner.validate_head_admission(self.plan)
        self.assertEqual(head["available_asset_count"], 144)
        self.assertEqual(head["request_attempt_count"], 144)
        self.assertEqual(head["total_content_length_bytes"], runner.HEAD_TOTAL_BYTES)
        self.assertEqual(head["response_body_bytes_read"], 0)

    def test_no_decode_model_faro_or_training_authority(self) -> None:
        authority = runner.EXPECTED_AUTHORITY
        self.assertTrue(authority["source_download"])
        for key in ("archive_decode", "source_frame_decode", "model_execution", "faro_read", "truth_scoring", "training", "device", "deployment", "product", "safety"):
            self.assertFalse(authority[key])

    def test_formal_entrypoint_is_importable_module_form(self) -> None:
        self.assertEqual(
            runner.EXPECTED_ARGV,
            [
                "-m",
                "scripts.research.taro_o1r_r11_abstention_runtime.run_pool_download",
                "--execution-lock",
                runner.LOCK_RELATIVE,
            ],
        )

    def test_download_receipt_binds_file_hash_crc_and_head(self) -> None:
        row = runner.expanded_download_plan(self.plan)[0]
        payload = b"exact source payload"
        head = {"content_length_bytes": len(payload), "etag": "E", "last_modified": "L"}
        with tempfile.TemporaryDirectory() as directory:
            source_root = Path(directory)
            target = source_root / row["relative_path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            receipt = {
                "schema": "blindassist.taro.o1r.r11_source_asset_download_receipt.v1",
                **row,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
                "crc32": f"{binascii.crc32(payload) & 0xFFFFFFFF:08X}",
                "head_content_length_bytes": len(payload),
                "head_etag": "E",
                "head_last_modified": "L",
                "redirect_chain": [],
                "attempt_count": 1,
                "prior_transport_errors": [],
            }
            receipt["content_sha256"] = adapter.canonical_sha256(receipt)
            runner.validate_download_receipt(row, head, receipt, source_root=source_root)
            receipt["head_etag"] = "MUTATED"
            with self.assertRaisesRegex(runner.PoolDownloadError, "HEAD binding drift"):
                runner.validate_download_receipt(row, head, reseal(receipt), source_root=source_root)

    def test_only_transient_transport_failure_retries(self) -> None:
        row = runner.expanded_download_plan(self.plan)[0]
        payload = b"retry payload"
        head = {"content_length_bytes": len(payload), "etag": "E", "last_modified": "L", "url": row["url"], "http_status": 200}
        calls = 0

        def fake(request, _head, *, source_root, timeout_seconds):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("timeout")
            target = source_root / request["relative_path"]
            target.parent.mkdir(parents=True)
            target.write_bytes(payload)
            return {
                "asset": request["asset"],
                "url": request["url"],
                "relative_path": request["relative_path"],
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest().upper(),
                "crc32": f"{binascii.crc32(payload) & 0xFFFFFFFF:08X}",
                "head_content_length_bytes": len(payload),
                "head_etag": "E",
                "head_last_modified": "L",
                "redirect_chain": [],
            }

        with tempfile.TemporaryDirectory() as directory:
            receipt = runner.download_with_transient_retries(row, head, source_root=Path(directory), timeout_seconds=1, maximum_attempts=3, download_fn=fake)
            self.assertEqual(receipt["attempt_count"], 2)
            self.assertEqual(receipt["prior_transport_errors"], ["TimeoutError"])
        with self.assertRaises(ValueError):
            runner.download_with_transient_retries(row, head, source_root=Path("."), timeout_seconds=1, maximum_attempts=3, download_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("semantic")))
        with self.assertRaises(urllib.error.HTTPError):
            runner.download_with_transient_retries(
                row,
                head,
                source_root=Path("."),
                timeout_seconds=1,
                maximum_attempts=3,
                download_fn=lambda *_args, **_kwargs: (_ for _ in ()).throw(urllib.error.HTTPError(row["url"], 403, "forbidden", {}, None)),
            )

    def test_second_root_reservation_failure_is_sealed_and_manifested(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT / "artifacts.local") as directory:
            base = Path(directory)
            evidence_root = base / "evidence"
            source_root = base / "source"
            source_root.mkdir()
            files: dict[str, dict[str, object]] = {}
            with self.assertRaises(FileExistsError):
                runner._reserve_execution_roots(
                    source_root,
                    evidence_root,
                    execution_lock_sha256="A" * 64,
                    head_receipt_sha256="B" * 64,
                    files=files,
                )
            self.assertEqual(set(path.name for path in evidence_root.iterdir()), {"start-receipt.json", "failure.json", "manifest.json"})
            start = json.loads((evidence_root / "start-receipt.json").read_text(encoding="utf-8"))
            failure = json.loads((evidence_root / "failure.json").read_text(encoding="utf-8"))
            manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
            runner._validate_content_seal(start, "TEST_START_SEAL")
            runner._validate_content_seal(failure, "TEST_FAILURE_SEAL")
            runner._validate_content_seal(manifest, "TEST_MANIFEST_SEAL")
            self.assertEqual(failure["terminal"], runner.INVALID_TERMINAL)
            self.assertEqual(set(manifest["files"]), {"start-receipt.json", "failure.json"})
            for relative, row in manifest["files"].items():
                target = evidence_root / relative
                self.assertEqual(target.stat().st_size, row["bytes"])
                self.assertEqual(runner.materializer.sha256_file(target), row["sha256"])

    def test_retries_share_one_asset_wall_budget(self) -> None:
        row = runner.expanded_download_plan(self.plan)[0]
        head = {"content_length_bytes": 1, "etag": "E", "last_modified": "L", "url": row["url"], "http_status": 200}
        now = [0.0]
        observed_timeouts: list[float] = []

        def fake(_request, _head, *, source_root, timeout_seconds):
            observed_timeouts.append(timeout_seconds)
            if len(observed_timeouts) == 1:
                now[0] += 200.0
                raise TimeoutError("first attempt")
            now[0] += 101.0
            return {}

        with self.assertRaisesRegex(runner.PoolDownloadError, "asset wall budget was exceeded"):
            runner.download_with_transient_retries(
                row,
                head,
                source_root=Path("."),
                timeout_seconds=300.0,
                maximum_attempts=3,
                download_fn=fake,
                monotonic_fn=lambda: now[0],
            )
        self.assertEqual(observed_timeouts, [300.0, 100.0])

    def test_evidence_projection_includes_success_manifest(self) -> None:
        files = {"start-receipt.json": {"bytes": 17}}
        manifest = runner._sealed_record({"schema": "test.manifest.v1", "files": files})
        self.assertEqual(runner._projected_evidence_bytes(files, manifest), 17 + len(runner._canonical_json_line(manifest)))


if __name__ == "__main__":
    unittest.main()
