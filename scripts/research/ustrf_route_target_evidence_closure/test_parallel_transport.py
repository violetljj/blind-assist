#!/usr/bin/env python3
"""Contract tests for byte-equivalent parallel Range transport."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import tempfile
import threading
import unittest
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import inspect_remote_zip_inventory as inventory
import stream_remote_zip_entry as transport


class RangeHandler(BaseHTTPRequestHandler):
    server: "RangeServer"

    def do_GET(self) -> None:
        range_header = self.headers.get("Range", "")
        if not range_header.startswith("bytes=") or "-" not in range_header:
            self.send_error(400)
            return
        start_text, end_text = range_header.removeprefix("bytes=").split("-", 1)
        start = int(start_text)
        end = int(end_text)
        size = self.server.archive.stat().st_size
        if start < 0 or end < start or end >= size:
            self.send_error(416)
            return
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        with self.server.archive.open("rb") as stream:
            stream.seek(start)
            remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk:
                    raise RuntimeError("test archive ended before requested range")
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def log_message(self, _format: str, *_args: object) -> None:
        return


class RangeServer(ThreadingHTTPServer):
    archive: Path


class ParallelTransportTest(unittest.TestCase):
    def test_inventory_range_probe_refuses_http_200_before_body_read(self) -> None:
        class NonRangeResponse:
            status = 200
            headers = {"Content-Length": "41664649988"}
            read_called = False

            def __enter__(self):
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, *_args: object) -> bytes:
                self.read_called = True
                raise AssertionError("HTTP 200 body must not be read")

        response = NonRangeResponse()
        with patch.object(inventory.urllib.request, "urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "before body read"):
                inventory.range_get("https://example.invalid/archive.zip", 100, 199)
        self.assertFalse(response.read_called)

    def test_partition_range_is_exact_and_contiguous(self) -> None:
        parts = transport.partition_range(101, 23, 4)
        self.assertEqual(parts[0][0], 101)
        self.assertEqual(parts[-1][1], 123)
        self.assertEqual(sum(end - start + 1 for start, end in parts), 23)
        for left, right in zip(parts, parts[1:]):
            self.assertEqual(left[1] + 1, right[0])
        twelve_parts = transport.partition_range(500, 10_001, 12)
        self.assertEqual(len(twelve_parts), 12)
        self.assertEqual(sum(end - start + 1 for start, end in twelve_parts), 10_001)
        thirty_two_parts = transport.partition_range(700, 100_003, 32)
        self.assertEqual(len(thirty_two_parts), 32)
        self.assertEqual(
            sum(end - start + 1 for start, end in thirty_two_parts),
            100_003,
        )

    def test_parallel_output_matches_single_stream(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = random.Random(7).randbytes(6 * 1024 * 1024)
            archive = root / "fixture.zip"
            entry_name = "fixture.bag"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(entry_name, payload)
            with zipfile.ZipFile(archive) as bundle:
                info = bundle.getinfo(entry_name)

            server = RangeServer(("127.0.0.1", 0), RangeHandler)
            server.archive = archive
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                inventory = root / "inventory.json"
                inventory.write_text(
                    json.dumps(
                        {
                            "url": f"http://127.0.0.1:{server.server_port}/fixture.zip",
                            "entries": [
                                {
                                    "name": entry_name,
                                    "compressed_size": info.compress_size,
                                    "uncompressed_size": info.file_size,
                                    "local_header_offset": info.header_offset,
                                    "crc32": f"{info.CRC:08x}",
                                    "is_directory": False,
                                }
                            ],
                        }
                    ),
                    encoding="utf-8",
                )
                script = Path(__file__).with_name("stream_remote_zip_entry.py")
                common = [
                    sys.executable,
                    str(script),
                    "--inventory",
                    str(inventory),
                    "--entry",
                    entry_name,
                    "--max-compressed-bytes",
                    str(info.compress_size),
                    "--max-uncompressed-bytes",
                    str(info.file_size),
                ]
                single_output = root / "single.bag"
                single_receipt = root / "single-receipt.json"
                subprocess.run(
                    common
                    + [
                        "--output",
                        str(single_output),
                        "--receipt",
                        str(single_receipt),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                parallel_output = root / "parallel.bag"
                parallel_receipt = root / "parallel-receipt.json"
                cache_root = root / "cache"
                subprocess.run(
                    common
                    + [
                        "--output",
                        str(parallel_output),
                        "--receipt",
                        str(parallel_receipt),
                        "--range-workers",
                        "12",
                        "--range-parts",
                        "32",
                        "--compressed-cache-root",
                        str(cache_root),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            single = json.loads(single_receipt.read_text(encoding="utf-8"))
            parallel = json.loads(parallel_receipt.read_text(encoding="utf-8"))
            expected_sha = hashlib.sha256(payload).hexdigest()
            self.assertEqual(single_output.read_bytes(), payload)
            self.assertEqual(parallel_output.read_bytes(), payload)
            self.assertEqual(single["output_sha256"], expected_sha)
            self.assertEqual(parallel["output_sha256"], expected_sha)
            self.assertEqual(single["zip_crc32"], parallel["zip_crc32"])
            self.assertEqual(parallel["transport"]["mode"], "parallel_range_prefetch")
            self.assertEqual(parallel["transport"]["range_workers"], 12)
            self.assertEqual(parallel["transport"]["range_parts"], 32)
            self.assertEqual(list(cache_root.rglob("*.bin")), [])

    def test_zero_byte_cached_part_resumes_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "fixture.bin"
            payload = random.Random(11).randbytes(1024 * 1024)
            archive.write_bytes(payload)
            output = root / "part-0000.bin"
            output.touch()

            server = RangeServer(("127.0.0.1", 0), RangeHandler)
            server.archive = archive
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = transport.download_range_part(
                    url=f"http://127.0.0.1:{server.server_port}/fixture.bin",
                    start=0,
                    end=len(payload) - 1,
                    output=output,
                    progress=lambda _delta: None,
                    request_timeout_seconds=15,
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)

            self.assertEqual(output.read_bytes(), payload)
            self.assertEqual(result["bytes"], len(payload))
            self.assertEqual(result["sha256"], hashlib.sha256(payload).hexdigest())


if __name__ == "__main__":
    unittest.main()
