"""Download one pinned large artifact resumably with verified HTTP ranges."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import requests


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _chunk_sha256(path: Path, start: int, length: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        handle.seek(start)
        remaining = length
        while remaining:
            block = handle.read(min(1024 * 1024, remaining))
            if not block:
                raise ValueError("partial file ended inside a completed chunk")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def download(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    part = output.with_suffix(output.suffix + ".part")
    journal_path = output.with_suffix(output.suffix + ".chunks.json")
    lock_path = output.with_suffix(output.suffix + ".download.lock")
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(lock_fd, f"pid={os.getpid()}\n".encode())
        os.close(lock_fd)
        lock_fd = -1
        if output.exists():
            actual = _sha256(output)
            if output.stat().st_size == args.expected_bytes and actual == args.expected_sha256:
                return {"terminal": "ALREADY_COMPLETE", "bytes": args.expected_bytes, "sha256": actual}
            raise FileExistsError("completed output exists but does not match the pinned identity")

        if journal_path.exists():
            journal = json.loads(journal_path.read_text(encoding="utf-8"))
            identity = journal["identity"]
            if identity != {
                "url": args.url,
                "expected_bytes": args.expected_bytes,
                "expected_sha256": args.expected_sha256,
                "chunk_bytes": args.chunk_bytes,
            }:
                raise ValueError("download journal identity drift")
        else:
            journal = {
                "schema_version": "blindassist_pinned_range_download_v0",
                "identity": {
                    "url": args.url,
                    "expected_bytes": args.expected_bytes,
                    "expected_sha256": args.expected_sha256,
                    "chunk_bytes": args.chunk_bytes,
                },
                "completed_chunks": {},
            }
            _atomic_json(journal_path, journal)

        if not part.exists():
            with part.open("wb") as handle:
                handle.truncate(args.expected_bytes)
        elif part.stat().st_size != args.expected_bytes:
            raise ValueError("partial file size does not match the preallocated pinned size")

        completed = journal["completed_chunks"]
        for key, record in list(completed.items()):
            start = int(record["start"])
            length = int(record["length"])
            if _chunk_sha256(part, start, length) != record["sha256"]:
                raise ValueError(f"completed chunk {key} failed resume verification")

        session = requests.Session()
        session.headers["User-Agent"] = "BlindAssist-pinned-range-download/0"
        chunk_count = (args.expected_bytes + args.chunk_bytes - 1) // args.chunk_bytes
        with part.open("r+b") as handle:
            for index in range(chunk_count):
                key = str(index)
                if key in completed:
                    continue
                start = index * args.chunk_bytes
                end = min(args.expected_bytes, start + args.chunk_bytes) - 1
                expected_length = end - start + 1
                last_error = None
                for attempt in range(args.max_attempts):
                    try:
                        response = session.get(
                            args.url,
                            headers={"Range": f"bytes={start}-{end}"},
                            timeout=(20, args.read_timeout_seconds),
                        )
                        if response.status_code != 206:
                            raise ValueError(f"expected HTTP 206, received {response.status_code}")
                        content_range = response.headers.get("Content-Range")
                        expected_range = f"bytes {start}-{end}/{args.expected_bytes}"
                        if content_range != expected_range:
                            raise ValueError(f"Content-Range drift: {content_range!r}")
                        payload = response.content
                        if len(payload) != expected_length:
                            raise ValueError(
                                f"chunk length drift: {len(payload)} != {expected_length}"
                            )
                        break
                    except Exception as error:
                        last_error = error
                        if attempt + 1 == args.max_attempts:
                            raise
                        time.sleep(min(30, 2 ** attempt))
                else:
                    raise RuntimeError(str(last_error))
                handle.seek(start)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
                completed[key] = {
                    "start": start,
                    "length": expected_length,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
                _atomic_json(journal_path, journal)
                print(f"CHUNK {index + 1}/{chunk_count} COMPLETE", flush=True)

        actual = _sha256(part)
        if actual != args.expected_sha256:
            raise ValueError(f"full SHA-256 mismatch: {actual}")
        os.replace(part, output)
        return {"terminal": "DOWNLOAD_COMPLETE", "bytes": output.stat().st_size, "sha256": actual}
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        lock_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--chunk-bytes", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--read-timeout-seconds", type=int, default=90)
    args = parser.parse_args()
    result = download(args)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
