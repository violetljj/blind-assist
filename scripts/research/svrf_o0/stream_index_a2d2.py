#!/usr/bin/env python3
"""Build a resumable A2D2 TAR member index without retaining archive payloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from typing import BinaryIO

from .probe_materialization_capability import A2D2_BUCKET, parse_tar_header, request_range


SCHEMA = "blindassist.svrf_o0.a2d2_stream_index.v1"
RESERVE_BYTES = 64 * 1024 * 1024 * 1024
COMMIT_MEMBER_INTERVAL = 100
PROGRESS_INTERVAL_SECONDS = 30


class UnexpectedStreamEnd(RuntimeError):
    pass


class CountingReader:
    def __init__(self, source: BinaryIO) -> None:
        self.source = source
        self.bytes_read = 0

    def read_exact(self, size: int) -> bytes:
        chunks = []
        remaining = size
        while remaining:
            chunk = self.source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise UnexpectedStreamEnd(f"stream ended with {remaining} bytes still required")
            chunks.append(chunk)
            self.bytes_read += len(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def hash_exact(self, size: int, *, retain_limit: int = 0) -> tuple[str, bytes]:
        digest = hashlib.sha256()
        retained = bytearray()
        remaining = size
        while remaining:
            chunk = self.source.read(min(1024 * 1024, remaining))
            if not chunk:
                raise UnexpectedStreamEnd(f"stream ended inside payload with {remaining} bytes required")
            digest.update(chunk)
            if retain_limit and len(retained) < retain_limit:
                retained.extend(chunk[: retain_limit - len(retained)])
            self.bytes_read += len(chunk)
            remaining -= len(chunk)
        return digest.hexdigest(), bytes(retained)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tar_header_checksum(block: bytes) -> int:
    return int(block[148:156].split(b"\0", 1)[0].strip() or b"0", 8)


def find_archive(lock: dict, archive_name: str) -> dict:
    source = next(item for item in lock["sources"] if item["source_id"] == "A2D2_SENSOR_FUSION")
    for parent in source["parents"]:
        for modality, archive in parent["archives"].items():
            if archive["name"] == archive_name:
                return {
                    "parent_id": parent["parent_id"],
                    "modality": modality,
                    **archive,
                }
    raise ValueError(f"archive is not in the frozen A2D2 source lock: {archive_name}")


def connect_index(path: Path, binding: dict, source_lock_sha256: str) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS members(
            header_offset INTEGER PRIMARY KEY,
            member_name TEXT NOT NULL,
            member_type TEXT NOT NULL,
            payload_offset INTEGER NOT NULL,
            payload_bytes INTEGER NOT NULL,
            padded_payload_bytes INTEGER NOT NULL,
            payload_sha256 TEXT NOT NULL,
            tar_header_checksum INTEGER NOT NULL,
            next_header_offset INTEGER NOT NULL,
            long_name_target TEXT
        );
        CREATE TABLE IF NOT EXISTS requests(
            request_id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at_utc TEXT NOT NULL,
            ended_at_utc TEXT NOT NULL,
            start_offset INTEGER NOT NULL,
            bytes_received INTEGER NOT NULL,
            curl_exit_code INTEGER NOT NULL,
            completed INTEGER NOT NULL,
            error TEXT
        );
        """
    )
    expected = {
        "schema": SCHEMA,
        "archive": binding["name"],
        "archive_bytes": str(binding["bytes"]),
        "official_md5": binding["md5"],
        "parent_id": binding["parent_id"],
        "modality": binding["modality"],
        "source_lock_sha256": source_lock_sha256,
        "url": A2D2_BUCKET + binding["name"],
    }
    try:
        existing = dict(connection.execute("SELECT key, value FROM metadata"))
        if existing:
            for key, value in expected.items():
                if existing.get(key) != str(value):
                    raise ValueError(f"existing stream index metadata drift: {key}")
        else:
            connection.executemany("INSERT INTO metadata(key, value) VALUES(?, ?)", expected.items())
            connection.execute("INSERT INTO metadata(key, value) VALUES('status', 'IN_PROGRESS')")
            connection.execute("INSERT INTO metadata(key, value) VALUES('next_header_offset', '0')")
            connection.execute("INSERT INTO metadata(key, value) VALUES('pending_long_name', '')")
            connection.commit()
    except Exception:
        connection.close()
        raise
    return connection


def metadata(connection: sqlite3.Connection) -> dict[str, str]:
    return dict(connection.execute("SELECT key, value FROM metadata"))


def set_metadata(connection: sqlite3.Connection, **values: str | int) -> None:
    connection.executemany(
        "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [(key, str(value)) for key, value in values.items()],
    )


def curl_command(url: str, start_offset: int) -> list[str]:
    executable = shutil.which("curl.exe") or shutil.which("curl")
    if executable is None:
        raise ValueError("curl is required for the resumable A2D2 stream index")
    return [
        executable,
        "-sS",
        "-fL",
        "--range",
        f"{start_offset}-",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "--user-agent",
        "BlindAssist-SVRF-O0-A2D2-Stream-Index/1",
        url,
    ]


def record_request(
    connection: sqlite3.Connection,
    *,
    started_at: str,
    start_offset: int,
    bytes_received: int,
    exit_code: int,
    completed: bool,
    error: str | None,
) -> None:
    connection.execute(
        """
        INSERT INTO requests(started_at_utc, ended_at_utc, start_offset, bytes_received,
                             curl_exit_code, completed, error)
        VALUES(?, ?, ?, ?, ?, ?, ?)
        """,
        (started_at, utc_now(), start_offset, bytes_received, exit_code, int(completed), error),
    )
    connection.commit()


def stream_once(connection: sqlite3.Connection, binding: dict) -> bool:
    state = metadata(connection)
    start_offset = int(state["next_header_offset"])
    expected_bytes = int(binding["bytes"])
    if start_offset >= expected_bytes:
        return True
    url = A2D2_BUCKET + binding["name"]
    range_probe = request_range(url, start_offset, start_offset)
    if range_probe.total != expected_bytes:
        raise ValueError("A2D2 stream object-size drift")
    pending_long_name = state.get("pending_long_name") or None
    started_at = utc_now()
    last_progress = time.monotonic()
    batch_members = 0
    logical_offset = start_offset
    with tempfile.NamedTemporaryFile(prefix="svrf-a2d2-curl-", suffix=".stderr", delete=False) as stderr_file:
        stderr_path = Path(stderr_file.name)
    stderr_stream = stderr_path.open("wb")
    process = subprocess.Popen(curl_command(url, start_offset), stdout=subprocess.PIPE, stderr=stderr_stream)
    if process.stdout is None:
        raise RuntimeError("curl stdout pipe was not created")
    reader = CountingReader(process.stdout)
    completed = False
    error = None
    try:
        connection.execute("BEGIN")
        while logical_offset < expected_bytes:
            header_offset = logical_offset
            header_block = reader.read_exact(512)
            logical_offset += 512
            if header_block == bytes(512):
                second = reader.read_exact(512)
                logical_offset += 512
                if second != bytes(512):
                    raise ValueError("A2D2 TAR has only one zero terminal block")
                remaining = expected_bytes - logical_offset
                while remaining:
                    chunk_size = min(1024 * 1024, remaining)
                    if reader.read_exact(chunk_size) != bytes(chunk_size):
                        raise ValueError("A2D2 TAR trailing padding is non-zero")
                    logical_offset += chunk_size
                    remaining -= chunk_size
                set_metadata(
                    connection,
                    status="COMPLETE",
                    next_header_offset=expected_bytes,
                    pending_long_name="",
                    completed_at_utc=utc_now(),
                )
                connection.commit()
                completed = True
                break
            header = parse_tar_header(header_block)
            if header is None:
                raise AssertionError("non-zero TAR header parsed as terminal")
            payload_bytes = int(header["size"])
            padded_bytes = ((payload_bytes + 511) // 512) * 512
            payload_sha256, retained = reader.hash_exact(
                payload_bytes,
                retain_limit=4096 if header["type"] == "L" else 0,
            )
            logical_offset += payload_bytes
            padding = padded_bytes - payload_bytes
            if padding:
                reader.read_exact(padding)
                logical_offset += padding
            long_target = None
            member_name = str(header["name"])
            if header["type"] == "L":
                long_target = retained.split(b"\0", 1)[0].decode("utf-8", "replace")
                pending_long_name = long_target
            elif pending_long_name is not None:
                member_name = pending_long_name
                pending_long_name = None
            connection.execute(
                """
                INSERT INTO members(header_offset, member_name, member_type, payload_offset,
                                    payload_bytes, padded_payload_bytes, payload_sha256,
                                    tar_header_checksum, next_header_offset, long_name_target)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    header_offset,
                    member_name,
                    str(header["type"]),
                    header_offset + 512,
                    payload_bytes,
                    padded_bytes,
                    payload_sha256,
                    tar_header_checksum(header_block),
                    logical_offset,
                    long_target,
                ),
            )
            batch_members += 1
            if batch_members >= COMMIT_MEMBER_INTERVAL:
                set_metadata(
                    connection,
                    next_header_offset=logical_offset,
                    pending_long_name=pending_long_name or "",
                )
                connection.commit()
                connection.execute("BEGIN")
                batch_members = 0
            now = time.monotonic()
            if now - last_progress >= PROGRESS_INTERVAL_SECONDS:
                count = connection.execute("SELECT COUNT(*) FROM members").fetchone()[0] + batch_members
                print(
                    f"PROGRESS archive={binding['name']} logical_bytes={logical_offset}/{expected_bytes} "
                    f"members={count} request_bytes={reader.bytes_read}",
                    flush=True,
                )
                last_progress = now
        if not completed:
            set_metadata(
                connection,
                next_header_offset=logical_offset,
                pending_long_name=pending_long_name or "",
            )
            connection.commit()
    except Exception as exc:
        connection.rollback()
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if process.poll() is None:
            if completed:
                process.wait(timeout=30)
            else:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        exit_code = int(process.returncode or 0)
        stderr_stream.close()
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        stderr_path.unlink(missing_ok=True)
        if exit_code != 0 and error is None:
            error = f"curl exit {exit_code}: {stderr}"
        record_request(
            connection,
            started_at=started_at,
            start_offset=start_offset,
            bytes_received=reader.bytes_read,
            exit_code=exit_code,
            completed=completed and exit_code == 0,
            error=error or (stderr if stderr else None),
        )
    if error is not None:
        print(f"RETRYABLE archive={binding['name']} offset={start_offset} error={error}", file=sys.stderr, flush=True)
        return False
    if completed and exit_code != 0:
        return False
    return completed


def export_receipt(connection: sqlite3.Connection, database_path: Path, receipt_path: Path) -> dict:
    state = metadata(connection)
    if state.get("status") != "COMPLETE":
        raise ValueError("cannot export an incomplete A2D2 stream index")
    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    connection.commit()
    request = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(bytes_received), 0), COALESCE(SUM(completed), 0) FROM requests"
    ).fetchone()
    member = connection.execute(
        "SELECT COUNT(*), COALESCE(SUM(payload_bytes), 0), MIN(header_offset), MAX(next_header_offset) FROM members"
    ).fetchone()
    result = {
        "schema": "blindassist.svrf_o0.a2d2_stream_index_receipt.v1",
        "status": "A2D2_TAR_STREAM_INDEX_COMPLETE",
        "archive": state["archive"],
        "archive_bytes": int(state["archive_bytes"]),
        "official_md5_bound_not_recomputed": state["official_md5"],
        "parent_id": state["parent_id"],
        "modality": state["modality"],
        "source_lock_sha256": state["source_lock_sha256"],
        "database_logical_path": database_path.as_posix(),
        "database_sha256": hashlib.sha256(database_path.read_bytes()).hexdigest(),
        "member_count": int(member[0]),
        "indexed_payload_bytes": int(member[1]),
        "first_header_offset": int(member[2]),
        "last_next_header_offset": int(member[3]),
        "request_count": int(request[0]),
        "network_bytes_received": int(request[1]),
        "completed_request_count": int(request[2]),
        "archive_payload_retained": False,
        "media_tensor_or_label_decode_count": 0,
        "candidate_run_count": 0,
        "outcome_accessed": False,
        "completed_at_utc": state["completed_at_utc"],
    }
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("artifacts.local"))
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()
    physical_root = args.artifact_root.resolve()
    if physical_root.drive.upper() != "F:":
        raise ValueError("A2D2 large-data stream index must use the canonical F-drive artifact root")
    if shutil.disk_usage(physical_root).free < RESERVE_BYTES:
        raise ValueError("F-drive safety reserve is not available")
    lock_bytes = args.lock.read_bytes()
    lock = json.loads(lock_bytes)
    binding = find_archive(lock, args.archive)
    source_lock_sha256 = hashlib.sha256(lock_bytes).hexdigest()
    database_path = args.output_root / f"{args.archive}.index.sqlite3"
    receipt_path = args.output_root / f"{args.archive}.receipt.json"
    connection = connect_index(database_path, binding, source_lock_sha256)
    try:
        if metadata(connection).get("status") != "COMPLETE":
            for _ in range(args.max_retries + 1):
                if stream_once(connection, binding):
                    break
            else:
                raise RuntimeError("A2D2 stream index exhausted the bounded retry budget")
        result = export_receipt(connection, database_path, receipt_path)
    finally:
        connection.close()
    print(result["status"])


if __name__ == "__main__":
    main()
