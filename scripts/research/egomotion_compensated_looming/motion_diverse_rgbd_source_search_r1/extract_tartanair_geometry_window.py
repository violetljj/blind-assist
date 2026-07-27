"""Extract one frozen TartanAir pose+depth window without reading RGB payloads."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--queue-index", type=int, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    amendment_path = args.amendment.resolve()
    queue_path = args.queue.resolve()
    archive_path = args.archive.resolve()
    amendment = load(amendment_path)
    queue = load(queue_path)
    if sha(archive_path) != amendment["source"]["archive_sha256"]:
        raise ValueError("TARTANAIR_ARCHIVE_IDENTITY")
    if queue["archive_sha256"] != amendment["source"]["archive_sha256"]:
        raise ValueError("TARTANAIR_QUEUE_ARCHIVE_IDENTITY")
    if queue["amendment_sha256"] != sha(amendment_path):
        raise ValueError("TARTANAIR_QUEUE_AMENDMENT_IDENTITY")
    windows = queue["positive_proxy_queue"]
    if args.queue_index < 0 or args.queue_index >= len(windows):
        raise ValueError("TARTANAIR_QUEUE_INDEX")
    window = windows[args.queue_index]
    trajectory = PurePosixPath(window["trajectory"])
    if trajectory.is_absolute() or ".." in trajectory.parts:
        raise ValueError("TARTANAIR_TRAJECTORY_PATH")
    expected: dict[str, tuple[str, str]] = {}
    for frame_id in window["frame_ids"]:
        for kind, suffix in (("pose", "_cam.npz"), ("depth", "_depth.npy")):
            member = f"{trajectory.as_posix()}/{frame_id}{suffix}"
            expected[member] = (kind, frame_id)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"OUTPUT_ROOT_EXISTS:{output_root}")
    records = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            target = expected.get(member.name)
            if target is None:
                continue
            if not member.isfile():
                raise ValueError(f"TARTANAIR_MEMBER_NOT_FILE:{member.name}")
            raw = archive.extractfile(member).read()
            kind, frame_id = target
            relative = Path(kind) / f"{frame_id}{Path(member.name).suffix}"
            path = output_root / relative
            write_exclusive(path, raw)
            records.append(
                {
                    "archive_member": member.name,
                    "bytes": len(raw),
                    "frame_id": frame_id,
                    "kind": kind,
                    "relative_path": relative.as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    if {row["archive_member"] for row in records} != set(expected):
        missing = sorted(set(expected) - {row["archive_member"] for row in records})
        raise ValueError(f"TARTANAIR_REQUIRED_MEMBER_MISSING:{missing[:3]}")
    records.sort(key=lambda row: (row["frame_id"], row["kind"]))
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.tartanair_geometry_extract.v1",
        "protocol_id": amendment["protocol_id"],
        "amendment_sha256": sha(amendment_path),
        "queue_sha256": sha(queue_path),
        "archive_sha256": sha(archive_path),
        "queue_index": args.queue_index,
        "window": window,
        "output_root": output_root.as_posix(),
        "member_count": len(records),
        "members": records,
        "rgb_members_read": 0,
        "rgb_bytes_accessed": 0,
    }
    write_exclusive(
        args.manifest.resolve(),
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    print(
        json.dumps(
            {
                "window_id": window["window_id"],
                "member_count": len(records),
                "extracted_bytes": sum(row["bytes"] for row in records),
                "rgb_bytes_accessed": 0,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
