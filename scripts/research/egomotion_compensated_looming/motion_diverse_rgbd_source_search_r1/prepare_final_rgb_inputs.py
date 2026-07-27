"""Prepare only the RGB members named by the frozen four-window cohort."""

from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
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


def parse_tum_rgb_index(path: Path) -> list[tuple[Decimal, str]]:
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != 2:
            raise ValueError("TUM_RGB_INDEX_COLUMNS")
        rows.append((Decimal(values[0]), values[1]))
    if any(left[0] >= right[0] for left, right in zip(rows, rows[1:])):
        raise ValueError("TUM_RGB_INDEX_ORDER")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--eth3d-rgb-inventory", type=Path, required=True)
    parser.add_argument("--eth3d-include-file", type=Path, required=True)
    parser.add_argument("--tartanair-archive", type=Path, required=True)
    parser.add_argument("--tartanair-rgb-root", type=Path, required=True)
    parser.add_argument("--tum-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort_path = args.cohort.resolve()
    cohort = load(cohort_path)
    if cohort["status"] != "FOUR_WINDOW_IDENTITIES_FROZEN_BEFORE_NEW_RGB_ACCESS":
        raise ValueError("COHORT_NOT_FROZEN")
    windows = {row["window_id"]: row for row in cohort["windows"]}
    eth3d = windows["desk_changing_1@4065.364250422"]
    expected_eth3d = [
        member.replace("/depth/", "/rgb/") for member in eth3d["depth_members"]
    ]
    inventory = load(args.eth3d_rgb_inventory.resolve())
    by_name = {row["path"]: row for row in inventory["members"] if int(row["size"]) > 0}
    if any(member not in by_name for member in expected_eth3d):
        raise ValueError("ETH3D_FROZEN_RGB_MEMBER_MISSING")
    include_payload = ("\n".join(expected_eth3d) + "\n").encode("utf-8")
    write_exclusive(args.eth3d_include_file.resolve(), include_payload)

    tartanair = windows["japanesealley/Hard/P002@000260"]
    archive_path = args.tartanair_archive.resolve()
    tartanair_input = load(Path(cohort["evidence"]["tartanair_extract"]["path"]))
    if sha(archive_path) != tartanair_input["archive_sha256"]:
        raise ValueError("TARTANAIR_ARCHIVE_IDENTITY")
    expected_tartan = {
        f"japanesealley/Hard/P002/{frame_id}_rgb.png": frame_id
        for frame_id in tartanair["frame_ids"]
    }
    tartan_root = args.tartanair_rgb_root.resolve()
    if tartan_root.exists():
        raise FileExistsError(f"TARTANAIR_RGB_ROOT_EXISTS:{tartan_root}")
    tartan_records = []
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            frame_id = expected_tartan.get(member.name)
            if frame_id is None:
                continue
            raw = archive.extractfile(member).read()
            relative = Path("rgb") / f"{frame_id}.png"
            write_exclusive(tartan_root / relative, raw)
            tartan_records.append(
                {
                    "archive_member": member.name,
                    "bytes": len(raw),
                    "frame_id": frame_id,
                    "relative_path": relative.as_posix(),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
    if {row["archive_member"] for row in tartan_records} != set(expected_tartan):
        raise ValueError("TARTANAIR_FROZEN_RGB_MEMBER_MISSING")
    tartan_records.sort(key=lambda row: row["frame_id"])

    tum_root = args.tum_root.resolve()
    tum_index = parse_tum_rgb_index(tum_root / "rgb.txt")
    tum_windows = []
    tum_seen: set[str] = set()
    for window_id in ("TUM_RGBD_FR2_RPY@2", "TUM_RGBD_FR2_RPY@7"):
        window = windows[window_id]
        start = Decimal(window["start_timestamp_s"])
        end = Decimal(window["end_timestamp_s"])
        selected = [(timestamp, relative) for timestamp, relative in tum_index if start <= timestamp < end]
        if len(selected) != int(window["geometry_summary"]["candidate_pair_count"]) + 1:
            raise ValueError(f"TUM_RGB_PAIR_COUNT:{window_id}")
        if any(not (Decimal("0") < right[0] - left[0] <= Decimal("0.1")) for left, right in zip(selected, selected[1:])):
            raise ValueError(f"TUM_RGB_DT:{window_id}")
        members = []
        for timestamp, relative in selected:
            path = (tum_root / relative).resolve()
            if tum_root != path and tum_root not in path.parents:
                raise ValueError("TUM_RGB_PATH_ESCAPE")
            if not path.is_file():
                raise ValueError(f"TUM_RGB_MISSING:{relative}")
            tum_seen.add(relative)
            members.append(
                {
                    "timestamp_s": str(timestamp),
                    "relative_path": relative,
                    "bytes": path.stat().st_size,
                    "sha256": sha(path),
                }
            )
        tum_windows.append({"window_id": window_id, "members": members})
    result = {
        "schema": "rcle.motion_diverse_rgbd.source_search.rgb_preparation.v1",
        "protocol_id": cohort["protocol_id"],
        "cohort_sha256": sha(cohort_path),
        "eth3d": {
            "include_file": args.eth3d_include_file.resolve().as_posix(),
            "include_file_sha256": hashlib.sha256(include_payload).hexdigest(),
            "expected_member_count": len(expected_eth3d),
            "expected_members": [
                {
                    "path": member,
                    "bytes": int(by_name[member]["size"]),
                    "crc32": by_name[member]["crc32"],
                }
                for member in expected_eth3d
            ],
        },
        "tartanair": {
            "archive_sha256": sha(archive_path),
            "rgb_root": tartan_root.as_posix(),
            "members": tartan_records,
        },
        "tum": {
            "rgb_root": tum_root.as_posix(),
            "unique_member_count": len(tum_seen),
            "windows": tum_windows,
        },
        "window_substitution": False,
        "rgb_visual_inspection": False,
        "algorithm_outcome_accessed": False,
    }
    write_exclusive(
        args.output.resolve(),
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )
    print(
        json.dumps(
            {
                "eth3d_members_to_fetch": len(expected_eth3d),
                "tartanair_rgb_members_extracted": len(tartan_records),
                "tum_rgb_members_reused": len(tum_seen),
                "rgb_visual_inspection": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
