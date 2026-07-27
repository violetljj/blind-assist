"""One-claim, depth-and-pose-only scan of frozen CID-SIMS floor3_1.

The scan partitions the complete sequence into frozen non-overlapping windows,
reports all windows, and chooses the earliest qualifying window. RGB member
existence is checked, but RGB bytes are never read.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

import numpy as np

from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import producer as geometry


PROTOCOL_ID = "RCLE_REAL_POSITIVE_APPROACH_ROLE_ADMISSION_R3_CID_SIMS"
EXPECTED_BYTES = 2_211_008_069
EXPECTED_MD5 = "585d38855ad7d04817991cdbbb72016b"
WINDOW_SECONDS = Decimal("10.0")
MAX_DT = 0.100
MAX_PAIRS = 24
MIN_COVERAGE = 0.80
MIN_EVALUABLE = 12
MIN_SIGNED_RADIAL = 0.05
MIN_POSITIVE_FRACTION = 0.75
ADMITTED = "REAL_POSITIVE_APPROACH_ROLE_ADMITTED / VALID"
HOLD = "HOLD_NO_QUALIFYING_APPROACH_WINDOW / VALID"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("R3_JSON_OBJECT_REQUIRED")
    return value


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def sampled_indices(count: int) -> list[int]:
    if count <= MAX_PAIRS:
        return list(range(count))
    return sorted({math.floor(k * (count - 1) / (MAX_PAIRS - 1)) for k in range(MAX_PAIRS)})


def median(rows: list[dict[str, Any]], field: str) -> float | None:
    return float(np.median([float(row[field]) for row in rows])) if rows else None


def shared_rgbd_rows(archive: geometry.CidSimsArchive) -> list[Decimal]:
    color: set[Decimal] = set()
    depth: set[Decimal] = set()
    for item in archive.inventory:
        name = PurePosixPath(str(item["name"]))
        if len(name.parts) != 3 or name.parts[0] != geometry.SEQUENCE_ID:
            continue
        if name.suffix.lower() != ".png":
            continue
        try:
            timestamp = Decimal(name.stem)
        except Exception:
            continue
        if name.parts[1] == "color":
            color.add(timestamp)
        elif name.parts[1] == "depth":
            depth.add(timestamp)
    result = sorted(color & depth)
    if len(result) < 2:
        raise ValueError("R3_NO_EXACT_RGBD_TIMESTAMP_INTERSECTION")
    return result


def verify_bindings(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    receipt = load_object(args.transport_receipt)
    claim = load_object(args.claim)
    archive_sha = sha256(args.archive)
    hashes = {
        "archive": archive_sha,
        "transport_receipt": sha256(args.transport_receipt),
        "source_lock": sha256(args.source_lock),
        "contract": sha256(args.contract),
        "scanner": sha256(Path(__file__)),
        "validator": sha256(Path(claim["validator_path"])),
        "geometry_implementation": sha256(Path(geometry.__file__)),
    }
    if args.archive.stat().st_size != EXPECTED_BYTES:
        raise ValueError("R3_ARCHIVE_BYTES")
    if (
        receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("archive_bytes") != EXPECTED_BYTES
        or receipt.get("archive_md5") != EXPECTED_MD5
        or receipt.get("archive_sha256") != archive_sha
        or receipt.get("zip_opened") is not False
        or receipt.get("archive_members_read") != 0
    ):
        raise ValueError("R3_TRANSPORT_RECEIPT")
    if claim.get("protocol_id") != PROTOCOL_ID or claim.get("claim_count") != 1:
        raise ValueError("R3_CLAIM_IDENTITY")
    if claim.get("bindings") != hashes:
        raise ValueError("R3_CLAIM_BINDINGS")
    return archive_sha, hashes


def evaluate_window(
    archive: geometry.CidSimsArchive,
    poses: list[geometry.PoseRow],
    rgbd_timestamps: list[Decimal],
    start: Decimal,
    index: int,
    workers: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    end = start + WINDOW_SECONDS
    rows = [timestamp for timestamp in rgbd_timestamps if start <= timestamp < end]
    universe = [
        (previous, current)
        for previous, current in zip(rows, rows[1:])
        if 0.0 < float(current - previous) <= MAX_DT
    ]
    selected = [universe[i] for i in sampled_indices(len(universe))]
    tasks: list[Any] = []
    records: list[dict[str, Any]] = []
    for source_order, (previous, current) in enumerate(selected):
        previous_depth = f"depth/{previous}.png"
        current_depth = f"depth/{current}.png"
        previous_color = f"color/{previous}.png"
        current_color = f"color/{current}.png"
        record = {
            "window_index": index,
            "window_start_s": float(start),
            "window_end_s": float(end),
            "source_order": source_order,
            "previous_timestamp_s": float(previous),
            "current_timestamp_s": float(current),
            "previous_depth_member": previous_depth,
            "current_depth_member": current_depth,
            "dt_s": float(current - previous),
        }
        if not all(
            archive.member_exists(name)
            for name in (previous_depth, current_depth, previous_color, current_color)
        ):
            record.update(evaluable=False, reason="POSE_INDEXED_MEMBER_MISSING")
            records.append(record)
            continue
        try:
            raw_depth = archive.read_depth(previous_depth)
            previous_pose = geometry._interpolate_pose(poses, previous)
            current_pose = geometry._interpolate_pose(poses, current)
        except (KeyError, OSError, RuntimeError, ValueError):
            record.update(evaluable=False, reason="DEPTH_OR_POSE_INVALID")
            records.append(record)
            continue
        tasks.append(
            (
                record,
                raw_depth,
                geometry.INTRINSIC,
                previous_pose,
                current_pose,
                record["dt_s"],
            )
        )
    if workers == 1:
        evaluated = map(geometry._pair_worker, tasks)
        records.extend(record for record, _ in evaluated)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records.extend(record for record, _ in executor.map(geometry._pair_worker, tasks))
    records.sort(key=lambda row: int(row["source_order"]))
    evaluable = [row for row in records if row.get("evaluable") is True]
    coverage = len(evaluable) / len(records) if records else 0.0
    signed = median(evaluable, "median_signed_radial_expansion_per_s")
    positive = median(evaluable, "radial_expansion_positive_fraction")
    passed = bool(
        coverage >= MIN_COVERAGE
        and len(evaluable) >= MIN_EVALUABLE
        and signed is not None
        and signed >= MIN_SIGNED_RADIAL
        and positive is not None
        and positive >= MIN_POSITIVE_FRACTION
    )
    summary = {
        "window_index": index,
        "window_start_s": float(start),
        "window_end_s": float(end),
        "candidate_pair_universe_count": len(universe),
        "sampled_pair_count": len(records),
        "evaluable_pair_count": len(evaluable),
        "coverage": coverage,
        "median_signed_radial_expansion_per_s": signed,
        "median_radial_expansion_positive_fraction": positive,
        "median_q90_time_normalized_parallax_rad_per_s": median(
            evaluable, "q90_time_normalized_parallax_rad_per_s"
        ),
        "admitted": passed,
    }
    return summary, records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--transport-receipt", required=True, type=Path)
    parser.add_argument("--source-lock", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--claim", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("R3_WORKERS")
    if args.output_dir.exists():
        raise FileExistsError("R3_OUTPUT_DIR_ALREADY_EXISTS")
    archive_sha, bindings = verify_bindings(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    windows: list[dict[str, Any]] = []
    ledger: list[dict[str, Any]] = []
    with geometry.CidSimsArchive(args.archive) as archive:
        poses = geometry._parse_poses(archive.read_control("pose.txt"))
        rgbd_timestamps = shared_rgbd_rows(archive)
        anchor = next(
            (
                timestamp
                for timestamp in rgbd_timestamps
                if poses[0].timestamp <= timestamp <= poses[-1].timestamp
            ),
            None,
        )
        if anchor is None:
            raise ValueError("R3_NO_POSE_INDEXED_RGBD")
        complete_count = int((min(poses[-1].timestamp, rgbd_timestamps[-1]) - anchor) // WINDOW_SECONDS)
        for index in range(complete_count):
            summary, records = evaluate_window(
                archive,
                poses,
                rgbd_timestamps,
                anchor + index * WINDOW_SECONDS,
                index,
                args.workers,
            )
            windows.append(summary)
            ledger.extend(records)
            print(
                f"window={index + 1}/{complete_count} admitted={summary['admitted']} "
                f"signed={summary['median_signed_radial_expansion_per_s']} "
                f"positive={summary['median_radial_expansion_positive_fraction']}",
                flush=True,
            )
        inventory_sha = canonical_sha(archive.inventory)
        depth_reads = len(archive.depth_members_read)

    admitted = [window for window in windows if window["admitted"]]
    selected = admitted[0] if admitted else None
    terminal = ADMITTED if selected else HOLD
    ledger_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for row in ledger
    )
    write_exclusive(args.output_dir / "pair_ledger.jsonl", ledger_text)
    result = {
        "schema_version": "rcle.r3.geometry_role_result.v1",
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "terminal": terminal,
        "authority": "REAL_GEOMETRY_SELECTED_DEVELOPMENT_CANARY_ONLY",
        "archive_sha256": archive_sha,
        "archive_inventory_sha256": inventory_sha,
        "pose_row_count": len(poses),
        "exact_shared_rgbd_row_count": len(rgbd_timestamps),
        "complete_window_count": len(windows),
        "evaluated_window_count": len(windows),
        "admitted_window_count": len(admitted),
        "selected_earliest_admitted_window": selected,
        "windows": windows,
        "pair_ledger_sha256": hashlib.sha256(ledger_text.encode()).hexdigest(),
        "depth_members_read": depth_reads,
        "rgb_pixels_read": False,
        "rgb_algorithm_outcome_read": False,
        "confirmation_eligible": False,
        "performance_qualification_authorized": False,
        "bindings": bindings,
    }
    result["result_payload_sha256"] = canonical_sha(result)
    write_exclusive(
        args.output_dir / "result.json",
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
