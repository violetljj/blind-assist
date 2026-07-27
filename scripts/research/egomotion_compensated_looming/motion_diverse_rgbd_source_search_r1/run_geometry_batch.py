"""Run a frozen cross-sequence ETH3D geometry batch with the unchanged formula."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
from typing import Any
import zlib

from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.run_eth3d_geometry import (
    longest,
    pair_worker,
)
from scripts.research.egomotion_compensated_looming.motion_diverse_rgbd_geometry_admission_r0.template import (
    DEFAULT_WORKERS,
    decimal_median,
)
from scripts.research.egomotion_compensated_looming.real_positive_approach_role_admission_r2_cid_sims import (
    producer as geometry,
)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON_OBJECT_REQUIRED")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_exclusive(path: Path, value: Any, *, jsonl: bool = False) -> str:
    if jsonl:
        payload = b"".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
            for row in value
        )
    else:
        payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(os.fspath(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()


def summarize(window: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    denominator = int(window["pair_count"])
    evaluable = [row for row in rows if row["geometry_evaluable"] is True]
    counts = Counter(row["geometry_band"] for row in evaluable)
    abstentions = Counter(
        row["geometry_abstention_reason"]
        for row in rows
        if row["geometry_evaluable"] is not True
    )
    positive_count, positive_duration = longest(rows, "POSITIVE_APPROACH_GEOMETRY")
    below_count, below_duration = longest(rows, "BELOW_TRIGGER_REFERENCE")
    coverage = Decimal(len(evaluable)) / Decimal(denominator)
    positive_fraction = Decimal(counts["POSITIVE_APPROACH_GEOMETRY"]) / Decimal(denominator)
    below_fraction = Decimal(counts["BELOW_TRIGGER_REFERENCE"]) / Decimal(denominator)
    positive_ok = coverage >= Decimal("0.8") and positive_fraction >= Decimal("0.8") and positive_duration >= Decimal("5")
    below_ok = coverage >= Decimal("0.8") and below_fraction >= Decimal("0.8") and below_duration >= Decimal("5")
    if positive_ok and below_ok:
        raise ValueError(f"ROLE_OVERLAP:{window['window_id']}")
    role = (
        "POSITIVE_APPROACH_WINDOW"
        if positive_ok
        else "BELOW_TRIGGER_REFERENCE_WINDOW"
        if below_ok
        else "AMBIGUOUS_OR_INELIGIBLE"
    )
    values = [Decimal(str(row["geometry_signed_radial_expansion_per_s"])) for row in evaluable]
    return {
        "window_id": window["window_id"],
        "sequence_id": window["sequence_id"],
        "proxy_queue": window["proxy_queue"],
        "proxy_queue_index": window["proxy_queue_index"],
        "start_timestamp_s": window["start_timestamp_s"],
        "end_timestamp_s": window["end_timestamp_s"],
        "candidate_pair_count": denominator,
        "geometry_evaluable_pair_count": len(evaluable),
        "geometry_pair_coverage_fixed_denominator": float(coverage),
        "geometry_abstention_reasons": dict(sorted(abstentions.items())),
        "geometry_band_counts": dict(sorted(counts.items())),
        "positive_fraction_fixed_denominator": float(positive_fraction),
        "below_fraction_fixed_denominator": float(below_fraction),
        "longest_positive_run_pair_count": positive_count,
        "longest_positive_run_duration_s": float(positive_duration),
        "longest_below_run_pair_count": below_count,
        "longest_below_run_duration_s": float(below_duration),
        "median_signed_radial_expansion_per_s": float(decimal_median(values)) if values else None,
        "role": role,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--implementation-lock", type=Path, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--depth-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()
    if args.workers != 8:
        raise ValueError("WORKERS_MUST_EQUAL_8")
    contract_path = args.contract.resolve()
    contract = load(contract_path)
    implementation_lock = load(args.implementation_lock.resolve())
    if implementation_lock.get("status") != "FROZEN_BEFORE_DEPTH_GEOMETRY":
        raise ValueError("IMPLEMENTATION_LOCK_STATUS")
    repo = Path(__file__).resolve().parents[4]
    for entry in implementation_lock["files"]:
        if sha(repo / entry["path"]) != entry["sha256"]:
            raise ValueError(f"IMPLEMENTATION_LOCK_DRIFT:{entry['path']}")
    batch_path = args.batch.resolve()
    batch = load(batch_path)
    if batch["contract_sha256"] != sha(contract_path) or batch["rgb_access_allowed"] is not False:
        raise ValueError("BATCH_CONTRACT_OR_RGB_BOUNDARY")
    inventory = load(args.depth_inventory.resolve())
    if (
        implementation_lock["local_inputs"]["geometry_batch_sha256"] != sha(batch_path)
        or implementation_lock["local_inputs"]["depth_inventory_sha256"]
        != sha(args.depth_inventory.resolve())
    ):
        raise ValueError("IMPLEMENTATION_LOCK_LOCAL_INPUT")
    expected = {member for window in batch["windows"] for member in window["depth_members"]}
    indexed = {row["path"]: row for row in inventory["members"]}
    if expected != set(indexed):
        raise ValueError("DEPTH_INVENTORY_SET")
    depth_root = args.depth_root.resolve()
    verification = []
    for member in sorted(expected):
        path = depth_root.joinpath(*Path(member).parts)
        data = path.read_bytes()
        info = indexed[member]
        if len(data) != int(info["size"]) or f"{zlib.crc32(data) & 0xFFFFFFFF:08x}" != info["crc32"]:
            raise ValueError(f"DEPTH_MEMBER_IDENTITY:{member}")
        verification.append({"path": member, "bytes": len(data), "crc32": info["crc32"], "sha256": hashlib.sha256(data).hexdigest()})
    output = args.output_dir.resolve()
    write_exclusive(output / "depth_verification.json", {"member_count": len(verification), "members": verification})
    metadata_root = args.metadata_root.resolve()
    source_cache: dict[str, tuple[Any, Any]] = {}
    all_rows: list[dict[str, Any]] = []
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for window in batch["windows"]:
            sequence_id = window["sequence_id"]
            if sequence_id not in source_cache:
                source_root = metadata_root / sequence_id / sequence_id
                source_cache[sequence_id] = (
                    geometry._parse_poses((source_root / "groundtruth.txt").read_bytes()),
                    geometry._parse_intrinsic((source_root / "calibration.txt").read_bytes()),
                )
            poses, intrinsic = source_cache[sequence_id]
            timestamps = [Decimal(Path(member).stem) for member in window["depth_members"]]
            tasks = []
            window_rows = []
            for pair_index, (previous, current) in enumerate(zip(timestamps, timestamps[1:])):
                base = {
                    "window_id": window["window_id"],
                    "sequence_id": sequence_id,
                    "pair_index": pair_index,
                    "previous_timestamp_s": float(previous),
                    "current_timestamp_s": float(current),
                    "dt_s": float(current - previous),
                }
                try:
                    previous_pose = geometry._interpolate_pose(poses, previous)
                    current_pose = geometry._interpolate_pose(poses, current)
                except ValueError as error:
                    window_rows.append({**base, "geometry_evaluable": False, "geometry_abstention_reason": str(error), "geometry_band": None})
                    continue
                member = window["depth_members"][pair_index]
                path = depth_root.joinpath(*Path(member).parts)
                tasks.append((base, str(path), intrinsic, previous_pose, current_pose, float(current - previous)))
            window_rows.extend(executor.map(pair_worker, tasks))
            window_rows.sort(key=lambda row: row["pair_index"])
            if len(window_rows) != int(window["pair_count"]):
                raise ValueError(f"PAIR_COUNT:{window['window_id']}")
            all_rows.extend(window_rows)
            summaries.append(summarize(window, window_rows))
    ledger_sha = write_exclusive(output / "geometry_pair_ledger.jsonl", all_rows, jsonl=True)
    positive = [row for row in summaries if row["role"] == "POSITIVE_APPROACH_WINDOW"]
    below = [row for row in summaries if row["role"] == "BELOW_TRIGGER_REFERENCE_WINDOW"]
    admitted = len(positive) >= 2 and len(below) >= 2
    result = {
        "protocol_id": contract["protocol_id"],
        "batch_id": batch["batch_id"],
        "batch_sha256": sha(batch_path),
        "workers": args.workers,
        "window_summaries": summaries,
        "positive_window_count": len(positive),
        "below_reference_window_count": len(below),
        "ambiguous_window_count": len(summaries) - len(positive) - len(below),
        "geometry_admitted": admitted,
        "selected_windows": (
            [*positive[:2], *below[:2]]
            if admitted
            else []
        ),
        "terminal": (
            "GEOMETRY_USABLE_FOUR_WINDOWS_FROZEN"
            if admitted
            else "ADVANCE_FROZEN_POSE_QUEUES"
        ),
        "geometry_pair_ledger_sha256": ledger_sha,
        "rgb_bytes_accessed": 0,
        "algorithm_changed": False,
    }
    write_exclusive(output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("positive_window_count", "below_reference_window_count", "ambiguous_window_count", "geometry_admitted", "terminal")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
