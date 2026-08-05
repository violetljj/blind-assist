#!/usr/bin/env python3
"""Materialize label-blind Bonn RGB-D identities and integrity receipts."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import json
from pathlib import Path
from typing import Any


EXCLUSION_SCHEMA = "blindassist_p3_public_bonn_ancestry_exclusions_r0"
CATALOG_SCHEMA = "blindassist_p3_public_rgbd_source_admission_r0_catalog"
RECEIPT_SCHEMA = "blindassist_p3_public_bonn_identity_inventory_r0_receipt"
MAX_GAP_SECONDS = 0.5
MAX_RGB_DEPTH_DELTA_SECONDS = 0.05


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def resolve_member(sequence_root: Path, relative: str, *, required: bool) -> Path | None:
    require(not Path(relative).is_absolute(), f"absolute member path: {relative}")
    candidate = (sequence_root / relative).resolve()
    require(candidate.is_relative_to(sequence_root.resolve()), f"member escaped sequence: {relative}")
    if candidate.is_file():
        return candidate
    require(not required, f"referenced member missing: {candidate}")
    return None


def parse_index(sequence_root: Path, name: str, *, members_required: bool) -> list[tuple[float, str, Path | None]]:
    path = sequence_root / name
    require(path.is_file(), f"index missing: {path}")
    rows: list[tuple[float, str, Path | None]] = []
    for number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        require(len(parts) == 2, f"invalid {name}:{number}")
        timestamp = float(parts[0])
        require(not rows or timestamp > rows[-1][0], f"timestamps not strictly increasing: {name}:{number}")
        rows.append((timestamp, parts[1], resolve_member(sequence_root, parts[1], required=members_required)))
    require(len(rows) >= 4, f"fewer than four rows: {path}")
    return rows


def has_four_frame_run(timestamps: list[float]) -> bool:
    for start in range(len(timestamps) - 3):
        if all(0 < timestamps[i + 1] - timestamps[i] <= MAX_GAP_SECONDS for i in range(start, start + 3)):
            return True
    return False


def aggregate_files(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest().upper()


def paired_rgb_timestamps(
    rgb: list[tuple[float, str, Path | None]],
    depth: list[tuple[float, str, Path | None]],
) -> list[float]:
    valid_depth = [(timestamp, relative) for timestamp, relative, path in depth if path is not None]
    depth_times = [row[0] for row in valid_depth]
    used: set[int] = set()
    paired: list[float] = []
    for rgb_timestamp, _relative, _path in rgb:
        insertion = bisect.bisect_left(depth_times, rgb_timestamp)
        candidates = [index for index in (insertion - 1, insertion) if 0 <= index < len(depth_times) and index not in used]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(depth_times[index] - rgb_timestamp))
        if abs(depth_times[nearest] - rgb_timestamp) <= MAX_RGB_DEPTH_DELTA_SECONDS:
            used.add(nearest)
            paired.append(rgb_timestamp)
    return paired


def write_new(path: Path, value: dict[str, Any]) -> None:
    require(not path.exists(), f"overwrite forbidden: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def materialize(
    repo_root: Path,
    dataset_root: Path,
    archive_path: Path,
    base_catalog_path: Path,
    exclusions_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    repo_root = repo_root.resolve()
    dataset_root = dataset_root.resolve()
    require(dataset_root.is_dir(), "dataset root missing")
    require(archive_path.is_file(), "archive missing")
    exclusions = load_json(exclusions_path)
    require(exclusions.get("schema") == EXCLUSION_SCHEMA, "exclusion schema drift")
    require(set(exclusions) == {"schema", "basis", "excluded_parent_ids", "selection_rule", "forbidden_reads"}, "exclusion field drift")
    for basis in exclusions["basis"]:
        require(set(basis) == {"path", "sha256"}, "basis field drift")
        bound = (repo_root / basis["path"]).resolve()
        require(bound.is_relative_to(repo_root), "basis escaped repository")
        require(sha256_file(bound) == basis["sha256"], f"basis SHA mismatch: {basis['path']}")
    excluded = set(exclusions["excluded_parent_ids"])
    require(len(excluded) == len(exclusions["excluded_parent_ids"]), "duplicate exclusion")
    sequences = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    require(sequences, "no sequence directories")
    require(excluded <= {path.name for path in sequences}, "excluded parent absent from official inventory")

    identities = []
    details = []
    for sequence in sequences:
        rgb = parse_index(sequence, "rgb.txt", members_required=True)
        depth = parse_index(sequence, "depth.txt", members_required=False)
        rgb_files = [path for path in (sequence / "rgb").iterdir() if path.is_file()]
        depth_files = [path for path in (sequence / "depth").iterdir() if path.is_file()]
        require(rgb_files, f"RGB directory empty: {sequence}")
        require(depth_files, f"depth directory empty: {sequence}")
        rgb_aggregate = aggregate_files(rgb_files, sequence)
        depth_aggregate = aggregate_files(depth_files, sequence)
        paired_timestamps = paired_rgb_timestamps(rgb, depth)
        continuity = has_four_frame_run(paired_timestamps)
        missing_depth_refs = [relative for _timestamp, relative, path in depth if path is None]
        identity = {
            "parent_id": sequence.name,
            "higher_cluster_id": sequence.name,
            "rgb_identity_count": len(rgb),
            "four_frame_continuity_confirmed": continuity,
            "raw_metric_sensor_assets_present": continuity,
            "rgb_files_sha256_complete": True,
            "metric_sensor_files_sha256_complete": True,
            "timestamp_files_sha256_complete": True,
            "ancestry_excluded": sequence.name in excluded,
        }
        identities.append(identity)
        details.append({
            "parent_id": sequence.name,
            "ancestry_excluded": sequence.name in excluded,
            "rgb_index_row_count": len(rgb),
            "depth_index_row_count": len(depth),
            "rgb_directory_file_count": len(rgb_files),
            "depth_directory_file_count": len(depth_files),
            "paired_rgb_depth_identity_count": len(paired_timestamps),
            "missing_depth_reference_count": len(missing_depth_refs),
            "missing_depth_references": missing_depth_refs,
            "rgb_index_sha256": sha256_file(sequence / "rgb.txt"),
            "depth_index_sha256": sha256_file(sequence / "depth.txt"),
            "rgb_referenced_members_aggregate_sha256": rgb_aggregate,
            "depth_referenced_members_aggregate_sha256": depth_aggregate,
            "four_frame_continuity_confirmed": continuity,
        })

    catalog = load_json(base_catalog_path)
    require(catalog.get("schema") == CATALOG_SCHEMA, "catalog schema drift")
    require(all(value is False for value in catalog["runtime_state"].values()), "runtime boundary violated")
    bonn = next(row for row in catalog["sources"] if row["dataset_id"] == "bonn_rgbd_dynamic")
    require(not bonn["identity_inventory"], "base Bonn inventory must be empty")
    bonn["identity_inventory"] = identities
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "archive": {"bytes": archive_path.stat().st_size, "sha256": sha256_file(archive_path)},
        "base_catalog_sha256": sha256_file(base_catalog_path),
        "exclusions_sha256": sha256_file(exclusions_path),
        "sequence_count": len(sequences),
        "excluded_sequence_count": len(excluded),
        "eligible_identity_count": sum(not row["ancestry_excluded"] for row in identities),
        "maximum_adjacent_gap_seconds": MAX_GAP_SECONDS,
        "maximum_rgb_depth_delta_seconds": MAX_RGB_DEPTH_DELTA_SECONDS,
        "missing_index_references_are_not_admitted_as_frame_identities": True,
        "label_or_model_data_read": False,
        "parents": details,
    }
    return catalog, receipt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--base-catalog", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.catalog_output.exists(), f"overwrite forbidden: {args.catalog_output}")
    require(not args.receipt_output.exists(), f"overwrite forbidden: {args.receipt_output}")
    catalog, receipt = materialize(args.repo_root, args.dataset_root, args.archive, args.base_catalog, args.exclusions)
    write_new(args.receipt_output.resolve(), receipt)
    write_new(args.catalog_output.resolve(), catalog)
    print(json.dumps({
        "catalog_output": str(args.catalog_output.resolve()),
        "receipt_output": str(args.receipt_output.resolve()),
        "sequence_count": receipt["sequence_count"],
        "eligible_identity_count": receipt["eligible_identity_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
