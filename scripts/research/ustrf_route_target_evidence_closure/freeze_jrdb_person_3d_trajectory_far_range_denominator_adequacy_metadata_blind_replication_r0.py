"""Freeze unseen JRDB sequence windows from metadata only.

This process must run before any candidate label or PCD payload is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any


SCHEMA = (
    "blindassist_ustrf_jrdb_person_3d_trajectory_far_range_denominator_"
    "adequacy_metadata_blind_replication_r0_sequence_freeze"
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: Any) -> None:
    require(not path.exists(), f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(value))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_bound(repo: Path, binding: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = repo / binding["path"]
    require(path.is_file(), f"metadata input missing: {path}")
    require(path.stat().st_size == int(binding["bytes"]), f"metadata bytes drift: {path}")
    require(sha256_file(path) == binding["sha256"], f"metadata hash drift: {path}")
    return path, json.loads(path.read_text(encoding="utf-8"))


def sequence_from_bag(name: str) -> str | None:
    if not name.startswith("rosbags/") or not name.endswith(".bag"):
        return None
    return Path(name).stem


def frame_stem(url: str) -> str:
    return Path(url).stem


def sensor_row(row: dict[str, Any], name: str) -> dict[str, Any]:
    matches = [value for value in row["pointclouds"] if value["name"] == name]
    require(len(matches) == 1, f"timestamp sensor multiplicity: {name}")
    return matches[0]


def inventory_entry(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": value["name"],
        "compression_method": value["compression_method"],
        "flags": value["flags"],
        "crc32": value["crc32"],
        "compressed_size": value["compressed_size"],
        "uncompressed_size": value["uncompressed_size"],
        "local_header_offset": value["local_header_offset"],
    }


def build_freeze(repo: Path, config_path: Path) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    boundary = config["freeze_boundary"]
    count = int(boundary["frame_count"])
    timestamps_path = repo / config["metadata_inputs"]["timestamps"]["path"]
    require(timestamps_path.is_file(), "timestamps archive missing")
    require(timestamps_path.stat().st_size == config["metadata_inputs"]["timestamps"]["bytes"], "timestamps bytes drift")
    require(sha256_file(timestamps_path) == config["metadata_inputs"]["timestamps"]["sha256"], "timestamps hash drift")
    label_path, labels = load_bound(repo, config["metadata_inputs"]["labels_inventory"])
    point_path, pointclouds = load_bound(repo, config["metadata_inputs"]["pointclouds_inventory"])
    bag_path, rosbags = load_bound(repo, config["metadata_inputs"]["rosbags_inventory"])
    for name, inventory in (
        ("labels", labels),
        ("pointclouds", pointclouds),
        ("rosbags", rosbags),
    ):
        require(inventory["archive_payload_bytes_downloaded"] == 0, f"{name} payload was downloaded")
        require(inventory["sequence_content_decoded"] is False, f"{name} sequence content decoded")
        require(inventory["filename_inventory_decoded"] is True, f"{name} names absent")
        require(inventory["range_supported"] is True, f"{name} range unsupported")

    label_entries = {row["name"]: row for row in labels["entries"] if not row["is_directory"]}
    point_entries = {row["name"]: row for row in pointclouds["entries"] if not row["is_directory"]}
    bag_entries = {
        sequence: row
        for row in rosbags["entries"]
        if not row["is_directory"]
        for sequence in [sequence_from_bag(row["name"])]
        if sequence is not None
    }
    excluded = set(boundary["previously_seen_sequences"])
    seed = boundary["selection_seed"]
    candidates: list[dict[str, Any]] = []
    with zipfile.ZipFile(timestamps_path) as bundle:
        timestamp_members = sorted(
            name
            for name in bundle.namelist()
            if name.startswith("timestamps/") and name.endswith("/frames_pc.json")
        )
        for member in timestamp_members:
            sequence = member.split("/")[1]
            if sequence in excluded or sequence not in bag_entries:
                continue
            label_3d = f"labels/labels_3d/{sequence}.json"
            label_2d = f"labels/labels_2d_stitched/{sequence}.json"
            if label_3d not in label_entries or label_2d not in label_entries:
                continue
            document = json.loads(bundle.read(member))
            rows = document["data"]
            if len(rows) < count:
                continue
            rank_hash = hashlib.sha256(f"{seed}|sequence|{sequence}".encode()).hexdigest()
            window_hash = hashlib.sha256(f"{seed}|window|{sequence}".encode()).hexdigest()
            start = int(window_hash, 16) % (len(rows) - count + 1)
            window = rows[start : start + count]
            upper = [sensor_row(row, "upper_velodyne") for row in window]
            lower = [sensor_row(row, "lower_velodyne") for row in window]
            upper_stems = [frame_stem(row["url"]) for row in upper]
            lower_stems = [frame_stem(row["url"]) for row in lower]
            if upper_stems != lower_stems:
                continue
            numeric = [int(stem) for stem in upper_stems]
            if any(right != left + 1 for left, right in zip(numeric, numeric[1:])):
                continue
            all_times = []
            valid = True
            maximum_gap = 0.0
            for sensor_values in (upper, lower):
                times = [float(row["timestamp"]) for row in sensor_values]
                gaps = [right - left for left, right in zip(times, times[1:])]
                if not all(math.isfinite(value) for value in times) or not all(
                    0.0 < gap <= 0.2 for gap in gaps
                ):
                    valid = False
                    break
                maximum_gap = max(maximum_gap, max(gaps, default=0.0))
                all_times.extend(times)
            if not valid:
                continue
            required_pcd_names = [
                f"pointclouds/{sensor}/{sequence}/{stem}.pcd"
                for sensor in ("upper_velodyne", "lower_velodyne")
                for stem in upper_stems
            ]
            if any(name not in point_entries for name in required_pcd_names):
                continue
            timestamp_inventory = [
                {
                    "stem": stem,
                    "upper_timestamp": float(up["timestamp"]),
                    "lower_timestamp": float(low["timestamp"]),
                }
                for stem, up, low in zip(upper_stems, upper, lower)
            ]
            candidates.append(
                {
                    "sequence": sequence,
                    "rank_hash": rank_hash,
                    "window_hash": window_hash,
                    "row_count": len(rows),
                    "window_first_position": start,
                    "window_last_position": start + count - 1,
                    "frame_first_stem": upper_stems[0],
                    "frame_last_stem": upper_stems[-1],
                    "frame_count": count,
                    "maximum_adjacent_sensor_timestamp_gap_seconds": maximum_gap,
                    "timestamp_window_inventory_sha256": hashlib.sha256(
                        canonical_bytes(timestamp_inventory)
                    ).hexdigest(),
                    "label_members": [
                        inventory_entry(label_entries[label_2d]),
                        inventory_entry(label_entries[label_3d]),
                    ],
                    "pcd_member_name_inventory_sha256": hashlib.sha256(
                        canonical_bytes(required_pcd_names)
                    ).hexdigest(),
                    "rosbag_member": inventory_entry(bag_entries[sequence]),
                }
            )
    candidates.sort(key=lambda row: (row["rank_hash"], row["sequence"]))
    selected_count = int(boundary["selected_sequence_count"])
    require(len(candidates) >= selected_count, "insufficient metadata-eligible sequences")
    selected = candidates[:selected_count]
    amendment = config.get("prepayload_amendment")
    if amendment:
        initial_path = repo / config["outputs"]["initial_metadata_freeze"]
        require(initial_path.is_file(), "initial metadata freeze missing for amendment")
        initial = json.loads(initial_path.read_text(encoding="utf-8"))
        identity_fields = (
            "sequence",
            "window_first_position",
            "window_last_position",
            "frame_first_stem",
            "frame_last_stem",
            "frame_count",
        )
        require(
            [
                {key: row[key] for key in identity_fields}
                for row in initial["selected"]
            ]
            == [{key: row[key] for key in identity_fields} for row in selected],
            "prepayload amendment changed frozen sequence/window identity",
        )
    return {
        "schema": SCHEMA,
        "stage": config["stage"],
        "status": "FROZEN_BEFORE_CANDIDATE_LABEL_OR_PCD_PAYLOAD",
        "config_sha256": sha256_file(config_path),
        "blindness": {
            "selection_inputs": "TIMESTAMP_AND_ARCHIVE_MEMBER_METADATA_ONLY",
            "candidate_label_payload_read": False,
            "candidate_pcd_payload_read": False,
            "candidate_support_or_range_result_read": False,
            "replacement_after_freeze_allowed": False,
        },
        "metadata_bindings": {
            "timestamps_sha256": sha256_file(timestamps_path),
            "labels_inventory_sha256": sha256_file(label_path),
            "pointclouds_inventory_sha256": sha256_file(point_path),
            "rosbags_inventory_sha256": sha256_file(bag_path),
        },
        "selection_seed": seed,
        "prepayload_amendment": amendment,
        "excluded_sequences": sorted(excluded),
        "eligible_sequence_count": len(candidates),
        "selected_sequence_count": len(selected),
        "selected": selected,
        "denominator_gate_preregistered": config["denominator_adequacy"],
        "authority": config["authority"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    config_path = (repo / args.config).resolve() if not args.config.is_absolute() else args.config
    config = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = build_freeze(repo, config_path)
    output = repo / config["outputs"]["sequence_freeze"]
    atomic_write(output, freeze)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "eligible": freeze["eligible_sequence_count"],
                "selected": [
                    [row["sequence"], row["window_first_position"], row["window_last_position"]]
                    for row in freeze["selected"]
                ],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
