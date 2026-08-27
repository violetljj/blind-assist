#!/usr/bin/env python3
"""Merge completed R1C-G1 collection shards without changing sample selection."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def merge(role_root: Path, manifest: Path, role: str, shard_count: int) -> dict[str, Any]:
    manifest_hash = sha256_file(manifest)
    manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    shards = []
    for index in range(shard_count):
        path = role_root / f"shard-{index:02d}-of-{shard_count:02d}" / "collection.json"
        if not path.exists():
            raise ValueError(f"Missing R1C-G1 shard: {path}")
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row.get("schema") != "blindassist_grail_r1c_g1_collection_v1"
                or row["manifest_sha256"] != manifest_hash or row["role"] != role
                or row["shard_index"] != index or row["shard_count"] != shard_count):
            raise ValueError(f"R1C-G1 shard identity mismatch: {path}")
        shards.append(row)
    views = [view for shard in shards for view in shard["views"]]
    scans = [scan for shard in shards for scan in shard["scans"]]
    samples = [sample for shard in shards for sample in shard["samples"]]
    receipts = [receipt for shard in shards for receipt in shard["scene_receipts"]]
    sample_range = manifest_value["collection"][f"{role}_sample_range"]
    if len(samples) < int(sample_range[0]):
        raise RuntimeError(
            f"R1C-G1_NOT_EVALUABLE_SAMPLE_QUOTA role={role} samples={len(samples)}/{sample_range[0]}"
        )
    if len(samples) > int(sample_range[1]):
        samples = sorted(
            samples,
            key=lambda row: hashlib.sha256(row["sample_id"].encode("utf-8")).hexdigest(),
        )[:int(sample_range[1])]
    view_ids = [row["view_id"] for row in views]
    scan_ids = [row["scan_id"] for row in scans]
    sample_ids = [row["sample_id"] for row in samples]
    if (len(view_ids) != len(set(view_ids)) or len(scan_ids) != len(set(scan_ids))
            or len(sample_ids) != len(set(sample_ids))):
        raise ValueError("R1C-G1 merged IDs are not unique")
    known = set(view_ids)
    for sample in samples:
        needed = set(sample["reference_view_ids"]) | {sample["query_view_id"]}
        if not needed <= known:
            raise ValueError(f"R1C-G1 sample references missing views: {sample['sample_id']}")
        if sample["query_view_id"] in sample["reference_view_ids"]:
            raise ValueError(f"R1C-G1 query leaked into reference triplet: {sample['sample_id']}")
        if len(sample["reference_view_ids"]) != 3 or len(set(sample["reference_view_ids"])) != 3:
            raise ValueError(f"R1C-G1 reference triplet is not three distinct views: {sample['sample_id']}")
    for scan in scans:
        geometry = scan["acquisition_geometry_audit_only"]
        if not (-0.45 <= geometry["left_lateral_m"] <= -0.20):
            raise ValueError(f"R1C-G1 left baseline out of bounds: {scan['scan_id']}")
        if not (0.20 <= geometry["right_lateral_m"] <= 0.45):
            raise ValueError(f"R1C-G1 right baseline out of bounds: {scan['scan_id']}")
        if abs(geometry["left_longitudinal_m"]) > 0.20 or abs(geometry["right_longitudinal_m"]) > 0.20:
            raise ValueError(f"R1C-G1 longitudinal drift out of bounds: {scan['scan_id']}")
    result = {
        "schema": "blindassist_grail_r1c_g1_collection_v1",
        "manifest_sha256": manifest_hash,
        "dataset_sha256": shards[0]["dataset_sha256"],
        "role": role, "shard_count": shard_count, "houses": len(receipts),
        "views": views, "scans": scans, "samples": samples, "scene_receipts": receipts,
        "summary": {
            "views": len(views), "samples": len(samples),
            "discriminative_samples": sum(len(row["valid_slot_modes"]) == 1 for row in samples),
            "ambiguous_samples": sum(len(row["valid_slot_modes"]) > 1 for row in samples),
            "flip_only_samples": sum(row["valid_slot_modes"] == ["FLIP"] for row in samples),
            "preserve_only_samples": sum(row["valid_slot_modes"] == ["PRESERVE"] for row in samples),
            "drawer_samples": sum(row["object_type"] == "Drawer" for row in samples),
            "doorway_samples": sum(row["object_type"] == "Doorway" for row in samples),
            "runtime_timeouts": sum(row["runtime_timeout"] for row in receipts),
        },
        "leakage_audit": shards[0]["leakage_audit"],
        "shard_collection_sha256": [
            sha256_file(role_root / f"shard-{index:02d}-of-{shard_count:02d}" / "collection.json")
            for index in range(shard_count)
        ],
    }
    _atomic_json(role_root / "collection.json", result)
    print(json.dumps(result["summary"], indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation"), required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    merge(args.role_root, args.manifest, args.role, args.shard_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
