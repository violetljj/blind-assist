#!/usr/bin/env python3
"""Deterministically merge complete R1C-L collection shards without rerendering."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any

from grail_procthor_native_m0 import sha256_file


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rank(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _materialize_asset(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise ValueError(f"R1C-L conflicting merged asset: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def merge(manifest_path: Path, dataset: Path, role: str, shard_root: Path,
          shard_count: int, output: Path, allow_under_minimum: bool = False) -> dict[str, Any]:
    if shard_count < 1:
        raise ValueError("R1C-L shard count must be positive")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha256 = sha256_file(manifest_path)
    dataset_sha256 = sha256_file(dataset)
    dataset_key = "test_sha256" if role == "final_test" else "train_sha256"
    if dataset_sha256 != manifest["source"][dataset_key]:
        raise ValueError("R1C-L merge dataset/manifest identity mismatch")

    views: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    for shard_index in range(shard_count):
        shard_role_root = shard_root / f"shard-{shard_index:02d}" / role
        shard_path = shard_role_root / "collection.json"
        row = json.loads(shard_path.read_text(encoding="utf-8"))
        expected_houses = {
            int(item["house_index"]) for item in manifest["rosters"][role][shard_index::shard_count]
        }
        actual_houses = {int(item["house_index"]) for item in row["scene_receipts"]}
        if (row.get("manifest_sha256") != manifest_sha256
                or row.get("dataset_sha256") != dataset_sha256
                or row.get("role") != role
                or row.get("shard_index") != shard_index
                or row.get("shard_count") != shard_count
                or actual_houses != expected_houses):
            raise ValueError(f"R1C-L shard {shard_index} identity or coverage mismatch")
        for view in row["views"]:
            for key in ("rgb", "owner_union_mask", "sibling_centroid_mask"):
                _materialize_asset(shard_role_root / view[key], output / role / view[key])
        views.extend(row["views"])
        pairs.extend(row["pairs"])
        receipts.extend(row["scene_receipts"])

    roster_order = {int(row["house_index"]): index for index, row in enumerate(manifest["rosters"][role])}
    receipts.sort(key=lambda row: roster_order[int(row["house_index"])])
    views.sort(key=lambda row: (roster_order[int(row["house_index"])], row["view_id"]))
    if len({row["view_id"] for row in views}) != len(views):
        raise ValueError("R1C-L duplicate view across shards")
    if len({row["pair_id"] for row in pairs}) != len(pairs):
        raise ValueError("R1C-L duplicate pair across shards")
    maximum = (manifest["collection"]["validation_pair_range" if role == "validation" else "train_pair_range"][1]
               if role != "final_test" else len(pairs))
    pairs = sorted(pairs, key=lambda row: _rank(row["pair_id"]))[:maximum]
    minimum = (manifest["collection"]["validation_pair_range" if role == "validation" else "train_pair_range"][0]
               if role != "final_test" else 0)
    if len(pairs) < minimum and not allow_under_minimum:
        raise RuntimeError(f"R1C-L_NOT_EVALUABLE_PAIR_QUOTA role={role} pairs={len(pairs)}/{minimum}")
    view_ids = {row["view_id"] for row in views}
    if any(row["reference_view_id"] not in view_ids or row["query_view_id"] not in view_ids for row in pairs):
        raise ValueError("R1C-L merged pair references a missing view")

    summary = {
        "views": len(views), "pairs": len(pairs),
        "drawer_pairs": sum(row["object_type"] == "Drawer" for row in pairs),
        "doorway_pairs": sum(row["object_type"] == "Doorway" for row in pairs),
        "preserve_pairs": sum("PRESERVE" in row["valid_slot_modes"] for row in pairs),
        "flip_pairs": sum("FLIP" in row["valid_slot_modes"] for row in pairs),
    }
    result = {
        "schema": "blindassist_grail_r1c_l_collection_v1", "role": role,
        "manifest_sha256": manifest_sha256, "dataset_sha256": dataset_sha256,
        "shard_count": shard_count, "houses": len(receipts), "views": views, "pairs": pairs,
        "scene_receipts": receipts, "summary": summary,
        "minimum_pairs": minimum, "pair_quota_met": len(pairs) >= minimum,
        "development_under_minimum": len(pairs) < minimum and allow_under_minimum,
    }
    _atomic_json(output / role / "collection.json", result)
    _atomic_json(output / role / "progress.json", {
        "phase": "collection_merge", "completed_units": shard_count, "total_units": shard_count,
        "eta_seconds": 0.0, "last_progress_at": _now(), "status": "complete", **summary,
    })
    print(json.dumps(summary, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--role", choices=("train", "validation", "final_test"), required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-under-minimum", action="store_true")
    args = parser.parse_args()
    merge(args.manifest, args.dataset, args.role, args.shard_root, args.shard_count, args.output,
          args.allow_under_minimum)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
