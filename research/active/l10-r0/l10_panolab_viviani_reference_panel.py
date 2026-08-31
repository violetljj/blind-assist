#!/usr/bin/env python3
"""Freeze and materialize an unseen two-scale reference bank for Le Viviani."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from l10_panolab_entrance_ray import projection_gate


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-viviani-reference-source-protocol-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(spec: dict[str, Any]) -> Path:
    path = resolve(spec["path"])
    require(path.is_file(), f"MISSING_INPUT:{path}")
    require(sha256(path) == spec["sha256"], f"HASH_MISMATCH:{path}")
    if "bytes" in spec:
        require(path.stat().st_size == int(spec["bytes"]), f"BYTE_COUNT_MISMATCH:{path}")
    return path


def freeze(protocol_path: Path, output_path: Path) -> None:
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    candidates = load(verify(protocol["inputs"]["candidate_metadata"]))
    fresh_selection = load(verify(protocol["inputs"]["fresh_selection"]))
    fresh_materialization = load(verify(protocol["inputs"]["fresh_materialization"]))
    orientation = load(verify(protocol["inputs"]["orientation_projection_protocol"]))
    require(fresh_materialization["selection_sha256"] == protocol["inputs"]["fresh_selection"]["sha256"], "FRESH_SELECTION_LINK_MISMATCH")

    way_id = int(protocol["target"]["way_id"])
    matches = [row for row in candidates["candidates"] if int(row["target_way"]["id"]) == way_id]
    require(len(matches) == 1, f"TARGET_WAY_NOT_UNIQUE:{way_id}:{len(matches)}")
    candidate = matches[0]
    query_ids = {row["item_id"] for row in fresh_materialization["images"]}
    current_episode = next(row for row in fresh_selection["episodes"] if int(row["target_way"]["id"]) == way_id)
    require({current_episode["start_item"]["id"], current_episode["after_item"]["id"]} <= query_ids, "TARGET_QUERY_RECEIPTS_MISSING")

    direct_rows = []
    for support in candidate["supports"]["direct"]:
        item_id = support["item_id"]
        if item_id in query_ids:
            continue
        item = candidate["items"][item_id]
        gate = projection_gate(item, orientation)
        if not gate["eligible"]:
            continue
        direct_rows.append(
            {
                "item_id": item_id,
                "collection": item["collection"],
                "camera_to_entrance_distance_m": round(float(support["first_intersection"]["distance_from_camera_m"]), 3),
                "provider_item": item,
                "projection_gate": gate,
            }
        )
    require(len(direct_rows) >= 2, "FEWER_THAN_TWO_ELIGIBLE_REFERENCE_ITEMS")
    selected = []
    remaining = list(direct_rows)
    for anchor in protocol["selection"]["distance_anchors_m"]:
        chosen = min(
            remaining,
            key=lambda row: (
                abs(float(row["camera_to_entrance_distance_m"]) - float(anchor)),
                row["item_id"],
            ),
        )
        chosen = {**chosen, "selection_anchor_m": float(anchor)}
        selected.append(chosen)
        remaining = [row for row in remaining if row["item_id"] != chosen["item_id"]]
    require(len({row["item_id"] for row in selected}) == 2, "REFERENCE_ITEMS_NOT_UNIQUE")
    require(not ({row["item_id"] for row in selected} & query_ids), "QUERY_REFERENCE_ITEM_OVERLAP")

    receipt = {
        "schema": "blindassist-l10-panolab-viviani-reference-selection-v1",
        "status": "FROZEN_BEFORE_SELECTED_REFERENCE_PIXEL_DOWNLOAD_OR_MODEL_CALL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "candidate_metadata_sha256": protocol["inputs"]["candidate_metadata"]["sha256"],
        "target": {
            "way": candidate["target_way"],
            "main_entrance_node": candidate["main_entrance_node"],
            "source_city": candidate["source_city"],
            "query_sequence_id": current_episode["sequence_id"],
            "query_item_ids": sorted([current_episode["start_item"]["id"], current_episode["after_item"]["id"]]),
        },
        "selection_rule": protocol["selection"],
        "eligible_reference_count": len(direct_rows),
        "selected_reference_count": len(selected),
        "selected_pixel_views_before_freeze": 0,
        "selected_model_calls_before_freeze": 0,
        "repo_json_exact_item_occurrences_before_freeze": 0,
        "local_exact_item_file_matches_before_freeze": 0,
        "references": selected,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selection": str(output_path), "references": [{"item_id": row["item_id"], "distance_m": row["camera_to_entrance_distance_m"]} for row in selected]}, ensure_ascii=False, indent=2))


def materialize(protocol_path: Path, selection_path: Path, output_path: Path) -> None:
    require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = load(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    require(sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    selection = load(selection_path)
    require(selection["protocol_sha256"] == sha256(protocol_path), "SELECTION_PROTOCOL_HASH_MISMATCH")
    require(selection["status"] == "FROZEN_BEFORE_SELECTED_REFERENCE_PIXEL_DOWNLOAD_OR_MODEL_CALL", "SELECTION_NOT_PIXEL_UNSEEN")
    require(selection["selected_pixel_views_before_freeze"] == 0, "PIXELS_VIEWED_BEFORE_FREEZE")
    require(selection["selected_model_calls_before_freeze"] == 0, "MODEL_CALLED_BEFORE_FREEZE")

    output_root = resolve(protocol["materialization"]["output_root"])
    require(not output_root.exists(), f"MATERIALIZATION_ROOT_ALREADY_EXISTS:{output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    images = []
    partials: list[Path] = []
    try:
        for reference in selection["references"]:
            item = reference["provider_item"]
            target = image_root / f"{reference['item_id']}.jpg"
            partial = target.with_suffix(".jpg.part")
            partials.append(partial)
            request = urllib.request.Request(item["assets"]["hd"]["href"], headers={"User-Agent": "BlindAssist-L10-Development/1"})
            with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
            os.replace(partial, target)
            with Image.open(target) as image:
                size = list(image.size)
            require(size == [5760, 2880], f"UNEXPECTED_IMAGE_SIZE:{target}:{size}")
            images.append(
                {
                    "item_id": reference["item_id"],
                    "collection": reference["collection"],
                    "camera_to_entrance_distance_m": reference["camera_to_entrance_distance_m"],
                    "selection_anchor_m": reference["selection_anchor_m"],
                    "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                    "sha256": sha256(target),
                    "bytes": target.stat().st_size,
                    "image_size": size,
                    "url": item["assets"]["hd"]["href"],
                }
            )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    manifest = {
        "schema": "blindassist-l10-panolab-viviani-reference-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": sha256(selection_path),
        "pixel_views_before_frozen_selection": 0,
        "model_calls_before_frozen_selection": 0,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "images": len(images), "bytes": sum(row["bytes"] for row in images)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--protocol", type=Path, required=True)
    materialize_parser.add_argument("--selection", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.protocol.resolve(), args.output.resolve())
    else:
        materialize(args.protocol.resolve(), args.selection.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
