#!/usr/bin/env python3
"""Freeze and materialize two unseen cross-collection positive targets."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import l10_panolab_viviani_reference_panel as source
from l10_panolab_entrance_ray import projection_gate


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-cross-collection-positive-source-protocol-v1"


def exclusions(payloads: dict[str, dict[str, Any]]) -> tuple[set[int], set[str]]:
    way_ids = {int(row["target_way"]["id"]) for row in payloads["fresh_selection"]["episodes"]}
    way_ids.update(int(row["target"]["building"]["id"]) for row in payloads["active_source"]["episodes"])
    way_ids.update(int(row["target_way"]["id"]) for row in payloads["reference_portal_source"]["episodes"])
    way_ids.update(int(row["target_way"]["id"]) for row in payloads["node_credential_source"]["episodes"])
    way_ids.update(int(row["target_way"]["id"]) for row in payloads["negative_selection"]["episodes"])
    item_ids = {row["item_id"] for row in payloads["fresh_materialization"]["images"]}
    item_ids.update(row["item_id"] for row in payloads["viviani_selection"]["references"])
    item_ids.update(row["item_id"] for row in payloads["negative_materialization"]["images"])
    item_ids.update(
        episode[phase]["panorama"]["provider_item"]["id"]
        for episode in payloads["active_source"]["episodes"]
        for phase in ("start", "after")
    )
    item_ids.update(
        entry["item_id"]
        for episode in payloads["reference_portal_source"]["episodes"]
        for entry in (episode["reference"], episode["query"])
    )
    item_ids.update(
        view["item_id"]
        for episode in payloads["node_credential_source"]["episodes"]
        for view in episode["views"]
    )
    return way_ids, item_ids


def freeze(protocol_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    payloads = {
        key: source.load(source.verify(spec))
        for key, spec in protocol["inputs"].items()
        if key not in {"orientation_projection_protocol", "positive_calibration_result"}
    }
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    calibration = source.load(source.verify(protocol["inputs"]["positive_calibration_result"]))
    expected_score = min(float(row["appearance"]["truth_score"]) for row in calibration["rows"])
    expected_margin = min(float(row["appearance"]["truth_margin"]) for row in calibration["rows"])
    source.require(expected_score == float(protocol["accept_contract"]["minimum_top1_score"]), "SCORE_THRESHOLD_DRIFT")
    source.require(expected_margin == float(protocol["accept_contract"]["minimum_top1_margin"]), "MARGIN_THRESHOLD_DRIFT")
    excluded_ways, excluded_items = exclusions(payloads)
    contract = protocol["selection"]

    grouped: dict[int, dict[str, Any]] = {}
    for candidate in payloads["candidate_metadata"]["candidates"]:
        way_id = int(candidate["target_way"]["id"])
        name = str(candidate["target_way"].get("tags", {}).get("name") or "").strip()
        if way_id in excluded_ways or not name:
            continue
        entry = grouped.setdefault(
            way_id,
            {
                "source_city": candidate["source_city"],
                "target_way": candidate["target_way"],
                "main_entrance_node": candidate["main_entrance_node"],
                "items": {},
            },
        )
        for support in candidate["supports"]["direct"]:
            item_id = support["item_id"]
            if item_id in excluded_items:
                continue
            distance = float(support["first_intersection"]["distance_from_camera_m"])
            if not (
                float(contract["minimum_camera_to_entrance_m"])
                <= distance
                <= float(contract["maximum_camera_to_entrance_m"])
            ):
                continue
            item = candidate["items"][item_id]
            gate = projection_gate(item, orientation)
            if not gate["eligible"]:
                continue
            entry["items"][item_id] = {
                "item_id": item_id,
                "collection": item["collection"],
                "camera_to_entrance_distance_m": round(distance, 3),
                "provider_item": item,
                "projection_gate": gate,
            }

    targets = []
    for way_id, entry in grouped.items():
        collections: dict[str, list[dict[str, Any]]] = {}
        for item in entry["items"].values():
            collections.setdefault(item["collection"], []).append(item)
        if len(collections) < 2:
            continue
        representatives = {
            collection: min(
                rows,
                key=lambda row: (
                    abs(float(row["camera_to_entrance_distance_m"]) - float(contract["query_distance_anchor_m"])),
                    row["item_id"],
                ),
            )
            for collection, rows in collections.items()
        }
        pair = min(
            (
                (left, right)
                for index, left in enumerate(sorted(collections))
                for right in sorted(collections)[index + 1 :]
            ),
            key=lambda pair: (
                abs(
                    float(representatives[pair[0]]["camera_to_entrance_distance_m"])
                    - float(representatives[pair[1]]["camera_to_entrance_distance_m"])
                ),
                pair,
            ),
        )
        reference_collection, query_collection = pair
        remaining = list(collections[reference_collection])
        references = []
        for anchor in contract["reference_distance_anchors_m"]:
            chosen = min(
                remaining,
                key=lambda row: (
                    abs(float(row["camera_to_entrance_distance_m"]) - float(anchor)),
                    row["item_id"],
                ),
            )
            references.append({**chosen, "selection_anchor_m": float(anchor)})
            remaining = [row for row in remaining if row["item_id"] != chosen["item_id"]]
        query = representatives[query_collection]
        targets.append(
            {
                "source_city": entry["source_city"],
                "target_way": entry["target_way"],
                "main_entrance_node": entry["main_entrance_node"],
                "reference_collection": reference_collection,
                "query_collection": query_collection,
                "reference_query_collection_disjoint": reference_collection != query_collection,
                "references": references,
                "query": query,
            }
        )
    targets.sort(key=lambda row: (int(row["target_way"]["id"]), row["query"]["item_id"]))
    source.require(len(targets) == int(contract["panel_size"]), f"UNEXPECTED_CROSS_COLLECTION_TARGET_COUNT:{len(targets)}")
    all_items = [
        item["item_id"]
        for target in targets
        for item in [*target["references"], target["query"]]
    ]
    source.require(len(all_items) == len(set(all_items)), "SELECTED_ITEMS_NOT_UNIQUE")
    source.require(not (set(all_items) & excluded_items), "SELECTED_ITEM_PRIOR_OVERLAP")

    receipt = {
        "schema": "blindassist-l10-panolab-cross-collection-positive-selection-v1",
        "status": "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_MODEL_CALL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": source.sha256(protocol_path),
        "candidate_metadata_sha256": protocol["inputs"]["candidate_metadata"]["sha256"],
        "excluded_target_way_count": len(excluded_ways),
        "excluded_item_count": len(excluded_items),
        "selection_rule": contract,
        "accept_contract_frozen_before_pixels": protocol["accept_contract"],
        "selected_pixel_views_before_freeze": 0,
        "selected_human_pixel_reviews_before_freeze": 0,
        "selected_model_calls_before_freeze": 0,
        "episodes": [
            {"episode_id": f"FP{index:02d}", **target}
            for index, target in enumerate(targets, start=1)
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selection": str(output_path),
                "episodes": [
                    {
                        "episode_id": row["episode_id"],
                        "way_id": row["target_way"]["id"],
                        "name": row["target_way"]["tags"]["name"],
                        "reference_items": [item["item_id"] for item in row["references"]],
                        "reference_collection": row["reference_collection"],
                        "query_item": row["query"]["item_id"],
                        "query_collection": row["query_collection"],
                    }
                    for row in receipt["episodes"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def materialize(protocol_path: Path, selection_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    selection = source.load(selection_path)
    source.require(selection["protocol_sha256"] == source.sha256(protocol_path), "SELECTION_PROTOCOL_HASH_MISMATCH")
    source.require(selection["status"] == "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_MODEL_CALL", "SELECTION_NOT_PIXEL_UNSEEN")
    source.require(selection["selected_pixel_views_before_freeze"] == 0, "PIXELS_VIEWED_BEFORE_FREEZE")
    source.require(selection["selected_human_pixel_reviews_before_freeze"] == 0, "PIXELS_REVIEWED_BEFORE_FREEZE")
    source.require(selection["selected_model_calls_before_freeze"] == 0, "MODEL_CALLED_BEFORE_FREEZE")
    output_root = source.resolve(protocol["materialization"]["output_root"])
    source.require(not output_root.exists(), f"MATERIALIZATION_ROOT_ALREADY_EXISTS:{output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    images = []
    partials: list[Path] = []
    try:
        for episode in selection["episodes"]:
            for role, rows in (("reference", episode["references"]), ("query", [episode["query"]])):
                for role_index, item in enumerate(rows, start=1):
                    target = image_root / f"{item['item_id']}.jpg"
                    partial = target.with_suffix(".jpg.part")
                    partials.append(partial)
                    url = item["provider_item"]["assets"]["hd"]["href"]
                    request = urllib.request.Request(url, headers={"User-Agent": "BlindAssist-L10-Development/1"})
                    with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                        while block := response.read(1024 * 1024):
                            stream.write(block)
                    os.replace(partial, target)
                    with Image.open(target) as image:
                        size = list(image.size)
                    source.require(size == [5760, 2880], f"UNEXPECTED_IMAGE_SIZE:{target}:{size}")
                    images.append(
                        {
                            "episode_id": episode["episode_id"],
                            "role": role,
                            "role_index": role_index,
                            "item_id": item["item_id"],
                            "collection": item["collection"],
                            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": source.sha256(target),
                            "bytes": target.stat().st_size,
                            "image_size": size,
                            "url": url,
                        }
                    )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    manifest = {
        "schema": "blindassist-l10-panolab-cross-collection-positive-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": source.sha256(selection_path),
        "pixel_views_before_frozen_selection": 0,
        "human_pixel_reviews_before_frozen_selection": 0,
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
