#!/usr/bin/env python3
"""Freeze and materialize a pixel-unseen geospatial open-set negative panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import l10_panolab_viviani_reference_panel as source
from l10_panolab_entrance_ray import projection_gate


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-open-set-negative-source-protocol-v1"


def haversine_m(a: list[float], b: list[float]) -> float:
    lon1, lat1 = (math.radians(float(value)) for value in a)
    lon2, lat2 = (math.radians(float(value)) for value in b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 6371000.0 * 2.0 * math.atan2(math.sqrt(value), math.sqrt(1.0 - value))


def item_ids_from_inputs(payloads: dict[str, dict[str, Any]]) -> set[str]:
    ids = {
        row["item_id"]
        for row in payloads["fresh_materialization"]["images"]
    }
    ids.update(row["item_id"] for row in payloads["viviani_selection"]["references"])
    ids.update(
        episode[phase]["panorama"]["provider_item"]["id"]
        for episode in payloads["active_source"]["episodes"]
        for phase in ("start", "after")
    )
    ids.update(
        entry["item_id"]
        for episode in payloads["reference_portal_source"]["episodes"]
        for entry in (episode["reference"], episode["query"])
    )
    ids.update(
        view["item_id"]
        for episode in payloads["node_credential_source"]["episodes"]
        for view in episode["views"]
    )
    return ids


def way_ids_from_inputs(payloads: dict[str, dict[str, Any]]) -> set[int]:
    ids = {
        int(episode["target_way"]["id"])
        for episode in payloads["fresh_selection"]["episodes"]
    }
    ids.update(int(episode["target"]["building"]["id"]) for episode in payloads["active_source"]["episodes"])
    ids.update(int(episode["target_way"]["id"]) for episode in payloads["reference_portal_source"]["episodes"])
    ids.update(int(episode["target_way"]["id"]) for episode in payloads["node_credential_source"]["episodes"])
    return ids


def freeze(protocol_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    payloads = {
        key: source.load(source.verify(spec))
        for key, spec in protocol["inputs"].items()
        if key != "orientation_projection_protocol" and key != "positive_calibration_result"
    }
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    calibration = source.load(source.verify(protocol["inputs"]["positive_calibration_result"]))
    candidate_metadata = payloads["candidate_metadata"]
    roster_nodes = [
        [float(row["main_entrance_node"]["lon"]), float(row["main_entrance_node"]["lat"])]
        for row in payloads["fresh_selection"]["episodes"]
    ]
    source.require(len(roster_nodes) == 4, "POSITIVE_ROSTER_NOT_FOUR")
    expected_min_score = min(float(row["appearance"]["truth_score"]) for row in calibration["rows"])
    expected_min_margin = min(float(row["appearance"]["truth_margin"]) for row in calibration["rows"])
    source.require(
        abs(expected_min_score - float(protocol["open_set_contract"]["minimum_top1_score"])) < 1e-12,
        "CALIBRATED_SCORE_THRESHOLD_MISMATCH",
    )
    source.require(
        abs(expected_min_margin - float(protocol["open_set_contract"]["minimum_top1_margin"])) < 1e-12,
        "CALIBRATED_MARGIN_THRESHOLD_MISMATCH",
    )

    excluded_items = item_ids_from_inputs(payloads)
    excluded_ways = way_ids_from_inputs(payloads)
    contract = protocol["selection"]
    by_way: dict[int, list[dict[str, Any]]] = {}
    way_meta: dict[int, dict[str, Any]] = {}
    for candidate in candidate_metadata["candidates"]:
        way_id = int(candidate["target_way"]["id"])
        name = str(candidate["target_way"].get("tags", {}).get("name") or "").strip()
        if way_id in excluded_ways or not name:
            continue
        for support in candidate["supports"]["direct"]:
            item_id = support["item_id"]
            if item_id in excluded_items:
                continue
            distance = float(support["first_intersection"]["distance_from_camera_m"])
            if not (
                float(contract["minimum_camera_to_target_entrance_m"])
                <= distance
                <= float(contract["maximum_camera_to_target_entrance_m"])
            ):
                continue
            item = candidate["items"][item_id]
            gate = projection_gate(item, orientation)
            if not gate["eligible"]:
                continue
            camera = [float(value) for value in item["geometry"]["coordinates"]]
            roster_distances = [haversine_m(camera, node) for node in roster_nodes]
            minimum_roster_distance = min(roster_distances)
            if minimum_roster_distance < float(contract["minimum_camera_to_any_positive_roster_entrance_m"]):
                continue
            by_way.setdefault(way_id, []).append(
                {
                    "item_id": item_id,
                    "collection": item["collection"],
                    "camera_to_target_entrance_distance_m": round(distance, 3),
                    "minimum_camera_to_positive_roster_entrance_m": round(minimum_roster_distance, 3),
                    "camera_to_positive_roster_entrances_m": [round(value, 3) for value in roster_distances],
                    "provider_item": item,
                    "projection_gate": gate,
                }
            )
            way_meta[way_id] = candidate
    target_rows = []
    for way_id, items in by_way.items():
        chosen = min(
            items,
            key=lambda row: (
                abs(float(row["camera_to_target_entrance_distance_m"]) - float(contract["target_camera_distance_m"])),
                row["item_id"],
            ),
        )
        candidate = way_meta[way_id]
        target_rows.append(
            {
                "source_city": candidate["source_city"],
                "target_way": candidate["target_way"],
                "main_entrance_node": candidate["main_entrance_node"],
                **chosen,
            }
        )
    source.require(len(target_rows) >= int(contract["panel_size"]), "INSUFFICIENT_ELIGIBLE_NEGATIVE_TARGETS")
    cities = sorted({row["source_city"] for row in target_rows})
    city_rows = {
        city: sorted(
            [row for row in target_rows if row["source_city"] == city],
            key=lambda row: (int(row["target_way"]["id"]), row["item_id"]),
        )
        for city in cities
    }
    selected = []
    used_collections: set[str] = set()
    round_index = 0
    while len(selected) < int(contract["panel_size"]):
        added = False
        for city in cities:
            if round_index >= len(city_rows[city]):
                continue
            for row in city_rows[city][round_index:]:
                if row["collection"] in used_collections:
                    continue
                selected.append(row)
                used_collections.add(row["collection"])
                added = True
                break
            if len(selected) == int(contract["panel_size"]):
                break
        source.require(added, "CITY_ROUND_ROBIN_STALLED")
        round_index += 1
    source.require(len({int(row["target_way"]["id"]) for row in selected}) == len(selected), "NEGATIVE_WAYS_NOT_UNIQUE")
    source.require(len({row["item_id"] for row in selected}) == len(selected), "NEGATIVE_ITEMS_NOT_UNIQUE")
    source.require(len({row["collection"] for row in selected}) == len(selected), "NEGATIVE_COLLECTIONS_NOT_UNIQUE")
    source.require(len({row["source_city"] for row in selected}) >= 3, "NEGATIVE_CITY_DIVERSITY_NOT_THREE")

    receipt = {
        "schema": "blindassist-l10-panolab-open-set-negative-selection-v1",
        "status": "FROZEN_BEFORE_SELECTED_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_MODEL_CALL",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": source.sha256(protocol_path),
        "candidate_metadata_sha256": protocol["inputs"]["candidate_metadata"]["sha256"],
        "excluded_target_way_count": len(excluded_ways),
        "excluded_item_count": len(excluded_items),
        "eligible_distinct_target_way_count": len(target_rows),
        "selection_rule": contract,
        "open_set_contract_frozen_before_pixels": protocol["open_set_contract"],
        "selected_pixel_views_before_freeze": 0,
        "selected_human_pixel_reviews_before_freeze": 0,
        "selected_model_calls_before_freeze": 0,
        "episodes": [
            {"episode_id": f"ON{index:02d}", **row}
            for index, row in enumerate(selected, start=1)
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
                        "city": row["source_city"],
                        "way_id": row["target_way"]["id"],
                        "name": row["target_way"]["tags"]["name"],
                        "item_id": row["item_id"],
                        "minimum_roster_distance_m": row["minimum_camera_to_positive_roster_entrance_m"],
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
            target = image_root / f"{episode['item_id']}.jpg"
            partial = target.with_suffix(".jpg.part")
            partials.append(partial)
            url = episode["provider_item"]["assets"]["hd"]["href"]
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
                    "item_id": episode["item_id"],
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
        "schema": "blindassist-l10-panolab-open-set-negative-materialization-v1",
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
