#!/usr/bin/env python3
"""Freeze reciprocal query neighbours and materialize a three-frame temporal panel."""

from __future__ import annotations

import argparse
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
PROTOCOL_SCHEMA = "blindassist-l10-panolab-temporal-query-source-protocol-v1"


def haversine_m(a: list[float], b: list[float]) -> float:
    radius = 6_371_000.0
    lon1, lat1 = map(math.radians, a)
    lon2, lat2 = map(math.radians, b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(value))


def linked_item(item: dict[str, Any], relation: str) -> dict[str, Any]:
    matches = [row for row in item["links"] if row.get("rel") == relation and row.get("id")]
    source.require(len(matches) == 1, f"LINK_NOT_UNIQUE:{item['id']}:{relation}:{len(matches)}")
    link = matches[0]
    request = urllib.request.Request(
        link["href"], headers={"User-Agent": "BlindAssist-L10-Development/1"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    source.require(payload["id"] == link["id"], f"LINK_ID_MISMATCH:{relation}")
    return payload


def reciprocal(item: dict[str, Any], relation: str, anchor_id: str) -> bool:
    inverse = "next" if relation == "prev" else "prev"
    return any(
        row.get("rel") == inverse and row.get("id") == anchor_id for row in item.get("links", [])
    )


def freeze(protocol_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(
        source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "EVALUATOR_HASH_MISMATCH",
    )
    positive_selection = source.load(source.verify(protocol["inputs"]["positive_selection"]))
    positive_materialization = source.load(
        source.verify(protocol["inputs"]["positive_materialization"])
    )
    positive_result = source.load(source.verify(protocol["inputs"]["positive_result"]))
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    source.require(
        positive_materialization["selection_sha256"]
        == protocol["inputs"]["positive_selection"]["sha256"],
        "POSITIVE_SELECTION_LINK_MISMATCH",
    )
    source.require(
        positive_result["decision"]
        == "L10_PANOLAB_CROSS_COLLECTION_FRESH_POSITIVE_ROUTER_DEVELOPMENT_GATE_NOT_MET",
        "EXPECTED_FAILURE_RESULT_NOT_FROZEN",
    )
    result_rows = {row["episode_id"]: row for row in positive_result["rows"]}
    source.require(
        result_rows["FP01"]["acceptance"]["route"] == "UNKNOWN_KEEP_SEARCHING",
        "EXPECTED_FP01_UNKNOWN_NOT_PRESENT",
    )
    anchor_receipts = {
        row["item_id"]: row for row in positive_materialization["images"] if row["role"] == "query"
    }
    episodes = []
    for episode in positive_selection["episodes"]:
        anchor = episode["query"]
        source.require(anchor["item_id"] in anchor_receipts, f"ANCHOR_RECEIPT_MISSING:{anchor['item_id']}")
        members = []
        for sequence_index, relation in enumerate(("prev", "anchor", "next")):
            if relation == "anchor":
                item = anchor["provider_item"]
                pixel_status = "CONSUMED_ANCHOR_REUSED"
                reciprocal_link = True
            else:
                item = linked_item(anchor["provider_item"], relation)
                pixel_status = "PIXEL_UNSEEN_AT_FREEZE"
                reciprocal_link = reciprocal(item, relation, anchor["item_id"])
            source.require(item["collection"] == anchor["collection"], f"COLLECTION_DRIFT:{item['id']}")
            source.require(reciprocal_link, f"NON_RECIPROCAL_LINK:{item['id']}:{relation}")
            gate = projection_gate(item, orientation)
            source.require(gate["eligible"], f"PROJECTION_GATE_FAILED:{item['id']}:{gate['failures']}")
            camera = [float(value) for value in item["geometry"]["coordinates"]]
            entrance = [
                float(episode["main_entrance_node"]["lon"]),
                float(episode["main_entrance_node"]["lat"]),
            ]
            members.append(
                {
                    "sequence_index": sequence_index,
                    "relation_to_anchor": relation,
                    "item_id": item["id"],
                    "collection": item["collection"],
                    "camera_to_entrance_distance_m": round(haversine_m(camera, entrance), 3),
                    "pixel_status_at_freeze": pixel_status,
                    "reciprocal_anchor_link": reciprocal_link,
                    "provider_item": item,
                    "projection_gate": gate,
                }
            )
        source.require(len({row["item_id"] for row in members}) == 3, "SEQUENCE_ITEMS_NOT_UNIQUE")
        episodes.append(
            {
                "episode_id": episode["episode_id"],
                "source_city": episode["source_city"],
                "target_way": episode["target_way"],
                "main_entrance_node": episode["main_entrance_node"],
                "query_collection": episode["query_collection"],
                "anchor_result_before_freeze": {
                    "prediction": result_rows[episode["episode_id"]]["appearance"]["prediction"],
                    "truth_score": result_rows[episode["episode_id"]]["appearance"]["truth_score"],
                    "truth_margin": result_rows[episode["episode_id"]]["appearance"]["truth_margin"],
                    "route": result_rows[episode["episode_id"]]["acceptance"]["route"],
                },
                "members": members,
            }
        )
    receipt = {
        "schema": "blindassist-l10-panolab-temporal-query-selection-v1",
        "status": "FROZEN_BEFORE_NEIGHBOR_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_MODEL_CALL_AFTER_ANCHOR_FAILURE_OBSERVED",
        "authority": "POSTHOC_MECHANISM_SELECTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": source.sha256(protocol_path),
        "selection_rule": protocol["selection"],
        "anchor_results_observed_before_freeze": True,
        "new_neighbor_pixel_views_before_freeze": 0,
        "new_neighbor_human_pixel_reviews_before_freeze": 0,
        "new_neighbor_model_calls_before_freeze": 0,
        "episodes": episodes,
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
                        "items": [member["item_id"] for member in row["members"]],
                        "distances_m": [member["camera_to_entrance_distance_m"] for member in row["members"]],
                    }
                    for row in episodes
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
    source.require(
        source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "EVALUATOR_HASH_MISMATCH",
    )
    selection = source.load(selection_path)
    source.require(selection["protocol_sha256"] == source.sha256(protocol_path), "PROTOCOL_LINK_MISMATCH")
    source.require(selection["new_neighbor_pixel_views_before_freeze"] == 0, "NEIGHBOR_PIXELS_VIEWED")
    source.require(selection["new_neighbor_model_calls_before_freeze"] == 0, "NEIGHBOR_MODEL_CALLED")
    existing = source.load(source.verify(protocol["inputs"]["positive_materialization"]))
    existing_queries = {
        row["item_id"]: row for row in existing["images"] if row["role"] == "query"
    }
    output_root = source.resolve(protocol["materialization"]["output_root"])
    source.require(not output_root.exists(), f"MATERIALIZATION_ROOT_ALREADY_EXISTS:{output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    images = []
    partials: list[Path] = []
    try:
        for episode in selection["episodes"]:
            for member in episode["members"]:
                if member["relation_to_anchor"] == "anchor":
                    receipt = existing_queries[member["item_id"]]
                    images.append(
                        {
                            "episode_id": episode["episode_id"],
                            "sequence_index": member["sequence_index"],
                            "relation_to_anchor": "anchor",
                            "item_id": member["item_id"],
                            "collection": member["collection"],
                            "path": receipt["path"],
                            "sha256": receipt["sha256"],
                            "bytes": receipt["bytes"],
                            "image_size": receipt["image_size"],
                            "url": receipt["url"],
                            "reused_existing_anchor": True,
                        }
                    )
                    continue
                target = image_root / f"{member['item_id']}.jpg"
                partial = target.with_suffix(".jpg.part")
                partials.append(partial)
                url = member["provider_item"]["assets"]["hd"]["href"]
                request = urllib.request.Request(
                    url, headers={"User-Agent": "BlindAssist-L10-Development/1"}
                )
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
                        "sequence_index": member["sequence_index"],
                        "relation_to_anchor": member["relation_to_anchor"],
                        "item_id": member["item_id"],
                        "collection": member["collection"],
                        "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                        "sha256": source.sha256(target),
                        "bytes": target.stat().st_size,
                        "image_size": size,
                        "url": url,
                        "reused_existing_anchor": False,
                    }
                )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    images.sort(key=lambda row: (row["episode_id"], row["sequence_index"]))
    manifest = {
        "schema": "blindassist-l10-panolab-temporal-query-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": source.sha256(selection_path),
        "new_neighbor_pixel_views_before_frozen_selection": 0,
        "new_neighbor_human_pixel_reviews_before_frozen_selection": 0,
        "new_neighbor_model_calls_before_frozen_selection": 0,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "images": len(images),
                "new_images": sum(not row["reused_existing_anchor"] for row in images),
                "new_bytes": sum(row["bytes"] for row in images if not row["reused_existing_anchor"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
