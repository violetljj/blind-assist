#!/usr/bin/env python3
"""Freeze and materialize a new-city federated router confirmation panel."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import l10_panolab_temporal_query_panel as temporal
import l10_panolab_viviani_reference_panel as source
from l10_panolab_entrance_ray import projection_gate


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-federated-router-confirmation-source-protocol-v1"


def member(
    episode: dict[str, Any],
    item: dict[str, Any],
    relation: str,
    sequence_index: int,
    pixel_status: str,
    orientation: dict[str, Any],
    anchor_id: str,
) -> dict[str, Any]:
    reciprocal_link = relation == "anchor" or temporal.reciprocal(item, relation, anchor_id)
    source.require(reciprocal_link, f"NON_RECIPROCAL_LINK:{item['id']}:{relation}")
    gate = projection_gate(item, orientation)
    source.require(gate["eligible"], f"PROJECTION_GATE_FAILED:{item['id']}:{gate['failures']}")
    entrance = [
        float(episode["main_entrance_node"]["lon"]),
        float(episode["main_entrance_node"]["lat"]),
    ]
    camera = [float(value) for value in item["geometry"]["coordinates"]]
    return {
        "sequence_index": sequence_index,
        "relation_to_anchor": relation,
        "item_id": item["id"],
        "collection": item["collection"],
        "camera_to_entrance_distance_m": round(temporal.haversine_m(camera, entrance), 3),
        "pixel_status_at_freeze": pixel_status,
        "reciprocal_anchor_link": reciprocal_link,
        "provider_item": item,
        "projection_gate": gate,
    }


def freeze(protocol_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(
        source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"],
        "EVALUATOR_HASH_MISMATCH",
    )
    source.require(
        source.sha256(source.resolve(protocol["inputs"]["temporal_panel_evaluator"]["path"]))
        == protocol["inputs"]["temporal_panel_evaluator"]["sha256"],
        "TEMPORAL_HELPER_HASH_MISMATCH",
    )
    federated = source.load(source.verify(protocol["inputs"]["federated_source"]))
    prior_router = source.load(source.verify(protocol["inputs"]["prior_router_result"]))
    prior_temporal_selection = source.load(
        source.verify(protocol["inputs"]["prior_temporal_selection"])
    )
    prior_temporal = source.load(source.verify(protocol["inputs"]["prior_temporal_materialization"]))
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    prior_items = {row["item_id"] for row in prior_router["crop_receipts"]}
    prior_items.update(row["item_id"] for row in prior_temporal["images"])
    prior_way_ids = {int(row["target_way_id"]) for row in prior_router["rows"]}
    prior_way_ids.update(
        int(row["target_way"]["id"]) for row in prior_temporal_selection["episodes"]
    )
    contract = protocol["selection"]
    eligible = [
        row
        for row in federated["episodes"]
        if row["source_city"] == contract["source_city"]
        and int(row["target_way_id"]) not in prior_way_ids
        and bool(row["reference"]["entrance_ray"]["projection_gate"]["strict_eligible"])
        and bool(row["query"]["entrance_ray"]["projection_gate"]["strict_eligible"])
    ]
    eligible.sort(key=lambda row: (int(row["target_way_id"]), row["reference"]["item_id"]))
    selected = eligible[: int(contract["panel_size"])]
    source.require(len(selected) == int(contract["panel_size"]), "INSUFFICIENT_STRICT_EPISODES")
    episodes = []
    for index, episode in enumerate(selected, start=1):
        reference_anchor = episode["reference"]["provider_item"]
        query_anchor = episode["query"]["provider_item"]
        reference_next = temporal.linked_item(reference_anchor, "next")
        query_prev = temporal.linked_item(query_anchor, "prev")
        query_next = temporal.linked_item(query_anchor, "next")
        source.require(reference_next["collection"] == reference_anchor["collection"], "REFERENCE_COLLECTION_DRIFT")
        source.require(query_prev["collection"] == query_anchor["collection"], "QUERY_PREV_COLLECTION_DRIFT")
        source.require(query_next["collection"] == query_anchor["collection"], "QUERY_NEXT_COLLECTION_DRIFT")
        references = [
            member(
                episode,
                reference_anchor,
                "anchor",
                0,
                "PRIOR_FEDERATED_PIXEL_REUSED_ROUTER_UNSEEN",
                orientation,
                reference_anchor["id"],
            ),
            member(
                episode,
                reference_next,
                "next",
                1,
                "PIXEL_UNSEEN_AT_FREEZE",
                orientation,
                reference_anchor["id"],
            ),
        ]
        queries = [
            member(
                episode,
                query_prev,
                "prev",
                0,
                "PIXEL_UNSEEN_AT_FREEZE",
                orientation,
                query_anchor["id"],
            ),
            member(
                episode,
                query_anchor,
                "anchor",
                1,
                "PRIOR_FEDERATED_PIXEL_REUSED_ROUTER_UNSEEN",
                orientation,
                query_anchor["id"],
            ),
            member(
                episode,
                query_next,
                "next",
                2,
                "PIXEL_UNSEEN_AT_FREEZE",
                orientation,
                query_anchor["id"],
            ),
        ]
        item_ids = {row["item_id"] for row in [*references, *queries]}
        source.require(len(item_ids) == 5, f"ITEMS_NOT_UNIQUE:{episode['episode_id']}")
        source.require(not (item_ids & prior_items), f"PRIOR_ROUTER_ITEM_OVERLAP:{episode['episode_id']}")
        source.require(
            references[0]["collection"] != queries[1]["collection"],
            f"REFERENCE_QUERY_COLLECTION_OVERLAP:{episode['episode_id']}",
        )
        episodes.append(
            {
                "episode_id": f"CF{index:02d}",
                "federated_episode_id": episode["episode_id"],
                "source_city": episode["source_city"],
                "target_way_id": episode["target_way_id"],
                "target_name": episode["target_name"],
                "main_entrance_node": episode["main_entrance_node"],
                "reference_collection": references[0]["collection"],
                "query_collection": queries[1]["collection"],
                "references": references,
                "queries": queries,
                "reused_reference_panorama": episode["reference"]["panorama"],
                "reused_query_panorama": episode["query"]["panorama"],
            }
        )
    receipt = {
        "schema": "blindassist-l10-panolab-federated-router-confirmation-selection-v1",
        "status": "FROZEN_BEFORE_NEW_NEIGHBOR_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_ROUTER_CALL",
        "authority": "NEW_CITY_FEDERATED_MODEL_UNSEEN_DEVELOPMENT_SELECTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": source.sha256(protocol_path),
        "selection_rule": contract,
        "reused_anchor_pixels_exist_from_prior_portal_source": True,
        "router_calls_on_all_selected_pixels_before_freeze": 0,
        "new_neighbor_pixel_views_before_freeze": 0,
        "new_neighbor_human_pixel_reviews_before_freeze": 0,
        "new_neighbor_router_calls_before_freeze": 0,
        "prior_router_target_way_overlap": 0,
        "prior_router_item_overlap": 0,
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
                        "city": row["source_city"],
                        "way_id": row["target_way_id"],
                        "name": row["target_name"],
                        "reference_items": [item["item_id"] for item in row["references"]],
                        "query_items": [item["item_id"] for item in row["queries"]],
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
    source.require(selection["new_neighbor_router_calls_before_freeze"] == 0, "NEIGHBOR_ROUTER_CALLED")
    output_root = source.resolve(protocol["materialization"]["output_root"])
    source.require(not output_root.exists(), f"MATERIALIZATION_ROOT_ALREADY_EXISTS:{output_root}")
    image_root = output_root / "images"
    image_root.mkdir(parents=True)
    images = []
    partials: list[Path] = []
    try:
        for episode in selection["episodes"]:
            for role, members in (("reference", episode["references"]), ("query", episode["queries"])):
                for item in members:
                    reused = item["relation_to_anchor"] == "anchor"
                    if reused:
                        receipt = (
                            episode["reused_reference_panorama"]
                            if role == "reference"
                            else episode["reused_query_panorama"]
                        )
                        path = source.resolve(receipt["path"])
                        source.require(source.sha256(path) == receipt["sha256"], f"REUSED_HASH:{path}")
                        source.require(path.stat().st_size == int(receipt["bytes"]), f"REUSED_BYTES:{path}")
                        images.append(
                            {
                                "episode_id": episode["episode_id"],
                                "role": role,
                                "sequence_index": item["sequence_index"],
                                "relation_to_anchor": item["relation_to_anchor"],
                                "item_id": item["item_id"],
                                "collection": item["collection"],
                                "path": str(path),
                                "sha256": receipt["sha256"],
                                "bytes": receipt["bytes"],
                                "image_size": receipt["image_size"],
                                "url": receipt["provenance"],
                                "reused_federated_anchor": True,
                            }
                        )
                        continue
                    target = image_root / f"{item['item_id']}.jpg"
                    partial = target.with_suffix(".jpg.part")
                    partials.append(partial)
                    url = item["provider_item"]["assets"]["hd"]["href"]
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
                            "role": role,
                            "sequence_index": item["sequence_index"],
                            "relation_to_anchor": item["relation_to_anchor"],
                            "item_id": item["item_id"],
                            "collection": item["collection"],
                            "path": str(target.relative_to(ROOT)).replace("\\", "/"),
                            "sha256": source.sha256(target),
                            "bytes": target.stat().st_size,
                            "image_size": size,
                            "url": url,
                            "reused_federated_anchor": False,
                        }
                    )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    images.sort(key=lambda row: (row["episode_id"], row["role"], row["sequence_index"]))
    manifest = {
        "schema": "blindassist-l10-panolab-federated-router-confirmation-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": source.sha256(selection_path),
        "new_neighbor_pixel_views_before_frozen_selection": 0,
        "new_neighbor_human_pixel_reviews_before_frozen_selection": 0,
        "new_neighbor_router_calls_before_frozen_selection": 0,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "images": len(images),
                "new_images": sum(not row["reused_federated_anchor"] for row in images),
                "new_bytes": sum(row["bytes"] for row in images if not row["reused_federated_anchor"]),
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
