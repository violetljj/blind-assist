#!/usr/bin/env python3
"""Freeze and materialize a producer-stratified, pixel-unseen L10 router panel."""

from __future__ import annotations

import argparse
import json
import os
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import l10_panolab_federated_router_confirmation_panel as confirmation
import l10_panolab_temporal_query_panel as temporal
import l10_panolab_viviani_reference_panel as source


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-producer-stratified-router-source-protocol-v1"


def producer(item: dict[str, Any]) -> str:
    return str(item.get("properties", {}).get("geovisio:producer") or "")


def candidate_items(row: dict[str, Any]) -> set[str]:
    return {str(row["reference"]["item_id"]), str(row["query"]["item_id"])}


def build_episode(
    candidate: dict[str, Any],
    episode_id: str,
    stratum: str,
    orientation: dict[str, Any],
    prior_items: set[str],
) -> dict[str, Any]:
    reference_anchor = candidate["reference"]["item"]
    query_anchor = candidate["query"]["item"]
    reference_next = temporal.linked_item(reference_anchor, "next")
    query_prev = temporal.linked_item(query_anchor, "prev")
    query_next = temporal.linked_item(query_anchor, "next")
    source.require(reference_next["collection"] == reference_anchor["collection"], "REFERENCE_COLLECTION_DRIFT")
    source.require(query_prev["collection"] == query_anchor["collection"], "QUERY_PREV_COLLECTION_DRIFT")
    source.require(query_next["collection"] == query_anchor["collection"], "QUERY_NEXT_COLLECTION_DRIFT")
    references = [
        confirmation.member(
            candidate,
            reference_anchor,
            "anchor",
            0,
            "PIXEL_UNSEEN_AT_FREEZE",
            orientation,
            reference_anchor["id"],
        ),
        confirmation.member(
            candidate,
            reference_next,
            "next",
            1,
            "PIXEL_UNSEEN_AT_FREEZE",
            orientation,
            reference_anchor["id"],
        ),
    ]
    queries = [
        confirmation.member(
            candidate,
            query_prev,
            "prev",
            0,
            "PIXEL_UNSEEN_AT_FREEZE",
            orientation,
            query_anchor["id"],
        ),
        confirmation.member(
            candidate,
            query_anchor,
            "anchor",
            1,
            "PIXEL_UNSEEN_AT_FREEZE",
            orientation,
            query_anchor["id"],
        ),
        confirmation.member(
            candidate,
            query_next,
            "next",
            2,
            "PIXEL_UNSEEN_AT_FREEZE",
            orientation,
            query_anchor["id"],
        ),
    ]
    members = [*references, *queries]
    item_ids = {row["item_id"] for row in members}
    source.require(len(item_ids) == 5, f"ITEMS_NOT_UNIQUE:{episode_id}")
    source.require(not (item_ids & prior_items), f"PRIOR_ROUTER_ITEM_OVERLAP:{episode_id}")
    source.require(
        references[0]["collection"] != queries[1]["collection"],
        f"REFERENCE_QUERY_COLLECTION_OVERLAP:{episode_id}",
    )
    return {
        "episode_id": episode_id,
        "candidate_ledger_rank": int(candidate["candidate_ledger_rank"]),
        "producer_stratum": stratum,
        "source_city": candidate["source_city"],
        "target_way_id": candidate["target_way_id"],
        "target_name": candidate["target_name"],
        "main_entrance_node": candidate["main_entrance_node"],
        "reference_collection": references[0]["collection"],
        "query_collection": queries[1]["collection"],
        "reference_producer": producer(reference_anchor),
        "query_producer": producer(query_anchor),
        "references": references,
        "queries": queries,
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
        source.sha256(source.resolve(protocol["inputs"]["confirmation_panel_helper"]["path"]))
        == protocol["inputs"]["confirmation_panel_helper"]["sha256"],
        "CONFIRMATION_HELPER_HASH_MISMATCH",
    )
    source.require(
        source.sha256(source.resolve(protocol["inputs"]["temporal_panel_helper"]["path"]))
        == protocol["inputs"]["temporal_panel_helper"]["sha256"],
        "TEMPORAL_HELPER_HASH_MISMATCH",
    )
    candidates_payload = source.load(source.verify(protocol["inputs"]["direct_candidates"]))
    prior_six = source.load(source.verify(protocol["inputs"]["prior_six_target_result"]))
    prior_temporal = source.load(source.verify(protocol["inputs"]["prior_temporal_materialization"]))
    prior_eight = source.load(source.verify(protocol["inputs"]["prior_eight_target_materialization"]))
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    prior_items = {row["item_id"] for row in prior_six["crop_receipts"]}
    prior_items.update(row["item_id"] for row in prior_temporal["images"])
    prior_items.update(row["item_id"] for row in prior_eight["images"])
    excluded_way_ids = {int(value) for value in protocol["selection"]["excluded_target_way_ids"]}

    eligible = []
    for rank, candidate in enumerate(candidates_payload["candidates"], start=1):
        if int(candidate["target_way_id"]) in excluded_way_ids:
            continue
        if not bool(candidate["reference"]["orientation_gate"]["strict_eligible"]):
            continue
        if not bool(candidate["query"]["orientation_gate"]["strict_eligible"]):
            continue
        if candidate_items(candidate) & prior_items:
            continue
        copied = dict(candidate)
        copied["candidate_ledger_rank"] = rank
        eligible.append(copied)

    episodes = []
    failures = []
    for stratum in protocol["selection"]["producer_strata"]:
        stratum_candidates = [
            row
            for row in eligible
            if producer(row["reference"]["item"]) == stratum
            and producer(row["query"]["item"]) == stratum
        ]
        selected = None
        for candidate in stratum_candidates:
            try:
                selected = build_episode(
                    candidate,
                    f"PS{len(episodes) + 1:02d}",
                    stratum,
                    orientation,
                    prior_items,
                )
                break
            except (OSError, RuntimeError, ValueError) as error:
                failures.append(
                    {
                        "producer_stratum": stratum,
                        "candidate_ledger_rank": candidate["candidate_ledger_rank"],
                        "target_way_id": candidate["target_way_id"],
                        "failure": str(error),
                    }
                )
        source.require(selected is not None, f"NO_METADATA_VALID_CANDIDATE_FOR_PRODUCER:{stratum}")
        episodes.append(selected)

    source.require(len(episodes) == len(protocol["selection"]["producer_strata"]), "STRATUM_COUNT_MISMATCH")
    all_items = {
        row["item_id"]
        for episode in episodes
        for row in [*episode["references"], *episode["queries"]]
    }
    source.require(len(all_items) == 5 * len(episodes), "CROSS_EPISODE_ITEM_OVERLAP")
    receipt = {
        "schema": "blindassist-l10-panolab-producer-stratified-router-selection-v1",
        "status": "FROZEN_BEFORE_ANY_SELECTED_PIXEL_DOWNLOAD_HUMAN_REVIEW_OR_ROUTER_CALL",
        "authority": "PRODUCER_STRATIFIED_PIXEL_UNSEEN_DEVELOPMENT_SELECTION",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(protocol_path),
        "protocol_sha256": source.sha256(protocol_path),
        "selection_rule": protocol["selection"],
        "selected_pixel_views_before_freeze": 0,
        "selected_human_pixel_reviews_before_freeze": 0,
        "selected_appearance_calls_before_freeze": 0,
        "selected_ocr_calls_before_freeze": 0,
        "prior_router_target_way_overlap": 0,
        "prior_router_item_overlap": 0,
        "metadata_candidate_failures_before_selection": failures,
        "episodes": episodes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selection": str(output_path),
                "metadata_candidate_failures": len(failures),
                "episodes": [
                    {
                        "episode_id": row["episode_id"],
                        "producer": row["producer_stratum"],
                        "ledger_rank": row["candidate_ledger_rank"],
                        "way_id": row["target_way_id"],
                        "name": row["target_name"],
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
    source.require(selection["selected_pixel_views_before_freeze"] == 0, "PIXELS_VIEWED_BEFORE_FREEZE")
    source.require(selection["selected_appearance_calls_before_freeze"] == 0, "APPEARANCE_CALLED_BEFORE_FREEZE")
    source.require(selection["selected_ocr_calls_before_freeze"] == 0, "OCR_CALLED_BEFORE_FREEZE")
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
                    gate = item["projection_gate"]
                    expected = [int(gate["sensor_width_pixels"]), int(gate["sensor_height_pixels"])]
                    source.require(size == expected, f"UNEXPECTED_IMAGE_SIZE:{target}:{size}:{expected}")
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
                        }
                    )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    images.sort(key=lambda row: (row["episode_id"], row["role"], row["sequence_index"]))
    manifest = {
        "schema": "blindassist-l10-panolab-producer-stratified-router-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": source.sha256(selection_path),
        "selected_pixel_views_before_frozen_selection": 0,
        "selected_human_pixel_reviews_before_frozen_selection": 0,
        "selected_appearance_calls_before_frozen_selection": 0,
        "selected_ocr_calls_before_frozen_selection": 0,
        "images": images,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "manifest": str(output_path),
                "images": len(images),
                "bytes": sum(row["bytes"] for row in images),
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
