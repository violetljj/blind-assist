#!/usr/bin/env python3
"""Freeze and materialize a fresh panel capable of testing distinctive tokens."""

from __future__ import annotations

import argparse
import json
import os
import re
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

import l10_panolab_producer_stratified_router_panel as base
import l10_panolab_viviani_reference_panel as source


ROOT = Path(__file__).resolve().parents[3]
PROTOCOL_SCHEMA = "blindassist-l10-panolab-distinctive-token-fresh-source-protocol-v1"


def name_tokens(value: str, ignored: set[str]) -> list[str]:
    folded = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii").lower()
    return [token for token in re.findall(r"[a-z0-9]+", folded) if token not in ignored]


def capability_units(name: str, selection: dict[str, Any]) -> int:
    tokens = name_tokens(name, set(selection["ignored_name_tokens"]))
    minimum = int(selection["minimum_target_token_length"])
    long_minimum = int(selection["long_token_length_for_two_units"])
    return sum(2 if len(token) >= long_minimum else 1 for token in tokens if len(token) >= minimum)


def freeze(protocol_path: Path, output_path: Path) -> None:
    source.require(not output_path.exists(), f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol = source.load(protocol_path)
    source.require(protocol.get("schema") == PROTOCOL_SCHEMA, "UNEXPECTED_PROTOCOL_SCHEMA")
    source.require(source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
    source.require(
        source.sha256(source.resolve(protocol["inputs"]["episode_builder"]["path"]))
        == protocol["inputs"]["episode_builder"]["sha256"],
        "EPISODE_BUILDER_HASH_MISMATCH",
    )
    candidates_payload = source.load(source.verify(protocol["inputs"]["direct_candidates"]))
    prior_six = source.load(source.verify(protocol["inputs"]["prior_six_target_result"]))
    prior_temporal = source.load(source.verify(protocol["inputs"]["prior_temporal_materialization"]))
    prior_eight = source.load(source.verify(protocol["inputs"]["prior_eight_target_materialization"]))
    prior_eleven = source.load(source.verify(protocol["inputs"]["prior_eleven_target_materialization"]))
    orientation = source.load(source.verify(protocol["inputs"]["orientation_projection_protocol"]))
    prior_items = {row["item_id"] for row in prior_six["crop_receipts"]}
    prior_items.update(row["item_id"] for row in prior_temporal["images"])
    prior_items.update(row["item_id"] for row in prior_eight["images"])
    prior_items.update(row["item_id"] for row in prior_eleven["images"])
    excluded_way_ids = {int(value) for value in protocol["selection"]["excluded_target_way_ids"]}

    eligible = []
    metadata_skips = []
    for rank, candidate in enumerate(candidates_payload["candidates"], start=1):
        units = capability_units(candidate["target_name"], protocol["selection"])
        reason = None
        if int(candidate["target_way_id"]) in excluded_way_ids:
            reason = "EXCLUDED_PRIOR_TARGET_WAY"
        elif not candidate["reference"]["orientation_gate"]["strict_eligible"]:
            reason = "REFERENCE_NOT_STRICT"
        elif not candidate["query"]["orientation_gate"]["strict_eligible"]:
            reason = "QUERY_NOT_STRICT"
        elif base.candidate_items(candidate) & prior_items:
            reason = "PRIOR_ROUTER_ANCHOR_ITEM_OVERLAP"
        elif units < int(protocol["selection"]["minimum_distinctive_evidence_units"]):
            reason = "TARGET_NAME_CANNOT_REACH_DISTINCTIVE_EVIDENCE_THRESHOLD"
        if reason:
            metadata_skips.append(
                {
                    "candidate_ledger_rank": rank,
                    "target_way_id": candidate["target_way_id"],
                    "target_name": candidate["target_name"],
                    "capability_units": units,
                    "reason": reason,
                }
            )
            continue
        copied = dict(candidate)
        copied["candidate_ledger_rank"] = rank
        copied["metadata_capability_units"] = units
        eligible.append(copied)

    episodes = []
    failures = []
    for stratum in protocol["selection"]["producer_strata"]:
        selected = None
        for candidate in eligible:
            if base.producer(candidate["reference"]["item"]) != stratum or base.producer(candidate["query"]["item"]) != stratum:
                continue
            try:
                selected = base.build_episode(
                    candidate,
                    f"DF{len(episodes) + 1:02d}",
                    stratum,
                    orientation,
                    prior_items,
                )
                selected["metadata_capability_units"] = candidate["metadata_capability_units"]
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
        source.require(selected is not None, f"NO_METADATA_VALID_CAPABLE_CANDIDATE:{stratum}")
        episodes.append(selected)

    all_items = {
        member["item_id"]
        for episode in episodes
        for member in [*episode["references"], *episode["queries"]]
    }
    source.require(len(episodes) == 2 and len(all_items) == 10, "FRESH_PANEL_SIZE_MISMATCH")
    receipt = {
        "schema": "blindassist-l10-panolab-distinctive-token-fresh-selection-v1",
        "status": "FROZEN_BEFORE_ANY_SELECTED_PIXEL_DOWNLOAD_HUMAN_REVIEW_APPEARANCE_OR_OCR_CALL",
        "authority": "PIXEL_AND_OCR_UNSEEN_METADATA_CAPABILITY_STRATIFIED_DEVELOPMENT_SELECTION",
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
        "metadata_skips_before_selection": metadata_skips,
        "metadata_candidate_failures_before_selection": failures,
        "episodes": episodes,
    }
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "selection": str(output_path),
                "episodes": [
                    {
                        "episode_id": row["episode_id"],
                        "producer": row["producer_stratum"],
                        "ledger_rank": row["candidate_ledger_rank"],
                        "way_id": row["target_way_id"],
                        "name": row["target_name"],
                        "metadata_capability_units": row["metadata_capability_units"],
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
    source.require(source.sha256(Path(__file__).resolve()) == protocol["evaluator"]["sha256"], "EVALUATOR_HASH_MISMATCH")
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
                    request = urllib.request.Request(
                        item["provider_item"]["assets"]["hd"]["href"],
                        headers={"User-Agent": "BlindAssist-L10-Development/1"},
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
                            "url": item["provider_item"]["assets"]["hd"]["href"],
                        }
                    )
    finally:
        for partial in partials:
            if partial.exists():
                partial.unlink()
    images.sort(key=lambda row: (row["episode_id"], row["role"], row["sequence_index"]))
    manifest = {
        "schema": "blindassist-l10-panolab-distinctive-token-fresh-materialization-v1",
        "selection": str(selection_path),
        "selection_sha256": source.sha256(selection_path),
        "selected_pixel_views_before_frozen_selection": 0,
        "selected_human_pixel_reviews_before_frozen_selection": 0,
        "selected_appearance_calls_before_frozen_selection": 0,
        "selected_ocr_calls_before_frozen_selection": 0,
        "images": images,
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(output_path), "images": len(images), "bytes": sum(row["bytes"] for row in images)}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    freeze_parser = commands.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, required=True)
    materialize_parser = commands.add_parser("materialize")
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
