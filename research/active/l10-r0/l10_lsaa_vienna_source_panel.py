#!/usr/bin/env python3
"""Freeze pixel-blind LSAA Vienna address-door development and confirmation panels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any


SCHEMAS = {
    "blindassist-l10-lsaa-vienna-source-protocol-v1",
    "blindassist-l10-lsaa-vienna-source-protocol-v2",
    "blindassist-l10-lsaa-vienna-source-protocol-v3",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_number(value: Any) -> str:
    return re.sub(r"[^0-9A-Z]", "", str(value).upper())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def overpass_ways(endpoint: str, way_ids: list[int], user_agent: str) -> tuple[list[dict[str, Any]], str]:
    query = f"[out:json][timeout:90];way(id:{','.join(map(str, way_ids))});out tags;"
    request = urllib.request.Request(
        endpoint,
        data=urllib.parse.urlencode({"data": query}).encode("ascii"),
        headers={"User-Agent": user_agent},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    return list(payload.get("elements", [])), hashlib.sha256(query.encode("utf-8")).hexdigest()


def stable_rank(seed: str, row: dict[str, Any]) -> str:
    key = f"{seed}|{row['facade_name']}|{row['building']}|{row['house_number']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--public-output", type=Path, required=True)
    parser.add_argument("--truth-output", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    require(protocol.get("schema") in SCHEMAS, "protocol schema mismatch")
    for output in (args.public_output, args.truth_output, args.receipt_output):
        require(not output.exists(), f"refusing to overwrite: {output}")

    inputs: dict[str, Path] = {}
    for name, spec in protocol["inputs"].items():
        path = Path(spec["path"]).resolve()
        require(path.is_file(), f"missing input: {path}")
        require(sha256_file(path) == spec["sha256"], f"input SHA-256 mismatch: {path}")
        inputs[name] = path

    doors = read_csv(inputs["door_filtered"])
    facades = read_csv(inputs["properties_23k"])
    detections = json.loads(inputs["door_detection"].read_text(encoding="utf-8"))
    facade_counts = Counter(row["facade_name"] for row in doors)
    facade_by_name = {row["name"]: row for row in facades}
    quality = protocol["selection"]["quality"]

    metadata_candidates = []
    for door in doors:
        facade = facade_by_name.get(door["facade_name"])
        if facade is None or facade_counts[door["facade_name"]] != 1:
            continue
        if door["city"] != protocol["selection"]["city"] or not door["building"].startswith("way"):
            continue
        if float(door["width"]) < quality["minimum_door_width_pixels"]:
            continue
        if float(door["height"]) < quality["minimum_door_height_pixels"]:
            continue
        if float(facade["width"]) < quality["minimum_facade_width_pixels"]:
            continue
        if float(facade["height"]) < quality["minimum_facade_height_pixels"]:
            continue
        if float(facade["noblur"]) < quality["minimum_noblur"]:
            continue
        if float(facade["total_occlusion"]) > quality["maximum_total_occlusion"]:
            continue
        if abs(float(facade["view_angle"])) > quality["maximum_absolute_view_angle_degrees"]:
            continue
        detection = detections.get(door["name"])
        require(detection is not None, f"missing official door detection: {door['name']}")
        metadata_candidates.append(
            {
                "building": door["building"],
                "way_id": int(door["building"][3:]),
                "facade_name": door["facade_name"],
                "door_asset_name": door["name"],
                "panorama_id": door["panoID"],
                "country": door["country"],
                "city": door["city"],
                "facade_lon_lat": [float(facade["Lon"]), float(facade["Lat"])],
                "facade_shape_hw": [int(float(facade["height"])), int(float(facade["width"]))],
                "official_door_box_xywh": [float(value) for value in detection["box"]],
                "quality": {
                    "door_shape_hw": [int(float(door["height"])), int(float(door["width"]))],
                    "noblur": float(facade["noblur"]),
                    "view_angle_degrees": float(facade["view_angle"]),
                    "total_occlusion": float(facade["total_occlusion"]),
                },
            }
        )

    metadata_candidates.sort(key=lambda row: (row["way_id"], row["facade_name"]))
    elements, query_sha256 = overpass_ways(
        protocol["providers"]["overpass"],
        [row["way_id"] for row in metadata_candidates],
        protocol["providers"]["user_agent"],
    )
    osm_by_id = {int(row["id"]): row.get("tags", {}) for row in elements}
    address_candidates = []
    address_pattern = re.compile(protocol["selection"]["house_number_pattern"])
    for row in metadata_candidates:
        tags = osm_by_id.get(row["way_id"], {})
        raw_house_number = str(tags.get("addr:housenumber", "")).strip().upper()
        if protocol["selection"].get("raw_house_number_must_match", False) and not address_pattern.fullmatch(
            raw_house_number
        ):
            continue
        house_number = canonical_number(raw_house_number)
        if not address_pattern.fullmatch(house_number):
            continue
        address_candidates.append(
            {
                **row,
                "house_number": house_number,
                "street_name": str(tags.get("addr:street", "")),
                "osm_building_class": str(tags.get("building", "")),
            }
        )

    seed = protocol["selection"]["stable_split_seed"]
    address_candidates.sort(key=lambda row: (stable_rank(seed, row), row["facade_name"]))
    unique_building_candidates = []
    selected_buildings: set[str] = set()
    for row in address_candidates:
        if protocol["selection"].get("unique_buildings_across_panel", False):
            if row["building"] in selected_buildings:
                continue
            selected_buildings.add(row["building"])
        unique_building_candidates.append(row)
    dev_count = int(protocol["selection"]["development_count"])
    confirm_count = int(protocol["selection"]["confirmation_count"])
    require(
        len(unique_building_candidates) >= dev_count + confirm_count,
        "insufficient address-bearing candidates after uniqueness filters",
    )
    selected = unique_building_candidates[: dev_count + confirm_count]
    if protocol["selection"].get("unique_buildings_across_panel", False):
        require(
            len({row["building"] for row in selected}) == len(selected),
            "selected panel contains repeated buildings",
        )

    rows = []
    truths = {}
    for index, row in enumerate(selected):
        split = "DEVELOPMENT" if index < dev_count else "CONFIRMATION_HOLDOUT"
        split_rows = selected[:dev_count] if split == "DEVELOPMENT" else selected[dev_count:]
        split_index = index if split == "DEVELOPMENT" else index - dev_count
        counterfactual = next(
            split_rows[(split_index + offset) % len(split_rows)]["house_number"]
            for offset in range(1, len(split_rows))
            if split_rows[(split_index + offset) % len(split_rows)]["house_number"]
            != row["house_number"]
        )
        item_id = f"LSAA-VIE-{split[0]}-{split_index + 1:03d}"
        rows.append(
            {
                "item_id": item_id,
                "split": split,
                "provider": "LSAA_FILTERED_FACADES_GOOGLE_PANORAMA",
                "city": row["city"],
                "country": row["country"],
                "building": row["building"],
                "way_id": row["way_id"],
                "facade_name": row["facade_name"],
                "remote_archive_member": f"Filtered_facades/{row['facade_name']}",
                "panorama_id": row["panorama_id"],
                "facade_lon_lat": row["facade_lon_lat"],
                "mission": {"street_name": row["street_name"], "house_number": row["house_number"]},
                "counterfactual_mission": {"house_number": counterfactual},
                "quality": row["quality"],
            }
        )
        x, y, width, height = row["official_door_box_xywh"]
        truths[item_id] = {
            "official_single_door_box_xyxy": [x, y, x + width, y + height],
            "target_house_number_visible": None,
            "target_credential_box_xyxy": None,
            "annotation_state": "PENDING_PIXEL_TRUTH_BEFORE_INFERENCE",
        }

    public = {
        "schema": "blindassist-l10-lsaa-vienna-public-source-v1",
        "selection_authority": str(args.protocol.resolve()),
        "remote_facade_archive": protocol["providers"]["filtered_facades_archive"],
        "rows": rows,
        "claim_boundary": "Pixel-blind source selection only. OSM address metadata plus one official LSAA door annotation does not establish visible text, model correctness, public access, traversability, waypoint, arrival, handoff, user benefit, or safety.",
    }
    truth = {
        "schema": "blindassist-l10-lsaa-vienna-evaluator-truth-draft-v1",
        "public_source": str(args.public_output.resolve()),
        "truth": truths,
        "state": "PENDING_PIXEL_TRUTH_BEFORE_MODEL_INFERENCE",
    }
    receipt = {
        "schema": "blindassist-l10-lsaa-vienna-selection-receipt-v1",
        "inputs": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in inputs.items()
        },
        "overpass_query_sha256": query_sha256,
        "counts": {
            "door_rows": len(doors),
            "facade_rows": len(facades),
            "metadata_quality_candidates": len(metadata_candidates),
            "overpass_returned_ways": len(elements),
            "numeric_address_candidates": len(address_candidates),
            "unique_building_address_candidates": len(unique_building_candidates),
            "development_rows": dev_count,
            "confirmation_holdout_rows": confirm_count,
        },
        "selection_pixel_access_count": 0,
        "selected_facade_overlap_between_splits": sorted(
            {row["facade_name"] for row in rows if row["split"] == "DEVELOPMENT"}
            & {row["facade_name"] for row in rows if row["split"] == "CONFIRMATION_HOLDOUT"}
        ),
        "selected_building_overlap_between_splits": sorted(
            {row["building"] for row in rows if row["split"] == "DEVELOPMENT"}
            & {row["building"] for row in rows if row["split"] == "CONFIRMATION_HOLDOUT"}
        ),
        "selected_duplicate_building_groups": {
            building: sorted(row["item_id"] for row in rows if row["building"] == building)
            for building, count in Counter(row["building"] for row in rows).items()
            if count > 1
        },
    }

    args.public_output.write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.truth_output.write_text(json.dumps(truth, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.receipt_output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt["counts"], indent=2))


if __name__ == "__main__":
    main()
