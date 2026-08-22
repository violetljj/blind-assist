"""Fetch Mapillary sequence metadata for a pre-metadata public goal roster."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import requests


FIELDS = "id,captured_at,computed_geometry,computed_compass_angle,compass_angle,camera_type,sequence"


def _get(session: requests.Session, url: str, params: Mapping[str, Any]) -> Any:
    response = session.get(url, params=dict(params), timeout=60)
    response.raise_for_status()
    value = response.json()
    if isinstance(value, Mapping) and value.get("error"):
        raise ValueError("Mapillary response contains an error")
    return value


def _add_rows(images: dict[str, dict[str, Any]], raw_rows: Sequence[Mapping[str, Any]]) -> None:
    for raw in raw_rows:
        geometry = raw.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        sequence = raw.get("sequence")
        if isinstance(sequence, Mapping):
            sequence = sequence.get("id")
        if not raw.get("id") or not sequence or not isinstance(coordinates, list) or len(coordinates) != 2:
            continue
        image_id = str(raw["id"])
        images[image_id] = {
            "image_id": image_id,
            "sequence_id": str(sequence),
            "captured_at_ms": int(raw.get("captured_at") or 0),
            "coordinates": [float(coordinates[0]), float(coordinates[1])],
            "computed_compass_angle": raw.get("computed_compass_angle"),
            "compass_angle": raw.get("compass_angle"),
            "camera_type": raw.get("camera_type"),
            "source_url": f"https://www.mapillary.com/app/?focus=photo&pKey={image_id}",
        }


def acquire(roster: Mapping[str, Any], token: str) -> dict[str, Any]:
    if roster.get("schema_version") != "blindassist_public_goal_roster_v0":
        raise ValueError("goal roster schema mismatch")
    if any(roster.get("precedence", {}).get(key) is not False for key in (
        "mapillary_metadata_accessed", "mapillary_pixels_accessed", "model_outputs_created", "evaluator_truth_created"
    )):
        raise ValueError("roster precedence is not clean")
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    image_ids: set[str] = set()
    images: dict[str, dict[str, Any]] = {}
    queries = []
    for goal in roster["goals"]:
        points = [row["coordinates"] for row in goal["public_entrance_candidates"]]
        lon = sum(float(row[0]) for row in points) / len(points)
        lat = sum(float(row[1]) for row in points) / len(points)
        lat_delta = 60.0 / 111_320.0
        lon_delta = lat_delta / max(0.2, abs(math.cos(math.radians(lat))))
        response = _get(session, "https://graph.mapillary.com/images", {
            "bbox": f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}",
            "limit": 100,
            "fields": "id",
        })
        goal_ids = {str(raw["id"]) for raw in response.get("data", []) if raw.get("id")}
        image_ids.update(goal_ids)
        queries.append({"goal_id": goal["goal_id"], "returned_image_ids": len(goal_ids)})
    sorted_ids = sorted(image_ids)
    raw_rows = []
    for offset in range(0, len(sorted_ids), 40):
        detail = _get(session, "https://graph.mapillary.com/", {
            "ids": ",".join(sorted_ids[offset : offset + 40]), "fields": FIELDS,
        })
        raw_rows.extend(value for value in detail.values() if isinstance(value, Mapping))
    _add_rows(images, raw_rows)
    nearby_counts = Counter(row["sequence_id"] for row in images.values())
    expanded_ids: set[str] = set(images)
    expanded_sequences = sorted(sequence for sequence, count in nearby_counts.items() if count >= 3)
    for sequence_id in expanded_sequences:
        sequence = _get(session, "https://graph.mapillary.com/image_ids", {"sequence_id": sequence_id})
        expanded_ids.update(str(row["id"]) for row in sequence.get("data", []) if row.get("id"))
    missing = sorted(expanded_ids - set(images))
    expanded_rows = []
    for offset in range(0, len(missing), 40):
        detail = _get(session, "https://graph.mapillary.com/", {
            "ids": ",".join(missing[offset : offset + 40]), "fields": FIELDS,
        })
        expanded_rows.extend(value for value in detail.values() if isinstance(value, Mapping))
    _add_rows(images, expanded_rows)
    return {
        "schema_version": "blindassist_mapillary_sequence_metadata_v0",
        "goal_roster_selection": roster["selection"],
        "queries": queries,
        "expanded_sequence_ids": expanded_sequences,
        "images": sorted(images.values(), key=lambda row: (row["sequence_id"], row["captured_at_ms"], row["image_id"])),
        "pixel_download_count": 0,
        "model_call_count": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goal-roster", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output already exists")
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise ValueError("Mapillary access token unavailable")
    payload = acquire(json.loads(args.goal_roster.read_text(encoding="utf-8")), token)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"image_count": len(payload["images"]), "query_count": len(payload["queries"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
