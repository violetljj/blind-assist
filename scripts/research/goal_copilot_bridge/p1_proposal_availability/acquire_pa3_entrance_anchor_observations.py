#!/usr/bin/env python3
"""Acquire a fresh PA3 development cohort around frozen OSM entrance nodes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import requests

from scripts.research.goal_copilot_bridge.p1_proposal_availability.acquire_pa3_public_observations import (
    USER_AGENT,
    _atomic_json,
    _distance_m,
    _graph_get,
    _require,
    select_frames,
)
from scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_inputs import CAPTURE_SCHEMA
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256, sha256


PLAN_SCHEMA = "blindassist_p1_pa3_entrance_anchor_acquisition_plan_v1"
ROSTER_SCHEMA = "blindassist_p1_pa3_osm_entrance_anchor_roster_v1"
ACQUISITION_SCHEMA = "blindassist_p1_pa3_entrance_anchor_observation_acquisition_v1"


def _entrance_priority(tags: Mapping[str, Any]) -> int:
    value = str(tags.get("entrance") or "").lower()
    return 0 if value == "main" else 1 if value == "yes" else 2


def resolve_entrance(raw: list[Mapping[str, Any]], place: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = []
    origin = (float(place["lat"]), float(place["lon"]))
    for item in raw:
        tags = item.get("tags") if isinstance(item.get("tags"), Mapping) else {}
        if not tags.get("entrance") or str(tags.get("access") or "").lower() in {"private", "no"}:
            continue
        lat, lon = float(item["lat"]), float(item["lon"])
        candidates.append({
            "osm_node_id": int(item["id"]), "lat": lat, "lon": lon,
            "entrance_tag": str(tags["entrance"]), "access_tag": tags.get("access"),
            "distance_to_place_anchor_m": _distance_m(origin, (lat, lon)),
            "priority": _entrance_priority(tags),
        })
    candidates.sort(key=lambda row: (row["priority"], row["distance_to_place_anchor_m"], row["osm_node_id"]))
    return candidates[0] if candidates else None


def osm_map_entrance_nodes(
    session: requests.Session,
    place: Mapping[str, Any],
    radius_m: int,
    endpoint: str,
) -> list[dict[str, Any]]:
    """Read entrance nodes from a small OSM map bbox and retain the circular radius."""
    lat = float(place["lat"])
    lon = float(place["lon"])
    lat_delta = radius_m / 111_320.0
    lon_delta = radius_m / (111_320.0 * max(0.01, abs(math.cos(math.radians(lat)))))
    bbox = f"{lon - lon_delta},{lat - lat_delta},{lon + lon_delta},{lat + lat_delta}"
    response = session.get(endpoint, params={"bbox": bbox}, timeout=90)
    response.raise_for_status()
    root = ET.fromstring(response.content)
    rows = []
    for node in root.findall("node"):
        node_lat = float(node.attrib["lat"])
        node_lon = float(node.attrib["lon"])
        if _distance_m((lat, lon), (node_lat, node_lon)) > radius_m:
            continue
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        if tags.get("entrance"):
            rows.append({"id": int(node.attrib["id"]), "lat": node_lat, "lon": node_lon, "tags": tags})
    return rows


def acquire(plan_path: Path, c0_path: Path, output_dir: Path, token: str) -> dict[str, Any]:
    _require(not output_dir.exists(), "entrance-anchor acquisition output already exists")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    _require(plan.get("schema_version") == PLAN_SCHEMA, "entrance-anchor plan schema mismatch")
    _require(plan.get("goal_receipt_file_sha256") == sha256(c0_path), "goal receipt file binding mismatch")
    _require(plan.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "goal receipt body binding mismatch")
    _require(plan.get("pixel_state_at_freeze") == "NOT_ACCESSED_FOR_THIS_COHORT", "pixel precedence drift")
    _require(plan.get("truth_state_at_freeze") == "NOT_CREATED", "truth precedence drift")
    goal_ids = {row["episode_id"] for row in c0["episodes"]}
    queries = plan["geocoding"]["queries"]
    _require({row["episode_id"] for row in queries} == goal_ids, "goal/query roster mismatch")

    output_dir.mkdir(parents=True, exist_ok=False)
    public_session = requests.Session()
    public_session.headers["User-Agent"] = USER_AGENT
    place_anchors = []
    for index, item in enumerate(queries):
        response = public_session.get("https://nominatim.openstreetmap.org/search", params={
            "q": item["query"], "format": "jsonv2", "limit": 1,
            "countrycodes": plan["geocoding"]["countrycodes"],
        }, timeout=60)
        response.raise_for_status()
        matches = response.json()
        _require(len(matches) == 1, f"geocoder did not resolve exactly one place: {item['episode_id']}")
        place_anchors.append({
            "episode_id": item["episode_id"], "query": item["query"],
            "lat": float(matches[0]["lat"]), "lon": float(matches[0]["lon"]),
            "display_name": matches[0].get("display_name"),
        })
        if index + 1 < len(queries):
            time.sleep(1.05)

    resolved = []
    radius = int(plan["entrance_anchor"]["search_radius_m"])
    backend = str(plan["entrance_anchor"].get("backend", "OVERPASS"))
    endpoint = str(plan["entrance_anchor"].get("endpoint", "https://overpass-api.de/api/interpreter"))
    _require(endpoint.startswith("https://"), "OSM entrance endpoint must use HTTPS")
    for index, place in enumerate(place_anchors):
        if backend == "OSM_MAP_API":
            raw = osm_map_entrance_nodes(public_session, place, radius, endpoint)
        else:
            _require(backend == "OVERPASS", "unsupported OSM entrance backend")
            query = f'[out:json][timeout:25];node(around:{radius},{place["lat"]},{place["lon"]})["entrance"];out body;'
            response = public_session.post(endpoint, data={"data": query}, timeout=90)
            response.raise_for_status()
            raw = response.json().get("elements", [])
        entrance = resolve_entrance(raw, place)
        resolved.append({
            **place,
            "entrance_candidate_count": len(raw),
            "selected_entrance": entrance,
            "status": "OSM_ENTRANCE_FROZEN" if entrance else "NO_OSM_ENTRANCE_NO_REPLACEMENT",
        })
        if index + 1 < len(place_anchors):
            time.sleep(1.0)
    roster = {
        "schema_version": ROSTER_SCHEMA, "plan_sha256": sha256(plan_path),
        "goal_receipt_file_sha256": sha256(c0_path),
        "frozen_before_mapillary_metadata_or_pixels": True, "episodes": resolved,
    }
    roster["roster_body_sha256"] = content_sha256(roster)
    roster_path = output_dir / "entrance_anchor_roster.json"
    _atomic_json(roster_path, roster)

    image_dir = output_dir / "images"
    image_dir.mkdir()
    mapillary = requests.Session()
    mapillary.headers.update({"User-Agent": USER_AGENT, "Authorization": f"OAuth {token}"})
    results = []
    capture_cases = []
    for row in resolved:
        entrance = row["selected_entrance"]
        if entrance is None:
            results.append({"episode_id": row["episode_id"], "status": row["status"]})
            continue
        query = plan["mapillary_query"]
        response = _graph_get(mapillary, "https://graph.mapillary.com/images", {
            "lat": entrance["lat"], "lng": entrance["lon"], "radius": query["radius_m"],
            "limit": query["limit"], "fields": ",".join(query["fields"]),
        })
        raw = response.get("data", [])
        selected_frames, eligible_count = select_frames(raw, entrance, plan["outcome_blind_selection"])
        if not selected_frames:
            results.append({
                "episode_id": row["episode_id"], "status": "NO_GEOMETRIC_CANDIDATE_NO_REPLACEMENT",
                "raw_metadata_count": len(raw), "geometry_eligible_count": 0,
            })
            continue
        materialized = []
        for frame_index, selected in enumerate(selected_frames, start=1):
            detail = _graph_get(mapillary, "https://graph.mapillary.com/", {"ids": selected["image_id"], "fields": "thumb_2048_url"})
            url = detail.get(selected["image_id"], {}).get("thumb_2048_url")
            _require(bool(url), f"selected frame lacks thumb URL: {selected['image_id']}")
            response_image = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
            response_image.raise_for_status()
            payload = response_image.content
            _require(payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9"), "downloaded frame is not a complete JPEG")
            path = image_dir / f"{selected['image_id']}.jpg"
            temporary = path.with_suffix(".jpg.tmp")
            temporary.write_bytes(payload)
            os.replace(temporary, path)
            acquired_at = datetime.now(timezone.utc).isoformat()
            source_capture = datetime.fromtimestamp(selected["source_captured_at_ms"] / 1000.0, tz=timezone.utc).isoformat()
            image_sha = hashlib.sha256(payload).hexdigest()
            selected.update({"image_path": str(path.resolve()), "image_sha256": image_sha})
            materialized.append(selected)
            capture_cases.append({
                "case_id": f"{row['episode_id']}-frame-{frame_index:02d}", "episode_id": row["episode_id"],
                "capture_created_at_utc": acquired_at,
                "capture_time_semantics": "FIRST_PROJECT_PIXEL_ACCESS_NOT_PHYSICAL_CAMERA_CAPTURE",
                "source_captured_at_utc": source_capture,
                "image_path": str(path.resolve()), "image_sha256": image_sha,
                "mapillary_image_id": selected["image_id"], "mapillary_sequence_id": selected["sequence_id"],
                "osm_entrance_node_id": entrance["osm_node_id"],
            })
        results.append({
            "episode_id": row["episode_id"], "status": "MATERIALIZED",
            "raw_metadata_count": len(raw), "geometry_eligible_count": eligible_count,
            "selected": materialized[0],
            "selected_frame_count": len(materialized), "selected_frames": materialized,
        })
    acquisition = {
        "schema_version": ACQUISITION_SCHEMA, "plan_sha256": sha256(plan_path),
        "entrance_anchor_roster_file_sha256": sha256(roster_path),
        "precedence_mode": plan["precedence_mode"], "physical_capture_after_goal_claimed": False,
        "episode_count": len(resolved), "materialized_case_count": len(capture_cases),
        "replacement_or_resampling_performed": False, "results": results,
    }
    acquisition["acquisition_body_sha256"] = content_sha256(acquisition)
    _atomic_json(output_dir / "acquisition.json", acquisition)
    _atomic_json(output_dir / "capture_manifest.json", {
        "schema_version": CAPTURE_SCHEMA, "precedence_mode": plan["precedence_mode"],
        "physical_capture_after_goal_claimed": False,
        "acquisition_body_sha256": acquisition["acquisition_body_sha256"], "cases": capture_cases,
    })
    return acquisition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--c0-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    _require(bool(token), "MAPILLARY_ACCESS_TOKEN missing")
    result = acquire(args.plan, args.c0_receipt, args.output_dir, str(token))
    print(json.dumps({
        "episode_count": result["episode_count"], "materialized_case_count": result["materialized_case_count"],
        "replacement_or_resampling_performed": result["replacement_or_resampling_performed"],
        "acquisition_body_sha256": result["acquisition_body_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
