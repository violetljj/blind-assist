#!/usr/bin/env python3
"""Acquire a PA3 observation cohort after goal freeze, without pixel-based selection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import requests

from scripts.research.goal_copilot_bridge.p1_proposal_availability.materialize_pa3_inputs import CAPTURE_SCHEMA
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import content_sha256, sha256


PLAN_SCHEMA = "blindassist_p1_pa3_public_observation_acquisition_plan_v1"
ACQUISITION_SCHEMA = "blindassist_p1_pa3_public_observation_acquisition_v1"
ROSTER_SCHEMA = "blindassist_p1_pa3_anchor_roster_v1"
AMENDMENT_SCHEMA = "blindassist_p1_pa3_mapillary_transport_amendment_v1"
USER_AGENT = "BlindAssist-PA3-research/1.0 (local outcome-blind cohort materialization)"


class AcquisitionError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionError(message)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _distance_m(left: tuple[float, float], right: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (left[0], left[1], right[0], right[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6_371_000.0 * math.asin(math.sqrt(value))


def _bearing_deg(origin: tuple[float, float], target: tuple[float, float]) -> float:
    lat1, lat2 = math.radians(origin[0]), math.radians(target[0])
    delta_lon = math.radians(target[1] - origin[1])
    y = math.sin(delta_lon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def select_frame(raw: Sequence[Mapping[str, Any]], anchor: Mapping[str, Any], policy: Mapping[str, Any]) -> tuple[dict[str, Any] | None, int]:
    candidates: list[dict[str, Any]] = []
    anchor_point = (float(anchor["lat"]), float(anchor["lon"]))
    for item in raw:
        geometry = item.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        heading = item.get("computed_compass_angle") if item.get("computed_compass_angle") is not None else item.get("compass_angle")
        if not isinstance(coordinates, list) or len(coordinates) != 2 or heading is None:
            continue
        camera_type = str(item.get("camera_type") or "").lower()
        is_pano = item.get("is_pano") is True or camera_type in {"spherical", "equirectangular"}
        if is_pano and policy.get("panoramas_allowed") is False:
            continue
        camera = (float(coordinates[1]), float(coordinates[0]))
        distance = _distance_m(camera, anchor_point)
        bearing = _bearing_deg(camera, anchor_point)
        error = _angle_error(float(heading), bearing)
        if not float(policy["minimum_distance_m"]) <= distance <= float(policy["maximum_distance_m"]):
            continue
        if error > float(policy["maximum_absolute_bearing_error_deg"]):
            continue
        candidates.append({
            "image_id": str(item["id"]),
            "sequence_id": str(item.get("sequence") or "UNKNOWN"),
            "source_captured_at_ms": int(item.get("captured_at") or 0),
            "camera_lat": camera[0],
            "camera_lon": camera[1],
            "compass_angle_deg": float(heading),
            "target_bearing_deg": bearing,
            "absolute_bearing_error_deg": error,
            "target_distance_m": distance,
            "is_pano": is_pano,
        })
    candidates.sort(key=lambda row: (
        row["absolute_bearing_error_deg"],
        abs(row["target_distance_m"] - 20.0),
        row["source_captured_at_ms"],
        row["image_id"],
    ))
    return (candidates[0] if candidates else None), len(candidates)


def _graph_get(session: requests.Session, endpoint: str, params: Mapping[str, Any]) -> Any:
    response = session.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def acquire(plan_path: Path, c0_path: Path, output_dir: Path, token: str, amendment_path: Path | None = None) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    c0 = json.loads(c0_path.read_text(encoding="utf-8"))
    _require(plan.get("schema_version") == PLAN_SCHEMA, "acquisition plan schema mismatch")
    _require(plan.get("goal_receipt_file_sha256") == sha256(c0_path), "goal receipt file binding mismatch")
    _require(plan.get("goal_receipt_body_sha256") == c0.get("receipt_body_sha256"), "goal receipt body binding mismatch")
    _require(plan.get("pixel_state_at_freeze") == "NOT_ACCESSED_FOR_THIS_COHORT", "pixel precedence drift")
    _require(plan.get("truth_state_at_freeze") == "NOT_CREATED", "truth precedence drift")
    effective_query = plan["mapillary_query"]
    amendment_sha = None
    if amendment_path is not None:
        amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
        _require(amendment.get("schema_version") == AMENDMENT_SCHEMA, "transport amendment schema mismatch")
        _require(amendment.get("original_plan_sha256") == sha256(plan_path), "transport amendment plan binding mismatch")
        diagnostic = amendment.get("diagnostic", {})
        _require(diagnostic.get("pixel_content_accessed") is False and diagnostic.get("truth_accessed") is False and diagnostic.get("provider_run") is False, "transport amendment was not outcome-blind")
        effective_query = amendment["effective_mapillary_query"]
        amendment_sha = sha256(amendment_path)
    goal_ids = {row["episode_id"] for row in c0.get("episodes", [])}
    queries = plan.get("geocoding", {}).get("queries", [])
    _require(bool(queries) and {row["episode_id"] for row in queries} == goal_ids, "goal/geocoding roster mismatch")

    resume_from_roster = output_dir.exists()
    if resume_from_roster:
        _require((output_dir / "anchor_roster.json").is_file(), "existing acquisition directory lacks frozen roster")
        _require(not (output_dir / "acquisition.json").exists(), "acquisition is already terminal")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Authorization": f"OAuth {token}"})
    roster_path = output_dir / "anchor_roster.json"
    if resume_from_roster:
        roster = json.loads(roster_path.read_text(encoding="utf-8"))
        _require(roster.get("schema_version") == ROSTER_SCHEMA and roster.get("plan_sha256") == sha256(plan_path), "frozen roster binding mismatch")
        anchors = roster["anchors"]
    else:
        anchors = []
        for index, item in enumerate(queries):
            response = session.get("https://nominatim.openstreetmap.org/search", params={
                "q": item["query"], "format": "jsonv2", "limit": 1,
                "countrycodes": plan["geocoding"]["countrycodes"],
            }, timeout=60)
            response.raise_for_status()
            matches = response.json()
            _require(len(matches) == 1, f"geocoder did not resolve exactly one anchor: {item['episode_id']}")
            match = matches[0]
            anchors.append({
                "episode_id": item["episode_id"], "query": item["query"],
                "lat": float(match["lat"]), "lon": float(match["lon"]),
                "osm_type": match.get("osm_type"), "osm_id": match.get("osm_id"),
                "display_name": match.get("display_name"),
            })
            if index + 1 < len(queries):
                time.sleep(1.05)
        roster = {
            "schema_version": ROSTER_SCHEMA,
            "plan_sha256": sha256(plan_path),
            "goal_receipt_file_sha256": sha256(c0_path),
            "frozen_before_mapillary_metadata_or_pixels": True,
            "anchors": anchors,
        }
        roster["roster_body_sha256"] = content_sha256(roster)
        _atomic_json(roster_path, roster)

    image_dir = output_dir / "images"
    if resume_from_roster and image_dir.exists():
        _require(not any(image_dir.iterdir()), "resume is only allowed before the first pixel download")
    image_dir.mkdir(exist_ok=True)
    results = []
    capture_cases = []
    for anchor in anchors:
        query = effective_query
        response = _graph_get(session, "https://graph.mapillary.com/images", {
            "lat": anchor["lat"], "lng": anchor["lon"], "radius": query["radius_m"],
            "limit": query["limit"], "fields": ",".join(query["fields"]),
        })
        raw = response.get("data", [])
        selected, eligible_count = select_frame(raw, anchor, plan["outcome_blind_selection"])
        if selected is None:
            results.append({
                "episode_id": anchor["episode_id"], "raw_metadata_count": len(raw),
                "geometry_eligible_count": 0, "status": "NO_GEOMETRIC_CANDIDATE_NO_REPLACEMENT",
            })
            continue
        details = _graph_get(session, "https://graph.mapillary.com/", {
            "ids": selected["image_id"], "fields": "thumb_2048_url",
        })
        url = details.get(selected["image_id"], {}).get("thumb_2048_url")
        _require(bool(url), f"selected frame lacks thumb URL: {selected['image_id']}")
        image_response = requests.get(url, timeout=60, headers={"User-Agent": USER_AGENT})
        image_response.raise_for_status()
        payload = image_response.content
        _require(payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9"), "downloaded frame is not a complete JPEG")
        image_path = image_dir / f"{selected['image_id']}.jpg"
        temporary = image_path.with_suffix(".jpg.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, image_path)
        acquired_at = datetime.now(timezone.utc).isoformat()
        source_capture = datetime.fromtimestamp(selected["source_captured_at_ms"] / 1000.0, tz=timezone.utc).isoformat()
        selected.update({"image_path": str(image_path.resolve()), "image_sha256": hashlib.sha256(payload).hexdigest()})
        results.append({
            "episode_id": anchor["episode_id"], "raw_metadata_count": len(raw),
            "geometry_eligible_count": eligible_count, "status": "MATERIALIZED", "selected": selected,
        })
        capture_cases.append({
            "case_id": f"{anchor['episode_id']}-frame-01", "episode_id": anchor["episode_id"],
            "capture_created_at_utc": acquired_at,
            "capture_time_semantics": "FIRST_PROJECT_PIXEL_ACCESS_NOT_PHYSICAL_CAMERA_CAPTURE",
            "source_captured_at_utc": source_capture,
            "image_path": str(image_path.resolve()), "image_sha256": selected["image_sha256"],
            "mapillary_image_id": selected["image_id"], "mapillary_sequence_id": selected["sequence_id"],
        })
    acquisition = {
        "schema_version": ACQUISITION_SCHEMA,
        "plan_sha256": sha256(plan_path), "anchor_roster_file_sha256": sha256(roster_path),
        "transport_amendment_sha256": amendment_sha,
        "precedence_mode": plan["precedence_mode"], "physical_capture_after_goal_claimed": False,
        "episode_count": len(anchors), "materialized_case_count": len(capture_cases),
        "replacement_or_resampling_performed": False, "results": results,
    }
    acquisition["acquisition_body_sha256"] = content_sha256(acquisition)
    _atomic_json(output_dir / "acquisition.json", acquisition)
    capture_manifest = {
        "schema_version": CAPTURE_SCHEMA,
        "precedence_mode": plan["precedence_mode"],
        "physical_capture_after_goal_claimed": False,
        "acquisition_body_sha256": acquisition["acquisition_body_sha256"],
        "cases": capture_cases,
    }
    _atomic_json(output_dir / "capture_manifest.json", capture_manifest)
    return acquisition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--c0-receipt", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--transport-amendment", type=Path)
    args = parser.parse_args()
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    _require(bool(token), "MAPILLARY_ACCESS_TOKEN missing")
    result = acquire(args.plan, args.c0_receipt, args.output_dir, str(token), args.transport_amendment)
    print(json.dumps({
        "episode_count": result["episode_count"],
        "materialized_case_count": result["materialized_case_count"],
        "replacement_or_resampling_performed": result["replacement_or_resampling_performed"],
        "acquisition_body_sha256": result["acquisition_body_sha256"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
