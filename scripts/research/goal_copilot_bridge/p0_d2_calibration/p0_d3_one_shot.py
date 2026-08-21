"""Freeze and acquire the single P0-D3 targeted venue batch.

Roster selection uses Overture source metadata and building geometry only. It
must run before any new Mapillary pixels are fetched. Acquisition never runs an
entrance detector or model and never replaces a frozen parent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer, source_slice


POLICY_ID = "P0-D3-TARGETED-SET-VALUED-ONE-SHOT-CLOSURE-V1"
PARENT_COUNT = 6
MIN_BUILDING_AREA_M2 = 350.0
MIN_LONGEST_EDGE_M = 20.0
MAX_PARENTS_PER_CITY = 2
MAPILLARY_RADIUS_M = 50
MAPILLARY_LIMIT = 100
FRAMES_PER_PARENT = 4

FAMILY_PRIORITY = {
    "SHOPPING_MALL": 0,
    "TRANSIT_STATION": 0,
    "HOSPITAL": 1,
    "CIVIC_PUBLIC": 1,
    "UNIVERSITY": 1,
    "CINEMA_THEATRE": 1,
    "EVENT_VENUE": 1,
    "STADIUM_ARENA": 1,
    "DEPARTMENT_STORE": 2,
    "SUPERMARKET": 2,
    "HOTEL": 3,
}

EXACT_CATEGORY_FAMILY = {
    "shopping_mall": "SHOPPING_MALL",
    "shopping_center": "SHOPPING_MALL",
    "department_store": "DEPARTMENT_STORE",
    "supermarket": "SUPERMARKET",
    "hospital": "HOSPITAL",
    "college_university": "UNIVERSITY",
    "university": "UNIVERSITY",
    "college": "UNIVERSITY",
    "train_station": "TRANSIT_STATION",
    "subway_station": "TRANSIT_STATION",
    "metro_station": "TRANSIT_STATION",
    "hotel": "HOTEL",
    "civic_center": "CIVIC_PUBLIC",
    "government_office": "CIVIC_PUBLIC",
    "town_hall": "CIVIC_PUBLIC",
    "community_and_government": "CIVIC_PUBLIC",
    "movie_theater": "CINEMA_THEATRE",
    "cinema": "CINEMA_THEATRE",
    "theatre_venue": "CINEMA_THEATRE",
    "performing_arts_venue": "CINEMA_THEATRE",
    "event_venue": "EVENT_VENUE",
    "auditorium": "EVENT_VENUE",
    "convention_center": "EVENT_VENUE",
    "stadium_arena": "STADIUM_ARENA",
}

MAPILLARY_FIELDS = (
    "id", "captured_at", "camera_type", "camera_parameters", "computed_geometry",
    "computed_compass_angle", "compass_angle", "width", "height", "sequence",
)


class D3Error(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise D3Error(message)


def _normalized_name(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _place_name(feature: Mapping[str, Any]) -> str:
    properties = feature.get("properties", {})
    names = properties.get("names", {})
    if isinstance(names, Mapping) and names.get("primary"):
        return str(names["primary"]).strip()
    return str(properties.get("names.primary") or "").strip()


def _place_point(feature: Mapping[str, Any]) -> dict[str, float]:
    coordinates = feature.get("geometry", {}).get("coordinates")
    _require(isinstance(coordinates, list) and len(coordinates) >= 2, "place point missing")
    return {"lon": float(coordinates[0]), "lat": float(coordinates[1])}


def _category_values(properties: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()
    basic = properties.get("basic_category")
    if basic:
        values.add(str(basic))
    taxonomy = properties.get("taxonomy")
    if isinstance(taxonomy, Mapping):
        if taxonomy.get("primary"):
            values.add(str(taxonomy["primary"]))
        hierarchy = taxonomy.get("hierarchy")
        if isinstance(hierarchy, list):
            values.update(str(value) for value in hierarchy)
    categories = properties.get("categories")
    if isinstance(categories, Mapping) and categories.get("primary"):
        values.add(str(categories["primary"]))
    return values


def _family(properties: Mapping[str, Any]) -> str | None:
    families = {
        EXACT_CATEGORY_FAMILY[value]
        for value in _category_values(properties)
        if value in EXACT_CATEGORY_FAMILY
    }
    if not families:
        return None
    return min(families, key=lambda value: (FAMILY_PRIORITY[value], value))


def _rings(geometry: Mapping[str, Any]) -> list[list[list[float]]]:
    return list(source_slice._rings(geometry))  # source adapter owns Polygon/MultiPolygon normalization


def _building_metrics(geometry: Mapping[str, Any]) -> dict[str, float]:
    rings = _rings(geometry)
    _require(bool(rings), "building geometry has no exterior ring")
    all_points = [point for ring in rings for point in ring]
    origin = {
        "lon": sum(float(point[0]) for point in all_points) / len(all_points),
        "lat": sum(float(point[1]) for point in all_points) / len(all_points),
    }
    area = 0.0
    perimeter = 0.0
    longest = 0.0
    for ring in rings:
        xy = [source_slice._local_xy({"lon": point[0], "lat": point[1]}, origin) for point in ring]
        area += abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]))) / 2.0
        edges = [math.hypot(x2 - x1, y2 - y1) for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1])]
        perimeter += sum(edges)
        longest = max(longest, max(edges, default=0.0))
    return {"area_m2": round(area, 3), "perimeter_m": round(perimeter, 3), "longest_edge_m": round(longest, 3)}


def _excluded_values(cohorts: Sequence[Mapping[str, Any]], rosters: Sequence[Mapping[str, Any]]) -> tuple[set[str], set[str], set[str]]:
    names: set[str] = set()
    buildings: set[str] = set()
    places: set[str] = set()
    for cohort in cohorts:
        for episode in cohort.get("episodes", []):
            evaluator = episode.get("evaluator_episode", {})
            names.add(_normalized_name(evaluator.get("goal_spec", {}).get("target_name")))
            buildings.add(str(episode.get("target_building_id") or ""))
    for roster in rosters:
        buildings.update(str(value) for value in roster.get("excluded_target_building_ids", []))
        for parent in roster.get("parents", []):
            buildings.add(str(parent.get("building_id") or ""))
            places.add(str(parent.get("place_id") or ""))
            names.add(_normalized_name(parent.get("place_name")))
    return {value for value in names if value}, {value for value in buildings if value}, {value for value in places if value}


def plan_roster(
    slices: Sequence[Mapping[str, Any]],
    excluded_cohorts: Sequence[Mapping[str, Any]],
    excluded_rosters: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    _require(len(slices) == 6, "D3 geographic scope must contain exactly six frozen city slices")
    excluded_names, excluded_buildings, excluded_places = _excluded_values(excluded_cohorts, excluded_rosters)
    eligible: list[dict[str, Any]] = []
    source_identities = []
    for item in slices:
        city = str(item["city"])
        report = item["source_report"]
        places = {source_slice._feature_id(feature): feature for feature in item["places"].get("features", [])}
        buildings = {source_slice._feature_id(feature): feature for feature in item["buildings"].get("features", [])}
        _require(report.get("overture_release") == "2026-08-19.0", f"{city} Overture release drift")
        source_identities.append({
            "city": city,
            "source_report_sha256": report.get("report_sha256"),
            "places_sha256": report.get("source_files", {}).get("overture_places", {}).get("sha256"),
            "buildings_sha256": report.get("source_files", {}).get("overture_buildings", {}).get("sha256"),
        })
        by_building: dict[str, list[dict[str, Any]]] = {}
        for crosswalk in report.get("place_building_crosswalk_candidates", []):
            building_ids = crosswalk.get("building_ids", [])
            if crosswalk.get("status") != "CANDIDATE_ONLY" or len(building_ids) != 1:
                continue
            building_id, place_id = str(building_ids[0]), str(crosswalk["place_id"])
            feature, building = places.get(place_id), buildings.get(building_id)
            if not feature or not building or building_id in excluded_buildings or place_id in excluded_places:
                continue
            properties = feature.get("properties", {})
            family = _family(properties)
            name = _place_name(feature)
            if family is None or not name or _normalized_name(name) in excluded_names:
                continue
            metrics = _building_metrics(building.get("geometry", {}))
            if metrics["area_m2"] < MIN_BUILDING_AREA_M2 or metrics["longest_edge_m"] < MIN_LONGEST_EDGE_M:
                continue
            candidate = {
                "city": city,
                "building_id": building_id,
                "place_id": place_id,
                "place_name": name,
                "venue_family": family,
                "place_confidence": float(properties.get("confidence") or 0.0),
                "venue_coordinate": _place_point(feature),
                "building_metrics": metrics,
                "category_values": sorted(_category_values(properties)),
            }
            by_building.setdefault(building_id, []).append(candidate)
        for candidates in by_building.values():
            candidates.sort(key=lambda row: (
                FAMILY_PRIORITY[row["venue_family"]], -row["place_confidence"],
                _normalized_name(row["place_name"]), row["place_id"],
            ))
            eligible.append(candidates[0])
    eligible.sort(key=lambda row: (
        FAMILY_PRIORITY[row["venue_family"]], -row["building_metrics"]["area_m2"],
        -row["building_metrics"]["longest_edge_m"], -row["place_confidence"],
        row["city"], _normalized_name(row["place_name"]), row["place_id"], row["building_id"],
    ))
    selected = []
    city_counts: Counter[str] = Counter()
    for row in eligible:
        if city_counts[row["city"]] >= MAX_PARENTS_PER_CITY:
            continue
        selected.append(row)
        city_counts[row["city"]] += 1
        if len(selected) == PARENT_COUNT:
            break
    _require(len(selected) == PARENT_COUNT, "insufficient fixed-scope targeted venue parents")
    roster = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "data_role": "CONSUMED_DEVELOPMENT_ONE_SHOT_CLOSURE_NOT_ADJUDICATION",
        "selection_visibility": "OVERTURE_PLACE_TAXONOMY_AND_BUILDING_GEOMETRY_ONLY_NO_OSM_ENTRANCE_NO_MAPILLARY_NO_MODEL_NO_TRUTH",
        "batch_semantics": {
            "fixed_parent_count": PARENT_COUNT,
            "replacement_allowed": False,
            "second_batch_allowed": False,
            "stop_early_on_success_allowed": False,
        },
        "geographic_scope": sorted(str(item["city"]) for item in slices),
        "overture_release": "2026-08-19.0",
        "source_identities": sorted(source_identities, key=lambda row: row["city"]),
        "criteria": {
            "allowed_family_priority": FAMILY_PRIORITY,
            "minimum_building_area_m2": MIN_BUILDING_AREA_M2,
            "minimum_longest_edge_m": MIN_LONGEST_EDGE_M,
            "maximum_parents_per_city": MAX_PARENTS_PER_CITY,
            "osm_entrance_fields_used": False,
        },
        "excluded_name_count": len(excluded_names),
        "excluded_building_count": len(excluded_buildings),
        "excluded_place_count": len(excluded_places),
        "eligible_parent_count": len(eligible),
        "parents": selected,
        "mapillary_request_plan": {
            "endpoint": "https://graph.mapillary.com/images",
            "radius_m": MAPILLARY_RADIUS_M,
            "limit_per_parent": MAPILLARY_LIMIT,
            "frames_per_parent": FRAMES_PER_PARENT,
            "fields": list(MAPILLARY_FIELDS),
            "selection": "METADATA_ONLY_TARGET_FACING_SEQUENCE_DIVERSE_3M_SPACING_NO_REPLACEMENT",
        },
        "claim_ceiling": "CONSUMED_DEVELOPMENT_ONE_SHOT_DATA_CLOSURE_ONLY_NO_MODEL_OR_CALIBRATION_PERFORMANCE_CLAIM",
    }
    roster["report_sha256"] = materializer.content_sha256(roster)
    return roster


def _bearing_deg(source: Sequence[float], target: Mapping[str, float]) -> float:
    lon1, lat1 = map(math.radians, (float(source[0]), float(source[1])))
    lon2, lat2 = map(math.radians, (float(target["lon"]), float(target["lat"])))
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.degrees(math.atan2(x, y)) % 360.0


def _angle_error(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def select_frames(raw: Sequence[Mapping[str, Any]], target: Mapping[str, float]) -> list[dict[str, Any]]:
    normalized = []
    for item in raw:
        geometry = item.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        heading = item.get("computed_compass_angle", item.get("compass_angle"))
        sequence = item.get("sequence")
        if not isinstance(coordinates, list) or len(coordinates) != 2 or not isinstance(heading, (int, float)) or not sequence:
            continue
        camera_type = str(item.get("camera_type") or "").lower()
        distance = materializer.metric_distance_m(
            {"lon": float(coordinates[0]), "lat": float(coordinates[1])}, target,
        )
        if not 3.0 <= distance <= MAPILLARY_RADIUS_M:
            continue
        bearing = _bearing_deg(coordinates, target)
        error = 0.0 if camera_type in {"spherical", "equirectangular"} else _angle_error(float(heading), bearing)
        if camera_type not in {"spherical", "equirectangular"} and error > 75.0:
            continue
        normalized.append({
            "id": str(item["id"]),
            "sequence_id": str(sequence),
            "coordinates": [float(coordinates[0]), float(coordinates[1])],
            "captured_at": int(item.get("captured_at") or 0),
            "camera_type": camera_type,
            "heading_deg": float(heading) % 360.0,
            "target_bearing_deg": round(bearing, 6),
            "target_angle_error_deg": round(error, 6),
            "target_distance_m": round(distance, 6),
            "source_width": int(item.get("width") or 0),
            "source_height": int(item.get("height") or 0),
        })
    normalized.sort(key=lambda row: (
        row["target_angle_error_deg"], row["target_distance_m"], -row["captured_at"], row["id"],
    ))
    selected: list[dict[str, Any]] = []
    used_sequences: set[str] = set()
    for prefer_new_sequence in (True, False):
        for item in normalized:
            if item in selected or (prefer_new_sequence and item["sequence_id"] in used_sequences):
                continue
            if all(materializer.metric_distance_m(
                {"lon": item["coordinates"][0], "lat": item["coordinates"][1]},
                {"lon": prior["coordinates"][0], "lat": prior["coordinates"][1]},
            ) >= 3.0 for prior in selected):
                selected.append(item)
                used_sequences.add(item["sequence_id"])
            if len(selected) == FRAMES_PER_PARENT:
                return selected
    return selected


def _graph_get(session: requests.Session, endpoint: str, params: Mapping[str, Any]) -> Any:
    response = session.get(endpoint, params=params, timeout=60)
    response.raise_for_status()
    return response.json()


def acquire(roster: Mapping[str, Any], output_dir: Path, token: str) -> dict[str, Any]:
    _require(roster.get("policy_id") == POLICY_ID, "roster policy drift")
    _require(len(roster.get("parents", [])) == PARENT_COUNT, "roster parent count drift")
    output_dir.mkdir(parents=True, exist_ok=False)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    results = []
    for parent in roster["parents"]:
        target = parent["venue_coordinate"]
        response = _graph_get(session, "https://graph.mapillary.com/images", {
            "lat": target["lat"], "lng": target["lon"], "radius": MAPILLARY_RADIUS_M,
            "limit": MAPILLARY_LIMIT, "fields": ",".join(MAPILLARY_FIELDS),
        })
        raw = response.get("data", [])
        selected = select_frames(raw, target)
        if selected:
            detail = _graph_get(session, "https://graph.mapillary.com/", {
                "ids": ",".join(item["id"] for item in selected), "fields": "thumb_2048_url",
            })
        else:
            detail = {}
        for item in selected:
            url = detail.get(item["id"], {}).get("thumb_2048_url")
            _require(bool(url), f"selected frame {item['id']} lacks thumb_2048_url")
            path = image_dir / f"{item['id']}.jpg"
            response_image = requests.get(url, timeout=60)
            response_image.raise_for_status()
            temporary = path.with_suffix(".jpg.tmp")
            temporary.write_bytes(response_image.content)
            temporary.replace(path)
            payload = path.read_bytes()
            _require(payload.startswith(b"\xff\xd8") and payload.endswith(b"\xff\xd9"), f"selected frame {item['id']} is not a complete JPEG")
            item["path"] = str(path.resolve())
            item["image_sha256"] = hashlib.sha256(payload).hexdigest()
        results.append({
            "building_id": parent["building_id"], "place_id": parent["place_id"],
            "place_name": parent["place_name"], "raw_result_count": len(raw),
            "selected_frame_count": len(selected), "frames": selected,
            "status": "MATERIALIZED" if selected else "NO_ELIGIBLE_FRAME_NO_REPLACEMENT",
        })
    report = {
        "schema_version": 1,
        "policy_id": POLICY_ID,
        "roster_sha256": roster["report_sha256"],
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "request_plan": roster["mapillary_request_plan"],
        "parent_count": PARENT_COUNT,
        "materialized_parent_count": sum(item["status"] == "MATERIALIZED" for item in results),
        "selected_frame_count": sum(item["selected_frame_count"] for item in results),
        "replacement_performed": False,
        "parents": results,
        "claim_ceiling": roster["claim_ceiling"],
    }
    report["report_sha256"] = materializer.content_sha256(report)
    materializer.write_json(output_dir / "acquisition.json", report)
    return report


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan")
    plan.add_argument("--slice", action="append", required=True, help="CITY=source-slice-root")
    plan.add_argument("--exclude-cohort", action="append", type=Path, default=[])
    plan.add_argument("--exclude-roster", action="append", type=Path, default=[])
    plan.add_argument("--output", required=True, type=Path)
    acquisition = subparsers.add_parser("acquire")
    acquisition.add_argument("--roster", required=True, type=Path)
    acquisition.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "plan":
        slices = []
        for value in args.slice:
            city, root_text = value.split("=", 1)
            root = Path(root_text)
            slices.append({
                "city": city,
                "source_report": _load(root / "source-report.json"),
                "places": _load(root / "overture-places.geojson"),
                "buildings": _load(root / "overture-buildings.geojson"),
            })
        roster = plan_roster(slices, [_load(path) for path in args.exclude_cohort], [_load(path) for path in args.exclude_roster])
        materializer.write_json(args.output, roster)
        print(json.dumps({"parents": [{key: row[key] for key in ("city", "place_name", "venue_family", "building_id")} for row in roster["parents"]], "report_sha256": roster["report_sha256"]}, ensure_ascii=False, indent=2))
        return 0
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    _require(bool(token), "MAPILLARY_ACCESS_TOKEN missing from process environment")
    report = acquire(_load(args.roster), args.output_dir, str(token))
    print(json.dumps({key: report[key] for key in ("materialized_parent_count", "selected_frame_count", "report_sha256")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
