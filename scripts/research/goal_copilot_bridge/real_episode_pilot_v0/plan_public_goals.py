"""Freeze unused Overture place + OSM entrance goals before Mapillary access."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


VENUE_TERMS = {
    "museum", "hospital", "university", "college", "hotel", "shopping_mall", "shopping_center",
    "train_station", "subway_station", "metro_station", "cinema", "movie_theater", "theatre_venue",
    "government_office", "town_hall", "civic_center", "community_center", "stadium_arena",
    "convention_center", "supermarket", "department_store", "library",
}


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _used_buildings(metadata_root: Path) -> set[str]:
    used: set[str] = set()
    for path in sorted(metadata_root.rglob("mapillary_metadata.json")):
        value = _read(path)
        for row in value.get("images", []):
            for key in ("target_building_id", "overture_building_id", "building_id"):
                if row.get(key):
                    used.add(str(row[key]))
    for path in sorted(metadata_root.rglob("roster.json")):
        value = _read(path)
        for row in value.get("parents", []):
            if row.get("building_id"):
                used.add(str(row["building_id"]))
    for path in sorted(metadata_root.rglob("goal-roster.json")):
        value = _read(path)
        for row in value.get("goals", []):
            if row.get("overture_building_id"):
                used.add(str(row["overture_building_id"]))
    return used


def _venue_family(feature: dict[str, Any]) -> str | None:
    properties = feature.get("properties", {})
    values = {str(properties.get("basic_category") or "")}
    taxonomy = properties.get("taxonomy") or {}
    values.add(str(taxonomy.get("primary") or ""))
    values.update(str(item) for item in taxonomy.get("hierarchy", []) or [])
    matches = sorted(values & VENUE_TERMS)
    return matches[0] if matches else None


def plan(source_root: Path, metadata_root: Path, count: int = 4) -> dict[str, Any]:
    used = _used_buildings(metadata_root)
    eligible = []
    source_receipts = []
    for report_path in sorted(source_root.glob("*-source-slice/source-report.json")):
        report = _read(report_path)
        city = report_path.parent.name.removeprefix("2026-08-21-").removesuffix("-source-slice")
        place_features = {
            str(row["id"]): row
            for row in _read(report_path.parent / "overture-places.geojson").get("features", [])
            if row.get("id")
        }
        places = {
            str(row["building_ids"][0]): row
            for row in report.get("place_building_crosswalk_candidates", [])
            if row.get("status") == "CANDIDATE_ONLY"
            and len(row.get("building_ids", [])) == 1
            and row.get("place_name")
            and row.get("place_id") in place_features
            and _venue_family(place_features[str(row["place_id"])]) is not None
        }
        entrances: dict[str, list[dict[str, Any]]] = {}
        for row in report.get("osm_entrance_building_crosswalk_candidates", []):
            building = str(row.get("overture_building_id", ""))
            if row.get("status") != "CANDIDATE_ONLY" or building not in places or building in used:
                continue
            entrances.setdefault(building, []).append({
                "candidate_id": str(row["osm_entrance_id"]),
                "coordinates": [float(row["point"]["lon"]), float(row["point"]["lat"])],
                "entrance_tag": str(row.get("entrance") or "yes"),
                "access_tag": row.get("access"),
                "authority": "OSM_OVERTURE_SOURCE_CROSSWALK_PRETRUTH",
            })
        for building, candidates in entrances.items():
            place = places[building]
            eligible.append({
                "goal_id": f"public-real-{city}-{building[:8]}",
                "goal_type": "NAMED_BUILDING_ENTRANCE",
                "target_name": str(place["place_name"]),
                "city": city,
                "overture_building_id": building,
                "overture_place_id": str(place["place_id"]),
                "venue_family": _venue_family(place_features[str(place["place_id"])]),
                "public_entrance_candidates": sorted(candidates, key=lambda row: row["candidate_id"]),
            })
        source_receipts.append({"path": str(report_path.resolve()), "sha256": _sha(report_path)})
    eligible.sort(key=lambda row: (row["city"], row["target_name"].casefold(), row["overture_building_id"]))
    selected, cities = [], set()
    for row in eligible:
        if row["city"] in cities:
            continue
        selected.append(row)
        cities.add(row["city"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError("insufficient unused crosswalked public goals")
    return {
        "schema_version": "blindassist_public_goal_roster_v0",
        "selection": "LEXICAL_CITY_THEN_PLACE_THEN_BUILDING_ONE_PER_CITY_EXCLUDING_PRIOR_MAPILLARY_AND_ROSTERS",
        "precedence": {
            "mapillary_metadata_accessed": False,
            "mapillary_pixels_accessed": False,
            "model_outputs_created": False,
            "evaluator_truth_created": False,
        },
        "used_building_exclusion_count": len(used),
        "eligible_goal_count": len(eligible),
        "source_receipts": source_receipts,
        "goals": selected,
        "claim_ceiling": "PUBLIC_GOAL_ROSTER_ONLY_NO_VISIBILITY_PERFORMANCE_OR_USER_CLAIM",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("output already exists")
    payload = plan(args.source_root, args.metadata_root, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"goal_count": len(payload["goals"]), "eligible_goal_count": payload["eligible_goal_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
