"""Normalize and summarize a bounded Overture/OSM P0-S0 source slice.

This adapter emits crosswalk *candidates*.  It never upgrades them to a silver
label without Mapillary frames and protocol-conforming visual candidates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from .materializer import content_sha256, metric_distance_m, write_json


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_geojson(path: Path) -> list[Mapping[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("type") != "FeatureCollection" or not isinstance(value.get("features"), list):
        raise ValueError(f"{path} is not a GeoJSON FeatureCollection")
    return value["features"]


def _rings(geometry: Mapping[str, Any]) -> Iterable[list[list[float]]]:
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates")
    if geometry_type == "Polygon" and isinstance(coordinates, list):
        yield from coordinates[:1]
    elif geometry_type == "MultiPolygon" and isinstance(coordinates, list):
        for polygon in coordinates:
            if polygon:
                yield polygon[0]


def point_in_ring(point: Mapping[str, float], ring: Sequence[Sequence[float]]) -> bool:
    x, y = float(point["lon"]), float(point["lat"])
    inside = False
    for index, current in enumerate(ring):
        previous = ring[index - 1]
        x1, y1 = float(previous[0]), float(previous[1])
        x2, y2 = float(current[0]), float(current[1])
        intersects = (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1
        if intersects:
            inside = not inside
    return inside


def point_in_geometry(point: Mapping[str, float], geometry: Mapping[str, Any]) -> bool:
    return any(point_in_ring(point, ring) for ring in _rings(geometry))


def _local_xy(point: Mapping[str, float], origin: Mapping[str, float]) -> tuple[float, float]:
    mean_lat = math.radians((float(point["lat"]) + float(origin["lat"])) / 2.0)
    x = math.radians(float(point["lon"]) - float(origin["lon"])) * math.cos(mean_lat) * 6_371_008.8
    y = math.radians(float(point["lat"]) - float(origin["lat"])) * 6_371_008.8
    return x, y


def point_to_ring_distance_m(point: Mapping[str, float], ring: Sequence[Sequence[float]]) -> float:
    px, py = 0.0, 0.0
    best = math.inf
    for index, current in enumerate(ring):
        previous = ring[index - 1]
        a = _local_xy({"lon": previous[0], "lat": previous[1]}, point)
        b = _local_xy({"lon": current[0], "lat": current[1]}, point)
        abx, aby = b[0] - a[0], b[1] - a[1]
        denominator = abx * abx + aby * aby
        t = 0.0 if denominator == 0.0 else max(0.0, min(1.0, (-(a[0]) * abx - a[1] * aby) / denominator))
        nearest = (a[0] + t * abx, a[1] + t * aby)
        best = min(best, math.hypot(nearest[0] - px, nearest[1] - py))
    return best


def point_to_geometry_boundary_m(point: Mapping[str, float], geometry: Mapping[str, Any]) -> float:
    distances = [point_to_ring_distance_m(point, ring) for ring in _rings(geometry)]
    return min(distances, default=math.inf)


def _feature_id(feature: Mapping[str, Any]) -> str:
    return str(feature.get("id") or feature.get("properties", {}).get("id") or "")


def _place_name(feature: Mapping[str, Any]) -> str | None:
    properties = feature.get("properties", {})
    flattened = properties.get("names.primary")
    if isinstance(flattened, str) and flattened:
        return flattened
    names = properties.get("names")
    if isinstance(names, Mapping):
        primary = names.get("primary")
        if isinstance(primary, str) and primary:
            return primary
    return None


def _point_from_feature(feature: Mapping[str, Any]) -> dict[str, float] | None:
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates")
    if geometry.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) < 2:
        return None
    return {"lon": float(coordinates[0]), "lat": float(coordinates[1])}


def parse_osm(path: Path) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    nodes: dict[str, dict[str, Any]] = {}
    entrances: list[dict[str, Any]] = []
    root = ET.parse(path).getroot()
    for node in root.findall("node"):
        node_id = str(node.attrib["id"])
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in node.findall("tag")}
        value = {"lon": float(node.attrib["lon"]), "lat": float(node.attrib["lat"]), "tags": tags}
        nodes[node_id] = value
        if "entrance" in tags:
            entrances.append({"osm_entrance_id": f"node/{node_id}", "point": {"lon": value["lon"], "lat": value["lat"]}, "entrance": tags["entrance"], "access": tags.get("access")})
    building_ways = 0
    node_building_ways: dict[str, list[str]] = {}
    for way in root.findall("way"):
        tags = {tag.attrib["k"]: tag.attrib["v"] for tag in way.findall("tag")}
        if "building" in tags:
            building_ways += 1
            way_id = str(way.attrib["id"])
            for item in way.findall("nd"):
                node_building_ways.setdefault(item.attrib["ref"], []).append(way_id)
    for entrance in entrances:
        node_id = entrance["osm_entrance_id"].split("/", 1)[1]
        entrance["osm_building_way_ids"] = sorted(node_building_ways.get(node_id, []))
        entrance["on_osm_building_way"] = bool(entrance["osm_building_way_ids"])
    metadata = {
        "generator": root.attrib.get("generator"),
        "copyright": root.attrib.get("copyright"),
        "attribution": root.attrib.get("attribution"),
        "license": root.attrib.get("license"),
        "bounds": dict(root.find("bounds").attrib) if root.find("bounds") is not None else None,
    }
    return nodes, entrances, {"building_way_count": building_ways, **metadata}


def overture_osm_way_ids(feature: Mapping[str, Any]) -> set[str]:
    result: set[str] = set()
    sources = feature.get("properties", {}).get("sources")
    if not isinstance(sources, list):
        return result
    for source in sources:
        if not isinstance(source, Mapping) or str(source.get("dataset", "")).lower() != "openstreetmap":
            continue
        match = re.fullmatch(r"w(\d+)(?:@\d+)?", str(source.get("record_id", "")))
        if match:
            result.add(match.group(1))
    return result


def overture_license_summary(features: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for feature in features:
        sources = feature.get("properties", {}).get("sources")
        if not isinstance(sources, list):
            continue
        for source in sources:
            if not isinstance(source, Mapping):
                continue
            key = (str(source.get("dataset") or "UNKNOWN"), str(source.get("license") or "UNKNOWN"))
            counts[key] = counts.get(key, 0) + 1
    return [
        {"dataset": dataset, "license": license_name, "record_count": count}
        for (dataset, license_name), count in sorted(counts.items())
    ]


def build_source_report(building_path: Path, place_path: Path, osm_path: Path, *, release: str) -> dict[str, Any]:
    buildings = load_geojson(building_path)
    places = load_geojson(place_path)
    _, entrances, osm_metadata = parse_osm(osm_path)
    place_crosswalks: list[dict[str, Any]] = []
    for place in places:
        point = _point_from_feature(place)
        if point is None:
            continue
        containing = [_feature_id(building) for building in buildings if point_in_geometry(point, building.get("geometry", {}))]
        place_crosswalks.append({
            "place_id": _feature_id(place),
            "place_name": _place_name(place),
            "method": "CONTAINED_UNIQUE" if len(containing) == 1 else "AMBIGUOUS",
            "status": "CANDIDATE_ONLY" if len(containing) == 1 else "REJECTED",
            "building_ids": containing,
        })
    anchor_crosswalks: list[dict[str, Any]] = []
    for entrance in entrances:
        source_matches = [
            _feature_id(building)
            for building in buildings
            if set(entrance["osm_building_way_ids"]) & overture_osm_way_ids(building)
        ]
        distances = sorted(
            ((point_to_geometry_boundary_m(entrance["point"], building.get("geometry", {})), _feature_id(building)) for building in buildings),
            key=lambda item: (item[0], item[1]),
        )
        nearest = distances[0] if distances else (math.inf, "")
        margin = distances[1][0] - nearest[0] if len(distances) > 1 else math.inf
        source_admissible = len(source_matches) == 1
        spatial_admissible = not source_matches and nearest[0] <= 3.0 and margin >= 5.0
        admissible = source_admissible or spatial_admissible
        anchor_crosswalks.append({
            **entrance,
            "method": "SOURCE_CROSSWALKED" if source_admissible else ("SPATIAL_CROSSWALKED" if spatial_admissible else "AMBIGUOUS"),
            "status": "CANDIDATE_ONLY" if admissible else "REJECTED",
            "overture_building_id": source_matches[0] if source_admissible else (nearest[1] or None),
            "boundary_distance_m": None if not math.isfinite(nearest[0]) else round(nearest[0], 6),
            "second_best_margin_m": None if not math.isfinite(margin) else round(margin, 6),
        })
    report = {
        "schema_version": 1,
        "area_role": "NON_FRESH_CANARY_SOURCE_SLICE",
        "overture_release": release,
        "source_files": {
            "overture_buildings": {"sha256": file_sha256(building_path), "feature_count": len(buildings), "licenses": overture_license_summary(buildings)},
            "overture_places": {"sha256": file_sha256(place_path), "feature_count": len(places), "licenses": overture_license_summary(places)},
            "osm": {"sha256": file_sha256(osm_path), "entrance_count": len(entrances), **osm_metadata},
        },
        "place_building_crosswalk_candidates": place_crosswalks,
        "osm_entrance_building_crosswalk_candidates": anchor_crosswalks,
        "admission_authority": "NONE_WITHOUT_MAPILLARY_AND_VISUAL_CANDIDATES",
    }
    report["report_sha256"] = content_sha256(report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--buildings", required=True, type=Path)
    parser.add_argument("--places", required=True, type=Path)
    parser.add_argument("--osm", required=True, type=Path)
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = build_source_report(args.buildings, args.places, args.osm, release=args.release)
    write_json(args.output, report)
    print(json.dumps({"report_sha256": report["report_sha256"], "source_files": report["source_files"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
