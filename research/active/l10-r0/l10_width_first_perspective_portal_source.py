#!/usr/bin/env python3
"""Freeze width-first portal intervals in source-disjoint perspective images."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
EARTH_RADIUS_M = 6_371_008.8
PROTOCOL_SCHEMA = "blindassist-l10-width-first-perspective-portal-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-width-first-perspective-portal-metadata-result-v1"
SOURCE_SCHEMA = "blindassist-l10-width-first-perspective-portal-source-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def artifact_path(artifact_root: Path, logical_path: str) -> Path:
    path = Path(logical_path)
    return path if path.is_absolute() else artifact_root / path


def wrap_degrees(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def to_east_north(origin: tuple[float, float], point: tuple[float, float]) -> tuple[float, float]:
    origin_lon, origin_lat = origin
    lon, lat = point
    mean_lat = math.radians((origin_lat + lat) / 2.0)
    east = math.radians(lon - origin_lon) * EARTH_RADIUS_M * math.cos(mean_lat)
    north = math.radians(lat - origin_lat) * EARTH_RADIUS_M
    return east, north


def from_east_north(origin: tuple[float, float], east: float, north: float) -> tuple[float, float]:
    origin_lon, origin_lat = origin
    lon = origin_lon + math.degrees(east / (EARTH_RADIUS_M * math.cos(math.radians(origin_lat))))
    lat = origin_lat + math.degrees(north / EARTH_RADIUS_M)
    return lon, lat


def parse_width(value: Any) -> float:
    width = float(str(value).strip().replace(",", "."))
    require(math.isfinite(width) and width > 0.0, f"INVALID_WIDTH:{value}")
    return width


def normalize_target(row: dict[str, Any], layout: str) -> dict[str, Any]:
    if layout == "targets":
        return {
            "source_city": row["source_city"],
            "target_way_id": int(row["target_way_id"]),
            "target_name": row["target_name"],
            "target_tags": row.get("target_tags") or {},
            "main_entrance_node": row["main_entrance_node"],
            "target_polygon_lon_lat": row["target_polygon_lon_lat"],
        }
    require(layout == "rows", f"UNKNOWN_ROSTER_LAYOUT:{layout}")
    return {
        "source_city": row["city"],
        "target_way_id": int(row["target_way"]["id"]),
        "target_name": (row["target_way"].get("tags") or {}).get("name"),
        "target_tags": row["target_way"].get("tags") or {},
        "main_entrance_node": row["main_entrance_node"],
        "target_polygon_lon_lat": row["target_polygon_lon_lat"],
    }


def roster_rows(payload: dict[str, Any], layout: str) -> list[dict[str, Any]]:
    key = "targets" if layout == "targets" else "rows"
    rows = payload.get(key)
    require(isinstance(rows, list), f"ROSTER_ROWS_MISSING:{key}")
    return [normalize_target(row, layout) for row in rows]


def find_target(rows: list[dict[str, Any]], way_id: int, entrance_id: int) -> dict[str, Any]:
    matches = [
        row
        for row in rows
        if int(row["target_way_id"]) == way_id
        and int(row["main_entrance_node"]["id"]) == entrance_id
    ]
    require(len(matches) == 1, f"TARGET_MATCH_COUNT:{way_id}:{entrance_id}:{len(matches)}")
    return matches[0]


def portal_geometry(target: dict[str, Any], tolerance_m: float) -> dict[str, Any]:
    entrance = target["main_entrance_node"]
    tags = entrance.get("tags") or {}
    width_m = parse_width(tags.get("width"))
    center = (float(entrance["lon"]), float(entrance["lat"]))
    polygon = [(float(point[0]), float(point[1])) for point in target["target_polygon_lon_lat"]]
    require(len(polygon) >= 3, "TARGET_POLYGON_TOO_SHORT")
    count = len(polygon) - 1 if polygon[0] == polygon[-1] else len(polygon)
    distances = [math.hypot(*to_east_north(center, polygon[index])) for index in range(count)]
    vertex_index = min(range(count), key=distances.__getitem__)
    require(distances[vertex_index] <= tolerance_m, f"ENTRANCE_NOT_HOST_VERTEX:{distances[vertex_index]}")
    previous = to_east_north(center, polygon[(vertex_index - 1) % count])
    following = to_east_north(center, polygon[(vertex_index + 1) % count])
    tangent_east = following[0] - previous[0]
    tangent_north = following[1] - previous[1]
    tangent_norm = math.hypot(tangent_east, tangent_north)
    require(tangent_norm > 0.0, "DEGENERATE_HOST_WALL_TANGENT")
    tangent_east /= tangent_norm
    tangent_north /= tangent_norm
    half = width_m / 2.0
    endpoints = [
        from_east_north(center, -half * tangent_east, -half * tangent_north),
        from_east_north(center, half * tangent_east, half * tangent_north),
    ]
    return {
        "entrance_center_lon_lat": list(center),
        "width_m": width_m,
        "host_vertex_index": vertex_index,
        "host_vertex_residual_m": distances[vertex_index],
        "wall_tangent_east_north": [tangent_east, tangent_north],
        "portal_endpoints_lon_lat": [list(point) for point in endpoints],
    }


def initial_bearing_degrees(camera: tuple[float, float], point: tuple[float, float]) -> float:
    east, north = to_east_north(camera, point)
    return math.degrees(math.atan2(east, north)) % 360.0


def pinhole_x(relative_bearing_degrees: float, width_pixels: int, horizontal_fov_degrees: float) -> float:
    return width_pixels / 2.0 + (
        math.tan(math.radians(relative_bearing_degrees))
        / math.tan(math.radians(horizontal_fov_degrees / 2.0))
        * width_pixels
        / 2.0
    )


def via_instance(item: dict[str, Any]) -> str | None:
    for link in item.get("links") or []:
        if link.get("rel") == "via":
            return link.get("instance_name")
    return None


def license_link(item: dict[str, Any]) -> str | None:
    for link in item.get("links") or []:
        if link.get("rel") == "license":
            return link.get("href")
    return None


def project_item(
    item: dict[str, Any], geometry: dict[str, Any], uncertainty_samples: int
) -> dict[str, Any]:
    failures: list[str] = []
    properties = item.get("properties") or {}
    interior = properties.get("pers:interior_orientation") or {}
    fov = interior.get("field_of_view")
    dimensions = interior.get("sensor_array_dimensions")
    azimuth = properties.get("view:azimuth")
    accuracy = properties.get("quality:horizontal_accuracy")
    hd_asset = (item.get("assets") or {}).get("hd") or {}
    coordinates = (item.get("geometry") or {}).get("coordinates")
    if not isinstance(fov, (int, float)) or not (0.0 < float(fov) < 180.0):
        failures.append("PERSPECTIVE_HORIZONTAL_FOV_MISSING_OR_NON_PERSPECTIVE")
    if not (
        isinstance(dimensions, list)
        and len(dimensions) == 2
        and all(isinstance(value, int) and value > 0 for value in dimensions)
    ):
        failures.append("SENSOR_DIMENSIONS_MISSING")
    if not isinstance(azimuth, (int, float)):
        failures.append("VIEW_AZIMUTH_MISSING")
    if not isinstance(accuracy, (int, float)) or float(accuracy) <= 0.0:
        failures.append("HORIZONTAL_ACCURACY_MISSING")
    if not isinstance(hd_asset.get("href"), str):
        failures.append("HD_ASSET_MISSING")
    if not (isinstance(coordinates, list) and len(coordinates) >= 2):
        failures.append("CAMERA_COORDINATES_MISSING")
    if failures:
        return {"eligible": False, "failures": failures, "item_id": item.get("id")}

    fov = float(fov)
    width_pixels, height_pixels = int(dimensions[0]), int(dimensions[1])
    azimuth = float(azimuth)
    accuracy = float(accuracy)
    camera = (float(coordinates[0]), float(coordinates[1]))
    center = tuple(geometry["entrance_center_lon_lat"])
    camera_east, camera_north = to_east_north(center, camera)
    camera_distance = math.hypot(camera_east, camera_north)
    endpoints = [tuple(point) for point in geometry["portal_endpoints_lon_lat"]]
    nominal_relative = [wrap_degrees(initial_bearing_degrees(camera, point) - azimuth) for point in endpoints]
    uncertainty_relative: list[float] = []
    for index in range(uncertainty_samples):
        angle = 2.0 * math.pi * index / uncertainty_samples
        offset_east = accuracy * math.cos(angle)
        offset_north = accuracy * math.sin(angle)
        sampled_camera = from_east_north(
            center, camera_east + offset_east, camera_north + offset_north
        )
        uncertainty_relative.extend(
            wrap_degrees(initial_bearing_degrees(sampled_camera, point) - azimuth)
            for point in endpoints
        )
    relative_min = min(uncertainty_relative)
    relative_max = max(uncertainty_relative)
    no_wrap = relative_max - relative_min < 180.0
    robust = (
        camera_distance > accuracy
        and no_wrap
        and relative_min >= -fov / 2.0
        and relative_max <= fov / 2.0
    )
    if not robust:
        failures.append("DECLARED_ACCURACY_ENVELOPE_NOT_FULLY_IN_SENSOR")
    nominal_x = sorted(pinhole_x(value, width_pixels, fov) for value in nominal_relative)
    uncertainty_x = sorted(
        [pinhole_x(relative_min, width_pixels, fov), pinhole_x(relative_max, width_pixels, fov)]
    )
    return {
        "eligible": robust,
        "failures": failures,
        "item_id": item["id"],
        "collection_id": item["collection"],
        "origin_instance": via_instance(item),
        "license": properties.get("license"),
        "license_href": license_link(item),
        "camera_lon_lat": list(camera),
        "camera_distance_m": camera_distance,
        "view_azimuth_degrees": azimuth,
        "horizontal_accuracy_m": accuracy,
        "horizontal_fov_degrees": fov,
        "sensor_dimensions": [width_pixels, height_pixels],
        "nominal_relative_bearing_interval_degrees": sorted(nominal_relative),
        "uncertainty_relative_bearing_interval_degrees": [relative_min, relative_max],
        "uncertainty_angular_width_degrees": relative_max - relative_min,
        "nominal_sensor_x_interval": nominal_x,
        "uncertainty_sensor_x_interval": uncertainty_x,
        "hd_asset": hd_asset["href"],
        "datetime": properties.get("datetime"),
    }


def choose_pair(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    robust = [row for row in rows if row["eligible"]]
    by_collection: dict[str, list[dict[str, Any]]] = {}
    for row in robust:
        by_collection.setdefault(row["collection_id"], []).append(row)
    representatives = []
    for collection_id, collection_rows in by_collection.items():
        representatives.append(
            min(
                collection_rows,
                key=lambda row: (
                    row["uncertainty_angular_width_degrees"],
                    row["camera_distance_m"],
                    row["item_id"],
                ),
            )
        )
    require(len(representatives) >= 2, "FEWER_THAN_TWO_ROBUST_COLLECTIONS")
    pair = min(
        combinations(representatives, 2),
        key=lambda rows_pair: (
            max(row["uncertainty_angular_width_degrees"] for row in rows_pair),
            sum(row["uncertainty_angular_width_degrees"] for row in rows_pair),
            tuple(sorted(row["collection_id"] for row in rows_pair)),
            tuple(sorted(row["item_id"] for row in rows_pair)),
        ),
    )
    ordered = sorted(pair, key=lambda row: (row["collection_id"], row["item_id"]))
    ordered[0] = {**ordered[0], "role": "REFERENCE"}
    ordered[1] = {**ordered[1], "role": "QUERY"}
    summary = {
        "returned_item_count": len(rows),
        "perspective_projectable_item_count": sum(
            "PERSPECTIVE_HORIZONTAL_FOV_MISSING_OR_NON_PERSPECTIVE" not in row["failures"]
            and "SENSOR_DIMENSIONS_MISSING" not in row["failures"]
            and "VIEW_AZIMUTH_MISSING" not in row["failures"]
            and "HORIZONTAL_ACCURACY_MISSING" not in row["failures"]
            and "HD_ASSET_MISSING" not in row["failures"]
            and "CAMERA_COORDINATES_MISSING" not in row["failures"]
            for row in rows
        ),
        "robust_item_count": len(robust),
        "robust_collection_count": len(by_collection),
    }
    return ordered, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol", default=str(HERE / "l10_width_first_perspective_portal_protocol_v1.json")
    )
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument(
        "--result", default=str(HERE / "l10_width_first_perspective_portal_source_result_v1.json")
    )
    parser.add_argument(
        "--source", default=str(HERE / "l10_width_first_perspective_portal_source_v1.json")
    )
    args = parser.parse_args()

    protocol_path = Path(args.protocol).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(int(protocol["selection"]["cohort_size"]) == len(protocol["frozen_targets"]), "COHORT_SIZE_MISMATCH")

    rosters: dict[str, list[dict[str, Any]]] = {}
    roster_receipts: dict[str, dict[str, Any]] = {}
    for name, spec in protocol["artifact_inputs"].items():
        path = artifact_path(artifact_root, spec["path"])
        actual_hash = sha256(path)
        require(actual_hash == spec["sha256"], f"ROSTER_HASH_MISMATCH:{name}")
        rosters[name] = roster_rows(load_json(path), spec["layout"])
        roster_receipts[name] = {
            "path": str(path),
            "sha256": actual_hash,
            "row_count": len(rosters[name]),
            "layout": spec["layout"],
        }

    episodes = []
    total_returned = 0
    total_projectable = 0
    total_robust = 0
    uncertainty_samples = int(protocol["projection"]["uncertainty_samples"])
    tolerance_m = float(protocol["geometry"]["entrance_vertex_tolerance_m"])
    for frozen in protocol["frozen_targets"]:
        target = find_target(
            rosters[frozen["roster"]],
            int(frozen["target_way_id"]),
            int(frozen["entrance_node_id"]),
        )
        geometry = portal_geometry(target, tolerance_m)
        response_path = artifact_path(artifact_root, frozen["response_path"])
        response_hash = sha256(response_path)
        require(response_hash == frozen["response_sha256"], f"RESPONSE_HASH_MISMATCH:{frozen['episode_id']}")
        features = load_json(response_path).get("features")
        require(isinstance(features, list), f"FEATURES_MISSING:{frozen['episode_id']}")
        projections = [project_item(item, geometry, uncertainty_samples) for item in features]
        roles, metrics = choose_pair(projections)
        total_returned += metrics["returned_item_count"]
        total_projectable += metrics["perspective_projectable_item_count"]
        total_robust += metrics["robust_item_count"]
        entrance = target["main_entrance_node"]
        episodes.append(
            {
                "episode_id": frozen["episode_id"],
                "source_city": target["source_city"],
                "target_way_id": target["target_way_id"],
                "target_name": target["target_name"],
                "entrance_node_id": int(entrance["id"]),
                "entrance_tags": entrance.get("tags") or {},
                "geometry": geometry,
                "response_receipt": {
                    "path": str(response_path),
                    "sha256": response_hash,
                    "feature_count": len(features),
                },
                "metadata_metrics": metrics,
                "roles": roles,
            }
        )

    distinct_role_collections = {
        role["collection_id"] for episode in episodes for role in episode["roles"]
    }
    gate_met = len(episodes) == int(protocol["selection"]["cohort_size"])
    status = (
        "L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_METADATA_GATE_MET_3_OF_3_BUILDINGS_6_DISTINCT_COLLECTIONS"
        if gate_met and len(distinct_role_collections) == 6
        else "L10_WIDTH_FIRST_PERSPECTIVE_PORTAL_METADATA_GATE_NOT_MET"
    )
    source = {
        "schema": SOURCE_SCHEMA,
        "authority": protocol["authority"],
        "protocol": str(protocol_path.relative_to(ROOT)).replace("\\", "/"),
        "protocol_sha256": sha256(protocol_path),
        "artifact_root": str(artifact_root),
        "roster_receipts": roster_receipts,
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    source_path = Path(args.source).resolve()
    write_json(source_path, source)
    result = {
        "schema": RESULT_SCHEMA,
        "authority": protocol["authority"],
        "status": status,
        "protocol": source["protocol"],
        "protocol_sha256": source["protocol_sha256"],
        "source": str(source_path.relative_to(ROOT)).replace("\\", "/"),
        "source_sha256": sha256(source_path),
        "metrics": {
            "building_disjoint_episode_count": len(episodes),
            "selected_role_image_count": sum(len(episode["roles"]) for episode in episodes),
            "selected_distinct_collection_count": len(distinct_role_collections),
            "returned_item_count": total_returned,
            "perspective_projectable_item_count": total_projectable,
            "declared_accuracy_robust_item_count": total_robust,
            "per_episode": [
                {"episode_id": episode["episode_id"], **episode["metadata_metrics"]}
                for episode in episodes
            ],
        },
        "next_action": "Materialize only the six frozen HD assets and run one disjoint role audit. Run the unchanged portal-transfer matcher only if all three episodes are jointly ADMITTED.",
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(Path(args.result).resolve(), result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
