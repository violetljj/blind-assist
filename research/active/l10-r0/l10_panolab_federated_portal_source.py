#!/usr/bin/env python3
"""Discover close cross-collection Panoramax entrance assets without pixels."""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent

from l10_panolab_entrance_ray import (  # noqa: E402
    circular_distance_degrees,
    initial_bearing_degrees,
    projection_gate as strict_projection_gate,
)
from l10_panolab_viewer_equivalent_projection import (  # noqa: E402
    projection_gate as viewer_projection_gate,
)


PROTOCOL_SCHEMA = "blindassist-l10-panolab-federated-portal-source-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-panolab-federated-portal-metadata-result-v1"
TARGETS_SCHEMA = "blindassist-l10-panolab-federated-target-roster-v1"
LEDGER_SCHEMA = "blindassist-l10-panolab-federated-search-ledger-v1"
CANDIDATES_SCHEMA = "blindassist-l10-panolab-federated-direct-candidates-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def fetch_json(
    url: str,
    raw_root: Path,
    label: str,
    *,
    form: dict[str, str] | None = None,
    timeout_seconds: int = 180,
    attempts: int = 3,
) -> tuple[dict[str, Any], dict[str, Any]]:
    body = None if form is None else urllib.parse.urlencode(form).encode("utf-8")
    request_key = sha256_bytes((url + "\n").encode("utf-8") + (body or b""))
    raw_path = raw_root / f"{label}-{request_key[:20]}.json"
    if raw_path.exists():
        payload_bytes = raw_path.read_bytes()
        return json.loads(payload_bytes), {
            "url": url,
            "method": "POST" if body is not None else "GET",
            "request_body_sha256": sha256_bytes(body) if body is not None else None,
            "response_path": str(raw_path.resolve()),
            "response_sha256": sha256_bytes(payload_bytes),
            "response_bytes": len(payload_bytes),
            "status": "REUSED_TASK_OWNED_RESPONSE",
        }

    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        headers = {
            "User-Agent": "BlindAssist-L10-Federated-Portal/1.0",
            "Accept": "application/json",
        }
        if body is not None:
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        request = urllib.request.Request(url, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload_bytes = response.read()
                status_code = int(response.status)
            payload = json.loads(payload_bytes)
            atomic_write(raw_path, payload_bytes)
            return payload, {
                "url": url,
                "method": "POST" if body is not None else "GET",
                "request_body_sha256": sha256_bytes(body) if body is not None else None,
                "response_path": str(raw_path.resolve()),
                "response_sha256": sha256_bytes(payload_bytes),
                "response_bytes": len(payload_bytes),
                "http_status": status_code,
                "attempt": attempt,
                "fetched_at_utc": utc_now(),
                "status": "FETCHED",
            }
        except (
            OSError,
            ValueError,
            http.client.HTTPException,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ) as error:
            last_error = error
            if attempt < attempts:
                time.sleep(float(attempt))
    raise RuntimeError(f"REQUEST_FAILED:{url}:{last_error}")


def overpass_target_query(city: dict[str, Any]) -> str:
    west, south, east, north = [float(value) for value in city["bbox"]]
    return (
        "[out:json][timeout:180];"
        f'way["building"]["name"]({south},{west},{north},{east});'
        "(._;>;);out body qt;"
    )


def extract_targets(
    payload: dict[str, Any], city: dict[str, Any], excluded_way_ids: set[int]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    nodes = {
        int(row["id"]): row
        for row in payload.get("elements", [])
        if row.get("type") == "node" and "lat" in row and "lon" in row
    }
    ways = [row for row in payload.get("elements", []) if row.get("type") == "way"]
    targets = []
    named_building_count = 0
    main_entrance_count = 0
    semantic_fields = ("amenity", "tourism", "shop", "office", "leisure", "healthcare", "historic")
    for way in ways:
        way_id = int(way["id"])
        tags = way.get("tags") or {}
        if "building" not in tags or not tags.get("name"):
            continue
        named_building_count += 1
        if way_id in excluded_way_ids:
            continue
        main_nodes = []
        for node_id in way.get("nodes", []):
            node = nodes.get(int(node_id))
            if node is not None and (node.get("tags") or {}).get("entrance") == "main":
                main_nodes.append(node)
        if not main_nodes:
            continue
        main_entrance_count += len(main_nodes)
        entrance = min(main_nodes, key=lambda row: int(row["id"]))
        polygon = [
            [float(nodes[node_id]["lon"]), float(nodes[node_id]["lat"])]
            for node_id in way.get("nodes", [])
            if int(node_id) in nodes
        ]
        semantic_rank = 0 if any(field in tags for field in semantic_fields) else 1
        targets.append(
            {
                "source_city": city["name"],
                "source_kind": "NEW_CITY_OVERPASS",
                "target_way_id": way_id,
                "target_name": str(tags["name"]),
                "target_tags": tags,
                "main_entrance_node": {
                    "id": int(entrance["id"]),
                    "lon": float(entrance["lon"]),
                    "lat": float(entrance["lat"]),
                    "tags": entrance.get("tags") or {},
                },
                "target_polygon_lon_lat": polygon,
                "selection_key": [semantic_rank, way_id, int(entrance["id"])],
            }
        )
    targets.sort(key=lambda row: tuple(row["selection_key"]))
    limit = int(city["target_limit"])
    return targets[:limit], {
        "named_building_ways": named_building_count,
        "main_entrance_nodes": main_entrance_count,
        "eligible_before_limit": len(targets),
        "selected_target_count": min(limit, len(targets)),
    }


def search_url(target: dict[str, Any], protocol: dict[str, Any]) -> str:
    entrance = target["main_entrance_node"]
    selection = protocol["selection"]
    params = {
        "place_position": f'{float(entrance["lon"]):.8f},{float(entrance["lat"]):.8f}',
        "place_distance": f'{selection["camera_distance_m"][0]}-{selection["camera_distance_m"][1]}',
        "place_fov_tolerance": "180",
        "filter": "field_of_view = 360",
        "limit": str(selection["search_limit"]),
        "sortby": "id",
    }
    return protocol["providers"]["federated_search"] + "?" + urllib.parse.urlencode(params)


def haversine_m(left: list[float], right: list[float]) -> float:
    lon1, lat1 = map(math.radians, left)
    lon2, lat2 = map(math.radians, right)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    value = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return 2.0 * 6371008.8 * math.asin(min(1.0, math.sqrt(value)))


def item_origin(item: dict[str, Any]) -> str | None:
    for link in item.get("links") or []:
        if link.get("rel") == "via" and isinstance(link.get("href"), str):
            return urllib.parse.urlparse(link["href"]).netloc
    href = (((item.get("assets") or {}).get("hd") or {}).get("href"))
    return urllib.parse.urlparse(href).netloc if isinstance(href, str) else None


def search_target(
    target: dict[str, Any],
    protocol: dict[str, Any],
    strict_protocol: dict[str, Any],
    successor_protocol: dict[str, Any],
    raw_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    url = search_url(target, protocol)
    label = f'search-{target["source_city"]}-{target["target_way_id"]}'
    try:
        payload, receipt = fetch_json(url, raw_root, label, timeout_seconds=120)
    except Exception as error:  # bounded per-target failure is retained in the ledger
        return {
            **target,
            "request_url": url,
            "error": f"{type(error).__name__}:{error}",
            "feature_count": 0,
            "eligible_items": [],
        }, {}
    entrance_lon_lat = [
        float(target["main_entrance_node"]["lon"]),
        float(target["main_entrance_node"]["lat"]),
    ]
    minimum, maximum = [float(value) for value in protocol["selection"]["camera_distance_m"]]
    eligible = []
    strict_items = 0
    item_failures: dict[str, int] = {}
    seen: set[str] = set()
    for item in payload.get("features") or []:
        item_id = str(item.get("id"))
        if not item_id or item_id in seen:
            continue
        seen.add(item_id)
        geometry = item.get("geometry") or {}
        coordinates = geometry.get("coordinates")
        if geometry.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) != 2:
            continue
        distance_m = haversine_m([float(coordinates[0]), float(coordinates[1])], entrance_lon_lat)
        if not minimum <= distance_m <= maximum:
            continue
        strict = strict_projection_gate(item, strict_protocol)
        gate = viewer_projection_gate(item, strict_protocol, successor_protocol)
        hd = ((item.get("assets") or {}).get("hd") or {}).get("href")
        if not isinstance(hd, str) or not hd.startswith("https://"):
            gate = {**gate, "eligible": False, "failures": gate["failures"] + ["HD_ASSET_MISSING"]}
        if strict["eligible"]:
            strict_items += 1
        if not gate["eligible"]:
            for failure in gate["failures"]:
                item_failures[failure] = item_failures.get(failure, 0) + 1
            continue
        collection = str(item.get("collection") or "")
        if not collection:
            continue
        camera_lon_lat = [float(coordinates[0]), float(coordinates[1])]
        eligible.append(
            {
                "item_id": item_id,
                "collection": collection,
                "origin": item_origin(item),
                "distance_m": round(distance_m, 3),
                "camera_to_entrance_bearing_degrees": round(
                    initial_bearing_degrees(camera_lon_lat, entrance_lon_lat), 6
                ),
                "orientation_gate": gate,
                "item": item,
            }
        )
    eligible.sort(key=lambda row: (row["collection"], row["item_id"]))
    return {
        **target,
        "request_url": url,
        "error": None,
        "feature_count": len(payload.get("features") or []),
        "strict_eligible_item_count": strict_items,
        "strict_eligible_collection_count": len(
            {
                str(row["collection"])
                for row in eligible
                if row["orientation_gate"]["strict_eligible"]
            }
        ),
        "viewer_eligible_item_count": len(eligible),
        "viewer_eligible_collection_count": len({row["collection"] for row in eligible}),
        "top_ineligible_failures": sorted(item_failures.items(), key=lambda row: (-row[1], row[0]))[:12],
        "eligible_items": eligible,
    }, receipt


def geometry_query(target: dict[str, Any], radius_m: float) -> str:
    entrance = target["main_entrance_node"]
    lat = float(entrance["lat"])
    lon = float(entrance["lon"])
    return (
        "[out:json][timeout:120];("
        f'way["building"](around:{radius_m},{lat},{lon});'
        f'relation["building"](around:{radius_m},{lat},{lon});'
        ");out geom;"
    )


def _rings(payload: dict[str, Any]) -> dict[str, list[list[list[float]]]]:
    output: dict[str, list[list[list[float]]]] = {}
    for row in payload.get("elements") or []:
        owner = f'{row.get("type")}/{row.get("id")}'
        if row.get("type") == "way":
            geometry = row.get("geometry") or []
            points = [[float(point["lon"]), float(point["lat"])] for point in geometry if "lon" in point and "lat" in point]
            if len(points) >= 2:
                output.setdefault(owner, []).append(points)
        elif row.get("type") == "relation":
            for member in row.get("members") or []:
                if member.get("role") not in ("", "outer"):
                    continue
                geometry = member.get("geometry") or []
                points = [[float(point["lon"]), float(point["lat"])] for point in geometry if "lon" in point and "lat" in point]
                if len(points) >= 2:
                    output.setdefault(owner, []).append(points)
    return output


def _xy(point: list[float], origin: list[float]) -> tuple[float, float]:
    scale_x = 111320.0 * math.cos(math.radians(origin[1]))
    return ((point[0] - origin[0]) * scale_x, (point[1] - origin[1]) * 110540.0)


def _cross(left: tuple[float, float], right: tuple[float, float]) -> float:
    return left[0] * right[1] - left[1] * right[0]


def _segment_intersection_t(
    start: tuple[float, float],
    end: tuple[float, float],
    left: tuple[float, float],
    right: tuple[float, float],
) -> float | None:
    ray = (end[0] - start[0], end[1] - start[1])
    edge = (right[0] - left[0], right[1] - left[1])
    denominator = _cross(ray, edge)
    if abs(denominator) <= 1e-9:
        return None
    delta = (left[0] - start[0], left[1] - start[1])
    t = _cross(delta, edge) / denominator
    u = _cross(delta, ray) / denominator
    if -1e-9 <= t <= 1.0 + 1e-9 and -1e-9 <= u <= 1.0 + 1e-9:
        return max(0.0, min(1.0, t))
    return None


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    inside = False
    x, y = point
    for index in range(len(ring)):
        x1, y1 = ring[index]
        x2, y2 = ring[(index + 1) % len(ring)]
        if (y1 > y) != (y2 > y):
            crossing_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < crossing_x:
                inside = not inside
    return inside


def classify_direct(
    item_row: dict[str, Any],
    target: dict[str, Any],
    rings: dict[str, list[list[list[float]]]],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    camera = [float(value) for value in item_row["item"]["geometry"]["coordinates"]]
    entrance = [
        float(target["main_entrance_node"]["lon"]),
        float(target["main_entrance_node"]["lat"]),
    ]
    target_owner = f'way/{target["target_way_id"]}'
    if target_owner not in rings:
        return {"status": "TARGET_BUILDING_GEOMETRY_MISSING"}
    origin = entrance
    camera_xy = _xy(camera, origin)
    entrance_xy = (0.0, 0.0)
    for ring in rings[target_owner]:
        polygon = [_xy(point, origin) for point in ring]
        if len(polygon) >= 3 and _point_in_ring(camera_xy, polygon):
            return {"status": "CAMERA_INSIDE_TARGET"}
    total_distance = math.hypot(camera_xy[0], camera_xy[1])
    if total_distance <= 0.0:
        return {"status": "ZERO_CAMERA_DISTANCE"}
    owner_distances: dict[str, float] = {}
    for owner, owner_rings in rings.items():
        for ring in owner_rings:
            points = [_xy(point, origin) for point in ring]
            if len(points) < 2:
                continue
            for index in range(len(points) - 1):
                t = _segment_intersection_t(camera_xy, entrance_xy, points[index], points[index + 1])
                if t is None:
                    continue
                distance = t * total_distance
                if owner not in owner_distances or distance < owner_distances[owner]:
                    owner_distances[owner] = distance
            if points[0] != points[-1]:
                t = _segment_intersection_t(camera_xy, entrance_xy, points[-1], points[0])
                if t is not None:
                    distance = t * total_distance
                    if owner not in owner_distances or distance < owner_distances[owner]:
                        owner_distances[owner] = distance
    if not owner_distances:
        return {"status": "NO_BUILDING_INTERSECTION"}
    first_distance = min(owner_distances.values())
    tie = float(protocol["geometry"]["owner_tie_tolerance_m"])
    first_owners = sorted(owner for owner, distance in owner_distances.items() if distance <= first_distance + tie)
    gap = max(0.0, total_distance - first_distance)
    if len(first_owners) != 1:
        status = "AMBIGUOUS_FIRST_OWNER"
    elif first_owners[0] != target_owner:
        status = "OTHER_BUILDING_FIRST"
    elif gap > float(protocol["geometry"]["direct_entrance_tolerance_m"]):
        status = "TARGET_SELF_OCCLUDED"
    else:
        status = "DIRECT"
    return {
        "status": status,
        "target_owner": target_owner,
        "first_owners": first_owners,
        "first_intersection_distance_m": round(first_distance, 3),
        "camera_to_entrance_planar_m": round(total_distance, 3),
        "first_intersection_to_entrance_m": round(gap, 3),
        "building_owner_count": len(owner_distances),
    }


def geometry_target(
    searched: dict[str, Any], protocol: dict[str, Any], raw_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    query = geometry_query(searched, float(protocol["geometry"]["building_radius_m"]))
    label = f'geometry-{searched["source_city"]}-{searched["target_way_id"]}'
    try:
        payload, receipt = fetch_json(
            protocol["providers"]["overpass_geometry"],
            raw_root,
            label,
            form={"data": query},
            timeout_seconds=180,
        )
    except Exception as error:
        return {**searched, "geometry_error": f"{type(error).__name__}:{error}"}, {}
    rings = _rings(payload)
    rows = []
    for item_row in searched["eligible_items"]:
        geometry = classify_direct(item_row, searched, rings, protocol)
        rows.append({**item_row, "geometry": geometry})
    return {**searched, "geometry_error": None, "eligible_items": rows}, receipt


def best_pair(rows: list[dict[str, Any]]) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]] | None:
    pairs = []
    direct = [row for row in rows if row.get("geometry", {}).get("status") == "DIRECT"]
    for left, right in combinations(direct, 2):
        if left["collection"] == right["collection"]:
            continue
        score = (
            max(float(left["distance_m"]), float(right["distance_m"])),
            circular_distance_degrees(
                float(left["camera_to_entrance_bearing_degrees"]),
                float(right["camera_to_entrance_bearing_degrees"]),
            ),
            abs(float(left["distance_m"]) - float(right["distance_m"])),
            tuple(sorted((left["collection"], right["collection"]))),
            tuple(sorted((left["item_id"], right["item_id"]))),
        )
        ordered = sorted((left, right), key=lambda row: (row["collection"], row["item_id"]))
        pairs.append((score, ordered[0], ordered[1]))
    return min(pairs, key=lambda row: row[0]) if pairs else None


def compact_search_row(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if key != "eligible_items"}


def main_run(protocol_path: Path, artifact_root: Path, result_path: Path) -> None:
    require(not result_path.exists(), f"RESULT_ALREADY_EXISTS:{result_path}")
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    strict_path = resolve(protocol["projection"]["strict_protocol_path"])
    successor_path = resolve(protocol["projection"]["successor_protocol_path"])
    require(sha256(strict_path) == protocol["projection"]["strict_protocol_sha256"], "STRICT_PROTOCOL_HASH_MISMATCH")
    require(sha256(successor_path) == protocol["projection"]["successor_protocol_sha256"], "SUCCESSOR_PROTOCOL_HASH_MISMATCH")
    strict_protocol = load_json(strict_path)
    successor_protocol = load_json(successor_path)
    artifact_root.mkdir(parents=True, exist_ok=True)
    raw_root = artifact_root / "raw"
    receipts: list[dict[str, Any]] = []

    excluded = {int(value) for value in protocol["selection"]["excluded_way_ids"]}
    targets = []
    city_audits = []
    for fixed in protocol["targets"]["fixed_targets"]:
        targets.append(
            {
                "source_city": fixed["source_city"],
                "source_kind": "FIXED_NEAR_STRICT_SEED",
                "target_way_id": int(fixed["target_way_id"]),
                "target_name": fixed["target_name"],
                "target_tags": {},
                "main_entrance_node": fixed["main_entrance_node"],
                "target_polygon_lon_lat": [],
                "selection_key": [-1, int(fixed["target_way_id"]), int(fixed["main_entrance_node"]["id"])],
            }
        )
    for city in protocol["targets"]["cities"]:
        payload, receipt = fetch_json(
            protocol["providers"]["overpass_targets"],
            raw_root,
            f'targets-{city["name"]}',
            form={"data": overpass_target_query(city)},
            timeout_seconds=240,
        )
        receipts.append(receipt)
        city_targets, audit = extract_targets(payload, city, excluded)
        targets.extend(city_targets)
        city_audits.append({"city": city["name"], "bbox": city["bbox"], **audit})
    target_roster = {
        "schema": TARGETS_SCHEMA,
        "protocol_sha256": sha256(protocol_path),
        "generated_at_utc": utc_now(),
        "city_audits": city_audits,
        "target_count": len(targets),
        "targets": targets,
    }
    targets_path = artifact_root / "targets.json"
    write_json(targets_path, target_roster)
    print(f"TARGET_ROSTER {len(targets)}", flush=True)

    searched = []
    workers = int(protocol["execution"]["network_workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                search_target,
                target,
                protocol,
                strict_protocol,
                successor_protocol,
                raw_root,
            )
            for target in targets
        ]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, receipt = future.result()
            searched.append(row)
            if receipt:
                receipts.append(receipt)
            if completed % 50 == 0 or completed == len(futures):
                print(f"SEARCH_PROGRESS {completed}/{len(futures)}", flush=True)
    searched.sort(key=lambda row: (row["source_city"], row["target_way_id"], row["main_entrance_node"]["id"]))
    search_errors = sum(1 for row in searched if row.get("error"))
    geometry_inputs = [row for row in searched if row.get("viewer_eligible_collection_count", 0) >= 2]
    search_ledger = {
        "schema": LEDGER_SCHEMA,
        "target_count": len(targets),
        "search_error_count": search_errors,
        "targets_with_any_strict_item": sum(1 for row in searched if row.get("strict_eligible_item_count", 0) > 0),
        "targets_with_strict_cross_collection": sum(1 for row in searched if row.get("strict_eligible_collection_count", 0) >= 2),
        "targets_with_viewer_item": sum(1 for row in searched if row.get("viewer_eligible_item_count", 0) > 0),
        "targets_with_viewer_cross_collection": len(geometry_inputs),
        "rows": [compact_search_row(row) for row in searched],
    }
    search_path = artifact_root / "search-ledger.json"
    write_json(search_path, search_ledger)
    print(
        "SEARCH_SUMMARY "
        f"errors={search_errors} strict_cross={search_ledger['targets_with_strict_cross_collection']} "
        f"viewer_cross={len(geometry_inputs)}",
        flush=True,
    )

    geometry_rows = []
    geometry_workers = int(protocol["execution"]["geometry_workers"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=geometry_workers) as executor:
        futures = [executor.submit(geometry_target, row, protocol, raw_root) for row in geometry_inputs]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row, receipt = future.result()
            geometry_rows.append(row)
            if receipt:
                receipts.append(receipt)
            if completed % 25 == 0 or completed == len(futures):
                print(f"GEOMETRY_PROGRESS {completed}/{len(futures)}", flush=True)
    geometry_rows.sort(key=lambda row: (row["source_city"], row["target_way_id"], row["main_entrance_node"]["id"]))
    geometry_errors = sum(1 for row in geometry_rows if row.get("geometry_error"))
    geometry_status_counts: dict[str, int] = {}
    geometry_ledger_rows = []
    for row in geometry_rows:
        compact_items = []
        for item in row["eligible_items"]:
            status = item.get("geometry", {}).get("status", "NOT_CLASSIFIED")
            geometry_status_counts[status] = geometry_status_counts.get(status, 0) + 1
            compact_items.append(
                {
                    "item_id": item["item_id"],
                    "collection": item["collection"],
                    "origin": item["origin"],
                    "distance_m": item["distance_m"],
                    "projection_mode": item["orientation_gate"]["projection_mode"],
                    "geometry": item.get("geometry"),
                }
            )
        geometry_ledger_rows.append(
            {
                "source_city": row["source_city"],
                "target_way_id": row["target_way_id"],
                "target_name": row["target_name"],
                "main_entrance_node": row["main_entrance_node"],
                "geometry_error": row.get("geometry_error"),
                "items": compact_items,
            }
        )
    geometry_ledger = {
        "schema": "blindassist-l10-panolab-federated-geometry-ledger-v1",
        "target_count": len(geometry_rows),
        "geometry_error_count": geometry_errors,
        "item_status_counts": dict(sorted(geometry_status_counts.items())),
        "rows": geometry_ledger_rows,
    }
    geometry_path = artifact_root / "geometry-ledger.json"
    write_json(geometry_path, geometry_ledger)
    eligible_candidates = []
    for row in geometry_rows:
        pair = best_pair(row["eligible_items"])
        if pair is None:
            continue
        score, reference, query = pair
        eligible_candidates.append(
            {
                "sort_key": score,
                "source_city": row["source_city"],
                "source_kind": row["source_kind"],
                "target_way_id": row["target_way_id"],
                "target_name": row["target_name"],
                "target_tags": row["target_tags"],
                "main_entrance_node": row["main_entrance_node"],
                "direct_item_count": sum(1 for item in row["eligible_items"] if item["geometry"]["status"] == "DIRECT"),
                "direct_collection_count": len({item["collection"] for item in row["eligible_items"] if item["geometry"]["status"] == "DIRECT"}),
                "reference": reference,
                "query": query,
            }
        )
    eligible_candidates.sort(key=lambda row: (row["sort_key"], row["target_way_id"], row["main_entrance_node"]["id"]))
    selected = eligible_candidates[: int(protocol["selection"]["cohort_size"])]
    episodes = []
    for index, row in enumerate(selected, 1):
        copied = {key: value for key, value in row.items() if key != "sort_key"}
        copied["episode_id"] = f"FP{index:02d}"
        copied["pair_score"] = {
            "maximum_camera_distance_m": round(float(row["sort_key"][0]), 3),
            "camera_bearing_delta_degrees": round(float(row["sort_key"][1]), 6),
            "camera_distance_delta_m": round(float(row["sort_key"][2]), 3),
        }
        episodes.append(copied)
    candidates_payload = {
        "schema": CANDIDATES_SCHEMA,
        "eligible_candidate_count": len(eligible_candidates),
        "selected_episode_count": len(episodes),
        "geometry_error_count": geometry_errors,
        "candidates": [{key: value for key, value in row.items() if key != "sort_key"} for row in eligible_candidates],
    }
    candidates_path = artifact_root / "direct-candidates.json"
    write_json(candidates_path, candidates_payload)
    receipts.sort(key=lambda row: (row.get("url", ""), row.get("request_body_sha256") or ""))
    receipts_path = artifact_root / "request-receipts.json"
    write_json(receipts_path, {"schema": "blindassist-l10-panolab-federated-request-receipts-v1", "receipts": receipts})

    minimum = int(protocol["selection"]["minimum_episode_count"])
    if search_errors or geometry_errors:
        decision = (
            "L10_PANOLAB_FEDERATED_PORTAL_SOURCE_INCOMPLETE_"
            f"SEARCH_ERRORS_{search_errors}_GEOMETRY_ERRORS_{geometry_errors}"
        )
        status = "INCOMPLETE_NETWORK_DENOMINATOR"
    elif len(episodes) < minimum:
        decision = f"L10_PANOLAB_FEDERATED_PORTAL_SOURCE_NOT_EVALUABLE_{len(episodes)}_OF_{minimum}"
        status = "NOT_EVALUABLE_INSUFFICIENT_CROSS_COLLECTION_DIRECT_WAYS"
    else:
        decision = "L10_PANOLAB_FEDERATED_VIEWER_EQUIVALENT_PORTAL_METADATA_GATE_MET"
        status = "METADATA_FROZEN_BEFORE_ANY_SELECTED_PIXEL_REQUEST_OR_INSPECTION"
    result = {
        "schema": RESULT_SCHEMA,
        "decision": decision,
        "status": status,
        "protocol": "research/active/l10-r0/" + protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "generated_at_utc": utc_now(),
        "information_change": protocol["information_change"],
        "metrics": {
            "cities": len(protocol["targets"]["cities"]),
            "fixed_seed_targets": len(protocol["targets"]["fixed_targets"]),
            "targets_scanned": len(targets),
            "search_errors": search_errors,
            "targets_with_any_strict_item": search_ledger["targets_with_any_strict_item"],
            "targets_with_strict_cross_collection": search_ledger["targets_with_strict_cross_collection"],
            "targets_with_viewer_item": search_ledger["targets_with_viewer_item"],
            "targets_with_viewer_cross_collection_before_geometry": search_ledger["targets_with_viewer_cross_collection"],
            "geometry_errors": geometry_errors,
            "cross_collection_direct_candidate_ways": len(eligible_candidates),
            "frozen_episode_count": len(episodes),
            "required_episode_count": minimum,
        },
        "city_target_audits": city_audits,
        "episodes": episodes,
        "durable_evidence": {
            "artifact_root": str(artifact_root.resolve()),
            "targets": {"path": str(targets_path.resolve()), "sha256": sha256(targets_path)},
            "search_ledger": {"path": str(search_path.resolve()), "sha256": sha256(search_path)},
            "geometry_ledger": {"path": str(geometry_path.resolve()), "sha256": sha256(geometry_path)},
            "direct_candidates": {"path": str(candidates_path.resolve()), "sha256": sha256(candidates_path)},
            "request_receipts": {"path": str(receipts_path.resolve()), "sha256": sha256(receipts_path)},
            "selected_pixel_requests": 0,
        },
        "next_action": (
            "Materialize only FP01-FP05, render the unchanged x=512 viewports, then run one role-separated exact-portal visibility audit."
            if len(episodes) >= minimum and not search_errors
            else "Replace the observation source; do not tune the matcher or resample this frozen metadata roster."
        ),
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(result_path, result)
    print(json.dumps({"decision": decision, **result["metrics"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    main_run(args.protocol.resolve(), args.artifact_root.resolve(), args.result.resolve())


if __name__ == "__main__":
    main()
