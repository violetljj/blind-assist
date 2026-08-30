"""Evaluate an entity-owned OSM entrance ray in a posed public panorama.

The evaluator is geometry-only.  It first proves that the requested entrance
node is a member of the named target-building way.  It then intersects the
camera-to-entrance segment with frozen OSM building footprints.  A horizontal
panorama ray is authorized only when the target entrance is the first facade
intersection; target self-occlusion and earlier non-target buildings fail
closed.  Human pixel intervals are evaluator-only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from l10_panolab_entrance_ray import (
    circular_distance_degrees,
    project_entrance_ray,
)


ROOT = Path(__file__).resolve().parents[3]
EARTH_RADIUS_M = 6_371_008.8
ROLE_COUNTS = Counter(
    {
        "VISIBLE_TARGET_ENTRANCE": 4,
        "TARGET_SELF_OCCLUDED": 2,
        "NON_TARGET_BUILDING_OCCLUDED": 2,
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(spec: dict[str, Any]) -> Path:
    path = _path(spec["path"])
    actual = _sha256(path)
    if actual != spec["sha256"]:
        raise ValueError(f"HASH_MISMATCH:{path}:{actual}:{spec['sha256']}")
    if "bytes" in spec and path.stat().st_size != int(spec["bytes"]):
        raise ValueError(f"BYTE_COUNT_MISMATCH:{path}")
    return path


def _enu(
    origin_lon_lat: Sequence[float], point_lon_lat: Sequence[float]
) -> tuple[float, float]:
    origin_lon, origin_lat = map(float, origin_lon_lat)
    point_lon, point_lat = map(float, point_lon_lat)
    mean_lat = math.radians(0.5 * (origin_lat + point_lat))
    east = EARTH_RADIUS_M * math.radians(point_lon - origin_lon) * math.cos(mean_lat)
    north = EARTH_RADIUS_M * math.radians(point_lat - origin_lat)
    return east, north


def _distance(origin: Sequence[float], point: Sequence[float]) -> float:
    east, north = _enu(origin, point)
    return math.hypot(east, north)


def _bearing_degrees(origin: Sequence[float], point: Sequence[float]) -> float:
    lon1, lat1 = map(math.radians, map(float, origin))
    lon2, lat2 = map(math.radians, map(float, point))
    delta_lon = lon2 - lon1
    east = math.sin(delta_lon) * math.cos(lat2)
    north = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(delta_lon)
    return math.degrees(math.atan2(east, north)) % 360.0


def _cross(left: Sequence[float], right: Sequence[float]) -> float:
    return float(left[0]) * float(right[1]) - float(left[1]) * float(right[0])


def _boundary_intersections(
    camera_lon_lat: Sequence[float],
    entrance_lon_lat: Sequence[float],
    nodes: list[dict[str, Any]],
) -> list[float]:
    ray = _enu(camera_lon_lat, entrance_lon_lat)
    ray_norm_sq = ray[0] * ray[0] + ray[1] * ray[1]
    if ray_norm_sq <= 1e-12:
        raise ValueError("CAMERA_ENTRANCE_DISTANCE_ZERO")
    values: list[float] = []
    for start, end in zip(nodes, nodes[1:]):
        q = _enu(camera_lon_lat, start["lon_lat"])
        edge_end = _enu(camera_lon_lat, end["lon_lat"])
        edge = (edge_end[0] - q[0], edge_end[1] - q[1])
        denominator = _cross(ray, edge)
        if abs(denominator) <= 1e-12:
            if abs(_cross(q, ray)) <= 1e-9:
                for point in (q, edge_end):
                    parameter = (point[0] * ray[0] + point[1] * ray[1]) / ray_norm_sq
                    if -1e-9 <= parameter <= 1.0 + 1e-9:
                        values.append(min(1.0, max(0.0, parameter)))
            continue
        parameter = _cross(q, edge) / denominator
        edge_parameter = _cross(q, ray) / denominator
        if -1e-9 <= parameter <= 1.0 + 1e-9 and -1e-9 <= edge_parameter <= 1.0 + 1e-9:
            values.append(min(1.0, max(0.0, parameter)))
    ordered = sorted(values)
    return [
        value
        for index, value in enumerate(ordered)
        if index == 0 or abs(value - ordered[index - 1]) > 1e-8
    ]


def _validate_polygon(building: dict[str, Any], label: str) -> None:
    nodes = building["nodes"]
    if not isinstance(nodes, list) or len(nodes) < 4:
        raise ValueError(f"{label}_POLYGON_TOO_SHORT")
    if nodes[0]["id"] != nodes[-1]["id"] or nodes[0]["lon_lat"] != nodes[-1]["lon_lat"]:
        raise ValueError(f"{label}_POLYGON_NOT_CLOSED")
    for node in nodes:
        lon_lat = node["lon_lat"]
        if not isinstance(lon_lat, list) or len(lon_lat) != 2:
            raise ValueError(f"{label}_NODE_COORDINATE_INVALID")
        if not all(math.isfinite(float(value)) for value in lon_lat):
            raise ValueError(f"{label}_NODE_COORDINATE_NONFINITE")


def _visibility(frame: dict[str, Any], clearance_m: float) -> dict[str, Any]:
    image = frame["panorama"]
    camera = image["camera_lon_lat"]
    target = frame["target"]
    entrance = target["entrance_node"]
    entrance_lon_lat = entrance["lon_lat"]
    distance_m = _distance(camera, entrance_lon_lat)
    if distance_m <= clearance_m:
        raise ValueError(f"CAMERA_TOO_CLOSE_TO_ENTRANCE:{frame['key']}")
    _validate_polygon(target["building"], f"TARGET:{frame['key']}")
    target_node_ids = {int(node["id"]) for node in target["building"]["nodes"]}
    if int(entrance["id"]) not in target_node_ids:
        raise ValueError(f"ENTRANCE_NOT_MEMBER_OF_TARGET_WAY:{frame['key']}")

    maximum_occluding_parameter = 1.0 - clearance_m / distance_m
    events = []
    target_parameters = _boundary_intersections(
        camera, entrance_lon_lat, target["building"]["nodes"]
    )
    if not any(abs(value - 1.0) <= 1e-7 for value in target_parameters):
        raise ValueError(f"ENTRANCE_NOT_ON_TARGET_BOUNDARY:{frame['key']}")
    for parameter in target_parameters:
        if 1e-8 < parameter < maximum_occluding_parameter:
            events.append(
                {
                    "parameter": parameter,
                    "distance_m": parameter * distance_m,
                    "kind": "TARGET_SELF_OCCLUDED",
                    "osm_way_id": int(target["building"]["id"]),
                    "osm_way_version": int(target["building"]["version"]),
                    "name": target["building"]["tags"].get("name"),
                }
            )
    for index, building in enumerate(frame["occluder_buildings"]):
        _validate_polygon(building, f"OCCLUDER:{frame['key']}:{index}")
        if int(building["id"]) == int(target["building"]["id"]):
            raise ValueError(f"TARGET_DUPLICATED_AS_OCCLUDER:{frame['key']}")
        for parameter in _boundary_intersections(camera, entrance_lon_lat, building["nodes"]):
            if 1e-8 < parameter < maximum_occluding_parameter:
                events.append(
                    {
                        "parameter": parameter,
                        "distance_m": parameter * distance_m,
                        "kind": "NON_TARGET_BUILDING_OCCLUDED",
                        "osm_way_id": int(building["id"]),
                        "osm_way_version": int(building["version"]),
                        "name": building["tags"].get("name"),
                    }
                )
    first = min(events, key=lambda row: (row["parameter"], row["osm_way_id"])) if events else None
    return {
        "class": first["kind"] if first is not None else "VISIBLE_TARGET_ENTRANCE",
        "camera_to_entrance_distance_m": distance_m,
        "target_boundary_parameters": target_parameters,
        "first_occlusion": first,
        "all_pre_entrance_intersections": sorted(
            events, key=lambda row: (row["parameter"], row["osm_way_id"])
        ),
    }


def _panorama_ray(
    frame: dict[str, Any], projection_protocol: dict[str, Any]
) -> dict[str, Any]:
    image = frame["panorama"]
    provider_item = image["provider_item"]
    if str(provider_item.get("id")) != str(image["image_id"]):
        raise ValueError(f"PROVIDER_ITEM_ID_MISMATCH:{frame['key']}")
    entrance = frame["target"]["entrance_node"]
    entrance_lon_lat = entrance["lon_lat"]
    projected = project_entrance_ray(
        provider_item,
        {
            "id": entrance["id"],
            "lon": entrance_lon_lat[0],
            "lat": entrance_lon_lat[1],
        },
        projection_protocol,
        downloaded_image_size=tuple(image["image_size"]),
    )
    camera = [float(value) for value in provider_item["geometry"]["coordinates"]]
    if any(
        abs(camera[index] - float(image["camera_lon_lat"][index])) > 1e-10
        for index in range(2)
    ):
        raise ValueError(f"PROVIDER_CAMERA_MISMATCH:{frame['key']}")
    bearing = float(projected["initial_bearing_degrees"])
    heading = float(image["raw_center_heading_degrees"])
    if circular_distance_degrees(
        heading, float(projected["view_azimuth_degrees"])
    ) > 1e-9:
        raise ValueError(f"RAW_CENTER_HEADING_NOT_PROVIDER_VIEW_AZIMUTH:{frame['key']}")
    relative = (bearing - heading + 180.0) % 360.0 - 180.0
    pixel_x = float(projected["raw_x_pixels"])
    basic_x = pixel_x / int(image["image_size"][0])
    return {
        "entrance_bearing_degrees": bearing,
        "raw_center_heading_degrees": heading,
        "relative_bearing_degrees": relative,
        "basic_x": basic_x,
        "pixel_x": pixel_x,
        "orientation_projection_gate": projected["projection_gate"],
    }


def _interval_member(pixel_x: float, interval: Sequence[float] | None) -> bool | None:
    if interval is None:
        return None
    return float(interval[0]) <= pixel_x <= float(interval[1])


def _validate_interval(
    interval: Sequence[float] | None, width: int, label: str
) -> None:
    if interval is None:
        return
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{label}_INVALID")
    lower, upper = map(float, interval)
    if not all(math.isfinite(value) for value in (lower, upper)):
        raise ValueError(f"{label}_NONFINITE")
    if not 0.0 <= lower <= upper < float(width):
        raise ValueError(f"{label}_OUT_OF_RANGE")


def _circular_pixel_error(pixel_x: float, center_x: float, width: int) -> float:
    delta = abs(pixel_x - center_x) % width
    return min(delta, width - delta)


def _token(
    frame: dict[str, Any], protocol_sha256: str, ray: dict[str, float]
) -> str:
    payload = json.dumps(
        {
            "authority": "ENTITY_LINKED_OSM_ENTRANCE_RAY",
            "provider": frame["panorama"]["provider"],
            "provider_image_id": str(frame["panorama"]["image_id"]),
            "osm_way_id": int(frame["target"]["building"]["id"]),
            "osm_way_version": int(frame["target"]["building"]["version"]),
            "entrance_node_id": int(frame["target"]["entrance_node"]["id"]),
            "basic_x": round(ray["basic_x"], 12),
            "protocol_sha256": protocol_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise ValueError(f"OUTPUT_ALREADY_EXISTS:{output_path}")
    protocol_sha256 = _sha256(protocol_path)
    protocol = _json(protocol_path)
    if _sha256(Path(__file__).resolve()) != protocol["evaluator"]["sha256"]:
        raise ValueError("EVALUATOR_HASH_MISMATCH")
    projection_evaluator_path = Path(__file__).with_name("l10_panolab_entrance_ray.py")
    if _sha256(projection_evaluator_path) != protocol["orientation_projection"]["evaluator"]["sha256"]:
        raise ValueError("ORIENTATION_PROJECTION_EVALUATOR_HASH_MISMATCH")
    projection_protocol_path = _verify_file(protocol["orientation_projection"]["protocol"])
    projection_protocol = _json(projection_protocol_path)
    cohort_path = _verify_file(protocol["source"])
    versions = {
        "python": sys.version.split()[0],
        "pillow": importlib.metadata.version("Pillow"),
    }
    if versions != protocol["runtime"]["versions"]:
        raise ValueError(f"RUNTIME_VERSION_MISMATCH:{versions}")
    cohort = _json(cohort_path)
    truth_freeze_audit = cohort["truth_freeze_audit"]
    required_truth_freeze = {
        "projection_calls_before_portal_truth_freeze": 0,
        "evaluator_calls_before_portal_truth_freeze": 0,
        "ocr_calls_for_portal_truth": 0,
        "marker_visible_during_portal_truth": False,
        "geometry_or_compass_visible_during_portal_truth": False,
    }
    if any(
        truth_freeze_audit.get(key) != value
        for key, value in required_truth_freeze.items()
    ):
        raise ValueError("PORTAL_TRUTH_NOT_MARKER_AND_GEOMETRY_BLIND_AT_FREEZE")
    frames = cohort["frames"]
    roles = Counter(frame["role"] for frame in frames)
    if roles != ROLE_COUNTS:
        raise ValueError(f"ROLE_COUNT_MISMATCH:{dict(roles)}")
    if len({frame["panorama"]["sequence_id"] for frame in frames}) != 8:
        raise ValueError("PANORAMA_SEQUENCES_NOT_UNIQUE")
    if len({int(frame["target"]["building"]["id"]) for frame in frames}) != 8:
        raise ValueError("TARGET_BUILDINGS_NOT_UNIQUE")

    clearance_m = float(protocol["geometry"]["pre_entrance_clearance_m"])
    provider_contract = protocol["panorama_provider"]
    rows = []
    for frame in frames:
        image_spec = frame["panorama"]
        if image_spec["provider"] != provider_contract["name"]:
            raise ValueError(f"PANORAMA_PROVIDER_MISMATCH:{frame['key']}")
        if (
            image_spec["heading_provenance"]
            not in provider_contract["allowed_heading_provenance"]
        ):
            raise ValueError(f"HEADING_PROVENANCE_NOT_ALLOWED:{frame['key']}")
        if (
            image_spec["raw_center_heading_rule"]
            != provider_contract["raw_center_heading_rule"]
        ):
            raise ValueError(f"RAW_CENTER_HEADING_RULE_MISMATCH:{frame['key']}")
        if image_spec["projection_type"] != "equirectangular":
            raise ValueError(f"NON_EQUIRECTANGULAR_IMAGE:{frame['key']}")
        rotation = image_spec["yaw_pitch_roll_degrees"]
        if len(rotation) != 3 or not all(
            math.isfinite(float(value)) for value in rotation
        ):
            raise ValueError(f"COMPUTED_ROTATION_INVALID:{frame['key']}")
        if any(abs(float(value)) > 1e-9 for value in rotation):
            raise ValueError(f"NONZERO_PANORAMA_POSE_FORBIDDEN:{frame['key']}")
        heading = float(image_spec["raw_center_heading_degrees"])
        if not math.isfinite(heading) or not 0.0 <= heading < 360.0:
            raise ValueError(f"COMPUTED_COMPASS_INVALID:{frame['key']}")
        image_path = _path(image_spec["local_path"])
        if _sha256(image_path) != image_spec["image_sha256"]:
            raise ValueError(f"IMAGE_HASH_MISMATCH:{frame['key']}")
        if image_path.stat().st_size != int(image_spec["image_bytes"]):
            raise ValueError(f"IMAGE_BYTE_COUNT_MISMATCH:{frame['key']}")
        with Image.open(image_path) as image:
            actual_size = [int(image.width), int(image.height)]
        if actual_size != image_spec["image_size"] or actual_size[0] != 2 * actual_size[1]:
            raise ValueError(f"PANORAMA_SIZE_CONTRACT_MISMATCH:{frame['key']}:{actual_size}")
        role = frame["role"]
        if role == "VISIBLE_TARGET_ENTRANCE" and "target_portal_interval_x" not in frame:
            raise ValueError(f"VISIBLE_TRUTH_INTERVAL_MISSING:{frame['key']}")
        if role != "VISIBLE_TARGET_ENTRANCE" and "target_portal_interval_x" in frame:
            raise ValueError(f"OCCLUDED_TARGET_INTERVAL_FORBIDDEN:{frame['key']}")
        if role != "NON_TARGET_BUILDING_OCCLUDED" and "decoy_portal_interval_x" in frame:
            raise ValueError(f"DECOY_INTERVAL_FORBIDDEN:{frame['key']}")

        target_interval = frame.get("target_portal_interval_x")
        decoy_interval = frame.get("decoy_portal_interval_x")
        _validate_interval(target_interval, actual_size[0], f"TARGET_INTERVAL:{frame['key']}")
        _validate_interval(decoy_interval, actual_size[0], f"DECOY_INTERVAL:{frame['key']}")

        visibility = _visibility(frame, clearance_m)
        ray = _panorama_ray(frame, projection_protocol)
        authorized = visibility["class"] == "VISIBLE_TARGET_ENTRANCE"
        target_hit = bool(authorized and _interval_member(ray["pixel_x"], target_interval))
        decoy_hit = bool(authorized and _interval_member(ray["pixel_x"], decoy_interval))
        token = _token(frame, protocol_sha256, ray) if authorized else None
        center_error_degrees = None
        hit_margin_degrees = None
        if target_interval is not None:
            width = int(image_spec["image_size"][0])
            center = 0.5 * (float(target_interval[0]) + float(target_interval[1]))
            center_error_degrees = _circular_pixel_error(ray["pixel_x"], center, width) * 360.0 / width
            if target_hit:
                hit_margin_degrees = min(
                    ray["pixel_x"] - float(target_interval[0]),
                    float(target_interval[1]) - ray["pixel_x"],
                ) * 360.0 / width
        if visibility["class"] == "VISIBLE_TARGET_ENTRANCE" and target_hit:
            state, action = "ENTITY_LINKED_ENTRANCE_RAY", None
        elif visibility["class"] == "VISIBLE_TARGET_ENTRANCE":
            state, action = "ENTRANCE_RAY_TRUTH_MISS", "SWEEP_ENTRANCE"
        elif visibility["class"] == "TARGET_SELF_OCCLUDED":
            state, action = "TARGET_ENTRANCE_SELF_OCCLUDED", "SIDESTEP_TO_ENTRANCE_FACE"
        else:
            state, action = "TARGET_ENTRANCE_OCCLUDED_BY_NON_TARGET", "SIDESTEP_CLEAR_OCCLUDER"
        rows.append(
            {
                "key": frame["key"],
                "role": role,
                "provider": image_spec["provider"],
                "provider_image_id": str(image_spec["image_id"]),
                "osm_way_id": int(frame["target"]["building"]["id"]),
                "entrance_node_id": int(frame["target"]["entrance_node"]["id"]),
                "visibility": visibility,
                "visibility_role_correct": visibility["class"] == role,
                "ray": ray,
                "target_portal_interval_x": target_interval,
                "decoy_portal_interval_x": decoy_interval,
                "authorized": authorized,
                "authority_token": token,
                "target_hit": target_hit,
                "decoy_hit": decoy_hit,
                "target_center_error_degrees": center_error_degrees,
                "target_hit_margin_degrees": hit_margin_degrees,
                "state": state,
                "deficit_action": action,
            }
        )

    visible = [row for row in rows if row["role"] == "VISIBLE_TARGET_ENTRANCE"]
    occluded = [row for row in rows if row["role"] != "VISIBLE_TARGET_ENTRANCE"]
    decoys = [row for row in rows if row["role"] == "NON_TARGET_BUILDING_OCCLUDED"]
    metrics = {
        "visibility_role_correct": sum(row["visibility_role_correct"] for row in rows),
        "visible_target_ray_hits": sum(row["target_hit"] for row in visible),
        "occluded_false_authorizations": sum(row["authorized"] for row in occluded),
        "annotated_non_target_decoys": sum(
            row["decoy_portal_interval_x"] is not None for row in decoys
        ),
        "non_target_decoy_ray_hits": sum(row["decoy_hit"] for row in decoys),
        "visible_mean_center_error_degrees": sum(
            float(row["target_center_error_degrees"]) for row in visible
        )
        / len(visible),
        "visible_max_center_error_degrees": max(
            float(row["target_center_error_degrees"]) for row in visible
        ),
        "visible_min_hit_margin_degrees": min(
            float(row["target_hit_margin_degrees"])
            for row in visible
            if row["target_hit_margin_degrees"] is not None
        )
        if any(row["target_hit_margin_degrees"] is not None for row in visible)
        else None,
    }
    metrics["balanced_accuracy"] = 0.5 * (
        metrics["visible_target_ray_hits"] / len(visible)
        + (len(occluded) - metrics["occluded_false_authorizations"]) / len(occluded)
    )
    gate = {
        "visibility_role_8_of_8": metrics["visibility_role_correct"] == 8,
        "visible_target_ray_hit_4_of_4": metrics["visible_target_ray_hits"] == 4,
        "zero_occluded_false_authorization": metrics["occluded_false_authorizations"] == 0,
        "balanced_accuracy_1": metrics["balanced_accuracy"] == 1.0,
    }
    passed = all(gate.values())
    decision = protocol["decision_names"]["gate_met" if passed else "gate_not_met"]
    result = {
        "schema": protocol["result_schema"],
        "decision": decision,
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "cohort_sha256": protocol["source"]["sha256"],
        "evaluator_sha256": protocol["evaluator"]["sha256"],
        "truth_freeze_audit": truth_freeze_audit,
        "metrics": metrics,
        "gate": {**gate, "passed": passed},
        "runtime": versions,
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"decision": decision, "metrics": metrics, "gate": result["gate"]}, indent=2))


if __name__ == "__main__":
    main()
