#!/usr/bin/env python3
"""Project an OSM entrance node into an applicability-gated Panoramax panorama."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_PROTOCOL_SCHEMA = "blindassist-l10-panolab-orientation-projection-protocol-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_number(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    require(isinstance(value, str) and value, "numeric value is missing")
    if "/" in value:
        numerator, denominator = value.split("/", 1)
        return float(numerator) / float(denominator)
    return float(value)


def wrap360(value: float) -> float:
    return value % 360.0


def circular_distance_degrees(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def initial_bearing_degrees(camera_lon_lat: list[float], node_lon_lat: list[float]) -> float:
    require(len(camera_lon_lat) == 2 and len(node_lon_lat) == 2, "coordinates must be lon/lat pairs")
    lon1, lat1 = map(math.radians, camera_lon_lat)
    lon2, lat2 = map(math.radians, node_lon_lat)
    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return wrap360(math.degrees(math.atan2(y, x)))


def _node_lon_lat(node: dict[str, Any]) -> list[float]:
    if "lon" in node and "lat" in node:
        return [float(node["lon"]), float(node["lat"])]
    geometry = node.get("geometry")
    require(isinstance(geometry, dict) and geometry.get("type") == "Point", "entrance node lacks lon/lat")
    coordinates = geometry.get("coordinates")
    require(isinstance(coordinates, list) and len(coordinates) == 2, "entrance node point is invalid")
    return [float(coordinates[0]), float(coordinates[1])]


def projection_gate(
    item: dict[str, Any],
    protocol: dict[str, Any],
    *,
    downloaded_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    require(protocol.get("schema") == EXPECTED_PROTOCOL_SCHEMA, "unexpected projection protocol schema")
    gate = protocol["applicability_gate"]
    properties = item.get("properties") or {}
    interior = properties.get("pers:interior_orientation") or {}
    dimensions = interior.get("sensor_array_dimensions")
    exif = properties.get("exif") or {}

    failures: list[str] = []
    if interior.get("field_of_view") != gate["field_of_view_degrees"]:
        failures.append("FIELD_OF_VIEW_NOT_360")
    if not isinstance(dimensions, list) or len(dimensions) != 2:
        failures.append("SENSOR_DIMENSIONS_MISSING")
        width = height = None
    else:
        width, height = int(dimensions[0]), int(dimensions[1])
        if width != 2 * height:
            failures.append("SENSOR_NOT_2_TO_1")
        if downloaded_image_size is not None and tuple(downloaded_image_size) != (width, height):
            failures.append("DOWNLOADED_IMAGE_NOT_FULL_SENSOR")
    visible_area = interior.get("visible_area")
    if visible_area is not None:
        if not isinstance(visible_area, list) or len(visible_area) != 4:
            failures.append("VISIBLE_AREA_INVALID")
        elif float(visible_area[0]) != 0.0 or float(visible_area[2]) != 0.0:
            failures.append("HORIZONTAL_CROP_PRESENT")

    azimuth_value = properties.get("view:azimuth")
    gps_direction_value = exif.get("Exif.GPSInfo.GPSImgDirection")
    try:
        azimuth = parse_number(azimuth_value)
    except (TypeError, ValueError, ZeroDivisionError):
        azimuth = None
        failures.append("VIEW_AZIMUTH_MISSING")
    try:
        gps_direction = parse_number(gps_direction_value)
    except (TypeError, ValueError, ZeroDivisionError):
        gps_direction = None
        failures.append("CAMERA_NATIVE_HEADING_MISSING")
    if exif.get("Exif.GPSInfo.GPSImgDirectionRef") != "T":
        failures.append("CAMERA_HEADING_NOT_TRUE_NORTH")
    heading_delta = None
    if azimuth is not None and gps_direction is not None:
        heading_delta = circular_distance_degrees(azimuth, gps_direction)
        if heading_delta > float(gate["maximum_circular_difference_between_view_azimuth_and_exif_degrees"]):
            failures.append("VIEW_AZIMUTH_EXIF_MISMATCH")

    pose_values: dict[str, float | None] = {}
    for field, failure in (
        ("pers:yaw", "YAW_NOT_EXPLICIT_ZERO"),
        ("pers:pitch", "PITCH_NOT_EXPLICIT_ZERO"),
        ("pers:roll", "ROLL_NOT_EXPLICIT_ZERO"),
    ):
        value = properties.get(field)
        if value is None:
            pose_values[field] = None
            failures.append(failure)
        else:
            pose_values[field] = float(value)
            if abs(float(value)) > 1e-9:
                failures.append(failure)
    yaw = pose_values["pers:yaw"]
    pitch = pose_values["pers:pitch"]
    roll = pose_values["pers:roll"]

    if exif.get("Xmp.GPano.ProjectionType") != "equirectangular":
        failures.append("PROJECTION_TYPE_NOT_EXPLICIT_EQUIRECTANGULAR")
    for exif_field, value, failure in (
        ("Xmp.GPano.PoseHeadingDegrees", yaw, "XMP_YAW_CONFLICT"),
        ("Xmp.GPano.PosePitchDegrees", pitch, "XMP_PITCH_CONFLICT"),
        ("Xmp.GPano.PoseRollDegrees", roll, "XMP_ROLL_CONFLICT"),
    ):
        if exif_field in exif and value is not None:
            try:
                if abs(parse_number(exif[exif_field]) - value) > 1e-9:
                    failures.append(failure)
            except (TypeError, ValueError, ZeroDivisionError):
                failures.append(failure)

    geometry = item.get("geometry") or {}
    coordinates = geometry.get("coordinates")
    if geometry.get("type") != "Point" or not isinstance(coordinates, list) or len(coordinates) != 2:
        failures.append("CAMERA_POINT_MISSING")

    return {
        "eligible": not failures,
        "failures": failures,
        "item_id": item.get("id"),
        "view_azimuth_degrees": azimuth,
        "camera_native_heading_degrees": gps_direction,
        "heading_circular_difference_degrees": round(heading_delta, 6) if heading_delta is not None else None,
        "pers_yaw_degrees": yaw,
        "pers_pitch_degrees": pitch,
        "pers_roll_degrees": roll,
        "sensor_width_pixels": width,
        "sensor_height_pixels": height,
        "downloaded_image_size": list(downloaded_image_size) if downloaded_image_size is not None else None,
    }


def project_entrance_ray(
    item: dict[str, Any],
    entrance_node: dict[str, Any],
    protocol: dict[str, Any],
    *,
    downloaded_image_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    gate = projection_gate(item, protocol, downloaded_image_size=downloaded_image_size)
    require(gate["eligible"], "projection gate failed: " + ", ".join(gate["failures"]))
    camera = [float(value) for value in item["geometry"]["coordinates"]]
    entrance = _node_lon_lat(entrance_node)
    bearing = initial_bearing_degrees(camera, entrance)
    relative_yaw = wrap360(bearing - float(gate["view_azimuth_degrees"]))
    raw_x_degrees = wrap360(relative_yaw + 180.0)
    raw_x_pixels = raw_x_degrees / 360.0 * int(gate["sensor_width_pixels"])
    return {
        "schema": "blindassist-l10-panolab-entrance-ray-v1",
        "item_id": item.get("id"),
        "entrance_node_id": entrance_node.get("id"),
        "camera_lon_lat": camera,
        "entrance_lon_lat": entrance,
        "initial_bearing_degrees": round(bearing, 6),
        "view_azimuth_degrees": gate["view_azimuth_degrees"],
        "relative_viewer_yaw_degrees": round(relative_yaw, 6),
        "raw_x_degrees": round(raw_x_degrees, 6),
        "raw_x_pixels": round(raw_x_pixels, 6),
        "world_horizon_raw_y_pixels": round(int(gate["sensor_height_pixels"]) / 2),
        "projection_gate": gate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--item", type=Path, required=True)
    parser.add_argument("--entrance", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--image-height", type=int)
    args = parser.parse_args()
    require((args.image_width is None) == (args.image_height is None), "image width and height must be supplied together")
    image_size = None if args.image_width is None else (args.image_width, args.image_height)
    item = json.loads(args.item.read_text(encoding="utf-8"))
    entrance = json.loads(args.entrance.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    print(json.dumps(project_entrance_ray(item, entrance, protocol, downloaded_image_size=image_size), indent=2))


if __name__ == "__main__":
    main()
