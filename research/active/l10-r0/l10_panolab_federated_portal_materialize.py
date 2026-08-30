#!/usr/bin/env python3
"""Materialize a frozen federated Panoramax portal cohort and ray viewports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2

from l10_panolab_close_portal_source import _ensure_image, _rectilinear_viewport, sha256
from l10_panolab_entrance_ray import initial_bearing_degrees, wrap360
from l10_panolab_viewer_equivalent_projection import projection_gate


RESULT_SCHEMA = "blindassist-l10-panolab-federated-portal-metadata-result-v1"
SOURCE_SCHEMA = "blindassist-l10-panolab-federated-portal-source-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def project_ray(
    item: dict[str, Any],
    entrance: dict[str, Any],
    strict_protocol: dict[str, Any],
    successor_protocol: dict[str, Any],
    image_size: tuple[int, int],
) -> dict[str, Any]:
    gate = projection_gate(
        item,
        strict_protocol,
        successor_protocol,
        downloaded_image_size=image_size,
    )
    require(gate["eligible"], f"PROJECTION_GATE_FAILED:{item.get('id')}:{gate['failures']}")
    camera = [float(value) for value in item["geometry"]["coordinates"]]
    target = [float(entrance["lon"]), float(entrance["lat"])]
    bearing = initial_bearing_degrees(camera, target)
    relative_yaw = wrap360(bearing - float(gate["view_azimuth_degrees"]))
    raw_x_degrees = wrap360(relative_yaw + 180.0)
    raw_x_pixels = raw_x_degrees / 360.0 * int(gate["sensor_width_pixels"])
    return {
        "schema": "blindassist-l10-panolab-viewer-equivalent-entrance-ray-v1",
        "item_id": item.get("id"),
        "entrance_node_id": entrance["id"],
        "camera_lon_lat": camera,
        "entrance_lon_lat": target,
        "initial_bearing_degrees": round(bearing, 6),
        "view_azimuth_degrees": gate["view_azimuth_degrees"],
        "relative_viewer_yaw_degrees": round(relative_yaw, 6),
        "raw_x_degrees": round(raw_x_degrees, 6),
        "raw_x_pixels": round(raw_x_pixels, 6),
        "world_horizon_raw_y_pixels": round(int(gate["sensor_height_pixels"]) / 2),
        "projection_gate": gate,
    }


def materialize_role(
    episode_id: str,
    role: str,
    row: dict[str, Any],
    entrance: dict[str, Any],
    strict_protocol: dict[str, Any],
    successor_protocol: dict[str, Any],
    asset_root: Path,
    viewport: dict[str, Any],
) -> dict[str, Any]:
    image = _ensure_image(row["item"], row["item_id"], asset_root / "images", {})
    panorama = cv2.imread(image["path"], cv2.IMREAD_COLOR)
    require(panorama is not None, f"IMAGE_DECODE_FAILED:{row['item_id']}")
    ray = project_ray(
        row["item"],
        entrance,
        strict_protocol,
        successor_protocol,
        tuple(image["image_size"]),
    )
    rendered = _rectilinear_viewport(
        panorama,
        float(ray["raw_x_pixels"]),
        int(viewport["width_pixels"]),
        int(viewport["height_pixels"]),
        float(viewport["horizontal_fov_degrees"]),
        float(viewport["center_pitch_degrees"]),
    )
    viewport_root = asset_root / "viewports" / role.lower()
    viewport_root.mkdir(parents=True, exist_ok=True)
    viewport_path = viewport_root / f"{episode_id}-{row['item_id']}.jpg"
    require(
        cv2.imwrite(
            str(viewport_path),
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, int(viewport["jpeg_quality"])],
        ),
        f"VIEWPORT_WRITE_FAILED:{row['item_id']}",
    )
    return {
        "role": role.upper(),
        "item_id": row["item_id"],
        "collection": row["collection"],
        "origin": row["origin"],
        "camera_to_entrance_distance_m": row["distance_m"],
        "provider_item": row["item"],
        "panorama": image,
        "entrance_ray": ray,
        "viewport": {
            "path": str(viewport_path.resolve()),
            "sha256": sha256(viewport_path),
            "bytes": viewport_path.stat().st_size,
            "image_size": [int(rendered.shape[1]), int(rendered.shape[0])],
            "projected_entrance_x_pixels": 512.0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-result", type=Path, required=True)
    parser.add_argument("--strict-protocol", type=Path, required=True)
    parser.add_argument("--successor-protocol", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    metadata_path = args.metadata_result.resolve()
    source_path = args.source.resolve()
    require(not source_path.exists(), f"SOURCE_ALREADY_EXISTS:{source_path}")
    metadata = load_json(metadata_path)
    require(metadata.get("schema") == RESULT_SCHEMA, "METADATA_RESULT_SCHEMA_MISMATCH")
    require(
        metadata.get("decision") == "L10_PANOLAB_FEDERATED_VIEWER_EQUIVALENT_PORTAL_METADATA_GATE_MET",
        "METADATA_GATE_NOT_MET",
    )
    require(len(metadata.get("episodes") or []) == 5, "FROZEN_EPISODE_COUNT_MISMATCH")
    strict_path = args.strict_protocol.resolve()
    successor_path = args.successor_protocol.resolve()
    strict_protocol = load_json(strict_path)
    successor_protocol = load_json(successor_path)
    viewport = {
        "width_pixels": 1024,
        "height_pixels": 768,
        "horizontal_fov_degrees": 90.0,
        "center_pitch_degrees": -10.0,
        "jpeg_quality": 95,
    }
    episodes = []
    for frozen in metadata["episodes"]:
        entrance = frozen["main_entrance_node"]
        episodes.append(
            {
                "episode_id": frozen["episode_id"],
                "source_city": frozen["source_city"],
                "target_way_id": frozen["target_way_id"],
                "target_name": frozen["target_name"],
                "main_entrance_node": entrance,
                "pair_score": frozen["pair_score"],
                "reference": materialize_role(
                    frozen["episode_id"],
                    "reference",
                    frozen["reference"],
                    entrance,
                    strict_protocol,
                    successor_protocol,
                    args.asset_root.resolve(),
                    viewport,
                ),
                "query": materialize_role(
                    frozen["episode_id"],
                    "query",
                    frozen["query"],
                    entrance,
                    strict_protocol,
                    successor_protocol,
                    args.asset_root.resolve(),
                    viewport,
                ),
            }
        )
        print(f"MATERIALIZED {frozen['episode_id']}", flush=True)
    source = {
        "schema": SOURCE_SCHEMA,
        "status": "FROZEN_FEDERATED_PAIR_MATERIALIZED_FOR_ROLE_SEPARATED_SOURCE_AUDIT",
        "metadata_result": "research/active/l10-r0/" + metadata_path.name,
        "metadata_result_sha256": sha256(metadata_path),
        "strict_projection_protocol": "research/active/l10-r0/" + strict_path.name,
        "strict_projection_protocol_sha256": sha256(strict_path),
        "successor_projection_protocol": "research/active/l10-r0/" + successor_path.name,
        "successor_projection_protocol_sha256": sha256(successor_path),
        "viewport": {**viewport, "projected_entrance_x_pixels": 512.0},
        "episode_count": len(episodes),
        "image_count": 2 * len(episodes),
        "distinct_target_way_count": len({row["target_way_id"] for row in episodes}),
        "cross_collection_episode_count": sum(
            row["reference"]["collection"] != row["query"]["collection"] for row in episodes
        ),
        "viewer_equivalent_image_count": sum(
            role["entrance_ray"]["projection_gate"]["projection_mode"]
            == "OFFICIAL_VIEWER_EFFECTIVE_ZERO"
            for row in episodes
            for role in (row["reference"], row["query"])
        ),
        "pixel_exposure_boundary": "All five target ways and ten item IDs were frozen in the hashed metadata result before any selected HD asset request. This file records deterministic materialization only; reference/query source admission remains unset.",
        "episodes": episodes,
        "claim_boundary": metadata["claim_boundary"],
    }
    write_json(source_path, source)
    print(
        json.dumps(
            {
                "source": str(source_path),
                "episodes": len(episodes),
                "images": 2 * len(episodes),
                "viewer_equivalent_images": source["viewer_equivalent_image_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
