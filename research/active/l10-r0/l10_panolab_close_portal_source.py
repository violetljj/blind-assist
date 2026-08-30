#!/usr/bin/env python3
"""Freeze and materialize close cross-collection Panoramax portal views."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import urllib.request
from itertools import combinations
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from l10_panolab_entrance_ray import (  # noqa: E402
    circular_distance_degrees,
    initial_bearing_degrees,
    projection_gate,
)


PROTOCOL_SCHEMA = "blindassist-l10-panolab-close-portal-source-protocol-v1"
FREEZE_SCHEMA = "blindassist-l10-panolab-close-portal-metadata-freeze-v1"
SOURCE_SCHEMA = "blindassist-l10-panolab-close-portal-source-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _collect_item_ids(value: Any, output: set[str]) -> None:
    if isinstance(value, dict):
        item_id = value.get("item_id")
        if isinstance(item_id, str):
            output.add(item_id)
        for child in value.values():
            _collect_item_ids(child, output)
    elif isinstance(value, list):
        for child in value:
            _collect_item_ids(child, output)


def _consumed_item_ids(protocol: dict[str, Any]) -> tuple[set[str], list[dict[str, Any]]]:
    item_ids: set[str] = set()
    receipts = []
    for row in protocol["source"]["consumed_source_manifests"]:
        path = resolve(row["path"])
        actual = sha256(path)
        require(actual == row["sha256"], f"CONSUMED_SOURCE_HASH_MISMATCH:{path}")
        payload = load_json(path)
        before = len(item_ids)
        _collect_item_ids(payload, item_ids)
        receipts.append(
            {
                "path": str(path.resolve()),
                "sha256": actual,
                "explicit_item_ids_added": len(item_ids) - before,
            }
        )
    return item_ids, receipts


def _eligible_direct_rows(
    candidate: dict[str, Any],
    orientation: dict[str, Any],
    consumed_item_ids: set[str],
    minimum_distance_m: float,
    maximum_distance_m: float,
) -> list[dict[str, Any]]:
    entrance = candidate["main_entrance_node"]
    entrance_lon_lat = [float(entrance["lon"]), float(entrance["lat"])]
    rows = []
    seen: set[str] = set()
    for support in candidate["supports"]["direct"]:
        item_id = str(support["item_id"])
        if item_id in seen or item_id in consumed_item_ids:
            continue
        seen.add(item_id)
        item = candidate["items"].get(item_id)
        if not isinstance(item, dict):
            continue
        distance_m = float(support["first_intersection"]["distance_from_camera_m"])
        if not minimum_distance_m <= distance_m <= maximum_distance_m:
            continue
        gate = projection_gate(item, orientation)
        if not gate["eligible"]:
            continue
        camera_lon_lat = [float(value) for value in item["geometry"]["coordinates"]]
        rows.append(
            {
                "item_id": item_id,
                "collection": str(item["collection"]),
                "camera_to_entrance_distance_m": round(distance_m, 3),
                "camera_to_entrance_bearing_degrees": round(
                    initial_bearing_degrees(camera_lon_lat, entrance_lon_lat), 6
                ),
                "orientation_gate": gate,
            }
        )
    return rows


def _best_cross_collection_pair(
    rows: list[dict[str, Any]],
) -> tuple[tuple[Any, ...], dict[str, Any], dict[str, Any]] | None:
    candidates = []
    for left, right in combinations(rows, 2):
        if left["collection"] == right["collection"]:
            continue
        bearing_delta = circular_distance_degrees(
            float(left["camera_to_entrance_bearing_degrees"]),
            float(right["camera_to_entrance_bearing_degrees"]),
        )
        score = (
            max(
                float(left["camera_to_entrance_distance_m"]),
                float(right["camera_to_entrance_distance_m"]),
            ),
            bearing_delta,
            abs(
                float(left["camera_to_entrance_distance_m"])
                - float(right["camera_to_entrance_distance_m"])
            ),
            tuple(sorted((left["collection"], right["collection"]))),
            tuple(sorted((left["item_id"], right["item_id"]))),
        )
        ordered = sorted((left, right), key=lambda row: (row["collection"], row["item_id"]))
        candidates.append((score, ordered[0], ordered[1]))
    return min(candidates, key=lambda row: row[0]) if candidates else None


def _image_size(path: Path) -> tuple[int, int]:
    import cv2  # noqa: PLC0415

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    require(image is not None, f"IMAGE_DECODE_FAILED:{path}")
    height, width = image.shape[:2]
    return width, height


def _download_index(manifests: list[str]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for value in manifests:
        path = resolve(value)
        if not path.exists():
            continue
        for row in load_json(path).get("images", []):
            image_path = Path(row["path"])
            if not image_path.exists():
                continue
            width, height = _image_size(image_path)
            index[str(row["item_id"])] = {
                "path": str(image_path.resolve()),
                "sha256": sha256(image_path),
                "bytes": image_path.stat().st_size,
                "image_size": [width, height],
                "provenance": str(path.resolve()),
            }
    return index


def _ensure_image(
    item: dict[str, Any], item_id: str, asset_root: Path, known: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    if item_id in known:
        return dict(known[item_id])
    href = ((item.get("assets") or {}).get("hd") or {}).get("href")
    require(isinstance(href, str) and href.startswith("https://"), f"HD_ASSET_MISSING:{item_id}")
    asset_root.mkdir(parents=True, exist_ok=True)
    target = asset_root / f"{item_id}.jpg"
    temporary: Path | None = None
    try:
        if not target.exists():
            descriptor, name = tempfile.mkstemp(
                prefix=f"{item_id}-", suffix=".partial", dir=asset_root
            )
            os.close(descriptor)
            temporary = Path(name)
            request = urllib.request.Request(
                href, headers={"User-Agent": "BlindAssist-L10-Close-Portal/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            os.replace(temporary, target)
            temporary = None
        width, height = _image_size(target)
        return {
            "path": str(target.resolve()),
            "sha256": sha256(target),
            "bytes": target.stat().st_size,
            "image_size": [width, height],
            "provenance": href,
        }
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _rectilinear_viewport(
    panorama: Any,
    center_x: float,
    width: int,
    height: int,
    horizontal_fov_degrees: float,
    center_pitch_degrees: float,
) -> Any:
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    pano_height, pano_width = panorama.shape[:2]
    focal = 0.5 * width / math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    screen_x = (np.arange(width, dtype=np.float32) + 0.5 - width / 2.0) / focal
    screen_y = -(np.arange(height, dtype=np.float32) + 0.5 - height / 2.0) / focal
    x, y = np.meshgrid(screen_x, screen_y)
    z = np.ones_like(x)
    pitch = math.radians(center_pitch_degrees)
    rotated_y = math.cos(pitch) * y + math.sin(pitch) * z
    rotated_z = -math.sin(pitch) * y + math.cos(pitch) * z
    yaw = np.arctan2(x, rotated_z)
    latitude = np.arctan2(rotated_y, np.sqrt(x * x + rotated_z * rotated_z))
    map_x = np.mod(center_x + yaw / (2.0 * math.pi) * pano_width, pano_width).astype(np.float32)
    map_y = np.clip(
        pano_height / 2.0 - latitude / math.pi * pano_height, 0, pano_height - 1
    ).astype(np.float32)
    return cv2.remap(
        panorama,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _viewport_record(
    role: str,
    row: dict[str, Any],
    candidate: dict[str, Any],
    orientation: dict[str, Any],
    image_root: Path,
    viewport_root: Path,
    known: dict[str, dict[str, Any]],
    viewport_spec: dict[str, Any],
) -> dict[str, Any]:
    import cv2  # noqa: PLC0415
    from l10_panolab_entrance_ray import project_entrance_ray  # noqa: PLC0415

    image = _ensure_image(row["item"], row["item_id"], image_root, known)
    full = cv2.imread(image["path"], cv2.IMREAD_COLOR)
    require(full is not None, f"IMAGE_DECODE_FAILED:{row['item_id']}")
    ray = project_entrance_ray(
        row["item"],
        candidate["main_entrance_node"],
        orientation,
        downloaded_image_size=tuple(image["image_size"]),
    )
    viewport = _rectilinear_viewport(
        full,
        float(ray["raw_x_pixels"]),
        int(viewport_spec["width_pixels"]),
        int(viewport_spec["height_pixels"]),
        float(viewport_spec["horizontal_fov_degrees"]),
        float(viewport_spec["center_pitch_degrees"]),
    )
    viewport_root.mkdir(parents=True, exist_ok=True)
    viewport_path = viewport_root / f"{row['item_id']}.jpg"
    encoded = cv2.imwrite(
        str(viewport_path),
        viewport,
        [cv2.IMWRITE_JPEG_QUALITY, int(viewport_spec["jpeg_quality"])],
    )
    require(encoded, f"VIEWPORT_WRITE_FAILED:{row['item_id']}")
    return {
        "role": role,
        "item_id": row["item_id"],
        "collection": row["collection"],
        "camera_to_entrance_distance_m": round(float(row["distance_m"]), 3),
        "provider_item": row["item"],
        "panorama": image,
        "entrance_ray": ray,
        "viewport": {
            "path": str(viewport_path.resolve()),
            "sha256": sha256(viewport_path),
            "bytes": viewport_path.stat().st_size,
            "image_size": [int(viewport.shape[1]), int(viewport.shape[0])],
            "projected_entrance_x_pixels": float(
                viewport_spec["projected_entrance_x_pixels"]
            ),
        },
    }


def freeze(protocol_path: Path, freeze_path: Path) -> dict[str, Any]:
    require(not freeze_path.exists(), f"FREEZE_ALREADY_EXISTS:{freeze_path}")
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    spec = protocol["source"]
    candidates_path = resolve(spec["candidates_path"])
    orientation_path = resolve(spec["orientation_protocol_path"])
    require(sha256(candidates_path) == spec["candidates_sha256"], "CANDIDATES_HASH_MISMATCH")
    require(
        sha256(orientation_path) == spec["orientation_protocol_sha256"],
        "ORIENTATION_HASH_MISMATCH",
    )
    candidates_payload = load_json(candidates_path)
    orientation = load_json(orientation_path)
    consumed_item_ids, consumed_receipts = _consumed_item_ids(protocol)
    selection = protocol["selection"]
    excluded_way_ids = {int(value) for value in selection["excluded_way_ids"]}
    minimum_distance_m = float(selection["camera_distance_m"][0])
    maximum_distance_m = float(selection["camera_distance_m"][1])

    eligible = []
    for candidate in candidates_payload["candidates"]:
        way_id = int(candidate["target_way"]["id"])
        if way_id in excluded_way_ids:
            continue
        rows = _eligible_direct_rows(
            candidate,
            orientation,
            consumed_item_ids,
            minimum_distance_m,
            maximum_distance_m,
        )
        pair = _best_cross_collection_pair(rows)
        if pair is None:
            continue
        score, reference, query = pair
        eligible.append(
            {
                "sort_key": score,
                "query_index": int(candidate["query_index"]),
                "source_city": str(candidate["source_city"]),
                "target_way_id": way_id,
                "target_name": candidate["target_way"].get("tags", {}).get("name"),
                "target_entrance_node_id": int(candidate["main_entrance_node"]["id"]),
                "strict_direct_item_count_within_distance": len(rows),
                "strict_direct_collection_count_within_distance": len(
                    {row["collection"] for row in rows}
                ),
                "reference": reference,
                "query": query,
            }
        )
    eligible.sort(
        key=lambda row: (
            row["sort_key"],
            row["target_way_id"],
            row["target_entrance_node_id"],
            row["query_index"],
        )
    )

    selected = []
    selected_ways: set[int] = set()
    selected_items: set[str] = set()
    for row in eligible:
        pair_items = {row["reference"]["item_id"], row["query"]["item_id"]}
        if row["target_way_id"] in selected_ways or pair_items & selected_items:
            continue
        selected.append(row)
        selected_ways.add(row["target_way_id"])
        selected_items.update(pair_items)
        if len(selected) == int(selection["cohort_size"]):
            break
    require(
        len(selected) == int(selection["cohort_size"]),
        f"INSUFFICIENT_CLOSE_CROSS_COLLECTION_COHORT:{len(selected)}",
    )

    episodes = []
    for index, row in enumerate(selected, 1):
        copied = dict(row)
        copied.pop("sort_key")
        copied["episode_id"] = f"CP{index:02d}"
        copied["pair_score"] = {
            "maximum_camera_distance_m": round(float(row["sort_key"][0]), 3),
            "camera_bearing_delta_degrees": round(float(row["sort_key"][1]), 6),
            "camera_distance_delta_m": round(float(row["sort_key"][2]), 3),
        }
        episodes.append(copied)

    result = {
        "schema": FREEZE_SCHEMA,
        "status": "METADATA_FROZEN_BEFORE_ANY_SELECTED_PIXEL_DOWNLOAD_OR_INSPECTION",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "orientation_protocol": str(orientation_path.resolve()),
        "orientation_protocol_sha256": sha256(orientation_path),
        "consumed_source_receipts": consumed_receipts,
        "consumed_item_id_count": len(consumed_item_ids),
        "excluded_way_id_count": len(excluded_way_ids),
        "candidate_count_scanned": len(candidates_payload["candidates"]),
        "eligible_candidate_count": len(eligible),
        "selected_episode_count": len(episodes),
        "strict_orientation_image_count": 2 * len(episodes),
        "cross_collection_episode_count": len(episodes),
        "pixel_views_before_freeze": 0,
        "world_bearing_to_raw_pixel_projection_calls_before_freeze": 0,
        "selection_rule": selection["rule"],
        "formal_source_admission_rule": protocol["source_admission"]["formal_rule"],
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(freeze_path, result)
    return result


def materialize(protocol_path: Path, freeze_path: Path, source_path: Path) -> dict[str, Any]:
    require(not source_path.exists(), f"SOURCE_ALREADY_EXISTS:{source_path}")
    protocol = load_json(protocol_path)
    frozen = load_json(freeze_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(frozen.get("schema") == FREEZE_SCHEMA, "FREEZE_SCHEMA_MISMATCH")
    require(frozen["protocol_sha256"] == sha256(protocol_path), "FREEZE_PROTOCOL_HASH_MISMATCH")
    spec = protocol["source"]
    candidates_path = resolve(spec["candidates_path"])
    orientation_path = resolve(spec["orientation_protocol_path"])
    require(frozen["candidates_sha256"] == sha256(candidates_path), "FREEZE_CANDIDATES_HASH_MISMATCH")
    require(
        frozen["orientation_protocol_sha256"] == sha256(orientation_path),
        "FREEZE_ORIENTATION_HASH_MISMATCH",
    )

    candidates_payload = load_json(candidates_path)
    candidates = {int(row["query_index"]): row for row in candidates_payload["candidates"]}
    orientation = load_json(orientation_path)
    known = _download_index(spec.get("download_manifests", []))
    asset_root = resolve(spec["new_asset_root"])
    episodes = []
    for frozen_episode in frozen["episodes"]:
        query_index = int(frozen_episode["query_index"])
        require(query_index in candidates, f"CANDIDATE_NOT_FOUND:{query_index}")
        candidate = candidates[query_index]
        require(
            int(candidate["target_way"]["id"]) == int(frozen_episode["target_way_id"]),
            f"TARGET_WAY_DRIFT:{query_index}",
        )
        require(
            int(candidate["main_entrance_node"]["id"])
            == int(frozen_episode["target_entrance_node_id"]),
            f"ENTRANCE_NODE_DRIFT:{query_index}",
        )
        direct = {str(row["item_id"]): row for row in candidate["supports"]["direct"]}
        role_rows = {}
        for role in ("reference", "query"):
            frozen_role = frozen_episode[role]
            item_id = str(frozen_role["item_id"])
            require(item_id in direct, f"FROZEN_DIRECT_ITEM_MISSING:{item_id}")
            item = candidate["items"][item_id]
            require(str(item["collection"]) == frozen_role["collection"], f"COLLECTION_DRIFT:{item_id}")
            distance_m = float(direct[item_id]["first_intersection"]["distance_from_camera_m"])
            require(
                abs(distance_m - float(frozen_role["camera_to_entrance_distance_m"])) <= 0.001,
                f"DISTANCE_DRIFT:{item_id}",
            )
            role_rows[role] = {
                "item_id": item_id,
                "collection": str(item["collection"]),
                "distance_m": distance_m,
                "item": item,
            }
        episodes.append(
            {
                "episode_id": frozen_episode["episode_id"],
                "candidate_index": query_index,
                "target_way": candidate["target_way"],
                "target_entrance_node": candidate["main_entrance_node"],
                "reference": _viewport_record(
                    "REFERENCE",
                    role_rows["reference"],
                    candidate,
                    orientation,
                    asset_root / "images",
                    asset_root / "viewports" / "reference",
                    known,
                    protocol["viewport"],
                ),
                "query": _viewport_record(
                    "QUERY",
                    role_rows["query"],
                    candidate,
                    orientation,
                    asset_root / "images",
                    asset_root / "viewports" / "query",
                    known,
                    protocol["viewport"],
                ),
            }
        )

    source = {
        "schema": SOURCE_SCHEMA,
        "status": "FROZEN_METADATA_PAIR_MATERIALIZED_FOR_ROLE_SEPARATED_SOURCE_AUDIT",
        "protocol": str(protocol_path.resolve()),
        "protocol_sha256": sha256(protocol_path),
        "metadata_freeze": str(freeze_path.resolve()),
        "metadata_freeze_sha256": sha256(freeze_path),
        "candidates": str(candidates_path.resolve()),
        "candidates_sha256": sha256(candidates_path),
        "orientation_protocol": str(orientation_path.resolve()),
        "orientation_protocol_sha256": sha256(orientation_path),
        "episode_count": len(episodes),
        "distinct_target_way_count": len({int(row["target_way"]["id"]) for row in episodes}),
        "strict_orientation_image_count": 2 * len(episodes),
        "cross_collection_episode_count": sum(
            row["reference"]["collection"] != row["query"]["collection"] for row in episodes
        ),
        "pixel_exposure_boundary": "Pair choice and order were frozen before any selected pixel download. Materialization only downloaded the ten frozen HD panoramas, rendered deterministic ray-centred viewports, and recorded hashes; source admission remains unset.",
        "formal_source_admission_rule": protocol["source_admission"]["formal_rule"],
        "episodes": episodes,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(source_path, source)
    return source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "materialize"), required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    args = parser.parse_args()
    protocol_path = args.protocol.resolve()
    freeze_path = args.freeze.resolve()
    if args.mode == "freeze":
        result = freeze(protocol_path, freeze_path)
        print(
            json.dumps(
                {
                    "freeze": str(freeze_path),
                    "eligible_candidate_count": result["eligible_candidate_count"],
                    "selected_episode_count": result["selected_episode_count"],
                    "episodes": [
                        {
                            "episode_id": row["episode_id"],
                            "target_way_id": row["target_way_id"],
                            "target_name": row["target_name"],
                            "pair_score": row["pair_score"],
                        }
                        for row in result["episodes"]
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    require(args.source is not None, "--source is required for materialize")
    result = materialize(protocol_path, freeze_path, args.source.resolve())
    print(
        json.dumps(
            {
                "source": str(args.source.resolve()),
                "episode_count": result["episode_count"],
                "strict_orientation_image_count": result["strict_orientation_image_count"],
                "cross_collection_episode_count": result["cross_collection_episode_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
