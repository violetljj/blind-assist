"""Run the small real-image P0-S0-R1 materialization canary.

Grounding DINO only proposes image-space regions.  Map, geometry, multiview,
conflict, and final admission remain owned by ``materializer.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import requests
import PIL
from PIL import Image, ImageDraw

from scripts.research.goal_copilot_bridge.p0_s0_materialization import materializer


MODEL_REPOSITORY = "IDEA-Research/grounding-dino-tiny"
MODEL_REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHTS_FILENAME = "model.safetensors"
WEIGHTS_SHA256 = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"
MODEL_LICENSE = "Apache-2.0"
PROMPT = "door . doorway . entrance . building entrance . storefront entrance . gate ."
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10
NMS_IOU_THRESHOLD = 0.50
MAX_PROPOSALS_PER_IMAGE = 100
IMAGE_COUNT = 30
MAPILLARY_FIELDS = (
    "id", "computed_geometry", "captured_at", "compass_angle", "computed_compass_angle",
    "camera_type", "camera_parameters", "width", "height",
)
GENERATOR_AUTHORITY = "VISUAL_PROPOSAL_ONLY"
GENERATOR_LINEAGE = "GROUNDING_DINO_TINY_A2BB814_PROMPT_V1"
TRAINING_PROVENANCE_LIMITATION = (
    "Official public materials identify broad training corpora, but complete per-source training-data provenance "
    "is not established; this is recorded as a proposal-only limitation, not a truth-authority gate."
)


class RunError(RuntimeError):
    pass


def source_bbox(source_report: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Read the actual bounded source slice instead of assuming the first Ghent canary."""
    try:
        bounds = source_report["source_files"]["osm"]["bounds"]
        bbox = (
            float(bounds["minlon"]), float(bounds["minlat"]),
            float(bounds["maxlon"]), float(bounds["maxlat"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise RunError("source report lacks a valid OSM bbox") from error
    if not (bbox[0] < bbox[2] and bbox[1] < bbox[3]):
        raise RunError("source report OSM bbox is empty or inverted")
    return bbox


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _chunks(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for index in range(0, len(values), size):
        yield values[index:index + size]


def _graph_get(session: requests.Session, url: str, *, params: Mapping[str, Any]) -> Any:
    for attempt in range(2):
        response = session.get(url, params=params, timeout=90)
        if response.status_code < 500 or attempt == 1:
            response.raise_for_status()
            return response.json()
        time.sleep(1.0)
    raise AssertionError("unreachable")


def metric_distance(left: Sequence[float], right: Sequence[float]) -> float:
    lon1, lat1 = map(float, left)
    lon2, lat2 = map(float, right)
    mean_lat = math.radians((lat1 + lat2) / 2.0)
    x = math.radians(lon2 - lon1) * math.cos(mean_lat)
    y = math.radians(lat2 - lat1)
    return 6_371_008.8 * math.hypot(x, y)


def _bearing_deg(origin: Sequence[float], target: Sequence[float]) -> float:
    lon1, lat1 = map(math.radians, origin)
    lon2, lat2 = map(math.radians, target)
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)
    return math.degrees(math.atan2(x, y)) % 360.0


def _angular_difference(left: float, right: float) -> float:
    delta = abs((left - right) % 360.0)
    return min(delta, 360.0 - delta)


def _eligible_target_anchors(
    source_report: Mapping[str, Any],
    target_building_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    places_by_building: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source_report["place_building_crosswalk_candidates"]:
        if item.get("status") == "CANDIDATE_ONLY" and len(item.get("building_ids", [])) == 1:
            places_by_building[str(item["building_ids"][0])].append(item)
    unique_place_buildings = {building_id for building_id, rows in places_by_building.items() if len(rows) == 1}
    requested_buildings = {str(value) for value in target_building_ids or []}
    result = sorted(
        [
            item for item in source_report["osm_entrance_building_crosswalk_candidates"]
            if item.get("status") == "CANDIDATE_ONLY"
            and item.get("entrance") not in {"exit", "no", "service", "emergency"}
            and str(item["overture_building_id"]) in unique_place_buildings
            and (not requested_buildings or str(item["overture_building_id"]) in requested_buildings)
        ],
        key=lambda item: item["osm_entrance_id"],
    )
    if requested_buildings:
        found = {str(item["overture_building_id"]) for item in result}
        missing = sorted(requested_buildings - found)
        if missing:
            raise RunError(f"target building has no eligible entrance anchor: {', '.join(missing)}")
    return result


def select_anchor_facing_images(
    metadata: Sequence[dict[str, Any]],
    anchors: Sequence[dict[str, Any]],
    *,
    requested_count: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    candidates_by_anchor: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for anchor in anchors:
        anchor_point = [anchor["point"]["lon"], anchor["point"]["lat"]]
        for item in metadata:
            distance = metric_distance(item["coordinates"], anchor_point)
            focal = float(item["camera_parameters"][0])
            half_fov = math.degrees(math.atan2(0.5, focal))
            angle_error = _angular_difference(item["heading_deg"], _bearing_deg(item["coordinates"], anchor_point))
            if 3.0 <= distance <= 45.0 and angle_error <= half_fov:
                candidates_by_anchor[anchor["osm_entrance_id"]].append(
                    dict(item, target_anchor_id=anchor["osm_entrance_id"], target_distance_m=distance, target_angle_error_deg=angle_error)
                )
    for values in candidates_by_anchor.values():
        values.sort(key=lambda item: (item["target_angle_error_deg"], item["target_distance_m"], item["captured_at"], item["id"]))
    anchor_order = sorted(anchors, key=lambda item: (-len(candidates_by_anchor[item["osm_entrance_id"]]), item["osm_entrance_id"]))
    selected: list[dict[str, Any]] = []
    used: set[str] = set()
    per_anchor_limit = max(2, math.ceil(requested_count / max(1, len(anchor_order))))
    for anchor in anchor_order:
        accepted: list[dict[str, Any]] = []
        for item in candidates_by_anchor[anchor["osm_entrance_id"]]:
            if item["id"] in used:
                continue
            if not accepted or all(metric_distance(item["coordinates"], prior["coordinates"]) >= 3.0 for prior in accepted):
                accepted.append(item)
                used.add(item["id"])
            if len(accepted) == per_anchor_limit:
                break
        selected.extend(accepted)
    if len(selected) < requested_count:
        remaining = sorted(
            (item for values in candidates_by_anchor.values() for item in values if item["id"] not in used),
            key=lambda item: (item["target_angle_error_deg"], item["target_distance_m"], item["id"]),
        )
        for item in remaining:
            selected.append(item)
            used.add(item["id"])
            if len(selected) == requested_count:
                break
    return selected[:requested_count], {anchor["osm_entrance_id"]: len(candidates_by_anchor[anchor["osm_entrance_id"]]) for anchor in anchors}


def fetch_mapillary_metadata(
    token: str,
    source_report: Mapping[str, Any],
    *,
    requested_count: int = IMAGE_COUNT,
    target_building_ids: Sequence[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    session = requests.Session()
    session.headers["Authorization"] = f"OAuth {token}"
    anchors = _eligible_target_anchors(source_report, target_building_ids)
    raw_by_id: dict[str, dict[str, Any]] = {}
    query_bboxes: list[list[float]] = []
    if len(anchors) > 40:
        min_lon, min_lat, max_lon, max_lat = source_bbox(source_report)
        mid_lon, mid_lat = (min_lon + max_lon) / 2.0, (min_lat + max_lat) / 2.0
        query_plan = [
            ([min_lon, min_lat, mid_lon, mid_lat], 500),
            ([mid_lon, min_lat, max_lon, mid_lat], 500),
            ([min_lon, mid_lat, mid_lon, max_lat], 500),
            ([mid_lon, mid_lat, max_lon, max_lat], 500),
        ]
        query_strategy = "FOUR_QUADRANT_SOURCE_SLICE_FOR_DENSE_ANCHOR_SET"
    else:
        query_plan = []
        for anchor in anchors:
            lon, lat = float(anchor["point"]["lon"]), float(anchor["point"]["lat"])
            lat_delta = 60.0 / 110_540.0
            lon_delta = 60.0 / (111_320.0 * math.cos(math.radians(lat)))
            query_plan.append(([lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta], 200))
        query_strategy = "PER_ANCHOR_60M"
    for query_bbox, limit in query_plan:
        query_bboxes.append(query_bbox)
        response = _graph_get(
            session,
            "https://graph.mapillary.com/images",
            params={"bbox": ",".join(str(value) for value in query_bbox), "fields": ",".join(MAPILLARY_FIELDS), "limit": limit},
        )
        for item in response.get("data", []):
            raw_by_id[str(item["id"])] = item
    raw = list(raw_by_id.values())
    normalized: list[dict[str, Any]] = []
    for item in raw:
        geometry = item.get("computed_geometry")
        coordinates = geometry.get("coordinates") if isinstance(geometry, Mapping) else None
        camera_type = str(item.get("camera_type") or "").lower()
        parameters = item.get("camera_parameters")
        heading = item.get("computed_compass_angle", item.get("compass_angle"))
        if (
            not isinstance(coordinates, list) or len(coordinates) != 2
            or camera_type not in {"perspective", "planar"}
            or not isinstance(parameters, list) or not parameters or float(parameters[0]) <= 0
            or not isinstance(heading, (int, float))
        ):
            continue
        normalized.append({
            "id": str(item["id"]),
            "coordinates": [float(coordinates[0]), float(coordinates[1])],
            "captured_at": int(item.get("captured_at") or 0),
            "heading_deg": float(heading) % 360.0,
            "heading_kind": "COMPUTED_COMPASS_ANGLE" if item.get("computed_compass_angle") is not None else "COMPASS_ANGLE",
            "camera_type": camera_type,
            "camera_parameters": [float(value) for value in parameters],
            "source_width": int(item.get("width") or 0),
            "source_height": int(item.get("height") or 0),
        })
    ids = [item["id"] for item in normalized]
    sequence_by_id: dict[str, str] = {}
    for chunk in _chunks(ids, 40):
        data = _graph_get(session, "https://graph.mapillary.com/", params={"ids": ",".join(chunk), "fields": "sequence"})
        for image_id, item in data.items():
            if item.get("sequence"):
                sequence_by_id[str(image_id)] = str(item["sequence"])
    normalized = [dict(item, sequence_id=sequence_by_id[item["id"]]) for item in normalized if item["id"] in sequence_by_id]

    selected, anchor_candidate_counts = select_anchor_facing_images(normalized, anchors, requested_count=requested_count)
    if len(selected) < requested_count:
        raise RunError(f"only {len(selected)} anchor-facing Mapillary images for requested {requested_count}")

    for chunk in _chunks([item["id"] for item in selected], 30):
        data = _graph_get(session, "https://graph.mapillary.com/", params={"ids": ",".join(chunk), "fields": "thumb_1024_url"})
        for item in selected:
            if item["id"] in data:
                item["download_url"] = data[item["id"]].get("thumb_1024_url")
    if any(not item.get("download_url") for item in selected):
        raise RunError("selected image lacks thumb_1024_url")
    acquisition = {
        "endpoint": "https://graph.mapillary.com/images",
        "bbox": list(source_bbox(source_report)),
        "anchor_query_bboxes": query_bboxes,
        "metadata_query_strategy": query_strategy,
        "requested_fields": list(MAPILLARY_FIELDS) + ["sequence", "thumb_1024_url"],
        "raw_eligible_count": len(normalized),
        "selected_count": len(selected),
        "selection": "DETERMINISTIC_ANCHOR_FACING_3_TO_45M_WITHIN_CAMERA_HFOV_AND_3M_VIEW_SPACING",
        "eligible_anchor_ids": [item["osm_entrance_id"] for item in anchors],
        "target_building_ids": sorted({str(item["overture_building_id"]) for item in anchors}) if target_building_ids else [],
        "anchor_facing_candidate_counts": anchor_candidate_counts,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    return selected, acquisition


def download_images(metadata: Sequence[dict[str, Any]], image_dir: Path) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    for item in metadata:
        path = image_dir / f"{item['id']}.jpg"
        if not path.is_file():
            response = session.get(item["download_url"], timeout=60)
            response.raise_for_status()
            temporary = path.with_suffix(".jpg.tmp")
            temporary.write_bytes(response.content)
            temporary.replace(path)
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        item["path"] = str(path.resolve())
        item["width"] = width
        item["height"] = height
        item["image_sha256"] = sha256_file(path)
        item.pop("download_url", None)


def _iou(left: Sequence[float], right: Sequence[float]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    if intersection == 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def deterministic_nms(proposals: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(proposals, key=lambda item: (-item["score"], item["label"], item["bbox_xyxy"]))
    kept: list[dict[str, Any]] = []
    for item in ordered:
        if all(_iou(item["bbox_xyxy"], prior["bbox_xyxy"]) <= NMS_IOU_THRESHOLD for prior in kept):
            kept.append(item)
            if len(kept) == MAX_PROPOSALS_PER_IMAGE:
                break
    return kept


def run_inference(model_dir: Path, metadata: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    weight_path = model_dir / WEIGHTS_FILENAME
    if not weight_path.is_file() or sha256_file(weight_path) != WEIGHTS_SHA256:
        raise RunError("pinned model.safetensors missing or SHA-256 mismatch")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        model_dir, local_files_only=True, use_safetensors=True,
    ).to(device).eval()
    outputs: list[dict[str, Any]] = []
    for index, item in enumerate(metadata, start=1):
        with Image.open(item["path"]) as source:
            image = source.convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            raw,
            inputs.input_ids,
            threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
            target_sizes=[(image.height, image.width)],
        )[0]
        proposals = []
        labels = result.get("text_labels", result.get("labels", []))
        for box, score, label in zip(result["boxes"], result["scores"], labels):
            coords = [round(float(value), 6) for value in box.detach().cpu().tolist()]
            proposals.append({"bbox_xyxy": coords, "score": round(float(score.detach().cpu()), 12), "label": str(label)})
        outputs.append({
            "image_id": item["id"],
            "image_sha256": item["image_sha256"],
            "raw_above_threshold_count": len(proposals),
            "proposals": deterministic_nms(proposals),
        })
        print(f"inference {index}/{len(metadata)} image={item['id']} proposals={len(outputs[-1]['proposals'])}", flush=True)
    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "pillow": PIL.__version__,
        "cuda": str(torch.version.cuda),
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "NONE",
    }
    return outputs, versions


def _local_xy(point: Sequence[float], origin: Sequence[float]) -> tuple[float, float]:
    lon, lat = map(float, point)
    lon0, lat0 = map(float, origin)
    return ((lon - lon0) * math.cos(math.radians(lat0)) * 111_320.0, (lat - lat0) * 110_540.0)


def _lonlat(point: Sequence[float], origin: Sequence[float]) -> dict[str, float]:
    x, y = map(float, point)
    lon0, lat0 = map(float, origin)
    return {"lon": lon0 + x / (math.cos(math.radians(lat0)) * 111_320.0), "lat": lat0 + y / 110_540.0}


def _ray_segment(origin: Sequence[float], direction: Sequence[float], left: Sequence[float], right: Sequence[float]) -> tuple[float, tuple[float, float]] | None:
    ax, ay = left
    bx, by = right
    dx, dy = direction
    sx, sy = bx - ax, by - ay
    determinant = dx * (-sy) - dy * (-sx)
    if abs(determinant) < 1e-12:
        return None
    rx, ry = ax - origin[0], ay - origin[1]
    t = (rx * (-sy) - ry * (-sx)) / determinant
    u = (dx * ry - dy * rx) / determinant
    if 0.1 <= t <= 60.0 and 0.0 <= u <= 1.0:
        return t, (origin[0] + t * dx, origin[1] + t * dy)
    return None


def _polygon_ring(feature: Mapping[str, Any]) -> list[list[float]] | None:
    geometry = feature.get("geometry", {})
    coordinates = geometry.get("coordinates")
    if geometry.get("type") == "Polygon" and coordinates:
        return coordinates[0]
    if geometry.get("type") == "MultiPolygon" and coordinates:
        return max((ring[0] for ring in coordinates if ring), key=len, default=None)
    return None


def build_materializer_bundle(
    metadata: Sequence[dict[str, Any]],
    inference: Sequence[dict[str, Any]],
    source_report: Mapping[str, Any],
    buildings_geojson: Mapping[str, Any],
    *,
    runtime_versions: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    bbox = source_bbox(source_report)
    origin = [(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0]
    buildings: dict[str, list[tuple[float, float]]] = {}
    for feature in buildings_geojson.get("features", []):
        ring = _polygon_ring(feature)
        building_id = feature.get("id") or feature.get("properties", {}).get("id")
        if ring and building_id:
            buildings[str(building_id)] = [_local_xy(point, origin) for point in ring]
    place_rows = [item for item in source_report["place_building_crosswalk_candidates"] if item.get("status") == "CANDIDATE_ONLY" and len(item.get("building_ids", [])) == 1]
    places_by_building: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in place_rows:
        places_by_building[str(item["building_ids"][0])].append(item)
    unique_places = {building_id: rows[0] for building_id, rows in places_by_building.items() if len(rows) == 1}
    anchors_by_building: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in source_report["osm_entrance_building_crosswalk_candidates"]:
        if item.get("status") == "CANDIDATE_ONLY" and item.get("entrance") not in {"exit", "no", "service", "emergency"}:
            anchors_by_building[str(item["overture_building_id"])].append(item)
    metadata_by_id = {item["id"]: item for item in metadata}
    inference_by_id = {item["image_id"]: item for item in inference}
    prompt_sha = sha256_bytes(PROMPT.encode("utf-8"))
    config = {
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "weights_sha256": WEIGHTS_SHA256,
        "prompt": PROMPT,
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "max_proposals_per_image": MAX_PROPOSALS_PER_IMAGE,
    }
    config_sha = sha256_bytes(canonical_bytes(config))
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    geometry_counts = Counter()
    for image_id, result in inference_by_id.items():
        frame = metadata_by_id[image_id]
        camera_xy = _local_xy(frame["coordinates"], origin)
        focal = float(frame["camera_parameters"][0])
        for rank, proposal in enumerate(result["proposals"], start=1):
            bbox = proposal["bbox_xyxy"]
            center_u = (bbox[0] + bbox[2]) / 2.0
            yaw_offset = math.degrees(math.atan2(center_u / frame["width"] - 0.5, focal))
            heading = (frame["heading_deg"] + yaw_offset) % 360.0
            theta = math.radians(heading)
            direction = (math.sin(theta), math.cos(theta))
            hits: list[tuple[float, str, tuple[float, float]]] = []
            for building_id, ring in buildings.items():
                for index in range(len(ring) - 1):
                    hit = _ray_segment(camera_xy, direction, ring[index], ring[index + 1])
                    if hit:
                        hits.append((hit[0], building_id, hit[1]))
            if not hits:
                geometry_counts["no_building_hit"] += 1
                continue
            ray_range, building_id, hit_xy = min(hits, key=lambda value: (value[0], value[1]))
            if building_id not in unique_places:
                geometry_counts["hit_without_unique_target"] += 1
                continue
            if building_id not in anchors_by_building:
                geometry_counts["hit_without_admitted_anchor"] += 1
                continue
            hit_geo = _lonlat(hit_xy, origin)
            distances = sorted(
                ((materializer.metric_distance_m(hit_geo, anchor["point"]), anchor) for anchor in anchors_by_building[building_id]),
                key=lambda value: (value[0], value[1]["osm_entrance_id"]),
            )
            anchor_distance, anchor = distances[0]
            second_margin = (distances[1][0] - anchor_distance) if len(distances) > 1 else 999.0
            if anchor_distance > 3.0 or second_margin < 2.0:
                geometry_counts["anchor_gate_failed"] += 1
                continue
            geometry_counts["map_geometry_anchored"] += 1
            candidate_id = f"gdino-{image_id}-{rank:03d}"
            grouped[(building_id, anchor["osm_entrance_id"])].append({
                "candidate_id": candidate_id,
                "frame_id": image_id,
                "building_id": building_id,
                "anchor_id": anchor["osm_entrance_id"],
                "bbox_xyxy": bbox,
                "proposal_rank": rank,
                "proposal_score": proposal["score"],
                "proposal_label": proposal["label"],
                "proposal_score_semantics": "MODEL_PROPOSAL_RANKING_SCORE_NOT_TRUTH",
                "predicted_entrance_geo": hit_geo,
                "ray_heading_deg": round(heading, 9),
                "ray_range_m": round(ray_range, 9),
                "candidate_anchor_distance_m": round(anchor_distance, 9),
                "second_anchor_margin_m": round(second_margin, 9),
                "geometry_verified": True,
                "map_anchored": True,
                "generator_provenance": {
                    "provider_id": MODEL_REPOSITORY,
                    "model_id": MODEL_REPOSITORY,
                    "model_version": MODEL_REVISION,
                    "weights_sha256": WEIGHTS_SHA256,
                    "config_sha256": config_sha,
                    "prompt_sha256": prompt_sha,
                    "input_sha256": frame["image_sha256"],
                    "candidate_source": "GROUNDING_DINO_TEXT_CONDITIONED_PROPOSAL",
                    "lineage_group": GENERATOR_LINEAGE,
                    "runtime_versions": dict(runtime_versions),
                },
            })
    frames = [{
        "frame_id": item["id"],
        "sequence_id": item["sequence_id"],
        "camera_position": {"lon": item["coordinates"][0], "lat": item["coordinates"][1]},
        "is_panorama_slice": False,
    } for item in metadata]
    mapillary_digest = sha256_bytes(canonical_bytes([
        {key: value for key, value in item.items() if key != "path"} for item in metadata
    ]))
    source_files = source_report["source_files"]
    records: list[dict[str, Any]] = []
    for (building_id, anchor_id), candidates in sorted(grouped.items()):
        place = unique_places[building_id]
        anchor = next(item for item in anchors_by_building[building_id] if item["osm_entrance_id"] == anchor_id)
        record_frames = [frame for frame in frames if frame["frame_id"] in {item["frame_id"] for item in candidates}]
        records.append({
            "record_id": f"{place['place_id']}--{anchor_id.replace('/', '-')}",
            "sources": {
                "mapillary": {
                    "source_name": "Mapillary",
                    "snapshot_or_release": "GRAPH_API_QUERY_2026-08-21",
                    "record_ids": sorted(frame["frame_id"] for frame in record_frames),
                    "license": "Mapillary Terms; imagery/open-data obligations retained",
                    "attribution": "Mapillary contributors; image IDs retained",
                    "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "content_sha256": mapillary_digest,
                },
                "overture": {
                    "source_name": "Overture Maps",
                    "snapshot_or_release": str(source_report["overture_release"]),
                    "record_ids": [building_id, place["place_id"]],
                    "license": "Per-record Overture source licenses retained in frozen source slice",
                    "attribution": "Overture Maps Foundation and record source providers",
                    "retrieved_at": "2026-08-21T00:00:00Z",
                    "content_sha256": sha256_bytes((source_files["overture_buildings"]["sha256"] + source_files["overture_places"]["sha256"]).encode("ascii")),
                },
                "osm": {
                    "source_name": "OpenStreetMap",
                    "snapshot_or_release": "API_MAP_SNAPSHOT_2026-08-21",
                    "record_ids": [anchor_id],
                    "license": "ODbL-1.0",
                    "attribution": "OpenStreetMap contributors",
                    "retrieved_at": "2026-08-21T00:00:00Z",
                    "content_sha256": source_files["osm"]["sha256"],
                },
            },
            "crosswalk": {"status": "ADMITTED", "unique": True, "building_id": building_id, "method": place["method"]},
            "anchor": {"status": "ADMITTED", "unique": True, "anchor_id": anchor_id, "method": anchor["method"]},
            "frames": record_frames,
            "candidates": candidates,
            "conflicts": [],
            "evaluated_system_overlap": "NO_KNOWN_OVERLAP",
            "target_visibility_identifiable": len(record_frames) >= 2,
            "entrance_semantics_match_goal": anchor.get("entrance") not in {"exit", "no", "service", "emergency"},
            "ancestry_deduplicated": True,
        })
    bundle = {
        "protocol_id": materializer.PROTOCOL_ID,
        "upstream_commit": materializer.UPSTREAM_COMMIT,
        "provider_input": {
            "goal": {"requested_relation": "entrance_of", "requested_entrance_type": "ANY"},
            "coarse_area": "Ghent fixed S0 source bbox",
        },
        "records": records[:20],
    }
    diagnostics = {"geometry_counts": dict(geometry_counts), "record_groups": len(grouped), "materializer_record_count": len(bundle["records"])}
    return bundle, diagnostics


def write_admitted_visualizations(run_dir: Path, bundle: Mapping[str, Any], result: Mapping[str, Any]) -> list[str]:
    positive_ids = {
        candidate_id
        for item in result.get("results", [])
        for candidate_id in item.get("positive_candidate_ids", [])
    }
    candidates = {
        item["candidate_id"]: item
        for record in bundle.get("records", [])
        for item in record.get("candidates", [])
        if item.get("candidate_id") in positive_ids
    }
    output_dir = run_dir / "admitted-proposal-visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[Image.Image] = []
    paths: list[str] = []
    for candidate_id in sorted(candidates):
        candidate = candidates[candidate_id]
        source_path = run_dir / "images" / f"{candidate['frame_id']}.jpg"
        with Image.open(source_path) as source:
            image = source.convert("RGB")
        draw = ImageDraw.Draw(image)
        box = candidate["bbox_xyxy"]
        draw.rectangle(box, outline=(255, 32, 32), width=5)
        text = f"PROPOSAL_ONLY {candidate['proposal_label']} {candidate['proposal_score']:.3f}"
        draw.rectangle((box[0], max(0, box[1] - 22), min(image.width, box[0] + 420), box[1]), fill=(255, 255, 255))
        draw.text((box[0] + 3, max(0, box[1] - 20)), text, fill=(180, 0, 0))
        output_path = output_dir / f"{candidate_id}.png"
        image.save(output_path, format="PNG")
        paths.append(str(output_path.resolve()))
        preview = image.copy()
        preview.thumbnail((512, 384))
        rendered.append(preview)
    if rendered:
        contact = Image.new("RGB", (1024, 768), "white")
        for index, image in enumerate(rendered[:4]):
            x = (index % 2) * 512
            y = (index // 2) * 384
            contact.paste(image, (x, y))
        contact_path = output_dir / "contact-sheet.png"
        contact.save(contact_path, format="PNG")
        paths.append(str(contact_path.resolve()))
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    token = os.environ.get("MAPILLARY_ACCESS_TOKEN") or os.environ.get("MAPILLARY_TOKEN")
    if not token:
        raise RunError("MAPILLARY_ACCESS_TOKEN missing from process environment")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    source_report = json.loads(args.source_report.read_text(encoding="utf-8"))
    buildings = json.loads(args.buildings.read_text(encoding="utf-8"))
    metadata, acquisition = fetch_mapillary_metadata(
        token,
        source_report,
        requested_count=args.image_count,
        target_building_ids=args.target_building_id,
    )
    download_images(metadata, run_dir / "images")
    write_json(run_dir / "mapillary_metadata.json", {"acquisition": acquisition, "images": metadata})
    inference, versions = run_inference(args.model_dir.resolve(), metadata)
    inference_receipt = {
        "schema_version": 1,
        "authority": GENERATOR_AUTHORITY,
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "weights_filename": WEIGHTS_FILENAME,
        "weights_sha256": WEIGHTS_SHA256,
        "license": MODEL_LICENSE,
        "training_data_provenance_limitation": TRAINING_PROVENANCE_LIMITATION,
        "prompt": PROMPT,
        "prompt_sha256": sha256_bytes(PROMPT.encode("utf-8")),
        "box_threshold": BOX_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "runtime_versions": versions,
        "images": inference,
        "truth_authority": False,
        "claim_ceiling": "PROPOSAL_ONLY_NO_MODEL_PERFORMANCE_OR_SILVER_TRUTH_CLAIM",
    }
    inference_receipt["receipt_sha256"] = sha256_bytes(canonical_bytes(inference_receipt))
    write_json(run_dir / "proposal-receipt.json", inference_receipt)
    bundle, diagnostics = build_materializer_bundle(metadata, inference, source_report, buildings, runtime_versions=versions)
    write_json(run_dir / "normalized-bundle.json", bundle)
    first = materializer.materialize_bundle(bundle)
    second = materializer.materialize_bundle(json.loads((run_dir / "normalized-bundle.json").read_text(encoding="utf-8")))
    replay_equal = materializer.canonical_bytes(first) == materializer.canonical_bytes(second)
    if not replay_equal:
        raise RunError("materializer deterministic replay mismatch")
    write_json(run_dir / "materialization-result.json", first)
    visualization_paths = write_admitted_visualizations(run_dir, bundle, first)
    summary = {
        "verdict": first["verdict"],
        "image_count": len(metadata),
        "sequence_count": len({item["sequence_id"] for item in metadata}),
        "proposal_count": sum(len(item["proposals"]) for item in inference),
        "images_with_proposals": sum(bool(item["proposals"]) for item in inference),
        "geometry_diagnostics": diagnostics,
        "primary_admitted_count": first["primary_admitted_count"],
        "secondary_admitted_count": first["secondary_admitted_count"],
        "rejected_count": first["rejected_count"],
        "deterministic_replay_equal": replay_equal,
        "proposal_receipt_sha256": inference_receipt["receipt_sha256"],
        "materialization_report_sha256": first["report_sha256"],
        "admitted_proposal_visualizations": visualization_paths,
        "claim_ceiling": first["claim_ceiling"],
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--buildings", required=True, type=Path)
    parser.add_argument("--image-count", type=int, default=IMAGE_COUNT, choices=range(20, 51))
    parser.add_argument(
        "--target-building-id",
        action="append",
        default=[],
        help="Restrict acquisition to an eligible Overture building; repeat for multiple buildings.",
    )
    args = parser.parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
