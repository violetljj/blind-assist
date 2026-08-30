#!/usr/bin/env python3
"""Freeze and replay the L10 3RScan registered-endpoint extent ceiling.

The replay is intentionally geometry-only.  It asks whether a complete reference
door credential, aligned with the provider scan-to-reference transform, repairs
the extent and centroid failure exposed by the edge-clipped SceneNN credentials.
It is not a learned RGB matcher or a traversability/arrival evaluator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from plyfile import PlyData


HERE = Path(__file__).resolve().parent
PROTOCOL_SCHEMA = "blindassist-l10-3rscan-registered-extent-protocol-v1"
COHORT_SCHEMA = "blindassist-l10-3rscan-registered-extent-cohort-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-registered-extent-result-v1"
PREFLIGHT_SCHEMA = "blindassist-l10-3rscan-registered-extent-preflight-v1"
INVENTORY_SCHEMA = "blindassist-l10-3rscan-registered-extent-inventory-v1"
DOOR_LABELS = {"door", "door frame", "doorframe"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_label(value: Any) -> str:
    return " ".join(str(value).lower().replace("_", " ").replace("-", " ").split())


def object_id(group: dict[str, Any]) -> int:
    value = group.get("objectId", group.get("id"))
    require(value is not None, "SEMSEG_OBJECT_ID_MISSING")
    return int(value)


def semseg_instances(path: Path) -> dict[int, dict[str, Any]]:
    require(path.is_file(), f"SEMSEG_MISSING:{path}")
    value = load_json(path)
    groups = value.get("segGroups", value.get("seg_groups", []))
    require(isinstance(groups, list), f"SEMSEG_GROUPS_INVALID:{path}")
    return {object_id(group): group for group in groups}


def door_instances(path: Path) -> dict[int, dict[str, Any]]:
    return {
        instance_id: group
        for instance_id, group in semseg_instances(path).items()
        if normalize_label(group.get("label", "")) in DOOR_LABELS
    }


def object_catalog(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    require(path.is_file(), f"OBJECT_CATALOG_MISSING:{path}")
    value = load_json(path)
    scans = value.get("scans", []) if isinstance(value, dict) else []
    require(isinstance(scans, list), f"OBJECT_CATALOG_INVALID:{path}")
    result: dict[str, dict[int, dict[str, Any]]] = {}
    for scan in scans:
        scan_id = str(scan.get("scan", ""))
        objects = scan.get("objects", [])
        result[scan_id] = {
            int(item["id"]): item
            for item in objects
            if normalize_label(item.get("label", "")) in DOOR_LABELS
        }
    return result


def obb_area_proxy(group: dict[str, Any]) -> float:
    axes = group.get("obb", {}).get("axesLengths", [])
    if not isinstance(axes, list) or len(axes) != 3:
        return 0.0
    lengths = sorted((abs(float(value)) for value in axes), reverse=True)
    return lengths[0] * lengths[1]


def collect_instance_ids(value: Any) -> set[int]:
    result: set[int] = set()
    if isinstance(value, bool) or value is None:
        return result
    if isinstance(value, (int, float)):
        result.add(int(value))
    elif isinstance(value, list):
        for item in value:
            result.update(collect_instance_ids(item))
    elif isinstance(value, dict):
        for key, item in value.items():
            if "instance" in str(key).lower() or str(key).lower() in {"id", "objectid"}:
                result.update(collect_instance_ids(item))
    return result


def changed_instance_ids(scene: dict[str, Any], rescan: dict[str, Any]) -> set[int]:
    changed: set[int] = set()
    for key in ("removed", "nonrigid", "rigid"):
        changed.update(collect_instance_ids(rescan.get(key, [])))
    changed.update(collect_instance_ids(scene.get("ambiguity", [])))
    return changed


def scan_paths(data_root: Path, scan_id: str) -> dict[str, Path]:
    scan_root = data_root / scan_id
    return {
        "root": scan_root,
        "semseg": scan_root / "semseg.v2.json",
        "instances": scan_root / "labels.instances.annotated.v2.ply",
    }


def provider_matrix(values: Iterable[Any]) -> np.ndarray:
    raw = np.asarray(list(values), dtype=np.float64)
    require(raw.shape == (16,), f"TRANSFORM_LENGTH:{raw.size}")
    # The official 3RScan C++ reader writes the flat JSON array through Eigen's
    # column-major storage before multiplying matrix * point.
    matrix = raw.reshape((4, 4), order="F")
    require(np.isfinite(matrix).all(), "TRANSFORM_NONFINITE")
    require(np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5), "TRANSFORM_AFFINE_ROW")
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    require(points.ndim == 2 and points.shape[1] == 3, "POINT_SHAPE")
    homogeneous = np.column_stack((points, np.ones(len(points), dtype=np.float64)))
    transformed = homogeneous @ matrix.T
    require(np.allclose(transformed[:, 3], 1.0, atol=1e-5), "TRANSFORM_HOMOGENEOUS")
    return transformed[:, :3]


def ply_instance_points(path: Path, wanted_ids: set[int]) -> dict[int, np.ndarray]:
    require(path.is_file(), f"INSTANCE_PLY_MISSING:{path}")
    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    names = vertices.dtype.names or ()
    label_name = next(
        (name for name in ("objectId", "objectid", "object_id") if name in names),
        None,
    )
    require(label_name is not None, f"INSTANCE_PROPERTY_MISSING:{path}:{names}")
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float64, copy=False
    )
    labels = np.asarray(vertices[label_name], dtype=np.int64)
    require(np.isfinite(xyz).all(), f"INSTANCE_PLY_NONFINITE:{path}")
    result: dict[int, np.ndarray] = {}
    for instance_id in wanted_ids:
        points = xyz[labels == instance_id]
        if len(points):
            result[instance_id] = np.ascontiguousarray(points)
    return result


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def preflight(data_root: Path, output_path: Path) -> dict[str, Any]:
    metadata = data_root / "3RScan.json"
    objects = data_root / "objects.json"
    semseg_scan_dirs = []
    geometry_scan_dirs = []
    if data_root.is_dir():
        semseg_scan_dirs = sorted(
            path.name
            for path in data_root.iterdir()
            if path.is_dir()
            and (path / "semseg.v2.json").is_file()
        )
        geometry_scan_dirs = sorted(
            scan_id
            for scan_id in semseg_scan_dirs
            if (data_root / scan_id / "labels.instances.annotated.v2.ply").is_file()
        )
    metadata_ready = metadata.is_file()
    inventory_ready = metadata_ready and (objects.is_file() or len(semseg_scan_dirs) >= 2)
    freeze_ready = metadata_ready and len(geometry_scan_dirs) >= 2
    if freeze_ready:
        status = "READY_FOR_GEOMETRY_FREEZE"
        required_next = "Run freeze with the approved local 3RScan data root."
    elif inventory_ready:
        status = "READY_FOR_ENDPOINT_INVENTORY"
        required_next = "Run inventory, then materialize only the shortlisted instance PLY assets."
    else:
        status = "NOT_EVALUABLE_3RSCAN_ACCESS_PENDING"
        required_next = "The researcher must accept the official 3RScan Terms of Use, receive approval, and materialize 3RScan.json plus semseg.v2.json metadata locally."
    result = {
        "schema": PREFLIGHT_SCHEMA,
        "status": status,
        "data_root": str(data_root.resolve()),
        "checks": {
            "data_root_exists": data_root.is_dir(),
            "metadata_exists": metadata.is_file(),
            "global_object_catalog_exists": objects.is_file(),
            "scan_directories_with_semseg": len(semseg_scan_dirs),
            "scan_directories_with_semseg_and_instance_ply": len(geometry_scan_dirs),
        },
        "required_next": required_next,
        "terms_url": "https://forms.gle/NvL5dvB4tSFrHfQH6",
        "claim_boundary": "This is an access/readiness check, not algorithm or dataset evidence.",
    }
    atomic_write_json(output_path, result)
    return result


def candidate_rows(
    protocol: dict[str, Any], data_root: Path, require_geometry: bool
) -> list[dict[str, Any]]:
    metadata_path = data_root / "3RScan.json"
    require(metadata_path.is_file(), f"METADATA_MISSING:{metadata_path}")
    metadata = load_json(metadata_path)
    require(isinstance(metadata, list), "METADATA_ROOT_INVALID")
    catalog_path = data_root / "objects.json"
    catalog = object_catalog(catalog_path) if not require_geometry and catalog_path.is_file() else {}

    def available_doors(scan_id: str, paths: dict[str, Path]) -> dict[int, dict[str, Any]]:
        if paths["semseg"].is_file():
            return door_instances(paths["semseg"])
        return catalog.get(scan_id, {})

    allowed_split = protocol["source_selector"]["split"]
    candidates: list[dict[str, Any]] = []
    for scene in sorted(metadata, key=lambda row: str(row.get("reference", ""))):
        if scene.get("type") != allowed_split:
            continue
        reference_id = str(scene.get("reference", ""))
        reference_paths = scan_paths(data_root, reference_id)
        if require_geometry and (
            not reference_paths["semseg"].is_file()
            or not reference_paths["instances"].is_file()
        ):
            continue
        reference_doors = available_doors(reference_id, reference_paths)
        if not reference_doors:
            continue
        for rescan in sorted(scene.get("scans", []), key=lambda row: str(row.get("reference", ""))):
            rescan_id = str(rescan.get("reference", ""))
            rescan_paths = scan_paths(data_root, rescan_id)
            if require_geometry and (
                not rescan_paths["semseg"].is_file()
                or not rescan_paths["instances"].is_file()
            ):
                continue
            rescan_doors = available_doors(rescan_id, rescan_paths)
            if len(rescan_doors) < int(protocol["source_selector"]["minimum_rescan_doors"]):
                continue
            changed = changed_instance_ids(scene, rescan)
            shared = sorted(set(reference_doors) & set(rescan_doors) - changed)
            for target_id in shared:
                area = min(
                    obb_area_proxy(reference_doors[target_id]),
                    obb_area_proxy(rescan_doors[target_id]),
                )
                candidates.append(
                    {
                        "reference_scan_id": reference_id,
                        "rescan_id": rescan_id,
                        "target_instance_id": target_id,
                        "target_label": normalize_label(reference_doors[target_id].get("label", "")),
                        "rescan_door_instance_ids": sorted(rescan_doors),
                        "area_proxy": area,
                        "transform": list(rescan.get("transform", [])),
                    }
                )
    return sorted(
        candidates,
        key=lambda row: (
            -float(row["area_proxy"]),
            row["reference_scan_id"],
            row["rescan_id"],
            int(row["target_instance_id"]),
        ),
    )


def inventory(protocol_path: Path, data_root: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    candidates = candidate_rows(protocol, data_root, require_geometry=False)
    inventory_size = int(protocol["source_selector"]["inventory_pairs"])
    selected: list[dict[str, Any]] = []
    used_references: set[str] = set()
    for candidate in candidates:
        if candidate["reference_scan_id"] in used_references:
            continue
        selected.append(candidate)
        used_references.add(candidate["reference_scan_id"])
        if len(selected) == inventory_size:
            break
    require(selected, "SOURCE_METADATA_NOT_EVALUABLE_ZERO_STABLE_ENDPOINTS")
    scan_ids = sorted(
        {
            scan_id
            for candidate in selected
            for scan_id in (candidate["reference_scan_id"], candidate["rescan_id"])
        }
    )
    value = {
        "schema": INVENTORY_SCHEMA,
        "status": "REGISTERED_ENDPOINT_GEOMETRY_SHORTLIST_READY",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "metadata_sha256": sha256(data_root / "3RScan.json"),
        "object_catalog_sha256": (
            sha256(data_root / "objects.json")
            if (data_root / "objects.json").is_file()
            else None
        ),
        "candidate_count": len(candidates),
        "shortlist": selected,
        "required_scan_ids": scan_ids,
        "required_next_assets": {
            scan_id: [
                f"{scan_id}/semseg.v2.json",
                f"{scan_id}/labels.instances.annotated.v2.ply",
            ]
            for scan_id in scan_ids
        },
        "claim_boundary": "Metadata-only source planning; no geometry, RGB, depth, or algorithm result.",
    }
    atomic_write_json(output_path, value)
    return value


def freeze(protocol_path: Path, data_root: Path, output_path: Path) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    candidates = candidate_rows(protocol, data_root, require_geometry=True)
    cohort_size = int(protocol["source_selector"]["cohort_size"])
    episodes: list[dict[str, Any]] = []
    used_references: set[str] = set()
    for candidate in candidates:
        reference_id = candidate["reference_scan_id"]
        if reference_id in used_references:
            continue
        reference_paths = scan_paths(data_root, reference_id)
        rescan_paths = scan_paths(data_root, candidate["rescan_id"])
        target_id = int(candidate["target_instance_id"])
        reference_points = ply_instance_points(reference_paths["instances"], {target_id})
        rescan_points = ply_instance_points(
            rescan_paths["instances"], set(candidate["rescan_door_instance_ids"])
        )
        if len(reference_points.get(target_id, [])) < 4 or len(rescan_points.get(target_id, [])) < 4:
            continue
        provider_matrix(candidate["transform"])
        episode_number = len(episodes) + 1
        files = {}
        for role, paths in (("reference", reference_paths), ("rescan", rescan_paths)):
            for name in ("semseg", "instances"):
                path = paths[name]
                files[f"{role}_{name}"] = {
                    "path": relative(path, data_root),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
        episodes.append(
            {
                "episode_id": f"RE{episode_number:02d}",
                **candidate,
                "reference_target_vertices": len(reference_points[target_id]),
                "rescan_target_vertices": len(rescan_points[target_id]),
                "files": files,
            }
        )
        used_references.add(reference_id)
        if len(episodes) == cohort_size:
            break
    require(len(episodes) == cohort_size, f"SOURCE_NOT_EVALUABLE:{len(episodes)}_OF_{cohort_size}")
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGB_REGISTERED_ENDPOINT_EXTENT_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "entrypoint_path": Path(__file__).name,
        "entrypoint_sha256": sha256(Path(__file__).resolve()),
        "source_root": str(data_root.resolve()),
        "metadata": {
            "path": "3RScan.json",
            "bytes": (data_root / "3RScan.json").stat().st_size,
            "sha256": sha256(data_root / "3RScan.json"),
        },
        "candidate_count": len(candidates),
        "episodes": episodes,
        "rgb_or_depth_opened": False,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, cohort)
    return cohort


def portal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(len(points) >= 4, "PORTAL_POINTS_TOO_FEW")
    origin = np.mean(points, axis=0)
    xy = points[:, :2] - origin[:2]
    covariance = xy.T @ xy / max(len(xy), 1)
    values, vectors = np.linalg.eigh(covariance)
    horizontal_xy = vectors[:, int(np.argmax(values))]
    if horizontal_xy[0] < 0 or (abs(horizontal_xy[0]) < 1e-12 and horizontal_xy[1] < 0):
        horizontal_xy = -horizontal_xy
    horizontal = np.array([horizontal_xy[0], horizontal_xy[1], 0.0], dtype=np.float64)
    horizontal /= np.linalg.norm(horizontal)
    vertical = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return origin, horizontal, vertical


def project_uv(
    points: np.ndarray, origin: np.ndarray, horizontal: np.ndarray, vertical: np.ndarray
) -> np.ndarray:
    centred = points - origin
    return np.column_stack((centred @ horizontal, centred @ vertical)).astype(np.float32)


def convex_hull(points_uv: np.ndarray) -> np.ndarray:
    require(len(points_uv) >= 3, "HULL_POINTS_TOO_FEW")
    hull = cv2.convexHull(points_uv.reshape(-1, 1, 2)).reshape(-1, 2)
    require(len(hull) >= 3 and cv2.contourArea(hull) > 0.0, "HULL_DEGENERATE")
    return hull.astype(np.float32)


def polygon_centroid(polygon: np.ndarray) -> np.ndarray:
    moments = cv2.moments(polygon)
    require(abs(moments["m00"]) > 1e-12, "POLYGON_ZERO_AREA")
    return np.array(
        [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
        dtype=np.float64,
    )


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_area = float(cv2.contourArea(first))
    second_area = float(cv2.contourArea(second))
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = first_area + second_area - float(intersection)
    return float(intersection / union) if union > 0.0 else 0.0


def partial_fragment(points: np.ndarray, uv: np.ndarray, fraction: float) -> np.ndarray:
    threshold = float(np.quantile(uv[:, 0], 1.0 - fraction))
    selected = points[uv[:, 0] >= threshold]
    if len(selected) < 4:
        count = max(4, int(np.ceil(len(points) * fraction)))
        selected = points[np.argsort(uv[:, 0])[-count:]]
    return selected


def evaluate_arm(
    prediction_points: np.ndarray,
    query_instances: dict[int, np.ndarray],
    target_id: int,
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> dict[str, Any]:
    origin, horizontal, vertical = frame
    prediction_hull = convex_hull(project_uv(prediction_points, *frame))
    truth_hull = convex_hull(project_uv(query_instances[target_id], *frame))
    prediction_uv_centroid = polygon_centroid(prediction_hull)
    truth_uv_centroid = polygon_centroid(truth_hull)
    prediction_world_centroid = (
        origin
        + horizontal * prediction_uv_centroid[0]
        + vertical * prediction_uv_centroid[1]
    )
    truth_world_centroid = (
        origin + horizontal * truth_uv_centroid[0] + vertical * truth_uv_centroid[1]
    )
    distances = {
        instance_id: float(np.linalg.norm(np.mean(points, axis=0) - prediction_world_centroid))
        for instance_id, points in query_instances.items()
    }
    selected_id = min(distances, key=lambda value: (distances[value], value))
    return {
        "planar_extent_iou": polygon_iou(prediction_hull, truth_hull),
        "world_centroid_error_metres": float(
            np.linalg.norm(prediction_world_centroid - truth_world_centroid)
        ),
        "centroid_inside_target_extent": bool(
            cv2.pointPolygonTest(
                truth_hull, tuple(float(value) for value in prediction_uv_centroid), False
            )
            >= 0
        ),
        "selected_instance_id": int(selected_id),
        "target_top1": bool(selected_id == target_id),
        "nearest_instance_distances_metres": {
            str(key): value for key, value in sorted(distances.items())
        },
        "prediction_polygon_uv": prediction_hull.tolist(),
        "truth_polygon_uv": truth_hull.tolist(),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "target_top1": sum(bool(row["target_top1"]) for row in rows),
        "wrong_instance_commit": sum(not bool(row["target_top1"]) for row in rows),
        "centroid_inside": sum(bool(row["centroid_inside_target_extent"]) for row in rows),
        "median_planar_extent_iou": float(
            np.median([float(row["planar_extent_iou"]) for row in rows])
        ),
        "median_world_centroid_error_metres": float(
            np.median([float(row["world_centroid_error_metres"]) for row in rows])
        ),
    }


def preview(path: Path, partial: dict[str, Any], complete: dict[str, Any]) -> None:
    polygons = [
        np.asarray(complete["truth_polygon_uv"], dtype=np.float32),
        np.asarray(partial["prediction_polygon_uv"], dtype=np.float32),
        np.asarray(complete["prediction_polygon_uv"], dtype=np.float32),
    ]
    joined = np.vstack(polygons)
    minimum = joined.min(axis=0)
    maximum = joined.max(axis=0)
    scale = 440.0 / max(float(np.max(maximum - minimum)), 1e-6)
    canvas = np.full((512, 512, 3), 248, dtype=np.uint8)
    colours = [(70, 170, 70), (40, 160, 230), (220, 110, 40)]
    for polygon, colour in zip(polygons, colours):
        pixels = np.rint((polygon - minimum) * scale + 36.0).astype(np.int32)
        cv2.polylines(canvas, [pixels], True, colour, 3, cv2.LINE_AA)
    cv2.putText(canvas, "truth=green partial=orange complete=blue", (16, 494), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(cv2.imwrite(str(path), canvas), f"PREVIEW_WRITE_FAILED:{path}")


def replay(
    protocol_path: Path,
    cohort_path: Path,
    data_root: Path,
    output_path: Path,
    preview_dir: Path | None,
) -> dict[str, Any]:
    protocol = load_json(protocol_path)
    cohort = load_json(cohort_path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort.get("protocol_sha256") == sha256(protocol_path), "PROTOCOL_HASH_MISMATCH")
    require(
        cohort.get("entrypoint_sha256") == sha256(Path(__file__).resolve()),
        "ENTRYPOINT_HASH_MISMATCH",
    )
    require(cohort.get("metadata", {}).get("sha256") == sha256(data_root / "3RScan.json"), "METADATA_HASH_MISMATCH")
    fraction = float(protocol["arms"]["partial_fragment"]["retained_width_fraction"])
    episode_results: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        for record in episode["files"].values():
            path = data_root / record["path"]
            require(path.stat().st_size == int(record["bytes"]), f"SOURCE_SIZE_MISMATCH:{path}")
            require(sha256(path) == record["sha256"], f"SOURCE_HASH_MISMATCH:{path}")
        target_id = int(episode["target_instance_id"])
        reference_path = data_root / episode["files"]["reference_instances"]["path"]
        rescan_path = data_root / episode["files"]["rescan_instances"]["path"]
        reference_points = ply_instance_points(reference_path, {target_id})[target_id]
        query_raw = ply_instance_points(
            rescan_path, set(int(value) for value in episode["rescan_door_instance_ids"])
        )
        matrix = provider_matrix(episode["transform"])
        query_instances = {
            instance_id: transform_points(points, matrix)
            for instance_id, points in query_raw.items()
        }
        require(target_id in query_instances, f"TARGET_MISSING_IN_RESCAN:{target_id}")
        frame = portal_frame(reference_points)
        uv = project_uv(reference_points, *frame)
        partial_points = partial_fragment(reference_points, uv, fraction)
        partial_result = evaluate_arm(partial_points, query_instances, target_id, frame)
        complete_result = evaluate_arm(reference_points, query_instances, target_id, frame)
        row = {
            "episode_id": episode["episode_id"],
            "reference_scan_id": episode["reference_scan_id"],
            "rescan_id": episode["rescan_id"],
            "target_instance_id": target_id,
            "reference_vertices": len(reference_points),
            "partial_vertices": len(partial_points),
            "rescan_target_vertices": len(query_instances[target_id]),
            "partial_fragment": partial_result,
            "complete_credential": complete_result,
        }
        episode_results.append(row)
        if preview_dir is not None:
            preview(preview_dir / f"{episode['episode_id']}.png", partial_result, complete_result)

    partial_aggregate = aggregate([row["partial_fragment"] for row in episode_results])
    complete_aggregate = aggregate([row["complete_credential"] for row in episode_results])
    gate = protocol["decision_gate"]
    conditions = {
        "complete_target_top1": complete_aggregate["target_top1"] == len(episode_results),
        "complete_wrong_commit_zero": complete_aggregate["wrong_instance_commit"] == 0,
        "complete_centroid_inside_all": complete_aggregate["centroid_inside"] == len(episode_results),
        "complete_median_iou": complete_aggregate["median_planar_extent_iou"]
        >= float(gate["complete_median_planar_extent_iou_minimum"]),
        "complete_median_centroid_error": complete_aggregate["median_world_centroid_error_metres"]
        <= float(gate["complete_median_world_centroid_error_metres_maximum"]),
        "extent_iou_effect": (
            complete_aggregate["median_planar_extent_iou"]
            - partial_aggregate["median_planar_extent_iou"]
        )
        >= float(gate["minimum_median_iou_gain_over_partial"]),
        "centroid_error_effect": (
            partial_aggregate["median_world_centroid_error_metres"]
            - complete_aggregate["median_world_centroid_error_metres"]
        )
        >= float(gate["minimum_median_centroid_error_reduction_metres"]),
    }
    passed = all(conditions.values())
    result = {
        "schema": RESULT_SCHEMA,
        "status": (
            "L10_3RSCAN_REGISTERED_ENDPOINT_EXTENT_DEVELOPMENT_GATE_MET"
            if passed
            else "L10_3RSCAN_REGISTERED_ENDPOINT_EXTENT_DEVELOPMENT_GATE_NOT_MET"
        ),
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "cohort_path": cohort_path.name,
        "cohort_sha256": sha256(cohort_path),
        "arms": {
            "partial_fragment": partial_aggregate,
            "complete_credential": complete_aggregate,
        },
        "effect": {
            "median_planar_extent_iou_gain": complete_aggregate["median_planar_extent_iou"]
            - partial_aggregate["median_planar_extent_iou"],
            "median_world_centroid_error_reduction_metres": partial_aggregate[
                "median_world_centroid_error_metres"
            ]
            - complete_aggregate["median_world_centroid_error_metres"],
        },
        "decision_conditions": conditions,
        "episodes": episode_results,
        "claim_boundary": protocol["claim_boundary"],
    }
    atomic_write_json(output_path, result)
    return result


def self_test() -> dict[str, Any]:
    transform_flat = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        1.5, -2.0, 0.25, 1.0,
    ]
    transformed = transform_points(np.array([[0.0, 0.0, 0.0]]), provider_matrix(transform_flat))[0]
    require(np.allclose(transformed, [1.5, -2.0, 0.25]), "SELF_TEST_TRANSFORM")

    xs = np.linspace(-0.5, 0.5, 21)
    zs = np.linspace(0.0, 2.0, 31)
    reference = np.array([[x, 0.0, z] for x in xs for z in zs], dtype=np.float64)
    query_target = reference + np.array([0.02, 0.01, 0.01])
    distractor = reference + np.array([2.0, 0.0, 0.0])
    query = {7: query_target, 9: distractor}
    frame = portal_frame(reference)
    partial = partial_fragment(reference, project_uv(reference, *frame), 0.25)
    partial_result = evaluate_arm(partial, query, 7, frame)
    complete_result = evaluate_arm(reference, query, 7, frame)
    require(complete_result["planar_extent_iou"] > 0.9, "SELF_TEST_COMPLETE_IOU")
    require(partial_result["planar_extent_iou"] < 0.35, "SELF_TEST_PARTIAL_IOU")
    require(complete_result["world_centroid_error_metres"] < 0.05, "SELF_TEST_COMPLETE_CENTROID")
    require(partial_result["world_centroid_error_metres"] > 0.25, "SELF_TEST_PARTIAL_CENTROID")
    require(complete_result["target_top1"], "SELF_TEST_TOP1")

    def write_fixture_ply(path: Path, instances: dict[int, np.ndarray]) -> None:
        rows = [
            (point, instance_id)
            for instance_id, points in sorted(instances.items())
            for point in points
        ]
        lines = [
            "ply",
            "format ascii 1.0",
            f"element vertex {len(rows)}",
            "property float x",
            "property float y",
            "property float z",
            "property ushort objectId",
            "end_header",
        ]
        lines.extend(
            f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} {instance_id}"
            for point, instance_id in rows
        )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol_path = HERE / "l10_3rscan_registered_extent_protocol_v1.json"
    require(protocol_path.is_file(), "SELF_TEST_PROTOCOL_MISSING")
    identity_flat = np.eye(4, dtype=np.float64).reshape(-1, order="F").tolist()
    with tempfile.TemporaryDirectory(prefix="l10-3rscan-fixture-") as temporary:
        root = Path(temporary)
        scenes = []
        for index in range(3):
            reference_id = f"0000000{index}-0000-0000-0000-00000000000{index}"
            rescan_id = f"1000000{index}-0000-0000-0000-00000000000{index}"
            scenes.append(
                {
                    "reference": reference_id,
                    "type": "train",
                    "ambiguity": [],
                    "scans": [
                        {
                            "reference": rescan_id,
                            "transform": identity_flat,
                            "removed": [],
                            "nonrigid": [],
                            "rigid": [],
                        }
                    ],
                }
            )
            for scan_id, is_rescan in ((reference_id, False), (rescan_id, True)):
                scan_root = root / scan_id
                scan_root.mkdir(parents=True)
                groups = [
                    {
                        "id": 7,
                        "objectId": 7,
                        "label": "door",
                        "obb": {"axesLengths": [1.0, 0.05, 2.0]},
                    }
                ]
                points = {7: query_target if is_rescan else reference}
                if is_rescan:
                    groups.append(
                        {
                            "id": 9,
                            "objectId": 9,
                            "label": "door",
                            "obb": {"axesLengths": [1.0, 0.05, 2.0]},
                        }
                    )
                    points[9] = distractor
                atomic_write_json(scan_root / "semseg.v2.json", {"segGroups": groups})
                write_fixture_ply(scan_root / "labels.instances.annotated.v2.ply", points)
        atomic_write_json(root / "3RScan.json", scenes)
        inventory_path = root / "inventory.json"
        cohort_path = root / "cohort.json"
        result_path = root / "result.json"
        fixture_inventory = inventory(protocol_path, root, inventory_path)
        fixture_cohort = freeze(protocol_path, root, cohort_path)
        fixture_result = replay(protocol_path, cohort_path, root, result_path, None)
        require(len(fixture_cohort["episodes"]) == 3, "SELF_TEST_FIXTURE_COHORT")
        require(len(fixture_inventory["shortlist"]) == 3, "SELF_TEST_FIXTURE_INVENTORY")
        require(
            fixture_result["status"]
            == "L10_3RSCAN_REGISTERED_ENDPOINT_EXTENT_DEVELOPMENT_GATE_MET",
            "SELF_TEST_FIXTURE_REPLAY",
        )
    return {
        "status": "PASS",
        "transform": transformed.tolist(),
        "partial_iou": partial_result["planar_extent_iou"],
        "complete_iou": complete_result["planar_extent_iou"],
        "partial_centroid_error_metres": partial_result["world_centroid_error_metres"],
        "complete_centroid_error_metres": complete_result["world_centroid_error_metres"],
        "fixture_inventory_freeze_and_replay": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--data-root", type=Path, required=True)
    preflight_parser.add_argument("--output", type=Path, required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_protocol_v1.json",
    )
    freeze_parser.add_argument("--data-root", type=Path, required=True)
    freeze_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_cohort_v1.json",
    )

    inventory_parser = subparsers.add_parser("inventory")
    inventory_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_protocol_v1.json",
    )
    inventory_parser.add_argument("--data-root", type=Path, required=True)
    inventory_parser.add_argument("--output", type=Path, required=True)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument(
        "--protocol",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_protocol_v1.json",
    )
    replay_parser.add_argument(
        "--cohort",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_cohort_v1.json",
    )
    replay_parser.add_argument("--data-root", type=Path, required=True)
    replay_parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "l10_3rscan_registered_extent_result_v1.json",
    )
    replay_parser.add_argument("--preview-dir", type=Path)

    subparsers.add_parser("self-test")
    arguments = parser.parse_args()
    if arguments.command == "preflight":
        value = preflight(arguments.data_root, arguments.output)
    elif arguments.command == "inventory":
        value = inventory(arguments.protocol, arguments.data_root, arguments.output)
    elif arguments.command == "freeze":
        value = freeze(arguments.protocol, arguments.data_root, arguments.output)
    elif arguments.command == "replay":
        value = replay(
            arguments.protocol,
            arguments.cohort,
            arguments.data_root,
            arguments.output,
            arguments.preview_dir,
        )
    else:
        value = self_test()
    print(json.dumps(value, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
