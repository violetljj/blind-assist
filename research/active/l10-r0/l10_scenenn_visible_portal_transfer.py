#!/usr/bin/env python3
"""Freeze and replay SceneNN strict-triangle visible portal transfer v3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from plyfile import PlyData


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / "l10_scenenn_real_posed_portal_transfer.py"
EXTRACTOR_SOURCE_PATH = HERE / "l10_scenenn_extract_selected_sync.cpp"
BASE_SPEC = importlib.util.spec_from_file_location("l10_scenenn_v2", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise RuntimeError(f"BASE_IMPORT_FAILED:{BASE_PATH}")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)

PROTOCOL_SCHEMA = "blindassist-l10-scenenn-visible-portal-protocol-v3"
COHORT_SCHEMA = "blindassist-l10-scenenn-visible-portal-cohort-v3"
RESULT_SCHEMA = "blindassist-l10-scenenn-visible-portal-result-v3"
RECEIPT_SCHEMA = "blindassist-l10-scenenn-selected-rgbd-receipt-v3"


def require(condition: bool, message: str) -> None:
    base.require(condition, message)


def sha256(path: Path) -> str:
    return base.sha256(path)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return base.load_json(path)


def write_json(path: Path, value: Any) -> None:
    base.write_json(path, value)


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = load_json(path)
    require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA_MISMATCH")
    for key in ("protocol", "cohort", "result"):
        predecessor = path.with_name(protocol["predecessor"][f"{key}_path"])
        require(predecessor.is_file(), f"PREDECESSOR_MISSING:{predecessor}")
        require(
            sha256(predecessor) == protocol["predecessor"][f"{key}_sha256"],
            f"PREDECESSOR_HASH_MISMATCH:{predecessor}",
        )
    return protocol


def input_paths(source_root: Path, scene_id: str) -> dict[str, Path]:
    scene = source_root / "payload" / scene_id
    return {
        "ply": scene / f"{scene_id}.ply",
        "xml": scene / f"{scene_id}.xml",
        "trajectory": scene / "trajectory.log",
        "intrinsic": source_root / "payload" / "intrinsic" / "asus.ini",
    }


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def read_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    require(path.is_file(), f"MISSING_PLY:{path}")
    ply = PlyData.read(str(path))
    vertices = ply["vertex"].data
    xyz = np.column_stack((vertices["x"], vertices["y"], vertices["z"])).astype(
        np.float32, copy=False
    )
    labels = np.asarray(vertices["label"], dtype=np.int64)
    face_data = ply["face"].data
    property_name = "vertex_indices" if "vertex_indices" in face_data.dtype.names else "vertex_index"
    faces = np.stack(face_data[property_name]).astype(np.int32, copy=False)
    require(faces.ndim == 2 and faces.shape[1] == 3, f"PLY_NON_TRIANGULAR:{path}")
    require(np.isfinite(xyz).all(), f"PLY_NONFINITE:{path}")
    require(int(faces.min()) >= 0 and int(faces.max()) < len(xyz), f"PLY_FACE_INDEX:{path}")
    return np.ascontiguousarray(xyz), labels, np.ascontiguousarray(faces)


CUDA_SOURCE = r"""
extern "C" __global__ void rasterize_depth(
    const float* vertices,
    const int* faces,
    const int face_count,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float near_z,
    unsigned int* depth_bits
) {
    int face_index = blockDim.x * blockIdx.x + threadIdx.x;
    if (face_index >= face_count) return;
    int i0 = faces[face_index * 3 + 0];
    int i1 = faces[face_index * 3 + 1];
    int i2 = faces[face_index * 3 + 2];
    float x0 = vertices[i0 * 3 + 0], y0 = vertices[i0 * 3 + 1], z0 = vertices[i0 * 3 + 2];
    float x1 = vertices[i1 * 3 + 0], y1 = vertices[i1 * 3 + 1], z1 = vertices[i1 * 3 + 2];
    float x2 = vertices[i2 * 3 + 0], y2 = vertices[i2 * 3 + 1], z2 = vertices[i2 * 3 + 2];
    if (z0 <= near_z || z1 <= near_z || z2 <= near_z) return;

    float u0 = fx * x0 / z0 + cx, v0 = fy * y0 / z0 + cy;
    float u1 = fx * x1 / z1 + cx, v1 = fy * y1 / z1 + cy;
    float u2 = fx * x2 / z2 + cx, v2 = fy * y2 / z2 + cy;
    float area = (u1 - u0) * (v2 - v0) - (v1 - v0) * (u2 - u0);
    if (!isfinite(area) || fabsf(area) < 1.0e-8f) return;

    int min_x = max(0, (int)floorf(fminf(u0, fminf(u1, u2))));
    int max_x = min(width - 1, (int)ceilf(fmaxf(u0, fmaxf(u1, u2))));
    int min_y = max(0, (int)floorf(fminf(v0, fminf(v1, v2))));
    int max_y = min(height - 1, (int)ceilf(fmaxf(v0, fmaxf(v1, v2))));
    if (min_x > max_x || min_y > max_y) return;

    float inverse_area = 1.0f / area;
    for (int py = min_y; py <= max_y; ++py) {
        float y = (float)py + 0.5f;
        for (int px = min_x; px <= max_x; ++px) {
            float x = (float)px + 0.5f;
            float w0 = ((u1 - x) * (v2 - y) - (v1 - y) * (u2 - x)) * inverse_area;
            float w1 = ((u2 - x) * (v0 - y) - (v2 - y) * (u0 - x)) * inverse_area;
            float w2 = 1.0f - w0 - w1;
            if (w0 < -1.0e-5f || w1 < -1.0e-5f || w2 < -1.0e-5f) continue;
            float inverse_z = w0 / z0 + w1 / z1 + w2 / z2;
            if (inverse_z <= 0.0f || !isfinite(inverse_z)) continue;
            float z = 1.0f / inverse_z;
            atomicMin(&depth_bits[py * width + px], __float_as_uint(z));
        }
    }
}
"""


class VisibilityRenderer:
    def __init__(
        self,
        xyz: np.ndarray,
        labels: np.ndarray,
        faces: np.ndarray,
        target_id: int,
        intrinsic: dict[str, float | int],
        renderer: dict[str, Any],
    ) -> None:
        import cupy as cp

        require(cp.cuda.runtime.getDeviceCount() > 0, "CUDA_DEVICE_MISSING")
        self.cp = cp
        self.xyz = cp.asarray(xyz, dtype=cp.float32)
        self.faces = cp.asarray(faces, dtype=cp.int32)
        target_face_mask = (
            (labels[faces[:, 0]] == target_id)
            & (labels[faces[:, 1]] == target_id)
            & (labels[faces[:, 2]] == target_id)
        )
        target_faces = faces[target_face_mask]
        require(len(target_faces) > 0, f"STRICT_TARGET_FACES_EMPTY:{target_id}")
        self.target_faces = cp.asarray(target_faces, dtype=cp.int32)
        self.target_face_count = int(len(target_faces))
        self.width = int(intrinsic["width"])
        self.height = int(intrinsic["height"])
        self.fx = np.float32(intrinsic["fx"])
        self.fy = np.float32(intrinsic["fy"])
        self.cx = np.float32(intrinsic["cx"])
        self.cy = np.float32(intrinsic["cy"])
        self.near = np.float32(renderer["near_plane_metres"])
        self.tolerance = np.float32(renderer["visibility_tolerance_metres"])
        self.kernel = cp.RawKernel(CUDA_SOURCE, "rasterize_depth")

    def runtime_identity(self) -> dict[str, Any]:
        cp = self.cp
        device_id = int(cp.cuda.Device().id)
        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = properties["name"]
        device_name = raw_name.decode("utf-8") if isinstance(raw_name, bytes) else str(raw_name)
        return {
            "cupy_version": cp.__version__,
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "device_id": device_id,
            "device_name": device_name,
        }

    def _depth(self, camera_xyz: Any, faces: Any) -> Any:
        cp = self.cp
        bits = cp.full(self.width * self.height, np.uint32(0x7F800000), dtype=cp.uint32)
        count = int(len(faces))
        self.kernel(
            ((count + 255) // 256,),
            (256,),
            (
                camera_xyz,
                faces,
                np.int32(count),
                np.int32(self.width),
                np.int32(self.height),
                self.fx,
                self.fy,
                self.cx,
                self.cy,
                self.near,
                bits,
            ),
        )
        return bits.view(cp.float32).reshape(self.height, self.width)

    def visible_mask(self, camera_to_world: np.ndarray) -> Any:
        cp = self.cp
        world_to_camera = np.linalg.inv(camera_to_world).astype(np.float32)
        rotation = cp.asarray(np.ascontiguousarray(world_to_camera[:3, :3]))
        translation = cp.asarray(np.ascontiguousarray(world_to_camera[:3, 3]))
        camera_xyz = self.xyz @ rotation.T + translation
        full_depth = self._depth(camera_xyz, self.faces)
        target_depth = self._depth(camera_xyz, self.target_faces)
        return cp.isfinite(target_depth) & (target_depth <= full_depth + self.tolerance), target_depth

    def statistics(
        self, camera_to_world: np.ndarray, return_mask: bool = False
    ) -> tuple[dict[str, Any], np.ndarray | None]:
        cp = self.cp
        visible, target_depth = self.visible_mask(camera_to_world)
        visible_pixels = int(cp.count_nonzero(visible).item())
        target_pixels = int(cp.count_nonzero(cp.isfinite(target_depth)).item())
        stats: dict[str, Any] = {
            "visible_pixels": visible_pixels,
            "target_raster_pixels": target_pixels,
            "visible_to_target_raster_ratio": (
                float(visible_pixels / target_pixels) if target_pixels else 0.0
            ),
            "visible_image_fraction": float(visible_pixels / (self.width * self.height)),
            "bbox_xyxy": None,
            "bbox_width": 0,
            "bbox_height": 0,
        }
        if visible_pixels:
            ys, xs = cp.nonzero(visible)
            x0, x1 = int(cp.min(xs).item()), int(cp.max(xs).item()) + 1
            y0, y1 = int(cp.min(ys).item()), int(cp.max(ys).item()) + 1
            stats.update(
                {
                    "bbox_xyxy": [x0, y0, x1, y1],
                    "bbox_width": x1 - x0,
                    "bbox_height": y1 - y0,
                }
            )
        mask = cp.asnumpy(visible) if return_mask else None
        return stats, mask


def mask_sha256(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(mask, bitorder="little").tobytes()).hexdigest()


def all_contour_plane_points(
    mask: np.ndarray,
    normal: np.ndarray,
    offset: float,
    intrinsic: dict[str, float | int],
) -> tuple[np.ndarray, np.ndarray]:
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    require(contours, "REFERENCE_CONTOUR_EMPTY")
    ordered = sorted(
        contours,
        key=lambda contour: (
            cv2.boundingRect(contour)[1],
            cv2.boundingRect(contour)[0],
            -cv2.contourArea(contour),
        ),
    )
    image_parts: list[np.ndarray] = []
    camera_parts: list[np.ndarray] = []
    for raw in ordered:
        contour = raw.reshape(-1, 2).astype(np.float64)
        rays = np.column_stack(
            (
                (contour[:, 0] - float(intrinsic["cx"])) / float(intrinsic["fx"]),
                (contour[:, 1] - float(intrinsic["cy"])) / float(intrinsic["fy"]),
                np.ones(len(contour), dtype=np.float64),
            )
        )
        denominator = rays @ normal
        scale = -offset / denominator
        valid = np.isfinite(scale) & (np.abs(denominator) > 1e-9) & (scale > 0.0)
        if np.any(valid):
            image_parts.append(contour[valid])
            camera_parts.append(rays[valid] * scale[valid, None])
    require(sum(len(part) for part in camera_parts) >= 3, "PLANE_CONTOUR_INTERSECTION_EMPTY")
    return np.concatenate(image_parts), np.concatenate(camera_parts)


def eligible_frame(
    stats: dict[str, Any], distance_m: float, rules: dict[str, Any]
) -> tuple[bool, str]:
    checks = (
        (stats["visible_pixels"] >= int(rules["minimum_visible_target_pixels"]), "PIXELS"),
        (stats["bbox_width"] >= int(rules["minimum_bbox_width_pixels"]), "WIDTH"),
        (stats["bbox_height"] >= int(rules["minimum_bbox_height_pixels"]), "HEIGHT"),
        (
            stats["visible_to_target_raster_ratio"]
            >= float(rules["minimum_visible_to_target_raster_ratio"]),
            "VISIBILITY",
        ),
        (
            stats["visible_image_fraction"]
            <= float(rules["maximum_visible_target_image_fraction"]),
            "FRACTION",
        ),
        (
            distance_m <= float(rules["maximum_camera_to_target_centroid_metres"]),
            "RANGE",
        ),
    )
    failures = [name for passed, name in checks if not passed]
    return not failures, "+".join(failures) if failures else "ELIGIBLE"


def select_pair(
    rows: list[dict[str, Any]], minimum_baseline: float = 0.5, minimum_gap: int = 30
) -> tuple[dict[str, Any], dict[str, Any], float] | None:
    eligible = [row for row in rows if row["eligible"]]
    best: tuple[tuple[float, int, float, int, int], dict[str, Any], dict[str, Any], float] | None = None
    for index, reference in enumerate(eligible[:-1]):
        for query in eligible[index + 1 :]:
            if int(query["frame"]) - int(reference["frame"]) < minimum_gap:
                continue
            baseline = float(
                np.linalg.norm(
                    np.asarray(query["camera_center_world"], dtype=np.float64)
                    - np.asarray(reference["camera_center_world"], dtype=np.float64)
                )
            )
            if baseline < minimum_baseline:
                continue
            key = (
                round(baseline, 9),
                min(int(reference["visible_pixels"]), int(query["visible_pixels"])),
                min(
                    float(reference["visible_to_target_raster_ratio"]),
                    float(query["visible_to_target_raster_ratio"]),
                ),
                -int(reference["frame"]),
                -int(query["frame"]),
            )
            if best is None or key > best[0]:
                best = (key, reference, query, baseline)
    return None if best is None else (best[1], best[2], best[3])


def validate_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    require(path.is_file(), f"MISSING_SOURCE:{path}")
    require(path.stat().st_size == int(expected["content_length"]), f"CONTENT_LENGTH:{path}")
    receipt = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    if "md5" in expected:
        digest = md5(path)
        require(digest == str(expected["md5"]).lower(), f"OFFICIAL_MD5:{path}")
        receipt["md5"] = digest
    return receipt


def freeze_cohort(protocol_path: Path, source_root: Path, output_path: Path) -> None:
    protocol = load_protocol(protocol_path)
    materialized = sorted((source_root / "payload").glob("**/*.oni"))
    materialized += sorted((source_root / "payload").glob("**/image*.png"))
    materialized += sorted((source_root / "payload").glob("**/depth*.png"))
    require(not materialized, f"RGBD_MATERIALIZED_BEFORE_FREEZE:{len(materialized)}")

    intrinsic_path = source_root / "payload" / "intrinsic" / "asus.ini"
    intrinsic = base.parse_intrinsic(intrinsic_path)
    require(intrinsic["width"] == protocol["provider"]["image_width"], "WIDTH_MISMATCH")
    require(intrinsic["height"] == protocol["provider"]["image_height"], "HEIGHT_MISMATCH")
    rules = protocol["pre_rgbd_selector"]["eligible_frame"]
    renderer_rules = protocol["pre_rgbd_selector"]["renderer"]
    audits: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    renderer_runtime: dict[str, Any] | None = None
    source_manifest: dict[str, dict[str, Any]] = {
        relative(intrinsic_path, source_root): validate_file(
            intrinsic_path, protocol["shared_intrinsic"]
        )
    }

    for frozen in protocol["fresh_scene_shortlist"]["candidates"]:
        scene_id = frozen["scene_id"]
        target_id = int(frozen["target_door_instance_id"])
        paths = input_paths(source_root, scene_id)
        for key, path in paths.items():
            if key == "intrinsic":
                continue
            file_name = "trajectory.log" if key == "trajectory" else path.name
            expected = {
                **frozen["files"][file_name],
                "md5": protocol["official_md5"][scene_id][key],
            }
            source_manifest[relative(path, source_root)] = validate_file(
                path, expected
            )
        require(
            sha256(paths["xml"]) == frozen["inventory_xml_sha256"],
            f"INVENTORY_XML_HASH:{scene_id}",
        )
        xml_labels = base.parse_xml_labels(paths["xml"])
        require(target_id in xml_labels, f"TARGET_NOT_IN_XML:{scene_id}:{target_id}")
        require(xml_labels[target_id]["text"].casefold() == "door", f"TARGET_NOT_DOOR:{scene_id}")
        require(xml_labels[target_id]["color_rgb"] == frozen["target_xml_color_rgb"], f"TARGET_COLOR:{scene_id}")
        require(xml_labels[target_id]["area"] == frozen["target_xml_area"], f"TARGET_AREA:{scene_id}")

        xyz, labels, faces = read_mesh(paths["ply"])
        target_points = xyz[labels == target_id]
        require(len(target_points) > 0, f"TARGET_NOT_IN_PLY:{scene_id}:{target_id}")
        target_centroid = np.mean(target_points.astype(np.float64), axis=0)
        renderer = VisibilityRenderer(xyz, labels, faces, target_id, intrinsic, renderer_rules)
        if renderer_runtime is None:
            renderer_runtime = renderer.runtime_identity()
        poses = base.parse_poses(paths["trajectory"])
        rows: list[dict[str, Any]] = []
        reasons: Counter[str] = Counter()
        for pose_row in poses:
            stats, _ = renderer.statistics(pose_row["camera_to_world"])
            camera_center = pose_row["camera_to_world"][:3, 3]
            distance_m = float(np.linalg.norm(camera_center - target_centroid))
            admitted, reason = eligible_frame(stats, distance_m, rules)
            reasons[reason] += 1
            rows.append(
                {
                    "frame": int(pose_row["frame"]),
                    "trajectory_header": pose_row["header"],
                    "camera_center_world": [float(value) for value in camera_center],
                    "camera_to_target_centroid_m": distance_m,
                    "eligible": admitted,
                    "reason": reason,
                    **stats,
                }
            )
        pair = select_pair(rows)
        audit = {
            "scene_id": scene_id,
            "target_door_instance_id": target_id,
            "target_xml_area": int(frozen["target_xml_area"]),
            "trajectory_frames": len(poses),
            "mesh_vertices": int(len(xyz)),
            "mesh_faces": int(len(faces)),
            "target_ply_vertices": int(len(target_points)),
            "strict_target_faces": renderer.target_face_count,
            "eligible_frames": sum(1 for row in rows if row["eligible"]),
            "frame_reason_counts": dict(sorted(reasons.items())),
            "ordered_pair_found": pair is not None,
        }
        if pair is not None:
            reference, query, baseline = pair
            reference_stats, reference_mask = renderer.statistics(
                poses[int(reference["frame"])]["camera_to_world"], return_mask=True
            )
            query_stats, query_mask = renderer.statistics(
                poses[int(query["frame"])]["camera_to_world"], return_mask=True
            )
            require(reference_mask is not None and query_mask is not None, "SELECTED_MASK_MISSING")
            require(reference_stats == {key: reference[key] for key in reference_stats}, f"REFERENCE_STATS_DRIFT:{scene_id}")
            require(query_stats == {key: query[key] for key in query_stats}, f"QUERY_STATS_DRIFT:{scene_id}")
            reference = {**reference, "visible_mask_sha256": mask_sha256(reference_mask)}
            query = {**query, "visible_mask_sha256": mask_sha256(query_mask)}
            smaller_pixels = min(
                int(reference["visible_pixels"]), int(query["visible_pixels"])
            )
            audit["selected_pair"] = {
                "reference_frame": int(reference["frame"]),
                "query_frame": int(query["frame"]),
                "camera_baseline_m": baseline,
                "smaller_visible_target_pixels": smaller_pixels,
            }
            candidates.append(
                {
                    "scene": frozen,
                    "target_xml": xml_labels[target_id],
                    "target_ply_vertices": int(len(target_points)),
                    "strict_target_faces": renderer.target_face_count,
                    "reference": reference,
                    "query": query,
                    "camera_baseline_m": baseline,
                    "smaller_visible_target_pixels": smaller_pixels,
                }
            )
        audits.append(audit)

    cohort_size = int(protocol["pre_rgbd_selector"]["cohort_size"])
    require(len(candidates) >= cohort_size, f"NOT_EVALUABLE_ONLY_{len(candidates)}_SCENES")
    candidates.sort(
        key=lambda row: (
            -int(row["smaller_visible_target_pixels"]),
            -round(float(row["camera_baseline_m"]), 9),
            -int(row["scene"]["target_xml_area"]),
            row["scene"]["scene_id"],
        )
    )
    selected = candidates[:cohort_size]
    episodes: list[dict[str, Any]] = []
    for rank, row in enumerate(selected, start=1):
        frozen = row["scene"]
        episodes.append(
            {
                "episode_id": f"SV{rank:02d}",
                "scene_id": frozen["scene_id"],
                "target_door_instance_id": int(frozen["target_door_instance_id"]),
                "target_xml": row["target_xml"],
                "target_ply_vertices": row["target_ply_vertices"],
                "strict_target_faces": row["strict_target_faces"],
                "reference": row["reference"],
                "query": row["query"],
                "camera_baseline_m": row["camera_baseline_m"],
                "playback_mapping": {
                    "reference": f"playback frame {row['reference']['frame'] + 1:05d} -> sealed frame {row['reference']['frame']:04d}",
                    "query": f"playback frame {row['query']['frame'] + 1:05d} -> sealed frame {row['query']['frame']:04d}",
                },
            }
        )
    cohort = {
        "schema": COHORT_SCHEMA,
        "authority": "FROZEN_PRE_RGBD_STRICT_TRIANGLE_VISIBILITY_COHORT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": sha256(protocol_path),
        "source": {
            "provider": protocol["provider"],
            "input_manifest": source_manifest,
            "rgbd_files_at_freeze": 0,
        },
        "selection": protocol["pre_rgbd_selector"],
        "implementation": {
            "path": Path(__file__).name,
            "sha256": sha256(Path(__file__)),
            "helper_path": BASE_PATH.name,
            "helper_sha256": sha256(BASE_PATH),
            "extractor_source_path": EXTRACTOR_SOURCE_PATH.name,
            "extractor_source_sha256": sha256(EXTRACTOR_SOURCE_PATH),
            "runtime": renderer_runtime,
        },
        "selection_audit": audits,
        "episodes": episodes,
        "materialize_after_freeze": {
            "oni": [
                {
                    "scene_id": row["scene"]["scene_id"],
                    **row["scene"]["files"][f"{row['scene']['scene_id']}.oni"],
                    "md5": protocol["official_md5"][row["scene"]["scene_id"]]["oni"],
                }
                for row in selected
            ],
            "sealed_frame_layout": "payload/<scene>/selected/{image,depth}/frame.<trajectory-index:04d>.png",
            "playback_index_rule": "Official Playback.exe output index is trajectory frame index plus one.",
        },
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json(output_path, cohort)


def selected_frame_path(source_root: Path, scene_id: str, kind: str, frame: int) -> Path:
    return source_root / "payload" / scene_id / "selected" / kind / f"frame.{frame:04d}.png"


def parse_selected_timestamps(path: Path) -> dict[int, dict[str, int]]:
    require(path.is_file(), f"TIMESTAMP_MISSING:{path}")
    rows: dict[int, dict[str, int]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        require(len(fields) == 6, f"TIMESTAMP_COLUMNS:{path}:{line_number}")
        try:
            values = [int(field) for field in fields]
        except ValueError as error:
            raise RuntimeError(f"TIMESTAMP_INTEGER:{path}:{line_number}") from error
        frame = values[0]
        require(frame not in rows, f"TIMESTAMP_DUPLICATE:{path}:{frame}")
        rows[frame] = {
            "trajectory_frame": frame,
            "playback_index": values[1],
            "color_timestamp": values[2],
            "depth_timestamp": values[3],
            "color_frame_id": values[4],
            "depth_frame_id": values[5],
        }
    return rows


def parse_extraction_summary(path: Path) -> dict[str, int]:
    require(path.is_file(), f"EXTRACTION_SUMMARY_MISSING:{path}")
    values: dict[str, int] = {}
    for field in path.read_text(encoding="utf-8").split():
        require("=" in field, f"EXTRACTION_SUMMARY_FIELD:{path}:{field}")
        key, raw_value = field.split("=", 1)
        require(key not in values, f"EXTRACTION_SUMMARY_DUPLICATE:{path}:{key}")
        try:
            values[key] = int(raw_value)
        except ValueError as error:
            raise RuntimeError(f"EXTRACTION_SUMMARY_INTEGER:{path}:{key}") from error
    required = {"scanned", "discarded_depth", "discarded_color", "requested", "saved"}
    require(set(values) == required, f"EXTRACTION_SUMMARY_KEYS:{path}")
    return values


def seal_selected_frames(
    cohort_path: Path,
    source_root: Path,
    extraction_root: Path,
    extractor_exe: Path,
    receipt_path: Path,
) -> None:
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(
        cohort["implementation"]["extractor_source_sha256"] == sha256(EXTRACTOR_SOURCE_PATH),
        "EXTRACTOR_SOURCE_HASH_MISMATCH",
    )
    require(extractor_exe.is_file(), f"EXTRACTOR_EXE_MISSING:{extractor_exe}")
    oni_by_scene = {row["scene_id"]: row for row in cohort["materialize_after_freeze"]["oni"]}
    receipt_scenes: list[dict[str, Any]] = []
    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        extracted = extraction_root / scene_id
        required_frames = {
            int(episode["reference"]["frame"]),
            int(episode["query"]["frame"]),
        }
        require(len(required_frames) == 2, f"SELECTED_FRAME_COLLISION:{scene_id}")
        expected_names = {f"frame.{frame:04d}.png" for frame in required_frames}
        image_files = sorted((extracted / "image").glob("*.png"))
        depth_files = sorted((extracted / "depth").glob("*.png"))
        require({path.name for path in image_files} == expected_names, f"SPARSE_IMAGE_SET:{scene_id}")
        require({path.name for path in depth_files} == expected_names, f"SPARSE_DEPTH_SET:{scene_id}")
        timestamp_rows = parse_selected_timestamps(extracted / "selected_timestamp.txt")
        require(set(timestamp_rows) == required_frames, f"SPARSE_TIMESTAMP_SET:{scene_id}")
        summary = parse_extraction_summary(extracted / "summary.txt")
        require(summary["requested"] == len(required_frames), f"SPARSE_REQUESTED_COUNT:{scene_id}")
        require(summary["saved"] == len(required_frames), f"SPARSE_SAVED_COUNT:{scene_id}")
        require(summary["scanned"] >= max(required_frames) + 1, f"SPARSE_SCAN_PREFIX:{scene_id}")
        pose_count = len(base.parse_poses(input_paths(source_root, scene_id)["trajectory"]))
        require(max(required_frames) < pose_count, f"SELECTED_FRAME_OUTSIDE_POSES:{scene_id}")
        oni_path = source_root / "payload" / scene_id / f"{scene_id}.oni"
        expected_oni = oni_by_scene[scene_id]
        oni_receipt = validate_file(oni_path, expected_oni)
        sealed: dict[str, dict[str, Any]] = {}
        for role in ("reference", "query"):
            frame = int(episode[role]["frame"])
            playback_index = frame + 1
            timestamp_row = timestamp_rows[frame]
            require(timestamp_row["playback_index"] == playback_index, f"PLAYBACK_INDEX:{scene_id}:{frame}")
            source_image = extracted / "image" / f"frame.{frame:04d}.png"
            source_depth = extracted / "depth" / f"frame.{frame:04d}.png"
            require(source_image.is_file() and source_depth.is_file(), f"SELECTED_PLAYBACK_FRAME_MISSING:{scene_id}:{frame}")
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
            depth = cv2.imread(str(source_depth), cv2.IMREAD_UNCHANGED)
            require(image is not None and image.shape == (480, 640, 3), f"SELECTED_IMAGE_FORMAT:{scene_id}:{frame}")
            require(depth is not None and depth.shape == (480, 640) and depth.dtype == np.uint16, f"SELECTED_DEPTH_FORMAT:{scene_id}:{frame}")
            target_image = selected_frame_path(source_root, scene_id, "image", frame)
            target_depth = selected_frame_path(source_root, scene_id, "depth", frame)
            base.atomic_write(target_image, source_image.read_bytes())
            base.atomic_write(target_depth, source_depth.read_bytes())
            sealed[role] = {
                "trajectory_frame": frame,
                "playback_index": playback_index,
                "timestamp_row": timestamp_row,
                "image_path": relative(target_image, source_root),
                "image_sha256": sha256(target_image),
                "depth_path": relative(target_depth, source_root),
                "depth_sha256": sha256(target_depth),
                "valid_depth_fraction": float(np.count_nonzero(depth) / depth.size),
            }
        receipt_scenes.append(
            {
                "scene_id": scene_id,
                "pose_frames": pose_count,
                "extraction_summary": summary,
                "oni_path": relative(oni_path, source_root),
                "oni_bytes": oni_receipt["bytes"],
                "oni_sha256": oni_receipt["sha256"],
                "oni_md5": oni_receipt["md5"],
                "sealed": sealed,
            }
        )
    write_json(
        receipt_path,
        {
            "schema": RECEIPT_SCHEMA,
            "authority": "POST_COHORT_FREEZE_MATERIALIZATION_RECEIPT",
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "extractor": {
                "executable_path": str(extractor_exe.resolve()),
                "executable_sha256": sha256(extractor_exe),
                "source_path": EXTRACTOR_SOURCE_PATH.name,
                "source_sha256": sha256(EXTRACTOR_SOURCE_PATH),
            },
            "playback_index_rule": "trajectory frame n maps to synchronized output n+1 after the official 33000 microsecond lag-only discard loop",
            "scenes": receipt_scenes,
        },
    )


def replay(
    protocol_path: Path,
    cohort_path: Path,
    receipt_path: Path,
    source_root: Path,
    output_path: Path,
    preview_dir: Path,
) -> None:
    protocol = load_protocol(protocol_path)
    cohort = load_json(cohort_path)
    require(cohort.get("schema") == COHORT_SCHEMA, "COHORT_SCHEMA_MISMATCH")
    require(cohort["protocol_sha256"] == sha256(protocol_path), "PROTOCOL_HASH_MISMATCH")
    require(cohort["implementation"]["sha256"] == sha256(Path(__file__)), "IMPLEMENTATION_HASH_MISMATCH")
    require(cohort["implementation"]["helper_sha256"] == sha256(BASE_PATH), "HELPER_HASH_MISMATCH")
    require(
        cohort["implementation"]["extractor_source_sha256"] == sha256(EXTRACTOR_SOURCE_PATH),
        "EXTRACTOR_SOURCE_HASH_MISMATCH",
    )
    receipt = load_json(receipt_path)
    require(receipt.get("schema") == RECEIPT_SCHEMA, "RECEIPT_SCHEMA_MISMATCH")
    require(receipt["cohort_path"] == cohort_path.name, "RECEIPT_COHORT_PATH_MISMATCH")
    require(receipt["cohort_sha256"] == sha256(cohort_path), "RECEIPT_COHORT_HASH_MISMATCH")
    require(
        receipt["extractor"]["source_path"] == cohort["implementation"]["extractor_source_path"],
        "RECEIPT_EXTRACTOR_SOURCE_PATH_MISMATCH",
    )
    require(
        receipt["extractor"]["source_sha256"]
        == cohort["implementation"]["extractor_source_sha256"],
        "RECEIPT_EXTRACTOR_SOURCE_HASH_MISMATCH",
    )
    receipt_scenes = {row["scene_id"]: row for row in receipt["scenes"]}
    require(
        set(receipt_scenes) == {episode["scene_id"] for episode in cohort["episodes"]},
        "RECEIPT_SCENE_SET_MISMATCH",
    )
    intrinsic_path = source_root / "payload" / "intrinsic" / "asus.ini"
    intrinsic = base.parse_intrinsic(intrinsic_path)
    k = base.intrinsic_matrix(intrinsic)
    width, height = int(intrinsic["width"]), int(intrinsic["height"])
    renderer_rules = protocol["pre_rgbd_selector"]["renderer"]
    results: list[dict[str, Any]] = []

    for episode in cohort["episodes"]:
        scene_id = episode["scene_id"]
        target_id = int(episode["target_door_instance_id"])
        paths = input_paths(source_root, scene_id)
        for relative_path, frozen in cohort["source"]["input_manifest"].items():
            if f"payload/{scene_id}/" not in relative_path and not relative_path.endswith("asus.ini"):
                continue
            current = source_root / relative_path
            require(current.is_file(), f"MISSING_FROZEN_INPUT:{current}")
            require(current.stat().st_size == int(frozen["bytes"]), f"FROZEN_INPUT_SIZE:{current}")
            require(sha256(current) == frozen["sha256"], f"FROZEN_INPUT_HASH:{current}")

        xyz, labels, faces = read_mesh(paths["ply"])
        xml_labels = base.parse_xml_labels(paths["xml"])
        target_points = xyz[labels == target_id].astype(np.float64)
        target_boundary = base.boundary_vertices(target_points)
        poses = {row["frame"]: row["camera_to_world"] for row in base.parse_poses(paths["trajectory"])}
        reference_frame = int(episode["reference"]["frame"])
        query_frame = int(episode["query"]["frame"])
        reference_pose = poses[reference_frame]
        query_pose = poses[query_frame]
        renderer = VisibilityRenderer(xyz, labels, faces, target_id, intrinsic, renderer_rules)
        reference_visibility, reference_mask = renderer.statistics(reference_pose, return_mask=True)
        require(reference_mask is not None, f"REFERENCE_MASK_MISSING:{scene_id}")
        require(mask_sha256(reference_mask) == episode["reference"]["visible_mask_sha256"], f"REFERENCE_MASK_HASH:{scene_id}")
        query_envelope = base.mesh_envelope(target_boundary, query_pose, k, width, height)
        require(query_envelope is not None, f"QUERY_TRUTH_MISSING:{scene_id}")
        truth_mask = query_envelope[0]

        reference_depth_path = selected_frame_path(source_root, scene_id, "depth", reference_frame)
        reference_rgb_path = selected_frame_path(source_root, scene_id, "image", reference_frame)
        query_rgb_path = selected_frame_path(source_root, scene_id, "image", query_frame)
        query_depth_path = selected_frame_path(source_root, scene_id, "depth", query_frame)
        for path in (reference_depth_path, reference_rgb_path, query_rgb_path, query_depth_path):
            require(path.is_file(), f"MISSING_SELECTED_FRAME:{path}")
        require(scene_id in receipt_scenes, f"RECEIPT_SCENE_MISSING:{scene_id}")
        receipt_scene = receipt_scenes[scene_id]
        sealed = receipt_scene["sealed"]
        require(set(sealed) == {"reference", "query"}, f"RECEIPT_ROLE_SET:{scene_id}")
        for role, frame, image_path, depth_path in (
            ("reference", reference_frame, reference_rgb_path, reference_depth_path),
            ("query", query_frame, query_rgb_path, query_depth_path),
        ):
            frozen_role = sealed[role]
            require(int(frozen_role["trajectory_frame"]) == frame, f"RECEIPT_FRAME:{scene_id}:{role}")
            require(int(frozen_role["playback_index"]) == frame + 1, f"RECEIPT_PLAYBACK_INDEX:{scene_id}:{role}")
            timestamp_row = frozen_role["timestamp_row"]
            require(int(timestamp_row["trajectory_frame"]) == frame, f"RECEIPT_TIMESTAMP_FRAME:{scene_id}:{role}")
            require(int(timestamp_row["playback_index"]) == frame + 1, f"RECEIPT_TIMESTAMP_INDEX:{scene_id}:{role}")
            require(
                abs(int(timestamp_row["color_timestamp"]) - int(timestamp_row["depth_timestamp"]))
                <= 33000,
                f"RECEIPT_TIMESTAMP_SYNC:{scene_id}:{role}",
            )
            require(frozen_role["image_path"] == relative(image_path, source_root), f"RECEIPT_IMAGE_PATH:{scene_id}:{role}")
            require(frozen_role["depth_path"] == relative(depth_path, source_root), f"RECEIPT_DEPTH_PATH:{scene_id}:{role}")
            require(frozen_role["image_sha256"] == sha256(image_path), f"RECEIPT_IMAGE_HASH:{scene_id}:{role}")
            require(frozen_role["depth_sha256"] == sha256(depth_path), f"RECEIPT_DEPTH_HASH:{scene_id}:{role}")
        depth = cv2.imread(str(reference_depth_path), cv2.IMREAD_UNCHANGED)
        reference_rgb = cv2.imread(str(reference_rgb_path), cv2.IMREAD_COLOR)
        query_rgb = cv2.imread(str(query_rgb_path), cv2.IMREAD_COLOR)
        require(depth is not None and depth.dtype == np.uint16 and depth.shape == (height, width), f"DEPTH_FORMAT:{reference_depth_path}")
        require(reference_rgb is not None and reference_rgb.shape[:2] == (height, width), f"REFERENCE_RGB_FORMAT:{reference_rgb_path}")
        require(query_rgb is not None and query_rgb.shape[:2] == (height, width), f"QUERY_RGB_FORMAT:{query_rgb_path}")

        reference_points_camera = base.backproject_mask(depth, reference_mask, intrinsic)
        normal_camera, offset_camera, plane_stats = base.fit_plane(reference_points_camera)
        _, contour_camera = all_contour_plane_points(
            reference_mask, normal_camera, offset_camera, intrinsic
        )
        contour_world = contour_camera @ reference_pose[:3, :3].T + reference_pose[:3, 3]
        predicted_pixels, _ = base.project_world(contour_world, query_pose, k)
        predicted_envelope = base.envelope_from_pixels(predicted_pixels, width, height)
        prediction_visible = predicted_envelope is not None
        if predicted_envelope is None:
            predicted_mask = np.zeros((height, width), dtype=bool)
            predicted_stats = {"pixels": 0}
        else:
            predicted_mask, _, predicted_stats = predicted_envelope

        candidates = base.instance_envelopes(xyz, labels, xml_labels, query_pose, k, width, height)
        require(target_id in candidates, f"TARGET_NOT_VISIBLE:{scene_id}:{target_id}")
        instance_iou = {
            str(instance_id): base.mask_iou(predicted_mask, mask)
            for instance_id, mask in candidates.items()
        }
        selected_id = min(candidates, key=lambda instance_id: (-instance_iou[str(instance_id)], instance_id))
        target_iou = instance_iou[str(target_id)]
        intersection = np.count_nonzero(predicted_mask & truth_mask)
        predicted_pixels_count = np.count_nonzero(predicted_mask)
        truth_pixels_count = np.count_nonzero(truth_mask)
        truth_centroid = base.mask_centroid(truth_mask)
        if prediction_visible:
            predicted_centroid = base.mask_centroid(predicted_mask)
            centroid_error = float(np.linalg.norm(predicted_centroid - truth_centroid))
            rounded_centroid = np.rint(predicted_centroid).astype(int)
            centroid_inside = bool(
                0 <= rounded_centroid[0] < width
                and 0 <= rounded_centroid[1] < height
                and truth_mask[rounded_centroid[1], rounded_centroid[0]]
            )
        else:
            predicted_centroid = None
            centroid_error = None
            centroid_inside = False
        normal_world = reference_pose[:3, :3] @ normal_camera
        target_normal = base.mesh_plane_normal(target_points)
        world_centroid_error = float(
            np.linalg.norm(np.mean(contour_world, axis=0) - np.mean(target_points, axis=0))
        )

        preview_path = preview_dir / f"{episode['episode_id'].lower()}-visible-portal-transfer.jpg"
        base.atomic_write(
            preview_path,
            base.overlay_preview(
                reference_rgb,
                query_rgb,
                reference_mask,
                predicted_mask,
                truth_mask,
                episode["episode_id"],
            ),
        )
        results.append(
            {
                "episode_id": episode["episode_id"],
                "scene_id": scene_id,
                "target_door_instance_id": target_id,
                "selected_instance_id": int(selected_id),
                "selected_instance_text": xml_labels[selected_id]["text"],
                "correct_target_instance": selected_id == target_id,
                "wrong_instance_commit": selected_id != target_id,
                "visible_query_instance_count": len(candidates),
                "prediction_visible_in_query": prediction_visible,
                "prediction_collapse_reason": None if prediction_visible else "PROJECTED_CONTOUR_OUTSIDE_OR_DEGENERATE_IN_QUERY",
                "target_envelope_iou": target_iou,
                "target_precision": float(intersection / predicted_pixels_count) if predicted_pixels_count else 0.0,
                "target_recall": float(intersection / truth_pixels_count),
                "centroid_error_pixels": centroid_error,
                "centroid_error_image_diagonal_fraction": float(centroid_error / math.hypot(width, height)) if centroid_error is not None else None,
                "centroid_inside_target_envelope": centroid_inside,
                "metric_world_centroid_error_m": world_centroid_error,
                "reference_plane_normal_error_degrees": base.angle_degrees(normal_world, target_normal),
                "reference_plane": plane_stats,
                "reference_visibility": reference_visibility,
                "reference_visible_mask_sha256": mask_sha256(reference_mask),
                "camera_baseline_m": float(episode["camera_baseline_m"]),
                "reference_frame": reference_frame,
                "query_frame": query_frame,
                "predicted_contour_points_in_front": int(len(predicted_pixels)),
                "predicted_envelope_pixels": int(predicted_stats["pixels"]),
                "target_truth_envelope_pixels": int(truth_pixels_count),
                "instance_envelope_iou": instance_iou,
                "preview_path": str(preview_path.resolve()),
                "input_sha256": {
                    relative(path, source_root): sha256(path)
                    for path in (
                        paths["ply"], paths["xml"], paths["trajectory"], intrinsic_path,
                        reference_depth_path, reference_rgb_path, query_depth_path, query_rgb_path,
                    )
                },
            }
        )

    visible_errors = [row["centroid_error_pixels"] for row in results if row["centroid_error_pixels"] is not None]
    summary = {
        "episodes": len(results),
        "correct_target_instance": sum(row["correct_target_instance"] for row in results),
        "wrong_instance_commit": sum(row["wrong_instance_commit"] for row in results),
        "prediction_visible_in_query": sum(row["prediction_visible_in_query"] for row in results),
        "centroid_inside_target_envelope": sum(row["centroid_inside_target_envelope"] for row in results),
        "median_target_envelope_iou": float(np.median([row["target_envelope_iou"] for row in results])),
        "minimum_target_envelope_iou": float(min(row["target_envelope_iou"] for row in results)),
        "mean_centroid_error_pixels_visible_predictions": float(np.mean(visible_errors)) if visible_errors else None,
        "median_metric_world_centroid_error_m": float(np.median([row["metric_world_centroid_error_m"] for row in results])),
        "mean_reference_plane_normal_error_degrees": float(np.mean([row["reference_plane_normal_error_degrees"] for row in results])),
        "median_camera_baseline_m": float(np.median([row["camera_baseline_m"] for row in results])),
    }
    gates = {
        "correct_target_instance_3_of_3": summary["correct_target_instance"] == 3,
        "wrong_instance_commit_0_of_3": summary["wrong_instance_commit"] == 0,
        "centroid_inside_target_envelope_3_of_3": summary["centroid_inside_target_envelope"] == 3,
        "median_target_envelope_iou_at_least_0_5": summary["median_target_envelope_iou"] >= 0.5,
        "median_metric_world_centroid_error_m_at_most_0_25": summary["median_metric_world_centroid_error_m"] <= 0.25,
    }
    gate_met = all(gates.values())
    write_json(
        output_path,
        {
            "schema": RESULT_SCHEMA,
            "authority": "FRESH_SCENE_REAL_INDOOR_RGBD_DEVELOPMENT_RESULT",
            "protocol_path": protocol_path.name,
            "protocol_sha256": sha256(protocol_path),
            "cohort_path": cohort_path.name,
            "cohort_sha256": sha256(cohort_path),
            "receipt_path": receipt_path.name,
            "receipt_sha256": sha256(receipt_path),
            "implementation": cohort["implementation"],
            "conclusion": "L10_SCENENN_VISIBLE_METRIC_PORTAL_TRANSFER_DEVELOPMENT_GATE_MET" if gate_met else "L10_SCENENN_VISIBLE_METRIC_PORTAL_TRANSFER_DEVELOPMENT_GATE_NOT_MET",
            "gate_met": gate_met,
            "gates": gates,
            "summary": summary,
            "episodes": results,
            "claim_boundary": protocol["claim_boundary"],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--protocol", type=Path, default=HERE / "l10_scenenn_visible_portal_protocol_v3.json")
    freeze_parser.add_argument("--source-root", type=Path, required=True)
    freeze_parser.add_argument("--output", type=Path, default=HERE / "l10_scenenn_visible_portal_cohort_v3.json")
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--cohort", type=Path, default=HERE / "l10_scenenn_visible_portal_cohort_v3.json")
    seal_parser.add_argument("--source-root", type=Path, required=True)
    seal_parser.add_argument("--extraction-root", type=Path, required=True)
    seal_parser.add_argument("--extractor-exe", type=Path, required=True)
    seal_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--protocol", type=Path, default=HERE / "l10_scenenn_visible_portal_protocol_v3.json")
    replay_parser.add_argument("--cohort", type=Path, default=HERE / "l10_scenenn_visible_portal_cohort_v3.json")
    replay_parser.add_argument("--receipt", type=Path, required=True)
    replay_parser.add_argument("--source-root", type=Path, required=True)
    replay_parser.add_argument("--output", type=Path, default=HERE / "l10_scenenn_visible_portal_result_v3.json")
    replay_parser.add_argument("--preview-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "freeze":
        freeze_cohort(args.protocol, args.source_root, args.output)
    elif args.command == "seal":
        seal_selected_frames(
            args.cohort,
            args.source_root,
            args.extraction_root,
            args.extractor_exe,
            args.receipt,
        )
    else:
        replay(
            args.protocol,
            args.cohort,
            args.receipt,
            args.source_root,
            args.output,
            args.preview_dir,
        )


if __name__ == "__main__":
    main()
