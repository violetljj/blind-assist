"""Batch candidate correspondence annotation for the Failure Atlas."""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import uuid
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    connected_components,
)

from .correspondence import (
    ABSTAIN,
    MATCH,
    NO_MATCH,
    CorrespondenceThresholds,
    EvidenceWeights,
    annotate_frame,
    bbox_iou,
    mask_box_metrics,
    _mask_iou,
    rasterize_box,
    warp_box,
    warp_mask,
)


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_INSTANCE_CORRESPONDENCE_CANDIDATE_R0"
SCHEMA_VERSION = "blindassist.dual_loop_segmentation_instance_correspondence.result.v1"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"expected JSON object at {path}:{line_number}")
                rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(dict(row)), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _decode_packed_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    try:
        packed = np.frombuffer(base64.b64decode(str(encoded), validate=True), dtype=np.uint8)
    except Exception as exc:  # pragma: no cover - exact decoder errors are library-specific
        raise ValueError("invalid base64 packed mask") from exc
    expected = int(np.prod(shape))
    unpacked = np.unpackbits(packed, bitorder="big")
    if unpacked.size < expected or unpacked.size - expected >= 8:
        raise ValueError("packed mask length does not match analysis shape")
    return unpacked[:expected].reshape(shape).astype(bool)


def _resolve(repo_root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()


def _assert_output_scope(repo_root: Path, output_root: Path) -> None:
    allowed = (repo_root / "artifacts.local").resolve()
    try:
        output_root.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("output-root must stay under repo-root/artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repo_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _pair_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["source_id"]), int(row["frame_id"])


def _identity_key(row: Mapping[str, Any]) -> tuple[str, int, str]:
    return str(row["source_id"]), int(row["frame_id"]), str(row.get("image_sha256", ""))


def _sequence_id(frame: Mapping[str, Any]) -> str:
    return str(
        frame.get("sequence_id")
        or frame.get("session_id")
        or frame.get("source_id")
    )


def _timestamp(frame: Mapping[str, Any]) -> int:
    return int(frame.get("source_capture_timestamp_ns", frame.get("frame_id", 0)))


def _load_flow(path: Path | None) -> dict[tuple[str, int], dict[str, Any]]:
    if path is None:
        return {}
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl([path]), start=1):
        key = _pair_key(row)
        if key in result:
            raise ValueError(f"duplicate optical-flow row at {path}:{index}: {key}")
        matrix = np.asarray(row.get("matrix_previous_to_current"), dtype=np.float64)
        if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
            raise ValueError(f"invalid optical-flow affine at {path}:{index}")
        result[key] = {
            "previous_source_id": str(row.get("previous_source_id", key[0])),
            "previous_frame_id": int(row["previous_frame_id"]),
            "matrix_previous_to_current": matrix.tolist(),
            "quality": row.get("quality"),
        }
    return result


def _load_depth_clusters(
    path: Path | None,
    *,
    shape: tuple[int, int],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    if path is None:
        return {}
    result: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(_read_jsonl([path]), start=1):
        key = _pair_key(row)
        cluster_id = row.get("cluster_id")
        if cluster_id is None:
            raise ValueError(f"depth row {index} is missing cluster_id")
        cluster_mask: np.ndarray | None = None
        if row.get("mask_packed") is not None or row.get("packed_mask") is not None:
            encoded = row.get("mask_packed") or row.get("packed_mask")
            cluster_shape = tuple(int(value) for value in row.get("shape", shape))
            if cluster_shape != shape:
                raise ValueError(f"depth row {index} mask shape differs from analysis shape")
            cluster_mask = _decode_packed_mask(str(encoded), shape)
        box = row.get("bbox_xyxy") or row.get("box")
        if cluster_mask is None and box is None:
            raise ValueError(f"depth row {index} needs bbox_xyxy or mask_packed")
        if box is not None:
            # Validate now; all cluster boxes are declared in analysis-grid coordinates.
            rasterize_box(box, shape)
        median_depth = row.get("median_depth")
        if median_depth is not None and not math.isfinite(float(median_depth)):
            raise ValueError(f"depth row {index} has non-finite median_depth")
        result[key].append(
            {
                "depth_cluster_id": str(cluster_id),
                "median_depth": float(median_depth) if median_depth is not None else None,
                "bbox_xyxy": list(box) if box is not None else None,
                "mask": cluster_mask,
            }
        )
    return dict(result)


def _entity_region(entity: Mapping[str, Any], shape: tuple[int, int]) -> np.ndarray:
    mask = entity.get("mask")
    if mask is not None:
        return np.asarray(mask, dtype=bool)
    box = entity.get("bbox_xyxy") or entity.get("box")
    if box is None:
        raise ValueError("entity needs mask or bbox_xyxy")
    return rasterize_box(box, shape)


def _depth_for_entity(
    entity: Mapping[str, Any],
    clusters: Sequence[Mapping[str, Any]],
    shape: tuple[int, int],
) -> dict[str, Any] | None:
    direct_id = entity.get("depth_cluster_id")
    direct_depth = entity.get("median_depth")
    if direct_id is not None:
        for cluster in clusters:
            if str(cluster["depth_cluster_id"]) == str(direct_id):
                return {
                    "depth_cluster_id": str(direct_id),
                    "median_depth": cluster.get("median_depth", direct_depth),
                }
        return {
            "depth_cluster_id": str(direct_id),
            "median_depth": direct_depth,
        }
    if not clusters:
        return None
    region = _entity_region(entity, shape)
    region_pixels = int(np.count_nonzero(region))
    if not region_pixels:
        return None
    candidates: list[tuple[float, str, Mapping[str, Any]]] = []
    for cluster in clusters:
        cluster_region = cluster.get("mask")
        if cluster_region is None:
            cluster_region = rasterize_box(cluster["bbox_xyxy"], shape)
        else:
            cluster_region = np.asarray(cluster_region, dtype=bool)
        intersection = int(np.count_nonzero(region & cluster_region))
        if intersection:
            coverage = intersection / region_pixels
            candidates.append((coverage, str(cluster["depth_cluster_id"]), cluster))
    if not candidates:
        return None
    _, _, best = max(candidates, key=lambda value: (value[0], value[1]))
    return {
        "depth_cluster_id": str(best["depth_cluster_id"]),
        "median_depth": best.get("median_depth"),
    }


def _detection_semantic(detection: Mapping[str, Any], class_mapping: Mapping[str, Any]) -> str | None:
    direct = detection.get("semantic_class") or detection.get("class_name")
    if direct is not None:
        return str(direct)
    label = str(detection.get("label") or "").strip().lower()
    mapping = class_mapping.get("yolo_label_to_semantic", {})
    value = mapping.get(label)
    return str(value) if value is not None else None


def _normalise_detections(
    frame: Mapping[str, Any],
    trace: Mapping[str, Any],
    *,
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    detections = trace.get("detections")
    if not isinstance(detections, list):
        raise ValueError("YOLO trace detections must be a list")
    result: list[dict[str, Any]] = []
    for index, source_detection in enumerate(detections):
        if not isinstance(source_detection, dict):
            raise ValueError(f"invalid YOLO detection at frame {_pair_key(frame)} index {index}")
        box = source_detection.get("bbox_xyxy")
        if box is None:
            fields = ("left", "top", "right", "bottom")
            if not all(field in source_detection for field in fields):
                raise ValueError(f"YOLO detection missing bbox at frame {_pair_key(frame)} index {index}")
            box = [source_detection[field] for field in fields]
        source_width = int(source_detection.get("frame_width", frame.get("width", shape[1])))
        source_height = int(source_detection.get("frame_height", frame.get("height", shape[0])))
        if source_width <= 0 or source_height <= 0:
            raise ValueError("YOLO detection frame dimensions must be positive")
        analysis_box = rasterize_box(box, shape, source_shape=(source_height, source_width))
        ys, xs = np.nonzero(analysis_box)
        if not len(xs):
            # Keep the detection row for auditability, but it cannot match a component.
            projected_box = [0.0, 0.0, 0.0, 0.0]
        else:
            projected_box = [float(np.min(xs)), float(np.min(ys)), float(np.max(xs) + 1), float(np.max(ys) + 1)]
        detection_id = f"{frame['source_id']}:{int(frame['frame_id'])}:detection-{index:04d}"
        result.append(
            {
                "detection_id": detection_id,
                "source_id": str(frame["source_id"]),
                "frame_id": int(frame["frame_id"]),
                "sequence_id": _sequence_id(frame),
                "frame_order": int(frame.get("_frame_order", 0)),
                "label": source_detection.get("label") or source_detection.get("name"),
                "class_id": source_detection.get("class_id"),
                "confidence": source_detection.get("confidence"),
                "semantic_class": source_detection.get("semantic_class") or source_detection.get("class_name"),
                "bbox_xyxy": projected_box,
                "raw_bbox_xyxy": list(box),
                "external_track_id": source_detection.get("track_id") or source_detection.get("temporal_track_id"),
                "frame_width": source_width,
                "frame_height": source_height,
                "_box_mask": analysis_box,
            }
        )
    return result


def _component_rows(
    frame: Mapping[str, Any],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    shape: tuple[int, int],
) -> list[dict[str, Any]]:
    packed = frame.get("packed_masks")
    if not isinstance(packed, dict):
        raise ValueError(f"frame {_pair_key(frame)} is missing packed_masks")
    result: list[dict[str, Any]] = []
    class_cache: dict[str, list[Any]] = {}
    for row in ledger_rows:
        class_name = str(row.get("class_name") or row.get("predicted_class") or "")
        if not class_name:
            raise ValueError("component row is missing class_name/predicted_class")
        key = f"candidate_{class_name}"
        if key not in packed:
            raise ValueError(f"frame {_pair_key(frame)} is missing {key}")
        if class_name not in class_cache:
            class_mask = _decode_packed_mask(str(packed[key]), shape)
            class_cache[class_name] = connected_components(class_mask, connectivity=8)
        components = class_cache[class_name]
        component_index = int(row["component_index"])
        if component_index < 0 or component_index >= len(components):
            raise ValueError(f"component index out of range: {row.get('component_id')}")
        component = components[component_index]
        if int(row.get("area_pixels", component.area)) != component.area:
            raise ValueError(f"component area mismatch: {row.get('component_id')}")
        if "bbox_xyxy" in row and list(row["bbox_xyxy"]) != list(component.bbox):
            raise ValueError(f"component bbox mismatch: {row.get('component_id')}")
        component_id = str(
            row.get("component_id")
            or f"{frame['source_id']}:{int(frame['frame_id'])}:{class_name}:{component_index}"
        )
        result.append(
            {
                "component_id": component_id,
                "source_id": str(frame["source_id"]),
                "frame_id": int(frame["frame_id"]),
                "sequence_id": _sequence_id(frame),
                "frame_order": int(frame.get("_frame_order", 0)),
                "semantic_class": class_name,
                "component_index": component_index,
                "bbox_xyxy": list(component.bbox),
                "mask": component.mask,
                "temporal_track_id": row.get("temporal_track_id"),
                "depth_cluster_id": row.get("depth_cluster_id"),
                "median_depth": row.get("median_depth"),
            }
        )
    return result


def _assign_component_tracks(
    components: Sequence[dict[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    flow_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    thresholds: CorrespondenceThresholds,
) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    frame_order = {(_sequence_id(frame), int(frame["frame_id"])): int(frame["_frame_order"]) for frame in frames}
    for row in components:
        groups[(str(row["sequence_id"]), str(row["semantic_class"]))].append(row)
    next_id = 0
    for (sequence_id, class_name), rows in sorted(groups.items()):
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_order[frame_order[(str(row["sequence_id"]), int(row["frame_id"]))]].append(row)
        previous: list[dict[str, Any]] = []
        previous_order: int | None = None
        for order in sorted(by_order):
            current = sorted(by_order[order], key=lambda item: (int(item["component_index"]), str(item["component_id"])))
            adjacent = previous_order is not None and order == previous_order + 1
            candidates: list[tuple[float, int, int]] = []
            if adjacent:
                for previous_index, previous_row in enumerate(previous):
                    for current_index, current_row in enumerate(current):
                        raw = _mask_iou(previous_row["mask"], current_row["mask"])
                        value = raw
                        flow = flow_rows.get((str(current_row["source_id"]), int(current_row["frame_id"])))
                        if flow and int(flow["previous_frame_id"]) == int(previous_row["frame_id"]):
                            warped = warp_mask(previous_row["mask"], flow["matrix_previous_to_current"])
                            value = max(value, _mask_iou(warped, current_row["mask"]))
                        if value >= thresholds.minimum_temporal_iou:
                            candidates.append((value, previous_index, current_index))
            used_previous: set[int] = set()
            used_current: set[int] = set()
            for _, previous_index, current_index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
                if previous_index in used_previous or current_index in used_current:
                    continue
                previous_row = previous[previous_index]
                current_row = current[current_index]
                if current_row.get("temporal_track_id") is None:
                    current_row["temporal_track_id"] = previous_row.get("temporal_track_id")
                used_previous.add(previous_index)
                used_current.add(current_index)
            for row in current:
                if row.get("temporal_track_id") is None:
                    row["temporal_track_id"] = f"{sequence_id}:{class_name}:component-track-{next_id:05d}"
                    next_id += 1
            previous, previous_order = current, order
    counts = Counter(str(row["temporal_track_id"]) for row in components)
    for row in components:
        row["track_persistence_frames"] = int(counts[str(row["temporal_track_id"])])


def _assign_detection_tracks(
    detections: Sequence[dict[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    flow_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    thresholds: CorrespondenceThresholds,
    class_mapping: Mapping[str, Any],
) -> None:
    frame_order = {(_sequence_id(frame), int(frame["frame_id"])): int(frame["_frame_order"]) for frame in frames}
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in detections:
        semantic = _detection_semantic(row, class_mapping)
        row["semantic_class"] = semantic
        groups[str(row["sequence_id"])].append(row)
    next_id = 0
    for sequence_id, rows in sorted(groups.items()):
        by_order: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_order[frame_order[(str(row["sequence_id"]), int(row["frame_id"]))]].append(row)
        previous: list[dict[str, Any]] = []
        previous_order: int | None = None
        for order in sorted(by_order):
            current = sorted(by_order[order], key=lambda item: str(item["detection_id"]))
            adjacent = previous_order is not None and order == previous_order + 1
            candidates: list[tuple[float, int, int]] = []
            if adjacent:
                for previous_index, previous_row in enumerate(previous):
                    for current_index, current_row in enumerate(current):
                        if previous_row.get("semantic_class") != current_row.get("semantic_class"):
                            continue
                        if previous_row.get("label") != current_row.get("label") and previous_row.get("semantic_class") is None:
                            continue
                        value = bbox_iou(previous_row["bbox_xyxy"], current_row["bbox_xyxy"])
                        flow = flow_rows.get((str(current_row["source_id"]), int(current_row["frame_id"])))
                        if flow and int(flow["previous_frame_id"]) == int(previous_row["frame_id"]):
                            value = max(value, bbox_iou(warp_box(previous_row["bbox_xyxy"], flow["matrix_previous_to_current"]), current_row["bbox_xyxy"]))
                        if value >= thresholds.detection_track_iou:
                            candidates.append((value, previous_index, current_index))
            used_previous: set[int] = set()
            used_current: set[int] = set()
            for _, previous_index, current_index in sorted(candidates, key=lambda item: (-item[0], item[1], item[2])):
                if previous_index in used_previous or current_index in used_current:
                    continue
                previous_row = previous[previous_index]
                current_row = current[current_index]
                if current_row.get("external_track_id") is None:
                    current_row["temporal_track_id"] = previous_row.get("temporal_track_id")
                used_previous.add(previous_index)
                used_current.add(current_index)
            for row in current:
                if row.get("temporal_track_id") is None:
                    external = row.get("external_track_id")
                    row["temporal_track_id"] = (
                        f"{sequence_id}:detector-track:{external}"
                        if external is not None
                        else f"{sequence_id}:detector-track-{next_id:05d}"
                    )
                    next_id += 1
            previous, previous_order = current, order
    counts = Counter(str(row["temporal_track_id"]) for row in detections)
    for row in detections:
        row["track_persistence_frames"] = int(counts[str(row["temporal_track_id"])])


def _attach_previous_entities(
    rows: Sequence[dict[str, Any]],
    *,
    track_field: str = "temporal_track_id",
) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["sequence_id"]), str(row.get(track_field)))].append(row)
    for group in grouped.values():
        ordered = sorted(group, key=lambda item: (int(item["frame_order"]), str(item.get("component_id") or item.get("detection_id"))))
        for previous, current in zip(ordered, ordered[1:]):
            if int(current["frame_order"]) == int(previous["frame_order"]) + 1:
                current["_previous_entity"] = previous


def _flow_evidence(
    component: Mapping[str, Any],
    detection: Mapping[str, Any],
    flow_rows: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any] | None:
    previous_component = component.get("_previous_entity")
    previous_detection = detection.get("_previous_entity")
    if previous_component is None or previous_detection is None:
        return None
    flow = flow_rows.get((str(component["source_id"]), int(component["frame_id"])))
    if flow is None:
        return None
    if int(flow["previous_frame_id"]) != int(previous_component["frame_id"]) or int(flow["previous_frame_id"]) != int(previous_detection["frame_id"]):
        return None
    matrix = flow["matrix_previous_to_current"]
    component_iou = _mask_iou(warp_mask(previous_component["mask"], matrix), component["mask"])
    detection_iou = bbox_iou(warp_box(previous_detection["bbox_xyxy"], matrix), detection["bbox_xyxy"])
    return {
        "component_iou": float(component_iou),
        "detection_iou": float(detection_iou),
        "support": float((component_iou + detection_iou) / 2.0),
        "previous_frame_id": int(flow["previous_frame_id"]),
        "quality": flow.get("quality"),
    }


def _temporal_values(
    components: Sequence[Mapping[str, Any]],
    detections: Sequence[Mapping[str, Any]],
    history: set[tuple[str, str]],
) -> dict[tuple[str, str], float | None]:
    component_history = {left for left, _ in history}
    detection_history = {right for _, right in history}
    result: dict[tuple[str, str], float | None] = {}
    for component in components:
        component_id = str(component["component_id"])
        component_track = component.get("temporal_track_id")
        for detection in detections:
            detection_id = str(detection["detection_id"])
            detection_track = detection.get("temporal_track_id")
            if component_track is None or detection_track is None:
                result[(component_id, detection_id)] = None
            elif (str(component_track), str(detection_track)) in history:
                result[(component_id, detection_id)] = 1.0
            elif str(component_track) in component_history or str(detection_track) in detection_history:
                result[(component_id, detection_id)] = 0.0
            else:
                result[(component_id, detection_id)] = None
    return result


def _frame_result_metadata(frame: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_id": str(frame["source_id"]),
        "session_id": frame.get("session_id"),
        "sequence_id": _sequence_id(frame),
        "frame_id": int(frame["frame_id"]),
        "view_row_id": frame.get("view_row_id"),
        "image_sha256": frame.get("image_sha256"),
    }


def run_batch(
    *,
    repo_root: Path,
    config_path: Path,
    frames_paths: Sequence[Path],
    components_paths: Sequence[Path],
    yolo_trace_paths: Sequence[Path],
    output_root: Path,
    depth_clusters_path: Path | None = None,
    optical_flow_path: Path | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    config_path = config_path.resolve()
    output_root = output_root.resolve()
    _assert_output_scope(repo_root, output_root)
    config = _read_json(config_path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected correspondence protocol_id")
    shape = tuple(int(value) for value in config.get("analysis_shape", [256, 256]))
    if len(shape) != 2 or min(shape) <= 0:
        raise ValueError("analysis_shape must contain positive height and width")
    thresholds = CorrespondenceThresholds.from_mapping(config.get("thresholds"))
    weights = EvidenceWeights.from_mapping(config.get("weights"))
    class_mapping = dict(config.get("class_compatibility", {}))
    frame_rows = _read_jsonl(frames_paths)
    component_ledger = _read_jsonl(components_paths)
    trace_rows = _read_jsonl(yolo_trace_paths)
    if not frame_rows:
        raise ValueError("no frame rows")
    frames_by_pair: dict[tuple[str, int], dict[str, Any]] = {}
    for frame in frame_rows:
        key = _pair_key(frame)
        if key in frames_by_pair:
            raise ValueError(f"duplicate frame identity {key}")
        frame_copy = dict(frame)
        frames_by_pair[key] = frame_copy
    ordered_frames = sorted(
        frames_by_pair.values(),
        key=lambda row: (_sequence_id(row), _timestamp(row), int(row["frame_id"]), str(row.get("view_row_id", ""))),
    )
    for index, frame in enumerate(ordered_frames):
        frame["_frame_order"] = index
    ledger_by_pair: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    seen_component_ids: set[str] = set()
    for row in component_ledger:
        component_id = str(row.get("component_id", ""))
        if not component_id or component_id in seen_component_ids:
            raise ValueError(f"duplicate or missing component_id: {component_id}")
        seen_component_ids.add(component_id)
        ledger_by_pair[_pair_key(row)].append(row)
    trace_by_identity: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in trace_rows:
        key = _identity_key(row)
        if key in trace_by_identity:
            raise ValueError(f"duplicate YOLO trace identity {key}")
        trace_by_identity[key] = row
    components: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    for frame in ordered_frames:
        pair = _pair_key(frame)
        trace = trace_by_identity.get(_identity_key(frame))
        if trace is None:
            raise ValueError(f"missing exact YOLO trace row for {pair}/{frame.get('image_sha256')}")
        frame_components = _component_rows(frame, ledger_by_pair.get(pair, []), shape=shape)
        frame_detections = _normalise_detections(frame, trace, shape=shape)
        components.extend(frame_components)
        detections.extend(frame_detections)
    flow_rows = _load_flow(optical_flow_path)
    depth_clusters = _load_depth_clusters(depth_clusters_path, shape=shape)
    _assign_component_tracks(components, ordered_frames, flow_rows, thresholds)
    _assign_detection_tracks(detections, ordered_frames, flow_rows, thresholds, class_mapping)
    _attach_previous_entities(components)
    _attach_previous_entities(detections)
    components_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    detections_by_pair: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in components:
        components_by_pair[_pair_key(row)].append(row)
    for row in detections:
        detections_by_pair[_pair_key(row)].append(row)

    pair_output: list[dict[str, Any]] = []
    component_output: list[dict[str, Any]] = []
    detection_output: list[dict[str, Any]] = []
    selected_track_pairs: set[tuple[str, str]] = set()
    depth_known_component = 0
    depth_known_detection = 0
    flow_pair_count = 0
    temporal_pair_count = 0
    class_state_counts: Counter[str] = Counter()
    for frame in ordered_frames:
        pair = _pair_key(frame)
        frame_components = sorted(components_by_pair.get(pair, []), key=lambda row: str(row["component_id"]))
        frame_detections = sorted(detections_by_pair.get(pair, []), key=lambda row: str(row["detection_id"]))
        cluster_rows = depth_clusters.get(pair, [])
        component_depth = {}
        detection_depth = {}
        for component in frame_components:
            value = _depth_for_entity(component, cluster_rows, shape)
            if value is not None:
                component_depth[str(component["component_id"])] = value
                depth_known_component += 1
        for detection in frame_detections:
            value = _depth_for_entity(detection, cluster_rows, shape)
            if value is not None:
                detection_depth[str(detection["detection_id"])] = value
                depth_known_detection += 1
        temporal_by_pair = _temporal_values(frame_components, frame_detections, selected_track_pairs)
        flow_by_pair: dict[tuple[str, str], dict[str, Any]] = {}
        for component in frame_components:
            for detection in frame_detections:
                value = _flow_evidence(component, detection, flow_rows)
                if value is not None:
                    flow_by_pair[(str(component["component_id"]), str(detection["detection_id"]))] = value
                    flow_pair_count += 1
                if temporal_by_pair.get((str(component["component_id"]), str(detection["detection_id"]))) is not None:
                    temporal_pair_count += 1
        frame_result = annotate_frame(
            frame_components,
            frame_detections,
            thresholds=thresholds,
            weights=weights,
            class_mapping=class_mapping,
            component_depth=component_depth,
            detection_depth=detection_depth,
            temporal_by_pair=temporal_by_pair,
            flow_by_pair=flow_by_pair,
        )
        metadata = _frame_result_metadata(frame)
        for row in frame_result["pair_rows"]:
            row = {**metadata, **row}
            class_state_counts[str(row["class_compatibility"]["state"])] += 1
            pair_output.append(row)
        for row in frame_result["component_rows"]:
            component_output.append({**metadata, **row})
        for row in frame_result["detection_rows"]:
            detection_output.append({**metadata, **row})
        for row in frame_result["component_rows"]:
            if row.get("state") == MATCH and row.get("component_track_id") and row.get("detection_track_id"):
                selected_track_pairs.add((str(row["component_track_id"]), str(row["detection_track_id"])))

    pair_counts = Counter(str(row["state"]) for row in pair_output)
    component_counts = Counter(str(row["state"]) for row in component_output)
    detection_counts = Counter(str(row["state"]) for row in detection_output)
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "authority": "CANDIDATE_ANNOTATION_ONLY",
        "claim_ceiling": config.get("claim_ceiling", "DEVELOPMENT_CANDIDATE_ANNOTATION_ONLY"),
        "drives_alerts": False,
        "input_counts": {
            "frames": len(ordered_frames),
            "components": len(components),
            "detections": len(detections),
            "pair_candidates": len(pair_output),
        },
        "state_counts": {
            "pair": dict(pair_counts),
            "component": dict(component_counts),
            "detection": dict(detection_counts),
        },
        "evidence_availability": {
            "depth_clusters_input_supplied": depth_clusters_path is not None,
            "optical_flow_input_supplied": optical_flow_path is not None,
            "component_depth_rows_assigned": depth_known_component,
            "detection_depth_rows_assigned": depth_known_detection,
            "pair_flow_rows_assigned": flow_pair_count,
            "pair_temporal_values_assigned": temporal_pair_count,
            "class_compatibility_states": dict(class_state_counts),
        },
        "terminals": {
            "candidate_annotation": "COMPLETE",
            "residual_labelability": "UNCHANGED_WEAKLY_LABELABLE_UNLESS_SEPARATE_CONTRACT",
            "product_or_safety": "NOT_EVALUABLE",
        },
    }
    input_paths: list[Path] = [config_path, *frames_paths, *components_paths, *yolo_trace_paths]
    if depth_clusters_path is not None:
        input_paths.append(depth_clusters_path)
    if optical_flow_path is not None:
        input_paths.append(optical_flow_path)
    provenance = {
        "schema_version": "blindassist.dual_loop_segmentation_instance_correspondence.provenance.v1",
        "inputs": [
            {"path": _repo_relative(repo_root, path), "sha256": _sha256(path.resolve())}
            for path in input_paths
        ],
        "implementation": [
            {"path": _repo_relative(repo_root, Path(__file__)), "sha256": _sha256(Path(__file__).resolve())},
            {"path": _repo_relative(repo_root, Path(__file__).parent / "correspondence.py"), "sha256": _sha256(Path(__file__).parent / "correspondence.py")},
        ],
        "config": {
            "path": _repo_relative(repo_root, config_path),
            "sha256": _sha256(config_path),
            "thresholds": _json_ready(config.get("thresholds", {})),
            "weights": _json_ready(config.get("weights", {})),
        },
    }
    summary["provenance"] = provenance
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(temporary / "pair_evidence.jsonl", pair_output)
        _write_jsonl(temporary / "component_annotations.jsonl", component_output)
        _write_jsonl(temporary / "detection_annotations.jsonl", detection_output)
        _write_json(temporary / "summary.json", summary)
        _write_json(temporary / "provenance.json", provenance)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", required=True, action="append")
    parser.add_argument("--components", required=True, action="append")
    parser.add_argument("--yolo-trace", required=True, action="append")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--depth-clusters")
    parser.add_argument("--optical-flow")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    summary = run_batch(
        repo_root=repo_root,
        config_path=_resolve(repo_root, args.config),  # type: ignore[arg-type]
        frames_paths=[_resolve(repo_root, value) for value in args.frames],  # type: ignore[list-item]
        components_paths=[_resolve(repo_root, value) for value in args.components],  # type: ignore[list-item]
        yolo_trace_paths=[_resolve(repo_root, value) for value in args.yolo_trace],  # type: ignore[list-item]
        output_root=_resolve(repo_root, args.output_root),  # type: ignore[arg-type]
        depth_clusters_path=_resolve(repo_root, args.depth_clusters),
        optical_flow_path=_resolve(repo_root, args.optical_flow),
    )
    print(json.dumps(_json_ready(summary), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
