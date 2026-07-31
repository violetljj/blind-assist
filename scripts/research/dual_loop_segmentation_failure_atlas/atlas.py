"""Build the R0 segmentation-failure atlas from consumed Development evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    aggregate_confusion,
    component_metrics,
    connected_components,
    mask_iou,
    pixel_metrics,
)

try:
    from scipy import ndimage as _ndimage
except ImportError:  # pragma: no cover - the NumPy fallback is tested
    _ndimage = None


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_FAILURE_ATLAS_AND_RESIDUAL_LABELABILITY_R0"
FRAME_ROLE = "r1_consumed_fresh"
SCHEMA_VERSION = "blindassist.dual_loop_segmentation_failure_atlas.result.v1"

_ACTIONABLE_MECHANISMS = (
    "SMALL_FRAGMENT_NOISE",
    "LARGE_WALKABLE_CONFUSION",
    "BOUNDARY_DILATION",
    "YOLO_ATTRIBUTION_AMBIGUITY",
    "TEMPORAL_FLICKER",
    "STABLE_HIGH_CONFIDENCE_ERROR",
    "UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY",
)

_EXPANSION_MECHANISMS = (
    "SMALL_FRAGMENT_NOISE",
    "YOLO_ATTRIBUTION_AMBIGUITY",
    "TEMPORAL_FLICKER",
    "STABLE_HIGH_CONFIDENCE_ERROR",
    "UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _as_paths(value: Path | Sequence[Path]) -> list[Path]:
    return [value] if isinstance(value, Path) else list(value)


def _read_jsonls(paths: Sequence[Path]) -> list[dict[str, Any]]:
    return [row for path in paths for row in read_jsonl(path)]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_packed_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    """Decode the frozen base64/packbits mask representation."""

    packed = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    expected_pixels = int(np.prod(shape))
    unpacked = np.unpackbits(packed, bitorder="big")
    if unpacked.size < expected_pixels or unpacked.size - expected_pixels >= 8:
        raise ValueError("packed mask length does not match analysis shape")
    return unpacked[:expected_pixels].reshape(shape).astype(bool)


def encode_packed_mask(mask: np.ndarray) -> str:
    """Encode a mask for unit-test fixtures."""

    flat = np.asarray(mask, dtype=bool).reshape(-1).astype(np.uint8)
    return base64.b64encode(np.packbits(flat, bitorder="big").tobytes()).decode("ascii")


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    value = np.asarray(mask, dtype=bool)
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return value.copy()
    if _ndimage is not None:
        return _ndimage.binary_dilation(
            value,
            structure=np.ones((radius * 2 + 1, radius * 2 + 1), dtype=bool),
        )
    padded = np.pad(value, radius, mode="constant", constant_values=False)
    result = np.zeros_like(value, dtype=bool)
    height, width = value.shape
    for dy in range(radius * 2 + 1):
        for dx in range(radius * 2 + 1):
            result |= padded[dy : dy + height, dx : dx + width]
    return result


def spatial_probe_mask(
    shape: tuple[int, int],
    probe_name: str,
    bands: dict[str, Any],
) -> np.ndarray:
    height, width = shape
    mask = np.zeros(shape, dtype=bool)
    if probe_name == "FULL_FRAME":
        mask[:] = True
    elif probe_name == "LOWER_FIELD":
        y0 = int(np.floor(height * float(bands["lower_field_y_fraction"])))
        mask[y0:, :] = True
    elif probe_name == "CENTRAL_BODY_CORRIDOR":
        x0_fraction, x1_fraction = bands["central_body_x_fraction"]
        x0 = int(np.floor(width * float(x0_fraction)))
        x1 = int(np.ceil(width * float(x1_fraction)))
        y0 = int(np.floor(height * float(bands["central_body_y_min_fraction"])))
        mask[y0:, x0:x1] = True
    elif probe_name == "UPPER_HEAD_BAND":
        y1 = int(np.ceil(height * float(bands["upper_head_y_max_fraction"])))
        mask[:y1, :] = True
    else:
        raise ValueError(f"unknown spatial probe: {probe_name}")
    return mask


def causal_temporal_probe(
    current: np.ndarray,
    previous: np.ndarray | None,
    previous_previous: np.ndarray | None,
    probe_name: str,
) -> np.ndarray:
    """Apply a causal gate while requiring a current-frame activation."""

    value = np.asarray(current, dtype=bool)
    previous_value = np.zeros_like(value) if previous is None else np.asarray(previous, dtype=bool)
    previous_previous_value = (
        np.zeros_like(value) if previous_previous is None else np.asarray(previous_previous, dtype=bool)
    )
    if probe_name == "CURRENT_FRAME":
        return value.copy()
    if probe_name == "CAUSAL_2_OF_3":
        return value & (previous_value | previous_previous_value)
    if probe_name == "CAUSAL_3_CONSECUTIVE":
        return value & previous_value & previous_previous_value
    raise ValueError(f"unknown temporal probe: {probe_name}")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.zeros(len(values), dtype=np.float64)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        average = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = average
        cursor = end
    return ranks


def spearman_rank_correlation(
    left: Sequence[float],
    right: Sequence[float],
) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_centered = left_ranks - float(np.mean(left_ranks))
    right_centered = right_ranks - float(np.mean(right_ranks))
    denominator = float(
        np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator == 0.0:
        return None
    return float(np.sum(left_centered * right_centered) / denominator)


def expansion_decision(
    *,
    mechanisms_reproduced: bool,
    aggregate_rank_correlation: float | None,
    minimum_rank_correlation: float,
    gating_axis: str,
    source_dependent: bool,
    terminals: dict[str, str] | None = None,
) -> str:
    names = terminals or {
        "mechanism_not_reproduced": "PILOT_NOT_GENERALIZABLE",
        "gating_sufficient": "GATING_SUFFICIENT",
        "gating_partial": "GATING_PARTIAL",
        "source_dependent": "SOURCE_DEPENDENT_FAILURE_STRUCTURE",
        "replicated": "RESIDUAL_TASK_REFORMULATION_JUSTIFIED",
    }
    if (
        not mechanisms_reproduced
        or aggregate_rank_correlation is None
        or aggregate_rank_correlation < minimum_rank_correlation
    ):
        return names["mechanism_not_reproduced"]
    if gating_axis == "SUFFICIENT":
        return names["gating_sufficient"]
    if gating_axis == "PARTIAL":
        return names["gating_partial"]
    if source_dependent:
        return names["source_dependent"]
    return names["replicated"]


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_json_ready(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(_json_ready(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _truth_histogram(mask: np.ndarray, component_mask: np.ndarray, names: dict[int, str]) -> dict[str, int]:
    values, counts = np.unique(mask[component_mask], return_counts=True)
    result = {names[int(value)]: int(count) for value, count in zip(values, counts)}
    for name in names.values():
        result.setdefault(name, 0)
    return result


def _dominant_truth_class(histogram: dict[str, int]) -> str:
    return sorted(histogram, key=lambda name: (-histogram[name], name))[0]


def _spatial_membership(
    component_mask: np.ndarray,
    spatial_masks: dict[str, np.ndarray],
) -> list[str]:
    return [
        name
        for name, mask in spatial_masks.items()
        if name != "FULL_FRAME" and np.count_nonzero(component_mask & mask) > 0
    ]


def _mechanism_tags(record: dict[str, Any], rules: dict[str, Any]) -> list[str]:
    if not record["false_activation"]:
        return []
    tags: list[str] = []
    if record["area_pixels"] <= int(rules["small_fragment_max_area_pixels"]):
        tags.append("SMALL_FRAGMENT_NOISE")
    if (
        record["area_pixels"] >= int(rules["large_component_min_area_pixels"])
        and record["dominant_truth_class"] == "walkable"
    ):
        tags.append("LARGE_WALKABLE_CONFUSION")
    if record["boundary_proximity_fraction"] >= float(rules["boundary_dilation_min_fraction"]):
        tags.append("BOUNDARY_DILATION")
    if (
        record["yolo_overlapped_truth_hazard_intersection_pixels"] > 0
        or (
            record["nearest_yolo_box_distance_pixels"] is not None
            and record["nearest_yolo_box_distance_pixels"] <= float(rules["yolo_attribution_gap_pixels"])
        )
    ):
        tags.append("YOLO_ATTRIBUTION_AMBIGUITY")
    if record["false_activation_run_observations"] == 1:
        tags.append("TEMPORAL_FLICKER")
    if (
        record["false_activation_run_observations"] >= int(rules["stable_minimum_frames"])
        and record["top1_confidence_median"] is not None
        and record["top1_confidence_median"] >= float(rules["high_confidence_minimum"])
    ):
        tags.append("STABLE_HIGH_CONFIDENCE_ERROR")
    if (
        "UPPER_HEAD_BAND" in record["spatial_bands"]
        and record["dominant_truth_class"] in {"walkable", "unknown_nonwalkable"}
    ):
        tags.append("UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY")
    return tags or ["OTHER_FALSE_ACTIVATION"]


def primary_mechanism(tags: list[str]) -> str | None:
    priority = (
        "YOLO_ATTRIBUTION_AMBIGUITY",
        "BOUNDARY_DILATION",
        "LARGE_WALKABLE_CONFUSION",
        "STABLE_HIGH_CONFIDENCE_ERROR",
        "TEMPORAL_FLICKER",
        "SMALL_FRAGMENT_NOISE",
        "UPPER_FIELD_BACKGROUND_ACTIVATION_PROXY",
        "OTHER_FALSE_ACTIVATION",
    )
    return next((name for name in priority if name in tags), None)


def assign_temporal_tracks(
    components: list[dict[str, Any]],
    frame_order: list[dict[str, Any]],
    minimum_iou: float,
) -> None:
    """Greedily associate same-class components across adjacent materialized observations."""

    order = {
        (row["sequence_id"], int(row["frame_id"])): index
        for index, row in enumerate(frame_order)
    }
    grouped: dict[tuple[str, str], dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for record in components:
        sequence_id = record["sequence_id"]
        position = order[(sequence_id, int(record["frame_id"]))]
        grouped[(sequence_id, record["predicted_class"])][position].append(record)

    next_track = 0
    track_members: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (sequence_id, class_name), by_position in sorted(grouped.items()):
        previous: list[dict[str, Any]] = []
        previous_position: int | None = None
        for position in sorted(by_position):
            current = sorted(by_position[position], key=lambda item: item["component_index"])
            adjacent = previous_position is not None and position == previous_position + 1
            pairs: list[tuple[float, int, int]] = []
            if adjacent:
                for previous_index, previous_record in enumerate(previous):
                    for current_index, current_record in enumerate(current):
                        iou = mask_iou(previous_record["_mask"], current_record["_mask"])
                        if iou >= minimum_iou:
                            pairs.append((iou, previous_index, current_index))
            used_previous: set[int] = set()
            used_current: set[int] = set()
            for iou, previous_index, current_index in sorted(
                pairs, key=lambda item: (-item[0], item[1], item[2])
            ):
                if previous_index in used_previous or current_index in used_current:
                    continue
                previous_record = previous[previous_index]
                current_record = current[current_index]
                current_record["temporal_track_id"] = previous_record["temporal_track_id"]
                current_record["previous_observation_iou"] = float(iou)
                previous_record["next_observation_iou"] = float(iou)
                used_previous.add(previous_index)
                used_current.add(current_index)
            for record in current:
                if record.get("temporal_track_id") is None:
                    record["temporal_track_id"] = (
                        f"{sequence_id}:{class_name}:track-{next_track:05d}"
                    )
                    next_track += 1
                track_members[record["temporal_track_id"]].append(record)
            previous = current
            previous_position = position

    for members in track_members.values():
        persistence = len(members)
        for member in members:
            member["persistence_observations"] = persistence
            member["false_activation_run_observations"] = 0
        false_run: list[dict[str, Any]] = []
        for member in [*members, {"false_activation": False}]:
            if member.get("false_activation", False):
                false_run.append(member)
                continue
            for false_member in false_run:
                false_member["false_activation_run_observations"] = len(false_run)
            false_run = []


def _aggregate_components(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "predicted_component_count": 0,
            "truth_component_count": 0,
            "hit_predicted_component_count": 0,
            "hit_truth_component_count": 0,
            "false_activation_component_count": 0,
            "component_precision": None,
            "component_recall": None,
        }
    predicted = sum(int(row["predicted_component_count"]) for row in rows)
    truth = sum(int(row["truth_component_count"]) for row in rows)
    predicted_hits = sum(int(row["hit_predicted_component_count"]) for row in rows)
    truth_hits = sum(int(row["hit_truth_component_count"]) for row in rows)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": predicted_hits,
        "hit_truth_component_count": truth_hits,
        "false_activation_component_count": predicted - predicted_hits,
        "component_precision": _safe_ratio(predicted_hits, predicted),
        "component_recall": _safe_ratio(truth_hits, truth),
    }


def _evaluate_masks(
    frame_data: list[dict[str, Any]],
    masks: list[np.ndarray],
) -> dict[str, Any]:
    pixel_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    by_session_pixels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_session_components: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame, mask in zip(frame_data, masks):
        pixel = pixel_metrics(mask, frame["residual_truth"])
        components = component_metrics(mask, frame["residual_truth"])
        pixel_rows.append(pixel)
        component_rows.append(components)
        by_session_pixels[frame["session_id"]].append(pixel)
        by_session_components[frame["session_id"]].append(components)
    sessions = {
        session_id: {
            "pixel": aggregate_confusion(rows),
            "component": _aggregate_components(by_session_components[session_id]),
        }
        for session_id, rows in sorted(by_session_pixels.items())
    }
    return {
        "pixel": aggregate_confusion(pixel_rows),
        "component": _aggregate_components(component_rows),
        "sessions": sessions,
    }


def _probe_comparison(
    probe: dict[str, Any],
    baseline: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    baseline_fp = int(baseline["pixel"]["fp"])
    probe_fp = int(probe["pixel"]["fp"])
    fp_reduction = _safe_ratio(baseline_fp - probe_fp, baseline_fp)
    recall_retention = _safe_ratio(
        float(probe["pixel"]["recall"] or 0.0),
        float(baseline["pixel"]["recall"] or 0.0),
    )
    session_retentions: dict[str, float | None] = {}
    for session_id, baseline_session in baseline["sessions"].items():
        baseline_recall = baseline_session["pixel"]["recall"]
        probe_recall = probe["sessions"][session_id]["pixel"]["recall"]
        session_retentions[session_id] = (
            _safe_ratio(float(probe_recall or 0.0), float(baseline_recall))
            if baseline_recall
            else None
        )
    comparable = [value for value in session_retentions.values() if value is not None]
    minimum_session_retention = min(comparable) if comparable else None

    status = "INSUFFICIENT"
    if (
        fp_reduction is not None
        and recall_retention is not None
        and minimum_session_retention is not None
        and fp_reduction >= float(rules["sufficient_min_fp_reduction"])
        and recall_retention >= float(rules["sufficient_min_recall_retention"])
        and minimum_session_retention >= float(rules["sufficient_min_session_recall_retention"])
    ):
        status = "SUFFICIENT"
    elif (
        fp_reduction is not None
        and recall_retention is not None
        and fp_reduction >= float(rules["partial_min_fp_reduction"])
        and recall_retention >= float(rules["partial_min_recall_retention"])
    ):
        status = "PARTIAL"
    return {
        "false_positive_reduction": fp_reduction,
        "recall_retention": recall_retention,
        "minimum_session_recall_retention": minimum_session_retention,
        "session_recall_retention": session_retentions,
        "decision": status,
    }


def _pareto_probe_ids(probes: list[dict[str, Any]]) -> list[str]:
    candidates = [
        probe
        for probe in probes
        if probe["comparison"]["false_positive_reduction"] is not None
        and probe["comparison"]["recall_retention"] is not None
    ]
    frontier: list[str] = []
    for probe in candidates:
        fp_reduction = probe["comparison"]["false_positive_reduction"]
        recall_retention = probe["comparison"]["recall_retention"]
        dominated = any(
            other["probe_id"] != probe["probe_id"]
            and other["comparison"]["false_positive_reduction"] >= fp_reduction
            and other["comparison"]["recall_retention"] >= recall_retention
            and (
                other["comparison"]["false_positive_reduction"] > fp_reduction
                or other["comparison"]["recall_retention"] > recall_retention
            )
            for other in candidates
        )
        if not dominated:
            frontier.append(probe["probe_id"])
    return sorted(frontier)


def _public_component(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def _input_provenance(
    repo_root: Path,
    paths: Sequence[Path],
) -> dict[str, str] | list[dict[str, str]]:
    rows = [
        {
            "path": str(path.relative_to(repo_root)),
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    return rows[0] if len(rows) == 1 else rows


def _validate_input_contract(
    *,
    config: dict[str, Any],
    frame_rows: list[dict[str, Any]],
    component_rows: list[dict[str, Any]],
) -> tuple[set[str], dict[str, int]]:
    contract = config.get("input_contract")
    if contract is None:
        if len(frame_rows) != 200:
            raise ValueError(f"pilot requires exactly 200 frame rows, got {len(frame_rows)}")
        if len(component_rows) != 5043:
            raise ValueError(
                f"pilot requires exactly 5043 component rows, got {len(component_rows)}"
            )
        expected_roles = {FRAME_ROLE}
        expected_sessions: dict[str, int] = {}
    else:
        expected_roles = {str(value) for value in contract["allowed_roles"]}
        expected_frames = int(contract["expected_frame_count"])
        if len(frame_rows) != expected_frames:
            raise ValueError(
                f"input contract requires exactly {expected_frames} frame rows, "
                f"got {len(frame_rows)}"
            )
        expected_sessions = {
            str(session_id): int(count)
            for session_id, count in contract["expected_session_frame_counts"].items()
        }
        observed_sessions = Counter(str(row.get("session_id")) for row in frame_rows)
        if observed_sessions != Counter(expected_sessions):
            raise ValueError(
                "frame session/count contract mismatch: "
                f"expected={dict(sorted(expected_sessions.items()))}, "
                f"observed={dict(sorted(observed_sessions.items()))}"
            )
    if any(row.get("rehearsal_role") not in expected_roles for row in frame_rows):
        raise ValueError("frame input contains a role outside the input contract")
    if any(row.get("rehearsal_role") not in expected_roles for row in component_rows):
        raise ValueError("component input contains a role outside the input contract")
    view_row_ids = [str(row.get("view_row_id")) for row in frame_rows]
    if len(view_row_ids) != len(set(view_row_ids)):
        raise ValueError("frame input contains duplicate view_row_id")
    source_frame_keys = [
        (str(row.get("source_id")), int(row.get("frame_id"))) for row in frame_rows
    ]
    if len(source_frame_keys) != len(set(source_frame_keys)):
        raise ValueError("frame input contains duplicate source_id/frame_id")
    return expected_roles, expected_sessions


def _read_bound_json(
    *,
    repo_root: Path,
    binding: dict[str, str],
) -> tuple[Path, dict[str, Any]]:
    path = _resolve_input(repo_root, binding["path"])
    observed = sha256_file(path)
    if observed != binding["sha256"]:
        raise ValueError(
            f"bound reference hash mismatch for {binding['path']}: {observed}"
        )
    return path, read_json(path)


def _frame_case_metrics(
    frame: dict[str, Any],
    baseline_mask: np.ndarray,
    probe_mask: np.ndarray,
) -> dict[str, Any]:
    truth = frame["residual_truth"]
    baseline_tp = int(np.count_nonzero(baseline_mask & truth))
    baseline_fp = int(np.count_nonzero(baseline_mask & ~truth))
    probe_tp = int(np.count_nonzero(probe_mask & truth))
    probe_fp = int(np.count_nonzero(probe_mask & ~truth))
    return {
        "view_row_id": frame["view_row_id"],
        "session_id": frame["session_id"],
        "role": frame["role"],
        "scene_bucket": frame["scene_bucket"],
        "frame_id": frame["frame_id"],
        "baseline_true_positive_pixels": baseline_tp,
        "baseline_false_positive_pixels": baseline_fp,
        "probe_true_positive_pixels": probe_tp,
        "probe_false_positive_pixels": probe_fp,
        "true_positive_pixels_lost": baseline_tp - probe_tp,
        "false_positive_pixels_removed": baseline_fp - probe_fp,
        "recall_retention": _safe_ratio(probe_tp, baseline_tp),
        "false_positive_reduction": _safe_ratio(baseline_fp - probe_fp, baseline_fp),
    }


def select_gate_cases(
    *,
    frame_data: list[dict[str, Any]],
    baseline_masks: list[np.ndarray],
    probe_masks: list[np.ndarray],
    success_minimum_recall_retention: float,
) -> dict[str, dict[str, Any] | None]:
    rows = [
        _frame_case_metrics(frame, baseline, probe)
        for frame, baseline, probe in zip(frame_data, baseline_masks, probe_masks)
    ]
    success_candidates = [
        row
        for row in rows
        if row["baseline_true_positive_pixels"] > 0
        and row["false_positive_pixels_removed"] > 0
        and row["recall_retention"] is not None
        and row["recall_retention"] >= success_minimum_recall_retention
    ]
    failure_candidates = [
        row for row in rows if row["baseline_true_positive_pixels"] > 0
    ]
    success = (
        sorted(
            success_candidates,
            key=lambda row: (
                -row["false_positive_pixels_removed"],
                -row["recall_retention"],
                row["view_row_id"],
            ),
        )[0]
        if success_candidates
        else None
    )
    failure = (
        sorted(
            failure_candidates,
            key=lambda row: (
                row["recall_retention"],
                -row["true_positive_pixels_lost"],
                row["view_row_id"],
            ),
        )[0]
        if failure_candidates
        else None
    )
    return {"success": success, "failure": failure}


def _tinted_panel(
    source: Image.Image,
    layers: Sequence[tuple[np.ndarray, tuple[int, int, int]]],
) -> Image.Image:
    array = np.asarray(source.convert("RGB"), dtype=np.float32).copy()
    for mask, color in layers:
        value = np.asarray(mask, dtype=bool)
        array[value] = array[value] * 0.42 + np.asarray(color, dtype=np.float32) * 0.58
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _render_gate_case(
    *,
    repo_root: Path,
    frame: dict[str, Any],
    baseline_mask: np.ndarray,
    probe_mask: np.ndarray,
    probe_id: str,
    case_kind: str,
    metrics: dict[str, Any],
    output_path: Path,
) -> None:
    with Image.open(repo_root / frame["image_repo_relative_path"]) as handle:
        source = handle.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    truth = frame["residual_truth"]
    baseline_fp = baseline_mask & ~truth
    kept_tp = probe_mask & truth
    kept_fp = probe_mask & ~truth
    rejected = baseline_mask & ~probe_mask
    panels = [
        ("source", source),
        ("residual truth / baseline FP", _tinted_panel(source, [(truth, (0, 210, 70)), (baseline_fp, (235, 45, 45))])),
        ("gate kept TP / FP", _tinted_panel(source, [(kept_tp, (0, 210, 70)), (kept_fp, (235, 45, 45))])),
        ("gate rejected", _tinted_panel(source, [(rejected, (255, 165, 0))])),
    ]
    canvas = Image.new("RGB", (512, 580), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 6), "DEVELOPMENT DIAGNOSTIC ONLY", fill=(255, 220, 70))
    draw.text((8, 22), f"{probe_id} / {case_kind.upper()}", fill=(255, 255, 255))
    draw.text(
        (8, 38),
        (
            f"session={frame['session_id']} frame={frame['frame_id']} "
            f"retention={metrics['recall_retention']} "
            f"fp_removed={metrics['false_positive_pixels_removed']}"
        ),
        fill=(210, 210, 210),
    )
    for index, (label, panel) in enumerate(panels):
        x = (index % 2) * 256
        y = 68 + (index // 2) * 256
        canvas.paste(panel, (x, y))
        draw.rectangle((x, y, x + 255, y + 17), fill=(0, 0, 0))
        draw.text((x + 4, y + 3), label, fill=(255, 255, 255))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG", optimize=True)


def _resolve_input(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _verify_output_scope(repo_root: Path, output_root: Path) -> None:
    allowed_root = (repo_root / "artifacts.local").resolve()
    try:
        output_root.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("output-root must be under repo-root/artifacts.local") from exc
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_root}")


def run_atlas(
    *,
    repo_root: Path,
    config_path: Path,
    frames_path: Path | Sequence[Path],
    components_path: Path | Sequence[Path],
    view_root: Path,
    yolo_trace_path: Path | Sequence[Path],
    output_root: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol_id")
    if config.get("stage") != "DEVELOPMENT_STANDARD":
        raise ValueError("R0 only accepts DEVELOPMENT_STANDARD")
    shape = tuple(int(value) for value in config["analysis_shape"])
    if len(shape) != 2:
        raise ValueError("analysis_shape must contain height and width")
    _verify_output_scope(repo_root, output_root)

    frames_paths = _as_paths(frames_path)
    components_paths = _as_paths(components_path)
    yolo_trace_paths = _as_paths(yolo_trace_path)
    frame_rows = _read_jsonls(frames_paths)
    component_rows = _read_jsonls(components_paths)
    manifest_rows = read_jsonl(view_root / "manifest.jsonl")
    yolo_rows = _read_jsonls(yolo_trace_paths)
    expected_roles, expected_session_counts = _validate_input_contract(
        config=config,
        frame_rows=frame_rows,
        component_rows=component_rows,
    )

    manifest_by_id = {
        row["id"]: row for row in manifest_rows if row.get("role") in expected_roles
    }
    if expected_session_counts:
        expected_manifest_rows = [
            row
            for row in manifest_rows
            if row.get("role") in expected_roles
            and row.get("session_id") in expected_session_counts
        ]
        manifest_session_counts = Counter(
            str(row["session_id"]) for row in expected_manifest_rows
        )
        if manifest_session_counts != Counter(expected_session_counts):
            raise ValueError(
                "input contract session counts do not exhaust the canonical manifest"
            )
        expected_view_row_ids = {str(row["id"]) for row in expected_manifest_rows}
        observed_view_row_ids = {str(row["view_row_id"]) for row in frame_rows}
        if observed_view_row_ids != expected_view_row_ids:
            missing = sorted(expected_view_row_ids - observed_view_row_ids)
            extra = sorted(observed_view_row_ids - expected_view_row_ids)
            raise ValueError(
                "frame membership does not match the frozen canonical session set: "
                f"missing={missing[:3]}, extra={extra[:3]}"
            )
    yolo_by_key = {
        (row["source_id"], int(row["frame_id"])): row for row in yolo_rows
    }
    if len(yolo_by_key) != len(yolo_rows):
        raise ValueError("YOLO trace input contains duplicate source_id/frame_id")
    ledger_by_id = {row["component_id"]: row for row in component_rows}
    if len(ledger_by_id) != len(component_rows):
        raise ValueError("duplicate component_id in component ledger")

    truth_names = {int(key): value for key, value in config["truth_classes"].items()}
    truth_ids_by_name = {value: key for key, value in truth_names.items()}
    hazard_ids = np.asarray(config["hazard_truth_ids"], dtype=np.uint8)
    rules = config["component_rules"]
    spatial_masks = {
        name: spatial_probe_mask(shape, name, config["spatial_bands"])
        for name in config["gating_probes"]["spatial"]
    }

    frame_data: list[dict[str, Any]] = []
    atlas_components: list[dict[str, Any]] = []
    observed_component_ids: set[str] = set()
    for frame_row in frame_rows:
        view_row_id = frame_row["view_row_id"]
        manifest = manifest_by_id.get(view_row_id)
        if manifest is None:
            raise ValueError(f"missing consumed canonical manifest row: {view_row_id}")
        for field in ("source_id", "session_id", "frame_id", "image_sha256", "canonical_mask_sha256"):
            if frame_row[field] != manifest[field]:
                raise ValueError(f"frame/manifest mismatch for {field}: {view_row_id}")
        if frame_row["rehearsal_role"] != manifest["role"]:
            raise ValueError(f"frame/manifest role mismatch: {view_row_id}")
        yolo = yolo_by_key.get((frame_row["source_id"], int(frame_row["frame_id"])))
        if yolo is None or yolo.get("image_sha256") != frame_row["image_sha256"]:
            raise ValueError(f"missing or mismatched frozen YOLO row: {view_row_id}")
        if tuple(frame_row["packed_masks"]["shape"]) != shape:
            raise ValueError(f"analysis shape mismatch: {view_row_id}")

        truth_path = view_root / manifest["canonical_mask_path"]
        if sha256_file(truth_path) != frame_row["canonical_mask_sha256"]:
            raise ValueError(f"canonical truth hash mismatch: {view_row_id}")
        truth = np.asarray(Image.open(truth_path), dtype=np.uint8)
        if truth.shape != shape:
            raise ValueError(f"canonical truth shape mismatch: {view_row_id}")
        full_truth_hazard = np.isin(truth, hazard_ids)
        a_mask = decode_packed_mask(frame_row["packed_masks"]["A"], shape)
        b_mask = decode_packed_mask(frame_row["packed_masks"]["B"], shape)
        class_masks = {
            class_name: decode_packed_mask(
                frame_row["packed_masks"][f"candidate_{class_name}"], shape
            )
            for class_name in config["candidate_classes"]
        }
        if not np.array_equal(b_mask, np.logical_or.reduce(list(class_masks.values()))):
            raise ValueError(f"candidate class union mismatch: {view_row_id}")
        residual_truth = full_truth_hazard & ~a_mask
        frame_components: list[dict[str, Any]] = []
        confidence_gate = np.zeros(shape, dtype=bool)

        for class_name, class_mask in class_masks.items():
            same_class_residual_truth = (truth == truth_ids_by_name[class_name]) & ~a_mask
            boundary_neighborhood = dilate_mask(
                same_class_residual_truth,
                int(rules["boundary_probe_radius_pixels"]),
            )
            for component in connected_components(
                class_mask, connectivity=int(rules["connectivity"])
            ):
                component_id = (
                    f"{frame_row['source_id']}:{int(frame_row['frame_id'])}:"
                    f"{class_name}:{component.index}"
                )
                ledger = ledger_by_id.get(component_id)
                if ledger is None:
                    raise ValueError(f"component missing from ledger: {component_id}")
                if (
                    int(ledger["area_pixels"]) != component.area
                    or list(ledger["bbox_xyxy"]) != list(component.bbox)
                    or ledger["class_name"] != class_name
                ):
                    raise ValueError(f"component ledger geometry mismatch: {component_id}")
                residual_intersection = int(
                    np.count_nonzero(component.mask & same_class_residual_truth)
                )
                if residual_intersection != int(ledger["truth_intersection_pixels"]):
                    raise ValueError(f"component ledger truth mismatch: {component_id}")
                observed_component_ids.add(component_id)
                histogram = _truth_histogram(truth, component.mask, truth_names)
                y_indices, x_indices = np.nonzero(component.mask)
                full_truth_intersection = int(np.count_nonzero(component.mask & full_truth_hazard))
                any_residual_intersection = int(
                    np.count_nonzero(component.mask & residual_truth)
                )
                yolo_overlapped_truth_intersection = int(
                    np.count_nonzero(component.mask & full_truth_hazard & a_mask)
                )
                record: dict[str, Any] = {
                    "schema_version": "blindassist.dual_loop_segmentation_failure_atlas.component.v1",
                    "protocol_id": PROTOCOL_ID,
                    "evidence_instance": config["evidence_instance"],
                    "view_row_id": view_row_id,
                    "source_id": frame_row["source_id"],
                    "session_id": frame_row["session_id"],
                    "sequence_id": manifest["sequence_id"],
                    "frame_id": int(frame_row["frame_id"]),
                    "source_capture_timestamp_ns": int(manifest["source_capture_timestamp_ns"]),
                    "component_id": component_id,
                    "component_index": int(component.index),
                    "predicted_class": class_name,
                    "area_pixels": int(component.area),
                    "bbox_xyxy": list(component.bbox),
                    "centroid_xy": [float(np.mean(x_indices)), float(np.mean(y_indices))],
                    "top1_confidence_median": ledger["top1_confidence_median"],
                    "top1_top2_margin_median": ledger["top1_top2_margin_median"],
                    "residual_truth_intersection_pixels": residual_intersection,
                    "any_class_residual_truth_intersection_pixels": any_residual_intersection,
                    "full_truth_hazard_intersection_pixels": full_truth_intersection,
                    "yolo_overlapped_truth_hazard_intersection_pixels": (
                        yolo_overlapped_truth_intersection
                    ),
                    "false_activation": residual_intersection == 0,
                    "truth_class_histogram": histogram,
                    "dominant_truth_class": _dominant_truth_class(histogram),
                    "spatial_bands": _spatial_membership(component.mask, spatial_masks),
                    "boundary_proximity_fraction": float(
                        np.count_nonzero(component.mask & boundary_neighborhood)
                        / component.area
                    ),
                    "nearest_yolo_box_distance_pixels": ledger[
                        "nearest_yolo_box_distance_pixels"
                    ],
                    "yolo_attribution_state": (
                        "ATTRIBUTION_UNCERTAIN_TRUTH_OVERLAP"
                        if yolo_overlapped_truth_intersection > 0
                        else "ATTRIBUTION_UNCERTAIN_NEAR_BOX"
                        if (
                            ledger["nearest_yolo_box_distance_pixels"] is not None
                            and ledger["nearest_yolo_box_distance_pixels"]
                            <= float(rules["yolo_attribution_gap_pixels"])
                        )
                        else "NO_YOLO_OVERLAPPED_OR_NEARBY_TRUTH_HAZARD"
                    ),
                    "previous_observation_iou": None,
                    "next_observation_iou": None,
                    "temporal_track_id": None,
                    "persistence_observations": None,
                    "false_activation_run_observations": None,
                    "depth_available": False,
                    "pose_available": False,
                    "distant_background_activation_status": "NOT_EVALUABLE_NO_DEPTH",
                    "texture_or_shadow_confusion_status": "NOT_EVALUABLE_NO_APPEARANCE_LABEL",
                    "_mask": component.mask,
                }
                if (
                    ledger["top1_confidence_median"] is not None
                    and ledger["top1_confidence_median"]
                    >= float(rules["high_confidence_minimum"])
                ):
                    confidence_gate |= component.mask
                frame_components.append(record)
                atlas_components.append(record)

        frame_data.append(
            {
                "view_row_id": view_row_id,
                "source_id": frame_row["source_id"],
                "session_id": frame_row["session_id"],
                "role": manifest["role"],
                "sequence_id": manifest["sequence_id"],
                "scene_bucket": manifest["scene_bucket"],
                "image_repo_relative_path": manifest["image_repo_relative_path"],
                "frame_id": int(frame_row["frame_id"]),
                "source_capture_timestamp_ns": int(manifest["source_capture_timestamp_ns"]),
                "a_mask": a_mask,
                "b_mask": b_mask,
                "residual_truth": residual_truth,
                "full_truth_hazard": full_truth_hazard,
                "confidence_gate": confidence_gate,
                "components": frame_components,
            }
        )
    if observed_component_ids != set(ledger_by_id):
        missing = sorted(set(ledger_by_id) - observed_component_ids)
        raise ValueError(f"unmatched component ledger rows: {missing[:3]}")

    frame_data.sort(
        key=lambda row: (
            row["sequence_id"],
            row["source_capture_timestamp_ns"],
            row["frame_id"],
        )
    )
    assign_temporal_tracks(
        atlas_components,
        frame_data,
        float(rules["temporal_match_iou"]),
    )
    for record in atlas_components:
        tags = _mechanism_tags(record, rules)
        record["mechanism_tags"] = tags
        record["primary_mechanism"] = primary_mechanism(tags)

    baseline_masks = [frame["b_mask"] for frame in frame_data]
    baseline = _evaluate_masks(frame_data, baseline_masks)
    probes: list[dict[str, Any]] = []
    probe_masks_by_id: dict[str, list[np.ndarray]] = {}
    for name in config["gating_probes"]["spatial"]:
        masks = [frame["b_mask"] & spatial_masks[name] for frame in frame_data]
        probe_masks_by_id[f"SPATIAL:{name}"] = masks
        evaluation = _evaluate_masks(frame_data, masks)
        probes.append(
            {
                "probe_id": f"SPATIAL:{name}",
                "family": "SPATIAL",
                "name": name,
                **evaluation,
                "comparison": _probe_comparison(
                    evaluation, baseline, config["gating_decision_rules"]
                ),
            }
        )

    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in frame_data:
        by_sequence[frame["sequence_id"]].append(frame)
    temporal_masks: dict[str, dict[str, np.ndarray]] = {
        name: {} for name in config["gating_probes"]["temporal"]
    }
    for sequence_frames in by_sequence.values():
        for index, frame in enumerate(sequence_frames):
            previous = sequence_frames[index - 1]["b_mask"] if index >= 1 else None
            previous_previous = sequence_frames[index - 2]["b_mask"] if index >= 2 else None
            for name in config["gating_probes"]["temporal"]:
                temporal_masks[name][frame["view_row_id"]] = causal_temporal_probe(
                    frame["b_mask"], previous, previous_previous, name
                )
    for name in config["gating_probes"]["temporal"]:
        masks = [temporal_masks[name][frame["view_row_id"]] for frame in frame_data]
        probe_masks_by_id[f"TEMPORAL:{name}"] = masks
        evaluation = _evaluate_masks(frame_data, masks)
        probes.append(
            {
                "probe_id": f"TEMPORAL:{name}",
                "family": "TEMPORAL",
                "name": name,
                **evaluation,
                "comparison": _probe_comparison(
                    evaluation, baseline, config["gating_decision_rules"]
                ),
            }
        )

    for name in config["gating_probes"]["confidence"]:
        if name == "ALL_COMPONENTS":
            masks = baseline_masks
        elif name == "COMPONENT_MEDIAN_CONFIDENCE_GE_0_65":
            masks = [frame["confidence_gate"] for frame in frame_data]
        else:
            raise ValueError(f"unknown confidence probe: {name}")
        probe_masks_by_id[f"CONFIDENCE:{name}"] = masks
        evaluation = _evaluate_masks(frame_data, masks)
        probes.append(
            {
                "probe_id": f"CONFIDENCE:{name}",
                "family": "CONFIDENCE",
                "name": name,
                **evaluation,
                "comparison": _probe_comparison(
                    evaluation, baseline, config["gating_decision_rules"]
                ),
            }
        )

    non_baseline = [
        probe
        for probe in probes
        if probe["name"] not in {"FULL_FRAME", "CURRENT_FRAME", "ALL_COMPONENTS"}
    ]
    decisions = Counter(probe["comparison"]["decision"] for probe in non_baseline)
    gating_axis = (
        "SUFFICIENT"
        if decisions["SUFFICIENT"]
        else "PARTIAL"
        if decisions["PARTIAL"]
        else "INSUFFICIENT"
    )
    gating_result = {
        "schema_version": "blindassist.dual_loop_segmentation_failure_atlas.gating.v1",
        "protocol_id": PROTOCOL_ID,
        "baseline": baseline,
        "probes": probes,
        "pareto_probe_ids": _pareto_probe_ids(non_baseline),
        "gating_axis": gating_axis,
        "interpretation": (
            "Predeclared independent probes only; no Cartesian search and no same-data best-gate selection."
        ),
    }

    total_hazard_pixels = sum(
        int(np.count_nonzero(frame["full_truth_hazard"])) for frame in frame_data
    )
    residual_pixels = sum(
        int(np.count_nonzero(frame["residual_truth"])) for frame in frame_data
    )
    uncertain_pixels = sum(
        int(np.count_nonzero(frame["full_truth_hazard"] & frame["a_mask"]))
        for frame in frame_data
    )
    residual_result = {
        "schema_version": "blindassist.dual_loop_segmentation_failure_atlas.residual.v1",
        "protocol_id": PROTOCOL_ID,
        "residual_labelability_axis": "WEAKLY_LABELABLE",
        "pixel_level_residual": {
            "status": "LABELABLE",
            "definition": "canonical_hazard AND NOT frozen_yolo_box_union",
            "pixels": residual_pixels,
            "share_of_truth_hazard": _safe_ratio(residual_pixels, total_hazard_pixels),
        },
        "three_state_attribution": {
            "A_EFFECTIVELY_COVERED": {
                "pixels": None,
                "status": "NOT_EVALUABLE_NO_INSTANCE_CORRESPONDENCE",
            },
            "RESIDUAL_HAZARD": {
                "pixels": residual_pixels,
                "status": "PIXEL_PROXY_AVAILABLE",
            },
            "ATTRIBUTION_UNCERTAIN": {
                "pixels": uncertain_pixels,
                "status": "YOLO_BOX_OVERLAP_WITH_CANONICAL_HAZARD",
            },
            "exhaustive_partition_status": "NOT_EVALUABLE",
        },
        "total_canonical_hazard_pixels": total_hazard_pixels,
        "depth_available": False,
        "pose_available": False,
        "instance_correspondence_available": False,
        "rationale": (
            "The pixel residual is reproducible, but YOLO-overlapped semantic hazard cannot be promoted "
            "to effective instance coverage."
        ),
    }

    false_components = [
        record for record in atlas_components if record["false_activation"]
    ]
    false_area_total = sum(int(record["area_pixels"]) for record in false_components)
    mechanism_stats: dict[str, dict[str, Any]] = {}
    for mechanism in (*_ACTIONABLE_MECHANISMS, "OTHER_FALSE_ACTIVATION"):
        matching = [
            record
            for record in false_components
            if mechanism in record["mechanism_tags"]
        ]
        area = sum(int(record["area_pixels"]) for record in matching)
        mechanism_stats[mechanism] = {
            "component_count": len(matching),
            "false_area_pixels_nonexclusive": area,
            "false_area_share_nonexclusive": _safe_ratio(area, false_area_total),
            "session_count": len({record["session_id"] for record in matching}),
        }
    expansion_rule = config["expansion_rule"]
    qualifying = [
        mechanism
        for mechanism in _ACTIONABLE_MECHANISMS
        if (mechanism_stats[mechanism]["false_area_share_nonexclusive"] or 0.0)
        >= float(expansion_rule["minimum_false_area_share"])
        and mechanism_stats[mechanism]["session_count"]
        >= int(expansion_rule["minimum_session_count"])
    ]

    allowed_expansion_rows = [
        row
        for row in manifest_rows
        if row.get("role") in {"dev", "consumed_old_blind"}
    ]
    candidate_sessions: dict[str, dict[str, Any]] = {}
    for row in allowed_expansion_rows:
        session = candidate_sessions.setdefault(
            row["session_id"],
            {
                "session_id": row["session_id"],
                "role": row["role"],
                "scene_buckets": set(),
                "available_frame_count": 0,
            },
        )
        session["scene_buckets"].add(row["scene_bucket"])
        session["available_frame_count"] += 1
    ranked_sessions = sorted(
        (
            {
                **session,
                "scene_buckets": sorted(session["scene_buckets"]),
            }
            for session in candidate_sessions.values()
        ),
        key=lambda row: (
            row["role"] != "dev",
            -len(row["scene_buckets"]),
            -row["available_frame_count"],
            row["session_id"],
        ),
    )
    selected_sessions = (
        ranked_sessions[: int(expansion_rule["maximum_target_sessions"])]
        if qualifying
        else []
    )
    expansion = {
        "decision": "TARGETED_EXPANSION_WARRANTED" if qualifying else "STOP_NO_INFORMATION_GAIN",
        "qualifying_mechanisms": qualifying,
        "selection_status": (
            "CANDIDATE_SESSION_LIST_ONLY_NO_INFERENCE_EXECUTED"
            if qualifying
            else "NO_EXPANSION"
        ),
        "selected_candidate_sessions": selected_sessions,
        "excluded_roles": ["train", "synthetic_canary", "r1_consumed_fresh"],
        "maximum_target_sessions": int(expansion_rule["maximum_target_sessions"]),
    }

    frame_summaries: list[dict[str, Any]] = []
    for frame in frame_data:
        frame_false = [
            record for record in frame["components"] if record["false_activation"]
        ]
        frame_summaries.append(
            {
                "schema_version": "blindassist.dual_loop_segmentation_failure_atlas.frame.v1",
                "protocol_id": PROTOCOL_ID,
                "view_row_id": frame["view_row_id"],
                "source_id": frame["source_id"],
                "session_id": frame["session_id"],
                "sequence_id": frame["sequence_id"],
                "frame_id": frame["frame_id"],
                "source_capture_timestamp_ns": frame["source_capture_timestamp_ns"],
                "candidate_pixels": int(np.count_nonzero(frame["b_mask"])),
                "residual_truth_pixels": int(np.count_nonzero(frame["residual_truth"])),
                "false_positive_pixels": int(
                    np.count_nonzero(frame["b_mask"] & ~frame["residual_truth"])
                ),
                "component_count": len(frame["components"]),
                "false_activation_component_count": len(frame_false),
                "false_activation_area_by_primary_mechanism": dict(
                    sorted(
                        Counter(
                            {
                                mechanism: sum(
                                    int(record["area_pixels"])
                                    for record in frame_false
                                    if record["primary_mechanism"] == mechanism
                                )
                                for mechanism in {
                                    record["primary_mechanism"] for record in frame_false
                                }
                            }
                        ).items()
                    )
                ),
            }
        )

    session_summaries: dict[str, dict[str, Any]] = {}
    for session_id in sorted({frame["session_id"] for frame in frame_data}):
        session_frames = [
            row for row in frame_summaries if row["session_id"] == session_id
        ]
        session_components = [
            row for row in atlas_components if row["session_id"] == session_id
        ]
        false_session_components = [
            row for row in session_components if row["false_activation"]
        ]
        session_summaries[session_id] = {
            "role": next(
                frame["role"] for frame in frame_data if frame["session_id"] == session_id
            ),
            "scene_buckets": sorted(
                {
                    frame["scene_bucket"]
                    for frame in frame_data
                    if frame["session_id"] == session_id
                }
            ),
            "frame_count": len(session_frames),
            "component_count": len(session_components),
            "false_activation_component_count": len(false_session_components),
            "false_activation_area_pixels": sum(
                int(row["area_pixels"]) for row in false_session_components
            ),
            "false_positive_pixels": sum(
                int(row["false_positive_pixels"]) for row in session_frames
            ),
            "false_positive_area_fraction": _safe_ratio(
                sum(int(row["false_positive_pixels"]) for row in session_frames),
                len(session_frames) * int(np.prod(shape)),
            ),
            "primary_mechanism_component_counts": dict(
                sorted(
                    Counter(
                        row["primary_mechanism"] for row in false_session_components
                    ).items()
                )
            ),
            "mechanism_false_area_pixels_nonexclusive": {
                mechanism: sum(
                    int(row["area_pixels"])
                    for row in false_session_components
                    if mechanism in row["mechanism_tags"]
                )
                for mechanism in (*_ACTIONABLE_MECHANISMS, "OTHER_FALSE_ACTIVATION")
            },
        }

    analysis_round = str(config.get("analysis_round", "PILOT_200"))
    replication: dict[str, Any] | None = None
    gate_cases: dict[str, Any] | None = None
    if analysis_round == "TARGETED_EXPANSION_320":
        references = config["pilot_reference"]
        pilot_result_path, pilot_result = _read_bound_json(
            repo_root=repo_root,
            binding=references["result"],
        )
        pilot_gating_path, pilot_gating = _read_bound_json(
            repo_root=repo_root,
            binding=references["gating"],
        )
        pilot_residual_path, pilot_residual = _read_bound_json(
            repo_root=repo_root,
            binding=references["residual"],
        )
        replication_rules = config["replication_decision_rules"]
        mechanisms = list(replication_rules["mechanisms"])
        if mechanisms != list(_EXPANSION_MECHANISMS):
            raise ValueError("expansion mechanism list must match the frozen five")
        pilot_profile = [
            float(
                pilot_result["false_activation"]["mechanisms"][mechanism][
                    "false_area_share_nonexclusive"
                ]
                or 0.0
            )
            for mechanism in mechanisms
        ]
        expansion_profile = [
            float(mechanism_stats[mechanism]["false_area_share_nonexclusive"] or 0.0)
            for mechanism in mechanisms
        ]
        aggregate_rank_correlation = spearman_rank_correlation(
            pilot_profile,
            expansion_profile,
        )
        mechanism_coverage: dict[str, dict[str, Any]] = {}
        for mechanism in mechanisms:
            matching = [
                row
                for row in false_components
                if mechanism in row["mechanism_tags"]
            ]
            sessions = sorted({row["session_id"] for row in matching})
            roles = sorted(
                {
                    frame["role"]
                    for frame in frame_data
                    if frame["session_id"] in sessions
                }
            )
            mechanism_coverage[mechanism] = {
                "component_count": len(matching),
                "false_area_pixels_nonexclusive": sum(
                    int(row["area_pixels"]) for row in matching
                ),
                "false_area_share_nonexclusive": mechanism_stats[mechanism][
                    "false_area_share_nonexclusive"
                ],
                "session_count": len(sessions),
                "session_coverage_rate": _safe_ratio(
                    len(sessions), len(session_summaries)
                ),
                "sessions": sessions,
                "role_count": len(roles),
                "roles": roles,
            }
        minimum_session_count = int(
            replication_rules["minimum_session_count_per_mechanism"]
        )
        minimum_role_count = int(
            replication_rules["minimum_role_count_per_mechanism"]
        )
        mechanisms_reproduced = all(
            row["session_count"] >= minimum_session_count
            and row["role_count"] >= minimum_role_count
            for row in mechanism_coverage.values()
        )
        session_rank_correlations: dict[str, float | None] = {}
        for session_id, summary in session_summaries.items():
            denominator = int(summary["false_activation_area_pixels"])
            session_profile = [
                float(
                    _safe_ratio(
                        summary["mechanism_false_area_pixels_nonexclusive"][mechanism],
                        denominator,
                    )
                    or 0.0
                )
                for mechanism in mechanisms
            ]
            session_rank_correlations[session_id] = spearman_rank_correlation(
                pilot_profile,
                session_profile,
            )
        divergence_threshold = float(
            replication_rules[
                "source_dependent_session_rank_correlation_below"
            ]
        )
        divergent_sessions = sorted(
            session_id
            for session_id, correlation in session_rank_correlations.items()
            if correlation is None or correlation < divergence_threshold
        )
        source_dependent = len(divergent_sessions) >= int(
            replication_rules["source_dependent_minimum_session_count"]
        )
        minimum_aggregate_correlation = float(
            replication_rules["minimum_aggregate_rank_correlation"]
        )
        decision_tree = config["decision_tree"]
        expected_branch_order = [
            "mechanism_not_reproduced",
            "gating_sufficient",
            "gating_partial",
            "source_dependent",
            "replicated",
        ]
        if decision_tree["branch_order"] != expected_branch_order:
            raise ValueError(
                "expansion decision_tree branch order does not match the frozen evaluator"
            )
        decision_terminals = {
            key: str(decision_tree["terminals"][key])
            for key in expected_branch_order
        }
        decision = expansion_decision(
            mechanisms_reproduced=mechanisms_reproduced,
            aggregate_rank_correlation=aggregate_rank_correlation,
            minimum_rank_correlation=minimum_aggregate_correlation,
            gating_axis=gating_axis,
            source_dependent=source_dependent,
            terminals=decision_terminals,
        )
        pilot_residual_share = pilot_residual["pixel_level_residual"][
            "share_of_truth_hazard"
        ]
        expansion_residual_share = residual_result["pixel_level_residual"][
            "share_of_truth_hazard"
        ]
        residual_change = {
            "pilot_residual_labelability_axis": pilot_residual[
                "residual_labelability_axis"
            ],
            "expansion_residual_labelability_axis": residual_result[
                "residual_labelability_axis"
            ],
            "pilot_pixel_level_residual_status": pilot_residual[
                "pixel_level_residual"
            ]["status"],
            "expansion_pixel_level_residual_status": residual_result[
                "pixel_level_residual"
            ]["status"],
            "pilot_residual_share_of_truth_hazard": pilot_residual_share,
            "expansion_residual_share_of_truth_hazard": expansion_residual_share,
            "residual_share_delta": (
                float(expansion_residual_share) - float(pilot_residual_share)
                if pilot_residual_share is not None
                and expansion_residual_share is not None
                else None
            ),
        }
        residual_change["labelability_status_changed"] = (
            residual_change["pilot_residual_labelability_axis"]
            != residual_change["expansion_residual_labelability_axis"]
            or residual_change["pilot_pixel_level_residual_status"]
            != residual_change["expansion_pixel_level_residual_status"]
        )
        replication = {
            "schema_version": (
                "blindassist.dual_loop_segmentation_failure_atlas.expansion.v1"
            ),
            "decision": decision,
            "questions": {
                "mechanisms_reproduced": mechanisms_reproduced,
                "aggregate_ranking_stable": (
                    aggregate_rank_correlation is not None
                    and aggregate_rank_correlation >= minimum_aggregate_correlation
                ),
                "simple_gating_failure_reproduced": gating_axis == "INSUFFICIENT",
            },
            "mechanism_order": {
                "mechanisms": mechanisms,
                "pilot_false_area_shares": dict(zip(mechanisms, pilot_profile)),
                "expansion_false_area_shares": dict(
                    zip(mechanisms, expansion_profile)
                ),
                "pilot_ranked": sorted(
                    mechanisms,
                    key=lambda mechanism: (
                        -pilot_profile[mechanisms.index(mechanism)],
                        mechanism,
                    ),
                ),
                "expansion_ranked": sorted(
                    mechanisms,
                    key=lambda mechanism: (
                        -expansion_profile[mechanisms.index(mechanism)],
                        mechanism,
                    ),
                ),
                "spearman_rank_correlation": aggregate_rank_correlation,
                "minimum_required": minimum_aggregate_correlation,
            },
            "mechanism_session_coverage": mechanism_coverage,
            "session_rank_correlations_vs_pilot": session_rank_correlations,
            "source_dependence": {
                "threshold": divergence_threshold,
                "minimum_divergent_session_count": int(
                    replication_rules["source_dependent_minimum_session_count"]
                ),
                "divergent_sessions": divergent_sessions,
                "source_dependent": source_dependent,
            },
            "decision_tree": {
                "branch_order": expected_branch_order,
                "terminals": decision_terminals,
            },
            "gating": {
                "pilot_gating_axis": pilot_gating["gating_axis"],
                "expansion_gating_axis": gating_axis,
                "failure_reproduced": gating_axis == "INSUFFICIENT",
            },
            "residual_labelability_change": residual_change,
            "pilot_reference_paths": {
                "result": str(pilot_result_path.relative_to(repo_root)),
                "gating": str(pilot_gating_path.relative_to(repo_root)),
                "residual": str(pilot_residual_path.relative_to(repo_root)),
            },
        }
        expansion = {
            "decision": decision,
            "selection_status": "FIXED_320_FRAME_EXPANSION_COMPLETE",
            "selected_candidate_sessions": [],
            "excluded_roles": ["train", "synthetic_canary", "r1_consumed_fresh"],
            "maximum_target_sessions": len(session_summaries),
        }

        gate_cases = {}
        success_minimum = float(
            config["case_selection"]["success_minimum_recall_retention"]
        )
        for probe in non_baseline:
            probe_id = probe["probe_id"]
            gate_cases[probe_id] = select_gate_cases(
                frame_data=frame_data,
                baseline_masks=baseline_masks,
                probe_masks=probe_masks_by_id[probe_id],
                success_minimum_recall_retention=success_minimum,
            )

    result = {
        "schema_version": (
            "blindassist.dual_loop_segmentation_failure_atlas.expansion_result.v1"
            if analysis_round == "TARGETED_EXPANSION_320"
            else SCHEMA_VERSION
        ),
        "protocol_id": PROTOCOL_ID,
        "analysis_round": analysis_round,
        "stage": config["stage"],
        "evidence_instance": config["evidence_instance"],
        "claim_ceiling": config["claim_ceiling"],
        "input_counts": {
            "frames": len(frame_data),
            "components": len(atlas_components),
            "sessions": len(session_summaries),
        },
        "false_activation": {
            "component_count": len(false_components),
            "component_area_pixels": false_area_total,
            "mechanisms": mechanism_stats,
        },
        "gating_axis": gating_axis,
        "residual_labelability_axis": residual_result["residual_labelability_axis"],
        "expansion": expansion,
        "replication": replication,
        "terminals": {
            "atlas": (
                "TARGETED_EXPANSION_COMPLETE"
                if analysis_round == "TARGETED_EXPANSION_320"
                else "PILOT_COMPLETE"
            ),
            "gating": gating_axis,
            "residual": residual_result["residual_labelability_axis"],
            "decision": replication["decision"] if replication else None,
            "confirmation": "NOT_ACTIVATED",
            "product_or_safety": "NOT_EVALUABLE",
        },
        "provenance": {
            "config": {
                "path": str(config_path.relative_to(repo_root)),
                "sha256": sha256_file(config_path),
            },
            "frames": _input_provenance(repo_root, frames_paths),
            "components": _input_provenance(repo_root, components_paths),
            "canonical_manifest": {
                "path": str((view_root / "manifest.jsonl").relative_to(repo_root)),
                "sha256": sha256_file(view_root / "manifest.jsonl"),
            },
            "yolo_trace": _input_provenance(repo_root, yolo_trace_paths),
            "implementation": {
                "path": str(Path(__file__).resolve().relative_to(repo_root)),
                "sha256": sha256_file(Path(__file__).resolve()),
            },
        },
    }

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(
            temporary / "atlas_components.jsonl",
            (_public_component(record) for record in atlas_components),
        )
        _write_jsonl(temporary / "frame_summary.jsonl", frame_summaries)
        _write_json(
            temporary / "session_summary.json",
            {
                "schema_version": "blindassist.dual_loop_segmentation_failure_atlas.sessions.v1",
                "protocol_id": PROTOCOL_ID,
                "sessions": session_summaries,
            },
        )
        _write_json(temporary / "gating_probes.json", gating_result)
        _write_json(temporary / "residual_labelability.json", residual_result)
        if replication is not None:
            _write_json(temporary / "replication.json", replication)
        if gate_cases is not None:
            frame_index = {
                frame["view_row_id"]: index for index, frame in enumerate(frame_data)
            }
            for probe_id, cases in gate_cases.items():
                safe_probe_id = probe_id.lower().replace(":", "-").replace("_", "-")
                for case_kind, metrics in cases.items():
                    if metrics is None:
                        continue
                    index = frame_index[metrics["view_row_id"]]
                    relative_figure = (
                        Path("case_figures")
                        / f"{safe_probe_id}-{case_kind}.png"
                    )
                    _render_gate_case(
                        repo_root=repo_root,
                        frame=frame_data[index],
                        baseline_mask=baseline_masks[index],
                        probe_mask=probe_masks_by_id[probe_id][index],
                        probe_id=probe_id,
                        case_kind=case_kind,
                        metrics=metrics,
                        output_path=temporary / relative_figure,
                    )
                    metrics["figure_path"] = relative_figure.as_posix()
            _write_json(
                temporary / "gate_cases.json",
                {
                    "schema_version": (
                        "blindassist.dual_loop_segmentation_failure_atlas.gate_cases.v1"
                    ),
                    "protocol_id": PROTOCOL_ID,
                    "selection_rule": config["case_selection"],
                    "cases": gate_cases,
                },
            )
            result["case_figures"] = {
                "manifest": "gate_cases.json",
                "directory": "case_figures",
            }
        _write_json(temporary / "result.json", result)
        temporary.replace(output_root)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--frames", required=True, action="append")
    parser.add_argument("--components", required=True, action="append")
    parser.add_argument("--view-root", required=True)
    parser.add_argument("--yolo-trace", required=True, action="append")
    parser.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    result = run_atlas(
        repo_root=repo_root,
        config_path=_resolve_input(repo_root, args.config),
        frames_path=[_resolve_input(repo_root, value) for value in args.frames],
        components_path=[_resolve_input(repo_root, value) for value in args.components],
        view_root=_resolve_input(repo_root, args.view_root),
        yolo_trace_path=[
            _resolve_input(repo_root, value) for value in args.yolo_trace
        ],
        output_root=_resolve_input(repo_root, args.output_root),
    )
    json.dump(_json_ready(result), sys.stdout, ensure_ascii=False, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
