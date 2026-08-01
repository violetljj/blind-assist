"""Fail-closed, read-only quality checks for RGB/segmentation-mask pairs.

This module audits existing labels.  It never edits, replaces, or writes a
source RGB/mask/proposal file.  A model mask is an optional proposal sidecar;
disagreement with it is evidence for review, never a truth-label replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError


STATUS_PASS = "PASS"
STATUS_REVIEW = "REVIEW"
STATUS_INVALID = "INVALID"
STATUSES = {STATUS_PASS, STATUS_REVIEW, STATUS_INVALID}

CLASS_ORDER = (
    "walkable",
    "blocking_obstacle",
    "boundary_level_change",
    "unknown_nonwalkable",
)

DEFAULT_CONFIG: dict[str, Any] = {
    "schema_version": "blindassist.segmentation_mask_quality_audit.config.v1",
    "expected_label_space": "blindassist_riskseg_v1",
    "class_order": list(CLASS_ORDER),
    "class_ids": {
        "walkable": 0,
        "blocking_obstacle": 1,
        "boundary_level_change": 2,
        "unknown_nonwalkable": 3,
    },
    "allowed_ids": [0, 1, 2, 3],
    "void_ids": [255],
    "default_mask_decoder": "canonical",
    "source_to_expected_mapping": {"0": 0, "1": 1, "2": 2, "3": 3},
    "accepted_rgb_modes": ["RGB", "RGBA"],
    "accepted_mask_modes": ["L", "P", "RGB", "RGBA"],
    "shape_policy": "exact_rgb",
    "fixed_mask_shape": None,
    "source_unknown_ids": [3],
    "require_input_hashes": True,
    "require_visual_review": True,
    "require_temporal_adjacency": True,
    "heuristics": {
        "flag_constant_masks": True,
        "far_region_row_fraction": 0.45,
        "far_region_walkable_fraction": 0.98,
        "lower_ground_start_fraction": 0.55,
        "lower_ground_min_walkable_fraction": 0.10,
        "lower_ground_max_components": 12,
        "lower_ground_largest_component_fraction": 0.45,
        "proposal_disagreement_fraction": 0.30,
        "proposal_swap_margin": 0.10,
        "proposal_class_min_pixels": 8,
        "boundary_iou_review_floor": 0.25,
        "unknown_walkable_conflict_fraction": 0.02,
        "thin_component_max_area": 256,
        "thin_component_max_thickness": 5,
        "thin_component_min_aspect": 4.0,
        "thin_component_missing_fraction": 0.75,
        "temporal_mask_change_fraction": 0.35,
        "temporal_rgb_change_fraction": 0.03,
    },
}


# INVALID means that the frame cannot be safely interpreted as the declared
# input. REVIEW means the original can remain evidence, but a person must
# inspect the RGB/mask pair before the frame is accepted for a clean set.
INVALID_REASON_CODES = {
    "MANIFEST_EMPTY",
    "MANIFEST_ROW_INVALID",
    "MANIFEST_DUPLICATE_ID",
    "MANIFEST_DUPLICATE_FRAME",
    "RGB_MISSING_OR_UNREADABLE",
    "MASK_MISSING_OR_UNREADABLE",
    "PROPOSAL_MISSING_OR_UNREADABLE",
    "RGB_UNSUPPORTED_MODE",
    "MASK_UNSUPPORTED_MODE",
    "MASK_NOT_2D_INTEGER",
    "MASK_EMPTY",
    "MASK_CLASS_ID_OUT_OF_RANGE",
    "MASK_SHAPE_MISMATCH",
    "RGB_MASK_DIMENSION_MISMATCH",
    "RGB_MASK_FRAME_KEY_MISMATCH",
    "RGB_HASH_MISMATCH",
    "MASK_HASH_MISMATCH",
    "RGB_HASH_UNVERIFIED",
    "MASK_HASH_UNVERIFIED",
    "CLASS_ID_ORDER_MISMATCH",
    "CLASS_ID_ORDER_UNVERIFIED",
    "CLASS_ID_MAPPING_MISMATCH",
    "CLASS_ID_MAPPING_UNVERIFIED",
    "LABEL_SPACE_UNVERIFIED",
    "SOURCE_MAPPING_MISSING",
    "SOURCE_MAPPING_UNSAFE_UNKNOWN_TO_WALKABLE",
    "MASK_NON_NEAREST_RESIZE",
    "MASK_RESIZE_INTERPOLATION_CONTAMINATION",
    "MASK_RESIZE_HISTORY_UNVERIFIED",
    "RGB_MASK_GEOMETRY_UNVERIFIED",
    "MASK_ALPHA_CONTENT_REQUIRES_EXPLICIT_POLICY",
    "MANUAL_REVIEW_INVALID",
    "MANUAL_REVIEW_MISSING_REASON",
    "MANUAL_REVIEW_ORPHAN_ID",
}

REVIEW_REASON_CODES = {
    "RGB_MASK_FRAME_KEY_UNVERIFIED",
    "PROPOSAL_INVALID",
    "MASK_CONSTANT_SINGLE_CLASS",
    "MASK_ALL_WALKABLE",
    "MASK_ALL_UNKNOWN",
    "PROPOSAL_DISAGREEMENT_REQUIRES_REVIEW",
    "BOUNDARY_DRIFT_SUSPECTED",
    "OBSTACLE_BOUNDARY_SWAP_SUSPECTED",
    "FAR_REGION_OVERFILL_SUSPECTED",
    "WALKABLE_DISCONTINUITY_SUSPECTED",
    "THIN_OBJECT_OR_BRANCH_MISSED_SUSPECTED",
    "UNKNOWN_AS_WALKABLE_SUSPECTED",
    "UNKNOWN_WALKABLE_SEMANTIC_CONFLICT",
    "ADJACENT_LABEL_FLICKER_SUSPECTED",
    "TEMPORAL_ADJACENCY_UNVERIFIED",
    "MASK_ALPHA_NOT_OPAQUE",
    "MANUAL_REVIEW_REQUIRED",
    "MANUAL_REVIEW_MARKED",
    "MANUAL_INVALID_MARKED",
}

ALL_REASON_CODES = INVALID_REASON_CODES | REVIEW_REASON_CODES

REASON_CODE_DETAILS: dict[str, dict[str, str]] = {
    code: {
        "severity": STATUS_INVALID,
        "description": "Hard input/provenance failure; exclude the frame until repaired or re-bound.",
    }
    for code in sorted(INVALID_REASON_CODES)
}
REASON_CODE_DETAILS.update(
    {
        code: {
            "severity": STATUS_REVIEW,
            "description": "Evidence is retained, but RGB/mask visual review or a declared contract is required.",
        }
        for code in sorted(REVIEW_REASON_CODES)
    }
)


class MaskQualityAuditError(ValueError):
    """Raised for an invalid audit contract or output request."""


class _DecodeError(MaskQualityAuditError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _merge_config(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = _deep_copy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_config(dict(merged[key]), value)
        else:
            merged[key] = _deep_copy(value)
    return merged


def _parse_mapping(value: Any, *, field_name: str) -> dict[int, int]:
    if not isinstance(value, Mapping):
        raise MaskQualityAuditError(f"{field_name} must be an object")
    try:
        parsed = {int(key): int(item) for key, item in value.items()}
    except (TypeError, ValueError) as error:
        raise MaskQualityAuditError(f"{field_name} contains a non-integer key/value") from error
    return parsed


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load and validate an audit config without touching any dataset file."""

    if path is None:
        config = _deep_copy(DEFAULT_CONFIG)
    else:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MaskQualityAuditError(f"cannot read audit config: {path}") from error
        if not isinstance(value, Mapping):
            raise MaskQualityAuditError("audit config must be a JSON object")
        config = _merge_config(DEFAULT_CONFIG, value)

    class_order = config.get("class_order")
    class_ids = config.get("class_ids")
    if (
        not isinstance(class_order, list)
        or tuple(class_order) != tuple(CLASS_ORDER)
        or len(set(class_order)) != len(class_order)
    ):
        raise MaskQualityAuditError(
            "audit class_order must be the explicit four-class semantic order "
            f"{list(CLASS_ORDER)!r}; create a separate config for a separately frozen taxonomy"
        )
    if not isinstance(class_ids, Mapping):
        raise MaskQualityAuditError("class_ids must be an object")
    parsed_class_ids = {str(key): int(value) for key, value in class_ids.items()}
    if set(parsed_class_ids) != set(CLASS_ORDER) or set(parsed_class_ids.values()) != {0, 1, 2, 3}:
        raise MaskQualityAuditError("class_ids must bind the four class names to IDs 0..3")
    config["class_ids"] = parsed_class_ids
    config["allowed_ids"] = sorted({int(value) for value in config.get("allowed_ids", [])})
    if config["allowed_ids"] != [0, 1, 2, 3]:
        raise MaskQualityAuditError("allowed_ids must be exactly [0, 1, 2, 3]")
    config["void_ids"] = sorted({int(value) for value in config.get("void_ids", [])})
    config["source_unknown_ids"] = sorted({int(value) for value in config.get("source_unknown_ids", [])})
    if parsed_class_ids["unknown_nonwalkable"] == parsed_class_ids["walkable"]:
        raise MaskQualityAuditError("unknown_nonwalkable cannot share walkable's ID")
    config["source_to_expected_mapping"] = _parse_mapping(
        config.get("source_to_expected_mapping"), field_name="source_to_expected_mapping"
    )
    if set(config["source_to_expected_mapping"].values()) - {0, 1, 2, 3}:
        raise MaskQualityAuditError("source_to_expected_mapping emits an unknown expected class ID")
    config["accepted_rgb_modes"] = [str(value) for value in config.get("accepted_rgb_modes", [])]
    config["accepted_mask_modes"] = [str(value) for value in config.get("accepted_mask_modes", [])]
    if not config["accepted_rgb_modes"] or not config["accepted_mask_modes"]:
        raise MaskQualityAuditError("accepted image modes cannot be empty")
    if config.get("shape_policy") not in {"exact_rgb", "fixed"}:
        raise MaskQualityAuditError("shape_policy must be exact_rgb or fixed")
    if config["shape_policy"] == "fixed":
        fixed = config.get("fixed_mask_shape")
        if (
            not isinstance(fixed, list)
            or len(fixed) != 2
            or any(int(value) <= 0 for value in fixed)
        ):
            raise MaskQualityAuditError("fixed shape policy requires fixed_mask_shape=[height,width]")
        config["fixed_mask_shape"] = [int(fixed[0]), int(fixed[1])]
    heuristics = config.get("heuristics")
    if not isinstance(heuristics, Mapping):
        raise MaskQualityAuditError("heuristics must be an object")
    config["heuristics"] = {str(key): value for key, value in heuristics.items()}
    return config


def reason_code_catalog() -> dict[str, dict[str, str]]:
    return _deep_copy(REASON_CODE_DETAILS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            rows: list[dict[str, Any]] = []
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MaskQualityAuditError(f"blank JSONL line at {path}:{line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise MaskQualityAuditError(f"JSONL row is not an object at {path}:{line_number}")
                rows.append(value)
    except (OSError, json.JSONDecodeError) as error:
        raise MaskQualityAuditError(f"cannot read JSONL: {path}") from error
    return rows


def _read_review_annotations(path: Path) -> dict[str, dict[str, Any]]:
    if path.suffix.lower() == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise MaskQualityAuditError(f"cannot read review annotations: {path}") from error
        if isinstance(value, dict):
            values = value.get("rows", value.get("annotations", []))
        else:
            values = value
    else:
        values = _read_jsonl(path)
    if not isinstance(values, list):
        raise MaskQualityAuditError("review annotations must be a JSON array or JSONL objects")
    result: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(values, start=1):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise MaskQualityAuditError(f"review annotation {index} has no string id")
        item_id = str(item["id"])
        if item_id in result:
            raise MaskQualityAuditError(f"duplicate review annotation id: {item_id}")
        result[item_id] = item
    return result


def _resolve_path(value: Any, base_root: Path, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _DecodeError("MANIFEST_ROW_INVALID", f"{field} must be a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        path = base_root / path
    return path.resolve()


def _image_array(path: Path, *, kind: str, config: Mapping[str, Any]) -> tuple[np.ndarray, str, dict[str, Any]]:
    try:
        with Image.open(path) as image:
            mode = image.mode
            image.load()
            raw = np.asarray(image)
            if kind == "rgb":
                if mode not in set(config["accepted_rgb_modes"]):
                    raise _DecodeError("RGB_UNSUPPORTED_MODE", f"RGB mode {mode!r} is not accepted")
                array = np.asarray(image.convert("RGB"), dtype=np.uint8)
                return array, mode, {"source_shape": list(raw.shape)}
            if mode not in set(config["accepted_mask_modes"]):
                raise _DecodeError("MASK_UNSUPPORTED_MODE", f"mask mode {mode!r} is not accepted")
            alpha_non_opaque = False
            if mode == "RGBA":
                if raw.ndim != 3 or raw.shape[2] != 4:
                    raise _DecodeError("MASK_NOT_2D_INTEGER", "RGBA mask tensor has an invalid shape")
                alpha_non_opaque = bool(np.any(raw[..., 3] != 255))
                array = raw[..., 0]
            elif mode == "RGB":
                if raw.ndim != 3 or raw.shape[2] != 3:
                    raise _DecodeError("MASK_NOT_2D_INTEGER", "RGB mask tensor has an invalid shape")
                array = raw[..., 0]
            else:
                array = raw
            if array.ndim != 2 or not np.issubdtype(array.dtype, np.integer):
                raise _DecodeError("MASK_NOT_2D_INTEGER", "mask must decode to a 2D integer array")
            return np.asarray(array, dtype=np.int64), mode, {
                "source_shape": list(raw.shape),
                "alpha_non_opaque": alpha_non_opaque,
            }
    except _DecodeError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as error:
        code = "RGB_MISSING_OR_UNREADABLE" if kind == "rgb" else "MASK_MISSING_OR_UNREADABLE"
        raise _DecodeError(code, f"cannot decode {kind} image: {path}") from error


def _decode_expected_mask(
    raw: np.ndarray,
    *,
    decoder: str,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if decoder == "canonical":
        mapping = {value: value for value in config["allowed_ids"]}
    elif decoder == "source_native":
        mapping = dict(config["source_to_expected_mapping"])
        if not mapping:
            raise _DecodeError("SOURCE_MAPPING_MISSING", "source-native mask has no frozen mapping")
    else:
        raise _DecodeError("CLASS_ID_MAPPING_MISMATCH", f"unsupported mask decoder: {decoder!r}")
    observed = {int(value) for value in np.unique(raw)}
    unknown = sorted(observed - set(mapping))
    if unknown:
        raise _DecodeError(
            "MASK_CLASS_ID_OUT_OF_RANGE",
            f"mask contains IDs not accepted by the declared decoder: {unknown}",
        )
    expected = np.full(raw.shape, 255, dtype=np.uint8)
    for source_id, expected_id in mapping.items():
        expected[raw == source_id] = int(expected_id)
    if np.any(expected == 255):
        raise _DecodeError("MASK_CLASS_ID_OUT_OF_RANGE", "mask mapping left pixels without a class")
    if np.any(expected == config["class_ids"]["walkable"]) and int(config["class_ids"]["unknown_nonwalkable"]) == int(config["class_ids"]["walkable"]):
        raise _DecodeError("SOURCE_MAPPING_UNSAFE_UNKNOWN_TO_WALKABLE", "unknown and walkable IDs collide")
    return expected, {"raw_unique_ids": sorted(observed), "mapping": {str(k): int(v) for k, v in mapping.items()}}


def _fraction(value: int, total: int) -> float:
    return float(value / total) if total else 0.0


def _class_fractions(mask: np.ndarray, class_ids: Mapping[str, int]) -> dict[str, float]:
    total = int(mask.size)
    return {
        name: _fraction(int(np.count_nonzero(mask == int(class_id))), total)
        for name, class_id in class_ids.items()
    }


def _component_stats(binary: np.ndarray) -> list[dict[str, int | float]]:
    """Return 4-connected component facts without requiring cv2/scipy."""

    value = np.asarray(binary, dtype=bool)
    if value.ndim != 2 or not np.any(value):
        return []
    height, width = value.shape
    visited = np.zeros(value.shape, dtype=bool)
    components: list[dict[str, int | float]] = []
    for y, x in np.argwhere(value):
        y = int(y)
        x = int(x)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        area = 0
        min_y = max_y = y
        min_x = max_x = x
        while stack:
            current_y, current_x = stack.pop()
            area += 1
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            for next_y, next_x in (
                (current_y - 1, current_x),
                (current_y + 1, current_x),
                (current_y, current_x - 1),
                (current_y, current_x + 1),
            ):
                if 0 <= next_y < height and 0 <= next_x < width and value[next_y, next_x] and not visited[next_y, next_x]:
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
        component_width = max_x - min_x + 1
        component_height = max_y - min_y + 1
        short_side = max(1, min(component_width, component_height))
        long_side = max(component_width, component_height)
        components.append(
            {
                "area": area,
                "min_y": min_y,
                "max_y": max_y,
                "min_x": min_x,
                "max_x": max_x,
                "width": component_width,
                "height": component_height,
                "thickness": short_side,
                "aspect": float(long_side / short_side),
            }
        )
    return components


def _iou(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=bool)
    right = np.asarray(right, dtype=bool)
    union = int(np.count_nonzero(left | right))
    if union == 0:
        return None
    return float(np.count_nonzero(left & right) / union)


def _semantic_metrics(mask: np.ndarray, config: Mapping[str, Any]) -> tuple[dict[str, Any], set[str]]:
    class_ids = config["class_ids"]
    heuristics = config["heuristics"]
    codes: set[str] = set()
    height, width = mask.shape
    metrics: dict[str, Any] = {
        "shape": [int(height), int(width)],
        "unique_ids": sorted(int(value) for value in np.unique(mask)),
        "class_fractions": _class_fractions(mask, class_ids),
    }
    if bool(heuristics.get("flag_constant_masks", True)) and len(metrics["unique_ids"]) == 1:
        codes.add("MASK_CONSTANT_SINGLE_CLASS")
        if metrics["unique_ids"] == [int(class_ids["walkable"])]:
            codes.add("MASK_ALL_WALKABLE")
        if metrics["unique_ids"] == [int(class_ids["unknown_nonwalkable"])]:
            codes.add("MASK_ALL_UNKNOWN")

    far_rows = max(1, int(round(height * float(heuristics["far_region_row_fraction"]))))
    far = mask[:far_rows]
    far_walkable_fraction = _fraction(
        int(np.count_nonzero(far == int(class_ids["walkable"]))), int(far.size)
    )
    metrics["far_region"] = {
        "rows": far_rows,
        "walkable_fraction": far_walkable_fraction,
    }
    if far_walkable_fraction >= float(heuristics["far_region_walkable_fraction"]):
        codes.add("FAR_REGION_OVERFILL_SUSPECTED")

    lower_start = max(0, min(height - 1, int(round(height * float(heuristics["lower_ground_start_fraction"])))) )
    lower = mask[lower_start:]
    lower_walkable = lower == int(class_ids["walkable"])
    lower_walkable_fraction = _fraction(int(np.count_nonzero(lower_walkable)), int(lower.size))
    components = _component_stats(lower_walkable)
    largest_fraction = 0.0
    if components and np.count_nonzero(lower_walkable):
        largest_fraction = float(max(int(item["area"]) for item in components) / np.count_nonzero(lower_walkable))
    metrics["lower_ground"] = {
        "start_row": lower_start,
        "walkable_fraction": lower_walkable_fraction,
        "component_count": len(components),
        "largest_component_fraction": largest_fraction,
    }
    if (
        lower_walkable_fraction >= float(heuristics["lower_ground_min_walkable_fraction"])
        and (
            len(components) > int(heuristics["lower_ground_max_components"])
            or largest_fraction < float(heuristics["lower_ground_largest_component_fraction"])
        )
    ):
        codes.add("WALKABLE_DISCONTINUITY_SUSPECTED")
    return metrics, codes


def _add_code(result: dict[str, Any], code: str, detail: str | None = None) -> None:
    codes = result.setdefault("_reason_codes", set())
    codes.add(code)
    if detail:
        result.setdefault("reason_details", {})[code] = detail


def _remove_code(result: dict[str, Any], code: str) -> None:
    result.setdefault("_reason_codes", set()).discard(code)
    details = result.get("reason_details", {})
    details.pop(code, None)


def _check_manifest_metadata(row: Mapping[str, Any], result: dict[str, Any], config: Mapping[str, Any]) -> None:
    expected_label_space = config.get("expected_label_space")
    if expected_label_space:
        if row.get("label_space") is None:
            _add_code(result, "LABEL_SPACE_UNVERIFIED", "row did not declare label_space")
        elif str(row.get("label_space")) != str(expected_label_space):
            _add_code(result, "CLASS_ID_MAPPING_MISMATCH", "row label_space differs from the frozen audit label space")

    declared_order = row.get("class_order")
    if declared_order is None:
        _add_code(result, "CLASS_ID_ORDER_UNVERIFIED", "row did not declare class_order")
    elif list(declared_order) != list(config["class_order"]):
        _add_code(result, "CLASS_ID_ORDER_MISMATCH", "row class_order differs from the frozen four-class order")

    declared_ids = row.get("class_ids")
    if declared_ids is not None:
        try:
            parsed_ids = {str(key): int(value) for key, value in dict(declared_ids).items()}
        except (TypeError, ValueError):
            _add_code(result, "CLASS_ID_MAPPING_MISMATCH", "row class_ids is not an integer mapping")
        else:
            if parsed_ids != {str(key): int(value) for key, value in config["class_ids"].items()}:
                _add_code(result, "CLASS_ID_MAPPING_MISMATCH", "row class_ids differs from the frozen class ID mapping")

    decoder = str(row.get("mask_decoder", config["default_mask_decoder"]))
    result["provenance"]["mask_decoder"] = decoder
    if decoder == "source_native":
        mapping = row.get("source_to_expected_mapping")
        if mapping is None:
            _add_code(result, "SOURCE_MAPPING_MISSING", "source_native rows must bind a frozen source-to-expected mapping")
        else:
            try:
                parsed = _parse_mapping(mapping, field_name="source_to_expected_mapping")
            except MaskQualityAuditError as error:
                _add_code(result, "CLASS_ID_MAPPING_MISMATCH", str(error))
            else:
                expected = dict(config["source_to_expected_mapping"])
                if parsed != expected:
                    _add_code(result, "CLASS_ID_MAPPING_MISMATCH", "row source-to-expected mapping differs from config")
                if any(
                    parsed.get(int(source_id)) == int(config["class_ids"]["walkable"])
                    for source_id in config.get("source_unknown_ids", [])
                ):
                    _add_code(result, "SOURCE_MAPPING_UNSAFE_UNKNOWN_TO_WALKABLE", "unknown source ID maps to walkable")
    elif decoder != "canonical":
        _add_code(result, "CLASS_ID_MAPPING_MISMATCH", f"unsupported row mask_decoder {decoder!r}")
    elif row.get("source_to_expected_mapping") is None:
        _add_code(result, "CLASS_ID_MAPPING_UNVERIFIED", "canonical row did not explicitly bind the identity mapping")
    else:
        try:
            parsed = _parse_mapping(row["source_to_expected_mapping"], field_name="source_to_expected_mapping")
        except MaskQualityAuditError as error:
            _add_code(result, "CLASS_ID_MAPPING_MISMATCH", str(error))
        else:
            identity = {int(value): int(value) for value in config["allowed_ids"]}
            if parsed != identity:
                _add_code(result, "CLASS_ID_MAPPING_MISMATCH", "canonical row declares a non-identity mapping")

    rgb_key = row.get("rgb_frame_key", row.get("rgb_frame_id"))
    mask_key = row.get("mask_frame_key", row.get("mask_frame_id"))
    if rgb_key is not None and mask_key is not None:
        if str(rgb_key) != str(mask_key):
            _add_code(result, "RGB_MASK_FRAME_KEY_MISMATCH", "RGB and mask frame keys differ")
    elif row.get("pairing_key") is None:
        _add_code(result, "RGB_MASK_FRAME_KEY_UNVERIFIED", "no explicit RGB/mask pairing key was supplied")

    resize_applied = bool(row.get("mask_resize_applied", False))
    interpolation = row.get("mask_resize_interpolation")
    if resize_applied and interpolation is None:
        _add_code(result, "MASK_RESIZE_HISTORY_UNVERIFIED", "resize was declared without interpolation provenance")
    elif interpolation is not None and str(interpolation).upper() not in {"NEAREST", "NEAREST_NEIGHBOR"}:
        _add_code(result, "MASK_NON_NEAREST_RESIZE", f"mask resize interpolation was {interpolation!r}")
    result["provenance"]["mask_resize_applied"] = resize_applied
    result["provenance"]["mask_resize_interpolation"] = interpolation


def _audit_proposal(
    *,
    original: np.ndarray,
    proposal: np.ndarray,
    config: Mapping[str, Any],
    result: dict[str, Any],
) -> None:
    if original.shape != proposal.shape:
        _add_code(result, "PROPOSAL_INVALID", "proposal shape differs from the audited original mask")
        return
    heuristic = config["heuristics"]
    disagreement = float(np.mean(original != proposal))
    swap = proposal.copy()
    obstacle_id = int(config["class_ids"]["blocking_obstacle"])
    boundary_id = int(config["class_ids"]["boundary_level_change"])
    proposal_obstacle = proposal == obstacle_id
    proposal_boundary = proposal == boundary_id
    swap[proposal_obstacle] = boundary_id
    swap[proposal_boundary] = obstacle_id
    direct_agreement = float(np.mean(original == proposal))
    swapped_agreement = float(np.mean(original == swap))
    result["proposal"]["metrics"] = {
        "disagreement_fraction": disagreement,
        "direct_agreement": direct_agreement,
        "swapped_obstacle_boundary_agreement": swapped_agreement,
        "original_class_fractions": _class_fractions(original, config["class_ids"]),
        "proposal_class_fractions": _class_fractions(proposal, config["class_ids"]),
        "boundary_iou": _iou(original == boundary_id, proposal == boundary_id),
    }
    if disagreement >= float(heuristic["proposal_disagreement_fraction"]):
        _add_code(result, "PROPOSAL_DISAGREEMENT_REQUIRES_REVIEW", "proposal differs materially from the original label")
    proposal_obstacle_pixels = int(np.count_nonzero(proposal == obstacle_id))
    proposal_boundary_pixels = int(np.count_nonzero(proposal == boundary_id))
    if (
        proposal_obstacle_pixels >= int(heuristic["proposal_class_min_pixels"])
        and proposal_boundary_pixels >= int(heuristic["proposal_class_min_pixels"])
        and swapped_agreement - direct_agreement >= float(heuristic["proposal_swap_margin"])
    ):
        _add_code(result, "OBSTACLE_BOUNDARY_SWAP_SUSPECTED", "proposal agrees better after swapping obstacle and boundary IDs")
    boundary_iou = result["proposal"]["metrics"]["boundary_iou"]
    if (
        boundary_iou is not None
        and proposal_boundary_pixels >= int(heuristic["proposal_class_min_pixels"])
        and int(np.count_nonzero(original == boundary_id)) >= int(heuristic["proposal_class_min_pixels"])
        and boundary_iou < float(heuristic["boundary_iou_review_floor"])
        and swapped_agreement - direct_agreement < float(heuristic["proposal_swap_margin"])
    ):
        _add_code(result, "BOUNDARY_DRIFT_SUSPECTED", "original/proposal boundary regions have low IoU")

    walkable_id = int(config["class_ids"]["walkable"])
    unknown_id = int(config["class_ids"]["unknown_nonwalkable"])
    original_walkable_proposal_unknown = float(np.mean((original == walkable_id) & (proposal == unknown_id)))
    original_unknown_proposal_walkable = float(np.mean((original == unknown_id) & (proposal == walkable_id)))
    result["proposal"]["metrics"]["original_walkable_proposal_unknown_fraction"] = original_walkable_proposal_unknown
    result["proposal"]["metrics"]["original_unknown_proposal_walkable_fraction"] = original_unknown_proposal_walkable
    if max(original_walkable_proposal_unknown, original_unknown_proposal_walkable) >= float(heuristic["unknown_walkable_conflict_fraction"]):
        _add_code(result, "UNKNOWN_WALKABLE_SEMANTIC_CONFLICT", "proposal and original disagree on UNKNOWN versus walkable")
    if original_unknown_proposal_walkable >= float(heuristic["unknown_walkable_conflict_fraction"]):
        _add_code(result, "UNKNOWN_AS_WALKABLE_SUSPECTED", "original unknown_nonwalkable pixels are proposed as walkable")

    proposal_hazard = (proposal == obstacle_id) | (proposal == boundary_id)
    original_walkable = original == walkable_id
    for component in _component_stats(proposal_hazard):
        area = int(component["area"])
        is_thin = (
            area <= int(heuristic["thin_component_max_area"])
            and (
                int(component["thickness"]) <= int(heuristic["thin_component_max_thickness"])
                or float(component["aspect"]) >= float(heuristic["thin_component_min_aspect"])
            )
        )
        if not is_thin:
            continue
        # Reconstructing the exact component mask is unnecessary for the
        # conservative review trigger: use its bounding box and count only
        # proposal hazard pixels in that box.  Any high original-walkable
        # overlap becomes a human-review candidate, not a relabel instruction.
        min_y, max_y = int(component["min_y"]), int(component["max_y"])
        min_x, max_x = int(component["min_x"]), int(component["max_x"])
        box = proposal_hazard[min_y : max_y + 1, min_x : max_x + 1]
        missing_fraction = float(np.count_nonzero(box & original_walkable[min_y : max_y + 1, min_x : max_x + 1]) / max(1, int(np.count_nonzero(box))))
        if missing_fraction >= float(heuristic["thin_component_missing_fraction"]):
            _add_code(result, "THIN_OBJECT_OR_BRANCH_MISSED_SUSPECTED", "thin proposal component is mostly walkable in the original")
            break


def _new_result(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "blindassist.segmentation_mask_quality_audit.frame_result.v1",
        "id": str(row.get("id", "")),
        "session_id": str(row.get("session_id", "")),
        "sequence_id": str(row.get("sequence_id", row.get("session_id", ""))),
        "frame_id": row.get("frame_id"),
        "status": STATUS_INVALID,
        "reason_codes": [],
        "reason_details": {},
        "original": {
            "rgb_path": row.get("rgb_path"),
            "mask_path": row.get("mask_path"),
            "rgb_sha256": None,
            "mask_sha256": None,
            "rgb_mode": None,
            "mask_mode": None,
            "rgb_shape": None,
            "mask_shape": None,
        },
        "proposal": {
            "present": bool(row.get("proposal_mask_path")),
            "path": row.get("proposal_mask_path"),
            "sha256": None,
            "authority": "PROPOSAL_ONLY" if row.get("proposal_mask_path") else "NONE",
            "replacement_applied": False,
            "metrics": {},
        },
        "provenance": {
            "original_label_immutable": True,
            "replacement_applied": False,
            "visual_review_status": "PENDING",
        },
        "metrics": {},
        "_reason_codes": set(),
    }


def _audit_frame(row: Mapping[str, Any], *, base_root: Path, config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    result = _new_result(row)
    if not isinstance(row.get("id"), str) or not row.get("id") or not isinstance(row.get("session_id"), str) or not row.get("session_id"):
        _add_code(result, "MANIFEST_ROW_INVALID", "id and session_id are required")
        return result, {"result": result, "rgb": None, "mask": None}
    try:
        frame_id = int(row.get("frame_id"))
        if frame_id < 0:
            raise ValueError
        result["frame_id"] = frame_id
    except (TypeError, ValueError):
        _add_code(result, "MANIFEST_ROW_INVALID", "frame_id must be a non-negative integer")
        return result, {"result": result, "rgb": None, "mask": None}

    _check_manifest_metadata(row, result, config)
    if bool(config.get("require_input_hashes", True)):
        if not isinstance(row.get("rgb_sha256"), str) or len(str(row.get("rgb_sha256"))) != 64:
            _add_code(result, "RGB_HASH_UNVERIFIED", "row did not bind a 64-character RGB SHA-256")
        if not isinstance(row.get("mask_sha256"), str) or len(str(row.get("mask_sha256"))) != 64:
            _add_code(result, "MASK_HASH_UNVERIFIED", "row did not bind a 64-character original-mask SHA-256")
    try:
        rgb_path = _resolve_path(row.get("rgb_path"), base_root, field="rgb_path")
        mask_path = _resolve_path(row.get("mask_path"), base_root, field="mask_path")
    except _DecodeError as error:
        _add_code(result, error.code, str(error))
        return result, {"result": result, "rgb": None, "mask": None}
    try:
        rgb, rgb_mode, rgb_meta = _image_array(rgb_path, kind="rgb", config=config)
        rgb_sha = sha256_file(rgb_path)
    except _DecodeError as error:
        _add_code(result, error.code, str(error))
        return result, {"result": result, "rgb": None, "mask": None}
    result["original"].update(
        {
            "rgb_sha256": rgb_sha,
            "rgb_mode": rgb_mode,
            "rgb_shape": [int(rgb.shape[0]), int(rgb.shape[1])],
        }
    )
    expected_rgb_sha = row.get("rgb_sha256")
    if expected_rgb_sha is not None and str(expected_rgb_sha).lower() != rgb_sha:
        _add_code(result, "RGB_HASH_MISMATCH", "decoded RGB hash differs from manifest")

    try:
        raw_mask, mask_mode, mask_meta = _image_array(mask_path, kind="mask", config=config)
        mask_sha = sha256_file(mask_path)
    except _DecodeError as error:
        _add_code(result, error.code, str(error))
        return result, {"result": result, "rgb": rgb, "mask": None}
    result["original"].update(
        {
            "mask_sha256": mask_sha,
            "mask_mode": mask_mode,
            "mask_shape": [int(raw_mask.shape[0]), int(raw_mask.shape[1])],
        }
    )
    expected_mask_sha = row.get("mask_sha256")
    if expected_mask_sha is not None and str(expected_mask_sha).lower() != mask_sha:
        _add_code(result, "MASK_HASH_MISMATCH", "decoded mask hash differs from manifest")

    if raw_mask.size == 0:
        _add_code(result, "MASK_EMPTY", "mask has zero pixels")
        return result, {"result": result, "rgb": rgb, "mask": None}
    raw_unique = {int(value) for value in np.unique(raw_mask)}
    void_ids = set(int(value) for value in config.get("void_ids", []))
    if void_ids and raw_unique and raw_unique.issubset(void_ids):
        _add_code(result, "MASK_EMPTY", "mask contains only declared void IDs")
        return result, {"result": result, "rgb": rgb, "mask": None}
    if mask_meta.get("alpha_non_opaque"):
        _add_code(result, "MASK_ALPHA_NOT_OPAQUE", "mask alpha contains transparent pixels")
        _add_code(result, "MASK_ALPHA_CONTENT_REQUIRES_EXPLICIT_POLICY", "alpha is ignored by the declared class decoder")
    decoder = str(row.get("mask_decoder", config["default_mask_decoder"]))
    try:
        mask, decode_meta = _decode_expected_mask(raw_mask, decoder=decoder, config=config)
    except _DecodeError as error:
        _add_code(result, error.code, str(error))
        return result, {"result": result, "rgb": rgb, "mask": None}
    result["original"]["raw_unique_ids"] = decode_meta["raw_unique_ids"]
    result["original"]["mapping"] = decode_meta["mapping"]

    shape_policy = str(config["shape_policy"])
    if shape_policy == "exact_rgb" and tuple(raw_mask.shape) != tuple(rgb.shape[:2]):
        _add_code(result, "RGB_MASK_DIMENSION_MISMATCH", "mask and RGB decoded dimensions differ")
    elif shape_policy == "fixed":
        fixed_shape = tuple(int(value) for value in config["fixed_mask_shape"])
        if tuple(raw_mask.shape) != fixed_shape:
            _add_code(result, "MASK_SHAPE_MISMATCH", f"mask shape {tuple(raw_mask.shape)} differs from fixed shape {fixed_shape}")
        if tuple(rgb.shape[:2]) != fixed_shape and row.get("mask_to_rgb_transform") is None:
            _add_code(result, "RGB_MASK_GEOMETRY_UNVERIFIED", "fixed-size mask differs from RGB without transform provenance")
    if bool(row.get("mask_resize_applied", False)) and str(row.get("mask_resize_interpolation", "")).upper() not in {"NEAREST", "NEAREST_NEIGHBOR"}:
        _add_code(result, "MASK_RESIZE_INTERPOLATION_CONTAMINATION", "declared resize is not categorical nearest-neighbor")

    if tuple(mask.shape) == tuple(rgb.shape[:2]) or shape_policy == "fixed":
        semantic_metrics, semantic_codes = _semantic_metrics(mask, config)
        result["metrics"].update(semantic_metrics)
        for code in semantic_codes:
            _add_code(result, code)
    else:
        result["metrics"].update(
            {
                "shape": [int(mask.shape[0]), int(mask.shape[1])],
                "unique_ids": sorted(int(value) for value in np.unique(mask)),
                "class_fractions": _class_fractions(mask, config["class_ids"]),
            }
        )
    result["original"]["array_sha256"] = _sha256_array(mask)

    proposal = None
    proposal_path_value = row.get("proposal_mask_path")
    if proposal_path_value:
        try:
            proposal_path = _resolve_path(proposal_path_value, base_root, field="proposal_mask_path")
            proposal_raw, proposal_mode, proposal_meta = _image_array(proposal_path, kind="mask", config=config)
            proposal_sha = sha256_file(proposal_path)
            result["proposal"].update({"sha256": proposal_sha, "mode": proposal_mode, "shape": [int(proposal_raw.shape[0]), int(proposal_raw.shape[1])]})
            proposal_decoder = str(row.get("proposal_mask_decoder", "canonical"))
            proposal, _ = _decode_expected_mask(proposal_raw, decoder=proposal_decoder, config=config)
            if proposal_meta.get("alpha_non_opaque"):
                _add_code(result, "PROPOSAL_INVALID", "proposal alpha contains transparent pixels")
        except (_DecodeError, OSError, UnidentifiedImageError, ValueError) as error:
            _add_code(result, "PROPOSAL_INVALID", str(error))
    if proposal is not None:
        _audit_proposal(original=mask, proposal=proposal, config=config, result=result)

    return result, {"result": result, "rgb": rgb, "mask": mask, "proposal": proposal}


def _apply_annotation(result: dict[str, Any], annotation: Mapping[str, Any] | None, *, require_visual_review: bool) -> None:
    if annotation is None:
        if require_visual_review:
            _add_code(result, "MANUAL_REVIEW_REQUIRED", "no per-frame visual review annotation was supplied")
        result["provenance"]["visual_review_status"] = "PENDING"
        return
    status = annotation.get("status")
    if status not in STATUSES:
        _add_code(result, "MANUAL_REVIEW_INVALID", "review status must be PASS, REVIEW, or INVALID")
        return
    codes = annotation.get("reason_codes", [])
    if not isinstance(codes, list) or any(str(code) not in ALL_REASON_CODES for code in codes):
        _add_code(result, "MANUAL_REVIEW_INVALID", "review annotation contains an unknown reason code")
        return
    if status == STATUS_PASS and codes:
        _add_code(result, "MANUAL_REVIEW_INVALID", "PASS review cannot carry reason codes")
        return
    if status != STATUS_PASS and not codes:
        _add_code(result, "MANUAL_REVIEW_MISSING_REASON", "REVIEW/INVALID requires at least one reason code")
        return
    result["provenance"]["visual_review_status"] = status
    result["provenance"]["reviewer_id"] = annotation.get("reviewer_id")
    result["provenance"]["reviewer_type"] = annotation.get("reviewer_type")
    if status == STATUS_PASS:
        _remove_code(result, "MANUAL_REVIEW_REQUIRED")
    elif status == STATUS_REVIEW:
        _add_code(result, "MANUAL_REVIEW_MARKED", "reviewer marked the original mask for review")
        for code in codes:
            _add_code(result, str(code), "manual visual review annotation")
    elif status == STATUS_INVALID:
        _add_code(result, "MANUAL_INVALID_MARKED", "reviewer marked the original mask invalid")
        for code in codes:
            _add_code(result, str(code), "manual visual review annotation")


def _finalize_result(result: dict[str, Any]) -> dict[str, Any]:
    codes = sorted(str(code) for code in result.pop("_reason_codes", set()))
    result["reason_codes"] = codes
    invalid = sorted(code for code in codes if code in INVALID_REASON_CODES or code in {"MANUAL_INVALID_MARKED", "MANUAL_REVIEW_INVALID", "MANUAL_REVIEW_MISSING_REASON"})
    result["status"] = STATUS_INVALID if invalid else STATUS_REVIEW if codes else STATUS_PASS
    result["reason_details"] = {code: result.get("reason_details", {}).get(code) for code in codes if result.get("reason_details", {}).get(code)}
    return result


def _add_temporal_pair_code(result: dict[str, Any], detail: str) -> None:
    _add_code(result, "ADJACENT_LABEL_FLICKER_SUSPECTED", detail)


def _temporal_audit(states: list[dict[str, Any]], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for state in states:
        by_sequence[str(state["result"]["sequence_id"])].append(state)
    pairs: list[dict[str, Any]] = []
    heuristic = config["heuristics"]
    for sequence_id, sequence in by_sequence.items():
        sequence.sort(key=lambda item: int(item["result"]["frame_id"]))
        for previous, current in zip(sequence, sequence[1:]):
            previous_result = previous["result"]
            current_result = current["result"]
            previous_id = int(previous_result["frame_id"])
            current_id = int(current_result["frame_id"])
            delta = current_id - previous_id
            pair = {
                "sequence_id": sequence_id,
                "previous_id": previous_result["id"],
                "current_id": current_result["id"],
                "previous_frame_id": previous_id,
                "current_frame_id": current_id,
                "frame_delta": delta,
                "checked": False,
            }
            if delta != 1:
                if bool(config.get("require_temporal_adjacency", True)):
                    _add_code(current_result, "TEMPORAL_ADJACENCY_UNVERIFIED", "manifest does not contain the immediately adjacent frame")
                pairs.append(pair)
                continue
            old_rgb = previous.get("rgb")
            new_rgb = current.get("rgb")
            old_mask = previous.get("mask")
            new_mask = current.get("mask")
            if old_rgb is None or new_rgb is None or old_mask is None or new_mask is None:
                if bool(config.get("require_temporal_adjacency", True)):
                    _add_code(current_result, "TEMPORAL_ADJACENCY_UNVERIFIED", "one adjacent RGB/mask decode failed")
                pairs.append(pair)
                continue
            if old_rgb.shape != new_rgb.shape or old_mask.shape != new_mask.shape:
                _add_code(current_result, "TEMPORAL_ADJACENCY_UNVERIFIED", "adjacent shapes differ")
                pairs.append(pair)
                continue
            mask_change = float(np.mean(old_mask != new_mask))
            old_gray = np.mean(old_rgb.astype(np.float32), axis=2)
            new_gray = np.mean(new_rgb.astype(np.float32), axis=2)
            rgb_change = float(np.mean(np.abs(old_gray - new_gray)) / 255.0)
            pair.update({"checked": True, "mask_change_fraction": mask_change, "rgb_change_fraction": rgb_change})
            if (
                mask_change >= float(heuristic["temporal_mask_change_fraction"])
                and rgb_change <= float(heuristic["temporal_rgb_change_fraction"])
            ):
                detail = f"mask change {mask_change:.4f} with RGB change {rgb_change:.4f}"
                _add_temporal_pair_code(previous_result, detail)
                _add_temporal_pair_code(current_result, detail)
            pairs.append(pair)
    return pairs


def audit_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    base_root: Path,
    config: Mapping[str, Any] | None = None,
    review_annotations: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit manifest rows and return a JSON-serializable report."""

    config = load_config() if config is None else load_config_from_mapping(config)
    base_root = base_root.resolve()
    materialized = [dict(row) for row in rows]
    if not materialized:
        return {
            "schema_version": "blindassist.segmentation_mask_quality_audit.report.v1",
            "status": STATUS_INVALID,
            "manifest_errors": [{"code": "MANIFEST_EMPTY", "detail": "manifest has zero rows"}],
            "frame_count": 0,
            "status_counts": {},
            "reason_code_counts": {"MANIFEST_EMPTY": 1},
            "frames": [],
            "temporal_pairs": [],
            "provenance": {"original_labels_immutable": True, "proposal_replacement_applied": False},
        }
    manifest_errors: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_frames: set[tuple[str, str, int]] = set()
    states: list[dict[str, Any]] = []
    annotations = review_annotations or {}
    known_ids = {str(row.get("id", "")) for row in materialized}
    for annotation_id in sorted(set(annotations) - known_ids):
        manifest_errors.append({"code": "MANUAL_REVIEW_ORPHAN_ID", "detail": annotation_id})
    for row in materialized:
        item_id = str(row.get("id", ""))
        if item_id in seen_ids:
            manifest_errors.append({"code": "MANIFEST_DUPLICATE_ID", "detail": item_id})
        seen_ids.add(item_id)
        try:
            frame_key = (str(row.get("sequence_id", row.get("session_id", ""))), str(row.get("session_id", "")), int(row.get("frame_id")))
            if frame_key in seen_frames:
                manifest_errors.append({"code": "MANIFEST_DUPLICATE_FRAME", "detail": repr(frame_key)})
            seen_frames.add(frame_key)
        except (TypeError, ValueError):
            pass
        result, state = _audit_frame(row, base_root=base_root, config=config)
        _apply_annotation(result, annotations.get(item_id), require_visual_review=bool(config.get("require_visual_review", True)))
        states.append(state)
    temporal_pairs = _temporal_audit(states, config)
    frames = [_finalize_result(state["result"]) for state in states]
    overall_manifest_status = STATUS_INVALID if manifest_errors else None
    status_counts = Counter(str(frame["status"]) for frame in frames)
    reason_counts = Counter(code for frame in frames for code in frame["reason_codes"])
    for error in manifest_errors:
        reason_counts[error["code"]] += 1
    if (
        overall_manifest_status == STATUS_INVALID
        or any(frame["status"] == STATUS_INVALID for frame in frames)
        or any(code in INVALID_REASON_CODES for code in reason_counts)
    ):
        overall_status = STATUS_INVALID
    elif any(frame["status"] == STATUS_REVIEW for frame in frames):
        overall_status = STATUS_REVIEW
    else:
        overall_status = STATUS_PASS
    return {
        "schema_version": "blindassist.segmentation_mask_quality_audit.report.v1",
        "status": overall_status,
        "manifest_sha256": manifest_sha256,
        "base_root": str(base_root),
        "frame_count": len(frames),
        "status_counts": dict(sorted(status_counts.items())),
        "reason_code_counts": dict(sorted(reason_counts.items())),
        "manifest_errors": manifest_errors,
        "frames": frames,
        "temporal_pairs": temporal_pairs,
        "provenance": {
            "original_labels_immutable": all(frame["provenance"].get("original_label_immutable") is True for frame in frames),
            "proposal_replacement_applied": False,
            "proposal_authority": "PROPOSAL_ONLY",
        },
    }


def load_config_from_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MaskQualityAuditError("config must be an object")
    return load_config_from_value(value)


def load_config_from_value(value: Mapping[str, Any]) -> dict[str, Any]:
    config = _merge_config(DEFAULT_CONFIG, value)
    # Validate via a temporary in-memory shape rather than writing a config.
    # The checks intentionally mirror load_config; this helper exists for unit
    # tests and callers that already hold a parsed contract.
    class_order = config.get("class_order")
    if not isinstance(class_order, list) or tuple(class_order) != CLASS_ORDER:
        raise MaskQualityAuditError("config class_order is not the frozen four-class order")
    parsed_class_ids = {str(key): int(item) for key, item in dict(config["class_ids"]).items()}
    if set(parsed_class_ids) != set(CLASS_ORDER) or set(parsed_class_ids.values()) != {0, 1, 2, 3}:
        raise MaskQualityAuditError("config class_ids must bind 0..3")
    if parsed_class_ids["unknown_nonwalkable"] == parsed_class_ids["walkable"]:
        raise MaskQualityAuditError("unknown_nonwalkable cannot share walkable's ID")
    config["class_ids"] = parsed_class_ids
    config["allowed_ids"] = sorted({int(item) for item in config["allowed_ids"]})
    if config["allowed_ids"] != [0, 1, 2, 3]:
        raise MaskQualityAuditError("config allowed_ids must be exactly [0, 1, 2, 3]")
    config["void_ids"] = sorted({int(item) for item in config.get("void_ids", [])})
    config["source_unknown_ids"] = sorted({int(item) for item in config.get("source_unknown_ids", [])})
    config["source_to_expected_mapping"] = _parse_mapping(config["source_to_expected_mapping"], field_name="source_to_expected_mapping")
    if set(config["source_to_expected_mapping"].values()) - {0, 1, 2, 3}:
        raise MaskQualityAuditError("source_to_expected_mapping emits an unknown class ID")
    if config.get("shape_policy") not in {"exact_rgb", "fixed"}:
        raise MaskQualityAuditError("shape_policy must be exact_rgb or fixed")
    if config["shape_policy"] == "fixed":
        fixed = config.get("fixed_mask_shape")
        if not isinstance(fixed, list) or len(fixed) != 2:
            raise MaskQualityAuditError("fixed shape policy requires fixed_mask_shape")
        config["fixed_mask_shape"] = [int(fixed[0]), int(fixed[1])]
    return config


def audit_manifest(
    manifest_path: Path,
    *,
    base_root: Path,
    config: Mapping[str, Any] | None = None,
    review_path: Path | None = None,
) -> dict[str, Any]:
    rows = _read_jsonl(manifest_path)
    annotations = _read_review_annotations(review_path) if review_path is not None else None
    return audit_rows(
        rows,
        base_root=base_root,
        config=config,
        review_annotations=annotations,
        manifest_sha256=sha256_file(manifest_path),
    )


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode("utf-8")


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def write_report(report: Mapping[str, Any], output_root: Path) -> None:
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise MaskQualityAuditError(f"refusing to overwrite non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _write_atomic(output_root / "summary.json", dict(report))
    frame_path = output_root / "frame_results.jsonl"
    temporary = frame_path.with_suffix(frame_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for frame in report.get("frames", []):
            handle.write(json.dumps(frame, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(frame_path)
    queue = [
        {
            "id": frame["id"],
            "session_id": frame["session_id"],
            "frame_id": frame["frame_id"],
            "status": frame["status"],
            "reason_codes": frame["reason_codes"],
            "rgb_path": frame["original"].get("rgb_path"),
            "mask_path": frame["original"].get("mask_path"),
            "original_label_immutable": True,
            "proposal_authority": "PROPOSAL_ONLY",
        }
        for frame in report.get("frames", [])
        if frame["status"] != STATUS_PASS
    ]
    _write_atomic(output_root / "review_queue.json", queue)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit RGB/segmentation-mask quality without replacing labels")
    parser.add_argument("--manifest", type=Path, required=False)
    parser.add_argument("--base-root", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--review-file", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--print-reason-codes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.print_reason_codes:
        print(json.dumps(reason_code_catalog(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.manifest is None:
        parser.error("--manifest is required unless --print-reason-codes is used")
    try:
        config = load_config(args.config)
        manifest = args.manifest.resolve()
        base_root = (args.base_root or manifest.parent).resolve()
        report = audit_manifest(
            manifest,
            base_root=base_root,
            config=config,
            review_path=args.review_file.resolve() if args.review_file else None,
        )
        if args.output_root is None:
            print(json.dumps({"status": report["status"], "frame_count": report["frame_count"], "status_counts": report["status_counts"]}, ensure_ascii=False))
        else:
            output_root = args.output_root.resolve()
            try:
                output_root.relative_to((base_root / "artifacts.local").resolve())
            except ValueError:
                raise MaskQualityAuditError("--output-root must stay under base-root/artifacts.local")
            write_report(report, output_root)
            print(json.dumps({"status": report["status"], "frame_count": report["frame_count"], "output_root": str(output_root)}, ensure_ascii=False))
        return 0 if report["status"] == STATUS_PASS else 1
    except (MaskQualityAuditError, OSError) as error:
        print(json.dumps({"status": STATUS_INVALID, "error": str(error)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
