"""Shared deterministic contracts for learned component validator R0."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import shutil
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    Component,
    aggregate_confusion,
    component_metrics,
    connected_components,
    mask_iou,
    pixel_metrics,
)
from scripts.research.dual_loop_segmentation_conditional_gating.conditional_gating import (
    PRIMARY_CANDIDATE_ID,
    apply_frozen_candidate_to_component,
    causal_two_of_three,
    upper_head_band,
)


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_LEARNED_COMPONENT_VALIDATOR_R0"
CANDIDATE_ID = "FAILURE_AWARE_CAUSAL_COMPONENT_VALIDATOR"
TABLE_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0."
    "component_table.v1"
)
PREPARE_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0.prepare.v1"
)
EVALUATION_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0.evaluation.v1"
)
FRAME_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0.frame_metrics.v1"
)
MODEL_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0.fold_model.v1"
)
PREDICTION_SCHEMA_VERSION = (
    "blindassist.dual_loop_segmentation_learned_component_validator_r0."
    "held_out_prediction.v1"
)

RAW_ARM_ID = "RAW_DDRNET"
CAUSAL_ARM_ID = "CAUSAL_2_OF_3"
CONFIDENCE_ARM_ID = "CONFIDENCE_GE_0_65"
HISTORICAL_ARM_ID = "CLASS_CONDITIONED_MULTI_NEGATIVE"
LEARNED_ARM_ID = CANDIDATE_ID
ARM_IDS = (
    RAW_ARM_ID,
    CAUSAL_ARM_ID,
    CONFIDENCE_ARM_ID,
    HISTORICAL_ARM_ID,
    LEARNED_ARM_ID,
)
FEATURE_NAMES = (
    "predicted_class_is_obstacle",
    "log1p_area_pixels",
    "bbox_width_fraction",
    "bbox_height_fraction",
    "log_bbox_aspect_ratio",
    "centroid_x_fraction",
    "centroid_y_fraction",
    "intersects_upper_head_band",
    "intersects_central_body_corridor",
    "top1_confidence_median",
    "top1_confidence_missing",
    "top1_top2_margin_median",
    "top1_top2_margin_missing",
    "causal_previous_component_iou_max",
    "causal_previous_component_iou_missing",
    "causal_same_footprint_age_5",
    "recent_flicker_count_3",
    "nearest_yolo_union_bbox_gap_fraction",
    "nearest_yolo_union_bbox_gap_missing",
    "near_yolo_union_gap_le_3",
    "obstacle_x_near_yolo_union",
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL row: {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object: {path}:{line_number}")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def atomic_output_directory(output_root: Path) -> tuple[Path, Any]:
    """Return a temporary directory and a finalizer callable."""

    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)

    def finalize(success: bool) -> None:
        if success:
            temporary.replace(output_root)
        else:
            shutil.rmtree(temporary, ignore_errors=True)

    return temporary, finalize


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def verify_output_scope(repo_root: Path, output_root: Path) -> None:
    repo = repo_root.resolve()
    output = output_root.resolve()
    allowed = (repo / "artifacts.local").resolve()
    try:
        output.relative_to(allowed)
    except ValueError as exc:
        raise ValueError("output must stay below artifacts.local") from exc


def decode_packed_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    packed = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    expected = int(np.prod(shape))
    unpacked = np.unpackbits(packed, bitorder="big")
    if unpacked.size < expected or unpacked.size - expected >= 8:
        raise ValueError("packed mask length does not match shape")
    return unpacked[:expected].reshape(shape).astype(bool)


def load_bound_jsonl(
    repo_root: Path,
    specifications: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for specification in specifications:
        path = resolve(repo_root, specification["path"])
        observed_sha = sha256_file(path)
        if observed_sha != str(specification["sha256"]):
            raise ValueError(f"bound input SHA mismatch: {path}")
        local = read_jsonl(path)
        if len(local) != int(specification["row_count"]):
            raise ValueError(f"bound input row count mismatch: {path}")
        rows.extend(local)
        provenance.append(
            {
                "path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": observed_sha,
                "row_count": len(local),
            }
        )
    return rows, provenance


def validate_static_config(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol_id")
    if config.get("candidate_id") != CANDIDATE_ID:
        raise ValueError("unexpected candidate_id")
    if config.get("stage") != "DEVELOPMENT_STANDARD":
        raise ValueError("R0 must remain Development-only")
    if config["model_contract"].get("family") != "LOGISTIC_REGRESSION_ONLY":
        raise ValueError("only Logistic Regression is allowed")
    if config["grouped_evaluation"].get("outer_method") != "LEAVE_ONE_SOURCE_SESSION_OUT":
        raise ValueError("outer split must be leave-one-source-session-out")
    if config["grouped_evaluation"].get("inner_method") != (
        "LEAVE_ONE_SOURCE_SESSION_OUT_WITHIN_OUTER_TRAINING"
    ):
        raise ValueError("inner split must be grouped leave-one-session-out")
    thresholds = [float(value) for value in config["grouped_evaluation"]["threshold_grid"]]
    if thresholds != [value / 20.0 for value in range(1, 20)]:
        raise ValueError("threshold grid drifted")
    if config["comparators"] != list(ARM_IDS):
        raise ValueError("comparator order drifted")
    feature_names = config["feature_contract"]["feature_names"]
    if feature_names != list(FEATURE_NAMES):
        raise ValueError("feature allowlist drifted from the frozen 21 columns")
    ablation_names = [
        str(name)
        for names in config["feature_contract"][
            "diagnostic_ablation_blocks"
        ].values()
        for name in names
    ]
    if (
        len(ablation_names) != len(set(ablation_names))
        or set(ablation_names) != set(FEATURE_NAMES)
    ):
        raise ValueError("diagnostic ablation blocks must partition the allowlist")
    amendment = config["role_forward_amendment"]
    if (
        amendment.get("historical_r1_amendment_unchanged") is not True
        or amendment.get("freshness_restored") is not False
        or amendment.get("historical_r1_repair_or_rerun") is not False
    ):
        raise ValueError("consumed-role forward boundary drifted")
    forbidden_text = " ".join(str(value).lower() for value in config["forbidden"])
    for required in (
        "fresh_holdout_access",
        "random_component_or_frame_split",
        "outer_held_out_threshold_tuning",
        "model_family_search",
        "future_frame_feature",
        "truth_derived_inference_feature",
    ):
        if required not in forbidden_text:
            raise ValueError(f"missing forbidden boundary: {required}")


@dataclass
class BoundInputs:
    frame_rows: list[dict[str, Any]]
    component_rows: list[dict[str, Any]]
    atlas_rows: list[dict[str, Any]]
    manifest_rows: list[dict[str, Any]]
    historical_frame_rows: list[dict[str, Any]]
    historical_component_rows: list[dict[str, Any]]
    provenance: dict[str, Any]


def load_bound_inputs(repo_root: Path, config: dict[str, Any]) -> BoundInputs:
    validate_static_config(config)
    contract = config["input_contract"]
    frame_rows, frame_provenance = load_bound_jsonl(repo_root, contract["frames"])
    component_rows, component_provenance = load_bound_jsonl(
        repo_root, contract["components"]
    )
    atlas_rows, atlas_provenance = load_bound_jsonl(
        repo_root, contract["atlas_components"]
    )
    historical_frame_rows, historical_frame_provenance = load_bound_jsonl(
        repo_root, [contract["historical_conditional_frame_metrics"]]
    )
    historical_component_rows, historical_component_provenance = load_bound_jsonl(
        repo_root, [contract["historical_conditional_component_decisions"]]
    )
    manifest_spec = contract["canonical_manifest"]
    manifest_path = resolve(repo_root, manifest_spec["path"])
    observed_manifest_sha = sha256_file(manifest_path)
    if observed_manifest_sha != str(manifest_spec["sha256"]):
        raise ValueError("canonical manifest SHA mismatch")
    manifest_rows = read_jsonl(manifest_path)
    result = BoundInputs(
        frame_rows=frame_rows,
        component_rows=component_rows,
        atlas_rows=atlas_rows,
        manifest_rows=manifest_rows,
        historical_frame_rows=historical_frame_rows,
        historical_component_rows=historical_component_rows,
        provenance={
            "frames": frame_provenance,
            "components": component_provenance,
            "atlas_components": atlas_provenance,
            "canonical_manifest": {
                "path": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": observed_manifest_sha,
                "row_count": len(manifest_rows),
            },
            "historical_frame_metrics": historical_frame_provenance[0],
            "historical_component_decisions": historical_component_provenance[0],
        },
    )
    validate_input_membership(config, result)
    return result


def _unique_by(rows: Sequence[dict[str, Any]], field: str, label: str) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row[field])
        if key in mapping:
            raise ValueError(f"duplicate {label}: {key}")
        mapping[key] = row
    return mapping


def validate_input_membership(config: dict[str, Any], inputs: BoundInputs) -> None:
    contract = config["input_contract"]
    if len(inputs.frame_rows) != int(contract["expected_frame_count"]):
        raise ValueError("unexpected frame count")
    if len(inputs.component_rows) != int(contract["expected_component_count"]):
        raise ValueError("unexpected component count")
    if len(inputs.atlas_rows) != int(contract["expected_atlas_component_count"]):
        raise ValueError("unexpected Atlas component count")
    if len(inputs.historical_frame_rows) != int(contract["expected_frame_count"]):
        raise ValueError("unexpected historical frame count")
    if len(inputs.historical_component_rows) != int(
        contract["expected_component_count"]
    ):
        raise ValueError("unexpected historical component count")
    frames = _unique_by(inputs.frame_rows, "view_row_id", "view_row_id")
    components = _unique_by(inputs.component_rows, "component_id", "component_id")
    atlas = _unique_by(inputs.atlas_rows, "component_id", "Atlas component_id")
    historical_frames = _unique_by(
        inputs.historical_frame_rows, "view_row_id", "historical view_row_id"
    )
    historical_components = _unique_by(
        inputs.historical_component_rows,
        "component_id",
        "historical component_id",
    )
    if set(components) != set(atlas) or set(components) != set(historical_components):
        raise ValueError("component identity sets do not match")
    if set(frames) != set(historical_frames):
        raise ValueError("frame identity sets do not match historical evidence")
    session_counts = Counter(str(row["session_id"]) for row in inputs.frame_rows)
    expected_sessions = Counter(
        {
            str(key): int(value)
            for key, value in contract["expected_session_frame_counts"].items()
        }
    )
    if session_counts != expected_sessions:
        raise ValueError("source-session membership drifted")
    if len(session_counts) != int(contract["expected_session_count"]):
        raise ValueError("unexpected source-session count")
    roles_by_session: dict[str, str] = {}
    image_hashes: set[str] = set()
    for row in inputs.frame_rows:
        session_id = str(row["session_id"])
        role = str(row["rehearsal_role"])
        previous = roles_by_session.setdefault(session_id, role)
        if previous != role:
            raise ValueError("session spans multiple roles")
        image_sha = str(row["image_sha256"])
        if image_sha in image_hashes:
            raise ValueError("duplicate image identity")
        image_hashes.add(image_sha)
    role_counts = Counter(roles_by_session.values())
    if role_counts != Counter(
        {
            str(key): int(value)
            for key, value in contract["expected_role_session_counts"].items()
        }
    ):
        raise ValueError("role/session counts drifted")


def manifest_by_id(inputs: BoundInputs) -> dict[str, dict[str, Any]]:
    return _unique_by(inputs.manifest_rows, "id", "canonical manifest id")


def enriched_frame_order(inputs: BoundInputs) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    manifests = manifest_by_id(inputs)
    enriched: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for frame in inputs.frame_rows:
        view_row_id = str(frame["view_row_id"])
        manifest = manifests.get(view_row_id)
        if manifest is None:
            raise ValueError(f"missing canonical manifest row: {view_row_id}")
        for field in (
            "source_id",
            "session_id",
            "frame_id",
            "image_sha256",
            "canonical_mask_sha256",
        ):
            if frame[field] != manifest[field]:
                raise ValueError(f"frame/manifest mismatch: {view_row_id}/{field}")
        if str(frame["rehearsal_role"]) != str(manifest["role"]):
            raise ValueError(f"frame/manifest role mismatch: {view_row_id}")
        enriched.append((frame, manifest))
    enriched.sort(
        key=lambda pair: (
            str(pair[1]["session_id"]),
            str(pair[1]["sequence_id"]),
            int(pair[1]["source_capture_timestamp_ns"]),
            int(pair[1]["frame_id"]),
            str(pair[1]["id"]),
        )
    )
    return enriched


def raw_masks_for_frame(
    frame: dict[str, Any],
    candidate_classes: Sequence[str],
    shape: tuple[int, int],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if tuple(int(value) for value in frame["packed_masks"]["shape"]) != shape:
        raise ValueError(f"packed mask shape mismatch: {frame['view_row_id']}")
    detector = decode_packed_mask(frame["packed_masks"]["A"], shape)
    raw_class_masks = {
        class_name: decode_packed_mask(
            frame["packed_masks"][f"candidate_{class_name}"], shape
        )
        for class_name in candidate_classes
    }
    union = np.logical_or.reduce(list(raw_class_masks.values()))
    packed_union = decode_packed_mask(frame["packed_masks"]["B"], shape)
    if not np.array_equal(union, packed_union):
        raise ValueError(f"candidate union mismatch: {frame['view_row_id']}")
    if np.count_nonzero(detector & union):
        raise ValueError(f"candidate overlaps detector mask: {frame['view_row_id']}")
    if np.count_nonzero(
        raw_class_masks[candidate_classes[0]]
        & raw_class_masks[candidate_classes[1]]
    ):
        raise ValueError(f"candidate classes overlap: {frame['view_row_id']}")
    return detector, raw_class_masks


def component_id_for(
    frame: dict[str, Any],
    class_name: str,
    component: Component,
) -> str:
    return (
        f"{frame['source_id']}:{int(frame['frame_id'])}:"
        f"{class_name}:{int(component.index)}"
    )


def _component_centroid(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        raise ValueError("component mask is empty")
    return float(np.mean(xs)), float(np.mean(ys))


def _max_previous_iou(
    component: Component,
    previous: Sequence[Component],
) -> tuple[float, bool]:
    if not previous:
        return 0.0, True
    return max(mask_iou(component.mask, item.mask) for item in previous), False


def _same_footprint_age(
    component: Component,
    history: Sequence[np.ndarray],
    maximum: int,
) -> int:
    age = 1
    for prior in reversed(history[-(maximum - 1) :]):
        if np.count_nonzero(component.mask & prior) == 0:
            break
        age += 1
    return age


def _recent_flicker_count(
    component: Component,
    history: Sequence[np.ndarray],
    window: int,
) -> int:
    presence = [
        bool(np.count_nonzero(component.mask & prior))
        for prior in history[-window:]
    ]
    presence.append(True)
    return sum(left != right for left, right in zip(presence, presence[1:]))


def _intersects_rectangle(
    component: Component,
    *,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
) -> bool:
    local = component.mask[y_min:y_max, x_min:x_max]
    return bool(np.count_nonzero(local))


def build_component_table(
    *,
    config: dict[str, Any],
    inputs: BoundInputs,
) -> list[dict[str, Any]]:
    feature_contract = config["feature_contract"]
    feature_names = [str(value) for value in feature_contract["feature_names"]]
    shape = tuple(int(value) for value in config["analysis_shape"])
    if len(shape) != 2:
        raise ValueError("analysis_shape must contain height and width")
    height, width = shape
    diagonal = math.hypot(width, height)
    candidate_classes = [str(value) for value in config["candidate_classes"]]
    history_max = int(feature_contract["causal_history_max_observations"])
    flicker_window = int(feature_contract["recent_flicker_window_observations"])
    upper_end = int(math.ceil(height * float(feature_contract["upper_head_y_max_fraction"])))
    central_x = feature_contract["central_body_x_fraction"]
    central_x0 = int(math.floor(width * float(central_x[0])))
    central_x1 = int(math.ceil(width * float(central_x[1])))
    central_y0 = int(
        math.floor(height * float(feature_contract["central_body_y_min_fraction"]))
    )
    near_gap = float(feature_contract["near_yolo_union_gap_pixels"])

    ledger_by_id = _unique_by(inputs.component_rows, "component_id", "component_id")
    atlas_by_id = _unique_by(inputs.atlas_rows, "component_id", "Atlas component_id")
    histories: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    previous_components: dict[tuple[str, str, str], list[Component]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()

    for frame, manifest in enriched_frame_order(inputs):
        _, raw_class_masks = raw_masks_for_frame(frame, candidate_classes, shape)
        session_id = str(manifest["session_id"])
        sequence_id = str(manifest["sequence_id"])
        for class_name in candidate_classes:
            key = (session_id, sequence_id, class_name)
            current_components = connected_components(
                raw_class_masks[class_name],
                connectivity=int(feature_contract["connectivity"]),
            )
            history = histories[key]
            previous = previous_components[key]
            for component in current_components:
                component_id = component_id_for(frame, class_name, component)
                ledger = ledger_by_id.get(component_id)
                atlas = atlas_by_id.get(component_id)
                if ledger is None or atlas is None:
                    raise ValueError(f"missing component binding: {component_id}")
                observed.add(component_id)
                if (
                    str(ledger["class_name"]) != class_name
                    or int(ledger["area_pixels"]) != component.area
                    or list(ledger["bbox_xyxy"]) != list(component.bbox)
                ):
                    raise ValueError(f"raw component geometry drifted: {component_id}")
                if (
                    str(atlas["predicted_class"]) != class_name
                    or int(atlas["area_pixels"]) != component.area
                    or list(atlas["bbox_xyxy"]) != list(component.bbox)
                ):
                    raise ValueError(f"Atlas component geometry drifted: {component_id}")
                keep_label = int(bool(ledger["truth_intersects"]))
                truth_pixels = int(ledger["truth_intersection_pixels"])
                atlas_truth_pixels = int(atlas["residual_truth_intersection_pixels"])
                if (
                    keep_label != int(truth_pixels > 0)
                    or keep_label != int(atlas_truth_pixels > 0)
                    or bool(atlas["false_activation"]) == bool(keep_label)
                ):
                    raise ValueError(f"component label disagreement: {component_id}")

                bbox = tuple(int(value) for value in component.bbox)
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                if bbox_width <= 0 or bbox_height <= 0:
                    raise ValueError(f"invalid component bbox: {component_id}")
                centroid_x, centroid_y = _component_centroid(component.mask)
                atlas_centroid = [float(value) for value in atlas["centroid_xy"]]
                if not np.allclose(
                    np.asarray([centroid_x, centroid_y]),
                    np.asarray(atlas_centroid),
                    atol=1e-12,
                    rtol=0.0,
                ):
                    raise ValueError(f"component centroid drifted: {component_id}")
                confidence = ledger["top1_confidence_median"]
                confidence_missing = confidence is None
                confidence_value = 0.0 if confidence_missing else float(confidence)
                margin = ledger["top1_top2_margin_median"]
                margin_missing = margin is None
                margin_value = 0.0 if margin_missing else float(margin)
                for label, value in (
                    ("confidence", confidence_value),
                    ("margin", margin_value),
                ):
                    if not math.isfinite(value):
                        raise ValueError(f"nonfinite {label}: {component_id}")
                previous_iou, previous_missing = _max_previous_iou(
                    component, previous
                )
                footprint_age = _same_footprint_age(
                    component, history, history_max
                )
                flicker_count = _recent_flicker_count(
                    component, history, flicker_window
                )
                gap = ledger["nearest_yolo_box_distance_pixels"]
                gap_missing = gap is None
                gap_pixels = diagonal if gap_missing else float(gap)
                if not math.isfinite(gap_pixels) or gap_pixels < 0:
                    raise ValueError(f"invalid YOLO-union bbox gap: {component_id}")
                normalized_gap = min(gap_pixels / diagonal, 1.0)
                near = int((not gap_missing) and gap_pixels <= near_gap)
                features = {
                    "predicted_class_is_obstacle": float(class_name == "obstacle"),
                    "log1p_area_pixels": float(math.log1p(component.area)),
                    "bbox_width_fraction": float(bbox_width / width),
                    "bbox_height_fraction": float(bbox_height / height),
                    "log_bbox_aspect_ratio": float(math.log(bbox_width / bbox_height)),
                    "centroid_x_fraction": float(centroid_x / width),
                    "centroid_y_fraction": float(centroid_y / height),
                    "intersects_upper_head_band": float(
                        _intersects_rectangle(
                            component,
                            x_min=0,
                            x_max=width,
                            y_min=0,
                            y_max=upper_end,
                        )
                    ),
                    "intersects_central_body_corridor": float(
                        _intersects_rectangle(
                            component,
                            x_min=central_x0,
                            x_max=central_x1,
                            y_min=central_y0,
                            y_max=height,
                        )
                    ),
                    "top1_confidence_median": confidence_value,
                    "top1_confidence_missing": float(confidence_missing),
                    "top1_top2_margin_median": margin_value,
                    "top1_top2_margin_missing": float(margin_missing),
                    "causal_previous_component_iou_max": float(previous_iou),
                    "causal_previous_component_iou_missing": float(previous_missing),
                    "causal_same_footprint_age_5": float(footprint_age),
                    "recent_flicker_count_3": float(flicker_count),
                    "nearest_yolo_union_bbox_gap_fraction": float(normalized_gap),
                    "nearest_yolo_union_bbox_gap_missing": float(gap_missing),
                    "near_yolo_union_gap_le_3": float(near),
                    "obstacle_x_near_yolo_union": float(
                        near and class_name == "obstacle"
                    ),
                }
                if list(features) != feature_names:
                    raise ValueError("feature implementation order drifted")
                if any(not math.isfinite(value) for value in features.values()):
                    raise ValueError(f"nonfinite feature: {component_id}")
                rows.append(
                    {
                        "schema_version": TABLE_SCHEMA_VERSION,
                        "protocol_id": PROTOCOL_ID,
                        "identity": {
                            "component_id": component_id,
                            "view_row_id": str(frame["view_row_id"]),
                            "source_id": str(frame["source_id"]),
                            "session_id": session_id,
                            "sequence_id": sequence_id,
                            "frame_id": int(frame["frame_id"]),
                            "source_capture_timestamp_ns": int(
                                manifest["source_capture_timestamp_ns"]
                            ),
                            "role": str(frame["rehearsal_role"]),
                            "predicted_class": class_name,
                            "component_index": int(component.index),
                        },
                        "features": features,
                        "target": {
                            "keep_label": keep_label,
                            "same_class_residual_truth_intersection_pixels": truth_pixels,
                        },
                        "diagnostic": {
                            "atlas_false_activation": bool(atlas["false_activation"]),
                            "atlas_mechanism_tags": sorted(
                                str(value) for value in atlas["mechanism_tags"]
                            ),
                            "raw_area_pixels": int(component.area),
                            "raw_bbox_xyxy": list(component.bbox),
                        },
                    }
                )
            history.append(raw_class_masks[class_name].copy())
            if len(history) > history_max:
                del history[:-history_max]
            previous_components[key] = current_components
    if observed != set(ledger_by_id):
        missing = sorted(set(ledger_by_id) - observed)
        raise ValueError(f"unmatched raw components: {missing[:3]}")
    if len(rows) != int(config["input_contract"]["expected_component_count"]):
        raise ValueError("prepared row count mismatch")
    return rows


def validate_table_contract(
    config: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    feature_names = [str(value) for value in config["feature_contract"]["feature_names"]]
    forbidden = {
        str(value) for value in config["feature_contract"]["forbidden_model_fields"]
    }
    component_ids: set[str] = set()
    labels = Counter()
    class_labels: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if row.get("schema_version") != TABLE_SCHEMA_VERSION:
            raise ValueError("component table schema drifted")
        if row.get("protocol_id") != PROTOCOL_ID:
            raise ValueError("component table protocol drifted")
        features = row.get("features")
        if (
            not isinstance(features, dict)
            or len(features) != len(feature_names)
            or set(features) != set(feature_names)
        ):
            raise ValueError("component table feature allowlist drifted")
        if forbidden & set(features):
            raise ValueError("forbidden field entered feature namespace")
        values = np.asarray([features[name] for name in feature_names], dtype=np.float64)
        if values.shape != (len(feature_names),) or not np.isfinite(values).all():
            raise ValueError("feature vector is invalid")
        identity = row["identity"]
        component_id = str(identity["component_id"])
        if component_id in component_ids:
            raise ValueError(f"duplicate table component_id: {component_id}")
        component_ids.add(component_id)
        label = int(row["target"]["keep_label"])
        if label not in {0, 1}:
            raise ValueError("target label must be binary")
        labels["keep" if label else "reject"] += 1
        class_name = str(identity["predicted_class"])
        class_labels[class_name]["total"] += 1
        class_labels[class_name]["keep"] += label
    expected = config["input_contract"]["expected_label_counts"]
    if labels != Counter({key: int(value) for key, value in expected.items()}):
        raise ValueError(f"label distribution drifted: {labels}")
    expected_class = config["input_contract"]["expected_class_label_counts"]
    for class_name, counts in expected_class.items():
        if class_labels[class_name] != Counter(
            {key: int(value) for key, value in counts.items()}
        ):
            raise ValueError(f"class label distribution drifted: {class_name}")
    return {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "label_counts": dict(labels),
        "class_label_counts": {
            key: dict(value) for key, value in sorted(class_labels.items())
        },
        "unique_component_ids": len(component_ids),
    }


def feature_matrix(
    config: dict[str, Any],
    rows: Sequence[dict[str, Any]],
) -> np.ndarray:
    names = [str(value) for value in config["feature_contract"]["feature_names"]]
    matrix = np.asarray(
        [[float(row["features"][name]) for name in names] for row in rows],
        dtype=np.float64,
    )
    if matrix.shape != (len(rows), len(names)) or not np.isfinite(matrix).all():
        raise ValueError("feature matrix is invalid")
    return matrix


@dataclass
class FrameComponent:
    component_id: str
    predicted_class: str
    component: Component
    table_row: dict[str, Any]


@dataclass
class FrameContext:
    view_row_id: str
    source_id: str
    session_id: str
    sequence_id: str
    role: str
    frame_id: int
    timestamp_ns: int
    detector_mask: np.ndarray
    raw_class_masks: dict[str, np.ndarray]
    components: list[FrameComponent]
    full_truth_hazard: np.ndarray
    residual_truth: np.ndarray
    class_truth: dict[str, np.ndarray]


def build_frame_contexts(
    *,
    repo_root: Path,
    config: dict[str, Any],
    inputs: BoundInputs,
    table_rows: Sequence[dict[str, Any]],
) -> list[FrameContext]:
    shape = tuple(int(value) for value in config["analysis_shape"])
    candidate_classes = [str(value) for value in config["candidate_classes"]]
    table_by_id = {
        str(row["identity"]["component_id"]): row for row in table_rows
    }
    if len(table_by_id) != len(table_rows):
        raise ValueError("duplicate component table identity")
    truth_names = {
        int(key): str(value) for key, value in config["truth_classes"].items()
    }
    truth_ids_by_name = {value: key for key, value in truth_names.items()}
    hazard_ids = np.asarray(config["hazard_truth_ids"], dtype=np.uint8)
    view_root = resolve(repo_root, config["input_contract"]["canonical_view_root"])
    contexts: list[FrameContext] = []
    observed: set[str] = set()
    for frame, manifest in enriched_frame_order(inputs):
        detector, raw_class_masks = raw_masks_for_frame(
            frame, candidate_classes, shape
        )
        truth_path = view_root / str(manifest["canonical_mask_path"])
        if sha256_file(truth_path) != str(frame["canonical_mask_sha256"]):
            raise ValueError(f"canonical truth SHA mismatch: {frame['view_row_id']}")
        truth = np.asarray(Image.open(truth_path), dtype=np.uint8)
        if truth.shape != shape or np.any(truth > max(truth_names)):
            raise ValueError(f"canonical truth contract mismatch: {frame['view_row_id']}")
        full_truth = np.isin(truth, hazard_ids)
        residual_truth = full_truth & ~detector
        class_truth = {
            name: (truth == truth_ids_by_name[name]) & ~detector
            for name in candidate_classes
        }
        frame_components: list[FrameComponent] = []
        for class_name in candidate_classes:
            for component in connected_components(
                raw_class_masks[class_name],
                connectivity=int(config["feature_contract"]["connectivity"]),
            ):
                component_id = component_id_for(frame, class_name, component)
                table = table_by_id.get(component_id)
                if table is None:
                    raise ValueError(f"missing table row: {component_id}")
                observed.add(component_id)
                frame_components.append(
                    FrameComponent(
                        component_id=component_id,
                        predicted_class=class_name,
                        component=component,
                        table_row=table,
                    )
                )
        contexts.append(
            FrameContext(
                view_row_id=str(frame["view_row_id"]),
                source_id=str(frame["source_id"]),
                session_id=str(frame["session_id"]),
                sequence_id=str(manifest["sequence_id"]),
                role=str(frame["rehearsal_role"]),
                frame_id=int(frame["frame_id"]),
                timestamp_ns=int(manifest["source_capture_timestamp_ns"]),
                detector_mask=detector,
                raw_class_masks=raw_class_masks,
                components=frame_components,
                full_truth_hazard=full_truth,
                residual_truth=residual_truth,
                class_truth=class_truth,
            )
        )
    if observed != set(table_by_id):
        raise ValueError("component table/frame context membership mismatch")
    return contexts


def _empty_class_masks(
    candidate_classes: Sequence[str],
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    return {name: np.zeros(shape, dtype=bool) for name in candidate_classes}


def build_reference_masks(
    *,
    config: dict[str, Any],
    contexts: Sequence[FrameContext],
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    candidate_classes = [str(value) for value in config["candidate_classes"]]
    shape = tuple(int(value) for value in config["analysis_shape"])
    histories: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    union_histories: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    upper = upper_head_band(
        shape, float(config["feature_contract"]["upper_head_y_max_fraction"])
    )
    outputs: dict[str, dict[str, dict[str, np.ndarray]]] = {
        arm_id: {} for arm_id in (RAW_ARM_ID, CAUSAL_ARM_ID, CONFIDENCE_ARM_ID, HISTORICAL_ARM_ID)
    }
    for context in contexts:
        raw = {
            name: context.raw_class_masks[name].copy() for name in candidate_classes
        }
        outputs[RAW_ARM_ID][context.view_row_id] = raw
        union = np.logical_or.reduce(list(raw.values()))
        union_history = union_histories[(context.session_id, context.sequence_id)]
        union_causal = causal_two_of_three(
            union,
            union_history[-1] if len(union_history) >= 1 else None,
            union_history[-2] if len(union_history) >= 2 else None,
        )
        outputs[CAUSAL_ARM_ID][context.view_row_id] = {
            name: raw[name] & union_causal for name in candidate_classes
        }
        confidence_masks = _empty_class_masks(candidate_classes, shape)
        historical_masks = _empty_class_masks(candidate_classes, shape)
        same_class_causal: dict[str, np.ndarray] = {}
        for class_name in candidate_classes:
            history = histories[(context.session_id, context.sequence_id, class_name)]
            same_class_causal[class_name] = causal_two_of_three(
                raw[class_name],
                history[-1] if len(history) >= 1 else None,
                history[-2] if len(history) >= 2 else None,
            )
        for item in context.components:
            confidence = float(item.table_row["features"]["top1_confidence_median"])
            confidence_missing = bool(
                item.table_row["features"]["top1_confidence_missing"]
            )
            if not confidence_missing and confidence >= 0.65:
                confidence_masks[item.predicted_class] |= item.component.mask
            kept, _ = apply_frozen_candidate_to_component(
                candidate_id=PRIMARY_CANDIDATE_ID,
                predicted_class=item.predicted_class,
                component_mask=item.component.mask,
                same_class_causal_mask=same_class_causal[item.predicted_class],
                confidence_median=None if confidence_missing else confidence,
                raw_area_pixels=item.component.area,
                upper_band_mask=upper,
                confidence_minimum=0.65,
                small_fragment_max_area_pixels=63,
            )
            historical_masks[item.predicted_class] |= kept
        outputs[CONFIDENCE_ARM_ID][context.view_row_id] = confidence_masks
        outputs[HISTORICAL_ARM_ID][context.view_row_id] = historical_masks
        for class_name in candidate_classes:
            history = histories[(context.session_id, context.sequence_id, class_name)]
            history.append(raw[class_name].copy())
            if len(history) > 2:
                del history[:-2]
        union_history.append(union.copy())
        if len(union_history) > 2:
            del union_history[:-2]
    return outputs


def masks_from_component_decisions(
    *,
    config: dict[str, Any],
    contexts: Sequence[FrameContext],
    keep_by_component_id: Mapping[str, bool],
) -> dict[str, dict[str, np.ndarray]]:
    candidate_classes = [str(value) for value in config["candidate_classes"]]
    shape = tuple(int(value) for value in config["analysis_shape"])
    masks: dict[str, dict[str, np.ndarray]] = {}
    observed: set[str] = set()
    for context in contexts:
        class_masks = _empty_class_masks(candidate_classes, shape)
        for item in context.components:
            if item.component_id not in keep_by_component_id:
                raise ValueError(f"missing component decision: {item.component_id}")
            observed.add(item.component_id)
            if bool(keep_by_component_id[item.component_id]):
                class_masks[item.predicted_class] |= item.component.mask
        masks[context.view_row_id] = class_masks
    if observed != set(keep_by_component_id):
        extra = sorted(set(keep_by_component_id) - observed)
        raise ValueError(f"unexpected component decisions: {extra[:3]}")
    return masks


def compact_pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    value = pixel_metrics(predicted, truth)
    return {
        key: value[key]
        for key in (
            "tp",
            "fp",
            "fn",
            "tn",
            "predicted_pixels",
            "truth_pixels",
            "precision",
            "recall",
            "iou",
            "f1",
            "false_positive_area_fraction",
        )
    }


def frame_arm_metrics(
    context: FrameContext,
    class_masks: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    union = np.logical_or.reduce(list(class_masks.values()))
    if np.count_nonzero(union & context.detector_mask):
        raise ValueError(f"candidate recreated detector pixels: {context.view_row_id}")
    arm_c = context.detector_mask | union
    class_values: dict[str, Any] = {}
    for class_name, mask in class_masks.items():
        class_values[class_name] = {
            "pixel": compact_pixel_metrics(mask, context.class_truth[class_name]),
            "component": component_metrics(mask, context.class_truth[class_name]),
        }
    return {
        "candidate": compact_pixel_metrics(union, context.residual_truth),
        "A": compact_pixel_metrics(
            context.detector_mask, context.full_truth_hazard
        ),
        "C": compact_pixel_metrics(arm_c, context.full_truth_hazard),
        "components": component_metrics(union, context.residual_truth),
        "classes": class_values,
    }


def build_frame_metric_rows(
    *,
    contexts: Sequence[FrameContext],
    arm_masks: Mapping[str, Mapping[str, Mapping[str, np.ndarray]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context in contexts:
        rows.append(
            {
                "schema_version": FRAME_SCHEMA_VERSION,
                "protocol_id": PROTOCOL_ID,
                "view_row_id": context.view_row_id,
                "source_id": context.source_id,
                "session_id": context.session_id,
                "sequence_id": context.sequence_id,
                "role": context.role,
                "frame_id": context.frame_id,
                "source_capture_timestamp_ns": context.timestamp_ns,
                "arms": {
                    arm_id: frame_arm_metrics(
                        context, arm_masks[arm_id][context.view_row_id]
                    )
                    for arm_id in arm_masks
                },
            }
        )
    return rows


def _aggregate_components(
    values: Sequence[dict[str, Any]],
    *,
    frame_count: int,
) -> dict[str, Any]:
    predicted = sum(int(value["predicted_component_count"]) for value in values)
    truth = sum(int(value["truth_component_count"]) for value in values)
    hit_predicted = sum(
        int(value["hit_predicted_component_count"]) for value in values
    )
    hit_truth = sum(int(value["hit_truth_component_count"]) for value in values)
    false_count = sum(
        int(value["false_activation_component_count"]) for value in values
    )
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": (
            float(hit_predicted / predicted)
            if predicted
            else (1.0 if truth == 0 else None)
        ),
        "component_recall": (
            float(hit_truth / truth)
            if truth
            else (1.0 if predicted == 0 else None)
        ),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / frame_count),
    }


def aggregate_arm(
    frame_rows: Sequence[dict[str, Any]],
    arm_id: str,
    candidate_classes: Sequence[str],
) -> dict[str, Any]:
    if not frame_rows:
        raise ValueError("cannot aggregate zero frame rows")
    frame_count = len(frame_rows)
    result: dict[str, Any] = {
        name: aggregate_confusion(row["arms"][arm_id][name] for row in frame_rows)
        for name in ("candidate", "A", "C")
    }
    result["components"] = _aggregate_components(
        [row["arms"][arm_id]["components"] for row in frame_rows],
        frame_count=frame_count,
    )
    result["classes"] = {
        class_name: {
            "pixel": aggregate_confusion(
                row["arms"][arm_id]["classes"][class_name]["pixel"]
                for row in frame_rows
            ),
            "components": _aggregate_components(
                [
                    row["arms"][arm_id]["classes"][class_name]["component"]
                    for row in frame_rows
                ],
                frame_count=frame_count,
            ),
        }
        for class_name in candidate_classes
    }
    result["frame_count"] = frame_count
    result["delta_recall_C_minus_A"] = float(
        result["C"]["recall"] - result["A"]["recall"]
    )
    result["delta_false_positive_area_fraction_C_minus_A"] = float(
        result["C"]["false_positive_area_fraction"]
        - result["A"]["false_positive_area_fraction"]
    )
    return result


def aggregate_arm_report(
    frame_rows: Sequence[dict[str, Any]],
    arm_id: str,
    candidate_classes: Sequence[str],
) -> dict[str, Any]:
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        by_session[str(row["session_id"])].append(row)
        by_role[str(row["role"])].append(row)
    return {
        "overall": aggregate_arm(frame_rows, arm_id, candidate_classes),
        "by_session": {
            key: aggregate_arm(value, arm_id, candidate_classes)
            for key, value in sorted(by_session.items())
        },
        "by_role": {
            key: aggregate_arm(value, arm_id, candidate_classes)
            for key, value in sorted(by_role.items())
        },
    }


def safe_ratio(numerator: float, denominator: float, label: str) -> float:
    if denominator <= 0:
        raise ValueError(f"non-positive denominator: {label}")
    return float(numerator / denominator)


def utility_values(
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    cand = candidate["overall"]
    base = baseline["overall"]
    session_retentions = {
        session_id: safe_ratio(
            metrics["candidate"]["tp"],
            baseline["by_session"][session_id]["candidate"]["tp"],
            f"{session_id} baseline TP",
        )
        for session_id, metrics in candidate["by_session"].items()
    }
    return {
        "fp_pixel_reduction": 1.0
        - safe_ratio(cand["candidate"]["fp"], base["candidate"]["fp"], "baseline FP"),
        "overall_recall_retention": safe_ratio(
            cand["candidate"]["tp"], base["candidate"]["tp"], "baseline TP"
        ),
        "minimum_session_recall_retention": min(session_retentions.values()),
        "minimum_session_id": min(
            session_retentions,
            key=lambda value: (session_retentions[value], value),
        ),
        "session_recall_retentions": session_retentions,
        "boundary_recall_retention": safe_ratio(
            cand["classes"]["boundary_step_curb"]["pixel"]["tp"],
            base["classes"]["boundary_step_curb"]["pixel"]["tp"],
            "baseline boundary TP",
        ),
        "obstacle_recall_retention": safe_ratio(
            cand["classes"]["obstacle"]["pixel"]["tp"],
            base["classes"]["obstacle"]["pixel"]["tp"],
            "baseline obstacle TP",
        ),
        "delta_recall_C_minus_A": cand["delta_recall_C_minus_A"],
        "delta_false_positive_area_fraction_C_minus_A": cand[
            "delta_false_positive_area_fraction_C_minus_A"
        ],
        "candidate_component_recall": cand["components"]["component_recall"],
        "false_activation_components_per_frame": cand["components"][
            "false_activation_components_per_frame"
        ],
    }


LOWER_GATE_FIELDS = {
    "fp_pixel_reduction": "minimum_fp_pixel_reduction",
    "overall_recall_retention": "minimum_overall_recall_retention",
    "minimum_session_recall_retention": "minimum_session_recall_retention",
    "boundary_recall_retention": "minimum_boundary_recall_retention",
    "obstacle_recall_retention": "minimum_obstacle_recall_retention",
    "delta_recall_C_minus_A": "minimum_delta_recall_C_minus_A",
    "candidate_component_recall": "minimum_candidate_component_recall",
}
UPPER_GATE_FIELDS = {
    "delta_false_positive_area_fraction_C_minus_A": (
        "maximum_delta_false_positive_area_fraction_C_minus_A"
    ),
    "false_activation_components_per_frame": (
        "maximum_false_activation_components_per_frame"
    ),
}


def gate_checks(
    values: dict[str, Any],
    gates: Mapping[str, float],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for field, gate_field in LOWER_GATE_FIELDS.items():
        threshold = float(gates[gate_field])
        value = float(values[field])
        checks[field] = {
            "operator": ">=",
            "threshold": threshold,
            "value": value,
            "passed": value >= threshold,
        }
    for field, gate_field in UPPER_GATE_FIELDS.items():
        threshold = float(gates[gate_field])
        value = float(values[field])
        checks[field] = {
            "operator": "<=",
            "threshold": threshold,
            "value": value,
            "passed": value <= threshold,
        }
    return {
        "checks": checks,
        "passed_count": sum(bool(value["passed"]) for value in checks.values()),
        "all_passed": all(bool(value["passed"]) for value in checks.values()),
    }


def normalized_gate_margins(
    values: dict[str, Any],
    gates: Mapping[str, float],
) -> dict[str, float]:
    margins: dict[str, float] = {}
    for field, gate_field in LOWER_GATE_FIELDS.items():
        threshold = float(gates[gate_field])
        margins[field] = float((float(values[field]) - threshold) / threshold)
    for field, gate_field in UPPER_GATE_FIELDS.items():
        threshold = float(gates[gate_field])
        margins[field] = float((threshold - float(values[field])) / threshold)
    return margins


def sigmoid_scores(
    matrix: np.ndarray,
    *,
    mean: np.ndarray,
    scale: np.ndarray,
    coefficients: np.ndarray,
    intercept: float,
) -> np.ndarray:
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    standardized = (matrix - mean) / scale
    logits = standardized @ coefficients + float(intercept)
    logits = np.clip(logits, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-logits))


def stable_high_confidence_diagnostic(
    *,
    table_rows: Sequence[dict[str, Any]],
    keep_by_component_id: Mapping[str, bool],
) -> dict[str, Any]:
    retained_false_area = 0
    stable_area = 0
    stable_sessions: set[str] = set()
    retained_false_count = 0
    stable_count = 0
    for row in table_rows:
        component_id = str(row["identity"]["component_id"])
        if not keep_by_component_id.get(component_id, False):
            continue
        if int(row["target"]["keep_label"]) != 0:
            continue
        area = int(row["diagnostic"]["raw_area_pixels"])
        retained_false_area += area
        retained_false_count += 1
        if "STABLE_HIGH_CONFIDENCE_ERROR" in row["diagnostic"][
            "atlas_mechanism_tags"
        ]:
            stable_area += area
            stable_count += 1
            stable_sessions.add(str(row["identity"]["session_id"]))
    return {
        "retained_false_component_count": retained_false_count,
        "retained_false_component_area_pixels": retained_false_area,
        "stable_high_confidence_component_count": stable_count,
        "stable_high_confidence_component_area_pixels": stable_area,
        "stable_high_confidence_false_area_share": (
            float(stable_area / retained_false_area)
            if retained_false_area
            else 0.0
        ),
        "stable_high_confidence_session_count": len(stable_sessions),
        "stable_high_confidence_session_ids": sorted(stable_sessions),
        "diagnostic_only_truth_and_future_fields_not_model_features": True,
    }


def near_miss_eligible(
    *,
    values: dict[str, Any],
    utility_gate_result: dict[str, Any],
    engineering_passed: bool,
    stable_diagnostic: dict[str, Any],
    rule: dict[str, Any],
) -> tuple[bool, dict[str, Any]]:
    failed = [
        field
        for field, check in utility_gate_result["checks"].items()
        if not bool(check["passed"])
    ]
    tolerance_pass = True
    for field, minimum in rule["minimum_values_for_lower_bound_gates"].items():
        if field in failed and float(values[field]) < float(minimum):
            tolerance_pass = False
    for field, maximum in rule["maximum_values_for_upper_bound_gates"].items():
        if field in failed and float(values[field]) > float(maximum):
            tolerance_pass = False
    stable_pass = (
        float(stable_diagnostic["stable_high_confidence_false_area_share"])
        >= float(
            rule[
                "minimum_retained_false_area_share_tagged_stable_high_confidence"
            ]
        )
        and int(stable_diagnostic["stable_high_confidence_session_count"])
        >= int(rule["minimum_session_count_with_retained_stable_high_confidence_error"])
    )
    eligible = (
        engineering_passed
        and len(failed) <= int(rule["maximum_failed_utility_gates"])
        and len(failed) > 0
        and tolerance_pass
        and stable_pass
    )
    return eligible, {
        "failed_utility_gate_ids": failed,
        "failed_utility_gate_count": len(failed),
        "engineering_gates_passed": engineering_passed,
        "failed_gates_within_tolerance": tolerance_pass,
        "stable_high_confidence_rule_passed": stable_pass,
        "eligible": eligible,
        "successor_authority": rule["successor_authority"],
    }
