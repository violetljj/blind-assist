"""Evaluate frozen conditional segmentation gates on consumed Development evidence."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import shutil
import subprocess
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    aggregate_confusion,
    component_metrics,
    connected_components,
    pixel_metrics,
)


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_CONDITIONAL_GATING_R0"
SCHEMA_VERSION = "blindassist.dual_loop_segmentation_conditional_gating.result.v1"
BASELINE_ID = "BASELINE_UNFILTERED"
REFERENCE_CAUSAL_ID = "REFERENCE_CAUSAL_2_OF_3_UNION"
REFERENCE_CONFIDENCE_ID = "REFERENCE_CONFIDENCE_GE_0_65"
REFERENCE_IDS = (REFERENCE_CAUSAL_ID, REFERENCE_CONFIDENCE_ID)


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
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


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


def decode_packed_mask(encoded: str, shape: tuple[int, int]) -> np.ndarray:
    packed = np.frombuffer(base64.b64decode(encoded, validate=True), dtype=np.uint8)
    expected_pixels = int(np.prod(shape))
    unpacked = np.unpackbits(packed, bitorder="big")
    if unpacked.size < expected_pixels or unpacked.size - expected_pixels >= 8:
        raise ValueError("packed mask length does not match analysis shape")
    return unpacked[:expected_pixels].reshape(shape).astype(bool)


def encode_packed_mask(mask: np.ndarray) -> str:
    flat = np.asarray(mask, dtype=bool).reshape(-1).astype(np.uint8)
    return base64.b64encode(np.packbits(flat, bitorder="big").tobytes()).decode("ascii")


def causal_two_of_three(
    current: np.ndarray,
    previous: np.ndarray | None,
    previous_previous: np.ndarray | None,
) -> np.ndarray:
    """Keep a current pixel only when raw history contains that pixel."""

    value = np.asarray(current, dtype=bool)
    prior = np.zeros_like(value) if previous is None else np.asarray(previous, dtype=bool)
    prior_prior = (
        np.zeros_like(value)
        if previous_previous is None
        else np.asarray(previous_previous, dtype=bool)
    )
    return value & (prior | prior_prior)


def upper_head_band(shape: tuple[int, int], fraction: float) -> np.ndarray:
    if not 0.0 < fraction < 1.0:
        raise ValueError("upper head fraction must be inside (0, 1)")
    result = np.zeros(shape, dtype=bool)
    result[: int(math.ceil(shape[0] * fraction)), :] = True
    return result


def apply_frozen_candidate_to_component(
    *,
    candidate_id: str,
    predicted_class: str,
    component_mask: np.ndarray,
    same_class_causal_mask: np.ndarray,
    confidence_median: float | None,
    raw_area_pixels: int,
    upper_band_mask: np.ndarray,
    confidence_minimum: float,
    small_fragment_max_area_pixels: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply a gate using runtime-observable raw-component evidence only."""

    raw = np.asarray(component_mask, dtype=bool)
    causal = raw & np.asarray(same_class_causal_mask, dtype=bool)
    if raw.ndim != 2 or causal.shape != raw.shape or upper_band_mask.shape != raw.shape:
        raise ValueError("component and support masks must be equal two-dimensional shapes")
    if int(np.count_nonzero(raw)) != int(raw_area_pixels):
        raise ValueError("raw_area_pixels must describe the unfiltered component")
    if predicted_class not in {"boundary_step_curb", "obstacle"}:
        raise ValueError(f"unexpected predicted class: {predicted_class}")
    confidence_known = confidence_median is not None and math.isfinite(float(confidence_median))
    low_confidence = bool(
        confidence_known and float(confidence_median) < float(confidence_minimum)
    )
    small_fragment = int(raw_area_pixels) <= int(small_fragment_max_area_pixels)
    intersects_upper = bool(np.count_nonzero(raw & upper_band_mask))
    noncausal = raw & ~causal

    if candidate_id == "CLASS_CONDITIONED_MULTI_NEGATIVE":
        if predicted_class == "obstacle":
            proxy = small_fragment or intersects_upper
            reject_pixels = noncausal if low_confidence and proxy else np.zeros_like(raw)
            kept = raw & ~reject_pixels
            reason_bits = [
                "OBSTACLE_MULTI_NEGATIVE_PIXEL_REJECTION"
                if np.count_nonzero(reject_pixels)
                else "OBSTACLE_INSUFFICIENT_COMBINED_NEGATIVE_EVIDENCE"
            ]
        else:
            reject = low_confidence and small_fragment
            kept = np.zeros_like(raw) if reject else raw.copy()
            reason_bits = (
                ["BOUNDARY_LOW_CONFIDENCE_SMALL_FRAGMENT_REJECT"]
                if reject
                else ["BOUNDARY_PROTECTED_FROM_TEMPORAL_AND_UPPER_REJECTION"]
            )
    else:
        raise ValueError(f"unknown frozen candidate: {candidate_id}")

    kept_pixels = int(np.count_nonzero(kept))
    raw_pixels = int(raw_area_pixels)
    fragments = connected_components(kept, connectivity=8) if kept_pixels else []
    action = (
        "REJECT"
        if kept_pixels == 0
        else "KEEP"
        if kept_pixels == raw_pixels
        else "PARTIAL"
    )
    evidence = {
        "confidence_known": confidence_known,
        "low_confidence": low_confidence,
        "small_fragment": small_fragment,
        "intersects_upper_head_band": intersects_upper,
        "raw_pixels": raw_pixels,
        "causal_supported_pixels": int(np.count_nonzero(causal)),
        "noncausal_pixels": int(np.count_nonzero(noncausal)),
        "kept_pixels": kept_pixels,
        "rejected_pixels": raw_pixels - kept_pixels,
        "post_fragment_count": len(fragments),
        "action": action,
        "reason_bits": reason_bits,
    }
    return kept, evidence


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            )
            handle.write("\n")


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _metric_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "false_positive_reduction": _safe_ratio(
            int(baseline["fp"]) - int(candidate["fp"]),
            int(baseline["fp"]),
        ),
        "recall_retention": _safe_ratio(
            int(candidate["tp"]),
            int(baseline["tp"]),
        ),
    }


def _compact_pixel_metrics(predicted: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    metric = pixel_metrics(predicted, truth)
    return {
        key: metric[key]
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
        )
    }


def _frame_arm_metrics(
    class_masks: dict[str, np.ndarray],
    residual_truth: np.ndarray,
    class_truth: dict[str, np.ndarray],
) -> dict[str, Any]:
    union = np.logical_or.reduce(list(class_masks.values()))
    any_components = component_metrics(union, residual_truth)
    strict_false = 0
    post_components = 0
    class_metrics: dict[str, Any] = {}
    for class_name, mask in class_masks.items():
        metrics = component_metrics(mask, class_truth[class_name])
        strict_false += int(metrics["false_activation_component_count"])
        post_components += int(metrics["predicted_component_count"])
        class_metrics[class_name] = {
            "pixel": _compact_pixel_metrics(mask, class_truth[class_name]),
            "component": metrics,
        }
    return {
        "pixel": _compact_pixel_metrics(union, residual_truth),
        "post_component_count": post_components,
        "any_hazard_false_component_count": int(
            any_components["false_activation_component_count"]
        ),
        "class_strict_false_component_count": strict_false,
        "classes": class_metrics,
    }


def _aggregate_confusion_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot aggregate empty metric rows")
    return aggregate_confusion(rows)


def _aggregate_arm(
    frame_rows: Sequence[dict[str, Any]],
    arm_id: str,
    candidate_classes: Sequence[str],
) -> dict[str, Any]:
    pixel_rows = [row["arms"][arm_id]["pixel"] for row in frame_rows]
    aggregate = _aggregate_confusion_rows(pixel_rows)
    frame_count = len(frame_rows)
    post_components = sum(
        int(row["arms"][arm_id]["post_component_count"]) for row in frame_rows
    )
    any_false = sum(
        int(row["arms"][arm_id]["any_hazard_false_component_count"])
        for row in frame_rows
    )
    strict_false = sum(
        int(row["arms"][arm_id]["class_strict_false_component_count"])
        for row in frame_rows
    )
    classes: dict[str, Any] = {}
    for class_name in candidate_classes:
        class_rows = [
            row["arms"][arm_id]["classes"][class_name]["pixel"] for row in frame_rows
        ]
        class_components = [
            row["arms"][arm_id]["classes"][class_name]["component"] for row in frame_rows
        ]
        class_post = sum(
            int(metric["predicted_component_count"]) for metric in class_components
        )
        class_false = sum(
            int(metric["false_activation_component_count"]) for metric in class_components
        )
        classes[class_name] = {
            "pixel": _aggregate_confusion_rows(class_rows),
            "post_component_count": class_post,
            "class_strict_false_component_count": class_false,
            "post_components_per_frame": float(class_post / frame_count),
            "class_strict_false_components_per_frame": float(class_false / frame_count),
        }
    return {
        "frame_count": frame_count,
        "pixel": aggregate,
        "post_component_count": post_components,
        "any_hazard_false_component_count": any_false,
        "class_strict_false_component_count": strict_false,
        "post_components_per_frame": float(post_components / frame_count),
        "any_hazard_false_components_per_frame": float(any_false / frame_count),
        "class_strict_false_components_per_frame": float(strict_false / frame_count),
        "classes": classes,
    }


def _add_comparison(
    aggregate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    value = dict(aggregate)
    value["comparison_to_baseline"] = _metric_comparison(
        aggregate["pixel"],
        baseline["pixel"],
    )
    classes: dict[str, Any] = {}
    for class_name, metric in aggregate["classes"].items():
        class_value = dict(metric)
        class_value["comparison_to_baseline"] = _metric_comparison(
            metric["pixel"],
            baseline["classes"][class_name]["pixel"],
        )
        classes[class_name] = class_value
    value["classes"] = classes
    return value


def _group_rows(
    frame_rows: Sequence[dict[str, Any]],
    field: str,
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frame_rows:
        groups[str(row[field])].append(row)
    return dict(sorted(groups.items()))


def _aggregate_arm_report(
    *,
    frame_rows: Sequence[dict[str, Any]],
    arm_id: str,
    candidate_classes: Sequence[str],
    group_fields: Sequence[str],
) -> dict[str, Any]:
    overall_baseline = _aggregate_arm(frame_rows, BASELINE_ID, candidate_classes)
    overall = _add_comparison(
        _aggregate_arm(frame_rows, arm_id, candidate_classes),
        overall_baseline,
    )
    report: dict[str, Any] = {"overall": overall}
    for field in group_fields:
        field_groups: dict[str, Any] = {}
        for group_id, rows in _group_rows(frame_rows, field).items():
            baseline = _aggregate_arm(rows, BASELINE_ID, candidate_classes)
            field_groups[group_id] = _add_comparison(
                _aggregate_arm(rows, arm_id, candidate_classes),
                baseline,
            )
        report[f"by_{field}"] = field_groups
    return report


def _decision_checks(
    arm_report: dict[str, Any],
    baseline_report: dict[str, Any],
    rules: dict[str, Any],
) -> dict[str, Any]:
    overall = arm_report["overall"]
    sessions = arm_report["by_session_id"]
    overall_comparison = overall["comparison_to_baseline"]
    session_retentions = {
        session_id: metrics["comparison_to_baseline"]["recall_retention"]
        for session_id, metrics in sessions.items()
    }
    comparable = {
        key: value for key, value in session_retentions.items() if value is not None
    }
    minimum_session_id = (
        min(comparable, key=lambda key: (comparable[key], key)) if comparable else None
    )
    minimum_session_retention = (
        comparable[minimum_session_id] if minimum_session_id is not None else None
    )
    boundary_retention = overall["classes"]["boundary_step_curb"][
        "comparison_to_baseline"
    ]["recall_retention"]
    obstacle_retention = overall["classes"]["obstacle"]["comparison_to_baseline"][
        "recall_retention"
    ]
    strict_false_nonincrease = (
        overall["class_strict_false_components_per_frame"]
        <= baseline_report["overall"]["class_strict_false_components_per_frame"]
    )
    checks = {
        "false_positive_reduction": {
            "value": overall_comparison["false_positive_reduction"],
            "threshold": float(rules["minimum_false_positive_reduction"]),
        },
        "overall_recall_retention": {
            "value": overall_comparison["recall_retention"],
            "threshold": float(rules["minimum_overall_recall_retention"]),
        },
        "minimum_session_recall_retention": {
            "value": minimum_session_retention,
            "threshold": float(rules["minimum_session_recall_retention"]),
            "session_id": minimum_session_id,
        },
        "boundary_step_curb_recall_retention": {
            "value": boundary_retention,
            "threshold": float(
                rules["minimum_boundary_step_curb_recall_retention"]
            ),
        },
        "obstacle_recall_retention": {
            "value": obstacle_retention,
            "threshold": float(rules["minimum_obstacle_recall_retention"]),
        },
    }
    for check in checks.values():
        value = check["value"]
        check["passed"] = value is not None and float(value) >= float(check["threshold"])
    checks["class_strict_false_components_nonincrease"] = {
        "value": strict_false_nonincrease,
        "passed": strict_false_nonincrease,
        "diagnostic_only": True,
    }
    passed = all(
        bool(check["passed"])
        for name, check in checks.items()
        if name != "class_strict_false_components_nonincrease"
    )
    return {"checks": checks, "sufficient": passed}


def _pareto_status(
    reports: dict[str, dict[str, Any]],
    reference_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> dict[str, Any]:
    ids = [*reference_ids, *candidate_ids]
    points = {
        arm_id: {
            "false_positive_reduction": reports[arm_id]["overall"][
                "comparison_to_baseline"
            ]["false_positive_reduction"],
            "recall_retention": reports[arm_id]["overall"]["comparison_to_baseline"][
                "recall_retention"
            ],
        }
        for arm_id in ids
    }
    frontier: list[str] = []
    for arm_id in ids:
        point = points[arm_id]
        dominated = any(
            other_id != arm_id
            and points[other_id]["false_positive_reduction"]
            >= point["false_positive_reduction"]
            and points[other_id]["recall_retention"] >= point["recall_retention"]
            and (
                points[other_id]["false_positive_reduction"]
                > point["false_positive_reduction"]
                or points[other_id]["recall_retention"] > point["recall_retention"]
            )
            for other_id in ids
        )
        if not dominated:
            frontier.append(arm_id)
    candidate_status: dict[str, Any] = {}
    for candidate_id in candidate_ids:
        dominates = [
            reference_id
            for reference_id in reference_ids
            if points[candidate_id]["false_positive_reduction"]
            >= points[reference_id]["false_positive_reduction"]
            and points[candidate_id]["recall_retention"]
            >= points[reference_id]["recall_retention"]
            and (
                points[candidate_id]["false_positive_reduction"]
                > points[reference_id]["false_positive_reduction"]
                or points[candidate_id]["recall_retention"]
                > points[reference_id]["recall_retention"]
            )
        ]
        candidate_status[candidate_id] = {
            "frontier_member": candidate_id in frontier,
            "dominates_predecessor_references": dominates,
            "new_frontier_point": candidate_id in frontier and bool(dominates),
        }
    return {
        "points": points,
        "frontier_arm_ids": sorted(frontier),
        "candidates": candidate_status,
    }


def _candidate_definition_hash(config: dict[str, Any]) -> str:
    frozen = {
        "candidate_order": config["candidate_order"],
        "candidate_definitions": config["candidate_definitions"],
        "thresholds": config["thresholds"],
        "implementation_contract": config["implementation_contract"],
        "forbidden_candidate_inputs": config["forbidden_candidate_inputs"],
    }
    return sha256_json(frozen)


def _resolve(repo_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_bound_jsonl(
    repo_root: Path,
    specifications: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for specification in specifications:
        path = _resolve(repo_root, specification["path"])
        observed_sha = sha256_file(path)
        if observed_sha != specification["sha256"]:
            raise ValueError(f"bound input SHA mismatch: {path}")
        file_rows = read_jsonl(path)
        if len(file_rows) != int(specification["row_count"]):
            raise ValueError(f"bound input row count mismatch: {path}")
        rows.extend(file_rows)
        provenance.append(
            {
                "path": str(path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": observed_sha,
                "row_count": len(file_rows),
            }
        )
    return rows, provenance


def _validate_static_config(config: dict[str, Any]) -> None:
    if config.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("unexpected protocol_id")
    if config.get("stage") != "DEVELOPMENT_STANDARD":
        raise ValueError("conditional gating R0 is Development only")
    expected_candidates = ["CLASS_CONDITIONED_MULTI_NEGATIVE"]
    if config.get("candidate_order") != expected_candidates:
        raise ValueError("candidate order must contain the exact single frozen candidate")
    thresholds = config["thresholds"]
    if (
        float(thresholds["confidence_minimum"]) != 0.65
        or int(thresholds["small_fragment_max_area_pixels"]) != 63
        or float(thresholds["upper_head_y_max_fraction"]) != 0.35
        or int(thresholds["connectivity"]) != 8
    ):
        raise ValueError("frozen thresholds drifted")
    contract = config["implementation_contract"]
    required_contract = {
        "temporal_unit": "PIXEL",
        "history_source": "RAW_CLASS_MASK",
        "class_isolation": True,
        "component_evidence_source": "RAW_CURRENT_COMPONENT",
        "upper_membership": "ANY_PIXEL_INTERSECTION",
        "post_filter_feature_recomputation": False,
        "candidate_selection_during_execution": False,
        "candidate_fit_count": 0,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"implementation contract drifted: {field}")
    def walk_keys(value: Any) -> Iterable[str]:
        if isinstance(value, dict):
            for key, child in value.items():
                yield str(key).lower()
                yield from walk_keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk_keys(child)

    observed_keys = set(walk_keys(config))
    for forbidden in ("threshold_grid", "threshold_search", "best_candidate"):
        if forbidden in observed_keys:
            raise ValueError(f"config contains forbidden selection field: {forbidden}")


def _validate_membership(
    config: dict[str, Any],
    frame_rows: Sequence[dict[str, Any]],
    component_rows: Sequence[dict[str, Any]],
) -> None:
    contract = config["input_contract"]
    if len(frame_rows) != int(contract["expected_frame_count"]):
        raise ValueError("unexpected frame count")
    if len(component_rows) != int(contract["expected_component_count"]):
        raise ValueError("unexpected component count")
    view_ids = [str(row["view_row_id"]) for row in frame_rows]
    if len(view_ids) != len(set(view_ids)):
        raise ValueError("duplicate view_row_id")
    component_ids = [str(row["component_id"]) for row in component_rows]
    if len(component_ids) != len(set(component_ids)):
        raise ValueError("duplicate component_id")
    session_counts = Counter(str(row["session_id"]) for row in frame_rows)
    if session_counts != Counter(
        {
            str(key): int(value)
            for key, value in contract["expected_session_frame_counts"].items()
        }
    ):
        raise ValueError("session membership does not match frozen contract")
    roles_by_session: dict[str, str] = {}
    for row in frame_rows:
        session_id = str(row["session_id"])
        role = str(row["rehearsal_role"])
        previous = roles_by_session.setdefault(session_id, role)
        if previous != role:
            raise ValueError("session contains multiple evidence roles")
    role_counts = Counter(roles_by_session.values())
    if role_counts != Counter(
        {
            str(key): int(value)
            for key, value in contract["expected_role_session_counts"].items()
        }
    ):
        raise ValueError("role/session counts do not match frozen contract")


def _current_git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _predecessor_probe(
    predecessor: dict[str, Any],
    probe_id: str,
) -> dict[str, Any]:
    for probe in predecessor["probes"]:
        if probe["probe_id"] == probe_id:
            return probe
    raise ValueError(f"missing predecessor probe: {probe_id}")


def _assert_confusion_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    label: str,
) -> None:
    for field in ("tp", "fp", "fn", "tn"):
        if int(actual[field]) != int(expected[field]):
            raise ValueError(
                f"predecessor reproduction mismatch for {label}.{field}: "
                f"{actual[field]} != {expected[field]}"
            )


def _verify_output_scope(repo_root: Path, output_root: Path) -> None:
    allowed = (repo_root / "artifacts.local" / "evidence").resolve()
    resolved = output_root.resolve()
    if resolved == allowed or allowed not in resolved.parents:
        raise ValueError("output_root must be below artifacts.local/evidence")


def run_conditional_gating(
    *,
    repo_root: Path,
    config_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    _validate_static_config(config)
    shape = tuple(int(value) for value in config["analysis_shape"])
    if len(shape) != 2:
        raise ValueError("analysis_shape must contain height and width")
    _verify_output_scope(repo_root, output_root)
    if output_root.exists():
        raise FileExistsError(f"output already exists: {output_root}")

    frame_rows, frame_provenance = _load_bound_jsonl(
        repo_root, config["input_contract"]["frames"]
    )
    component_rows, component_provenance = _load_bound_jsonl(
        repo_root, config["input_contract"]["components"]
    )
    _validate_membership(config, frame_rows, component_rows)
    manifest_spec = config["input_contract"]["canonical_manifest"]
    manifest_path = _resolve(repo_root, manifest_spec["path"])
    if sha256_file(manifest_path) != manifest_spec["sha256"]:
        raise ValueError("canonical manifest SHA mismatch")
    manifest_rows = read_jsonl(manifest_path)
    manifest_by_id = {str(row["id"]): row for row in manifest_rows}
    if len(manifest_by_id) != len(manifest_rows):
        raise ValueError("canonical manifest contains duplicate id")
    ledger_by_id = {str(row["component_id"]): row for row in component_rows}
    view_root = _resolve(repo_root, config["input_contract"]["canonical_view_root"])

    truth_names = {
        int(key): str(value) for key, value in config["truth_classes"].items()
    }
    truth_ids_by_name = {name: class_id for class_id, name in truth_names.items()}
    hazard_ids = np.asarray(config["hazard_truth_ids"], dtype=np.uint8)
    candidate_classes = [str(value) for value in config["candidate_classes"]]
    thresholds = config["thresholds"]
    upper_mask = upper_head_band(
        shape, float(thresholds["upper_head_y_max_fraction"])
    )

    enriched_rows: list[tuple[dict[str, Any], dict[str, Any]]] = []
    observed_image_hashes: set[str] = set()
    for frame_row in frame_rows:
        manifest = manifest_by_id.get(str(frame_row["view_row_id"]))
        if manifest is None:
            raise ValueError(f"missing canonical manifest row: {frame_row['view_row_id']}")
        for field in (
            "source_id",
            "session_id",
            "frame_id",
            "image_sha256",
            "canonical_mask_sha256",
        ):
            if frame_row[field] != manifest[field]:
                raise ValueError(
                    f"frame/manifest mismatch for {field}: {frame_row['view_row_id']}"
                )
        if str(frame_row["rehearsal_role"]) != str(manifest["role"]):
            raise ValueError("frame/manifest role mismatch")
        if tuple(frame_row["packed_masks"]["shape"]) != shape:
            raise ValueError("packed mask shape mismatch")
        image_sha = str(frame_row["image_sha256"])
        if image_sha in observed_image_hashes:
            raise ValueError("duplicate image identity across frozen frames")
        observed_image_hashes.add(image_sha)
        enriched_rows.append((frame_row, manifest))

    enriched_rows.sort(
        key=lambda pair: (
            str(pair[1]["session_id"]),
            str(pair[1]["sequence_id"]),
            int(pair[1]["source_capture_timestamp_ns"]),
            int(pair[1]["frame_id"]),
            str(pair[1]["id"]),
        )
    )

    histories: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)
    union_histories: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    output_frames: list[dict[str, Any]] = []
    output_decisions: list[dict[str, Any]] = []
    observed_component_ids: set[str] = set()
    role_frame_indices: dict[str, list[int]] = defaultdict(list)
    cross_class_credit = Counter()
    cross_class_credit_sessions: dict[str, set[str]] = defaultdict(set)

    arm_ids = [
        BASELINE_ID,
        REFERENCE_CAUSAL_ID,
        REFERENCE_CONFIDENCE_ID,
        *config["candidate_order"],
    ]
    for frame_row, manifest in enriched_rows:
        view_row_id = str(frame_row["view_row_id"])
        session_id = str(frame_row["session_id"])
        sequence_id = str(manifest["sequence_id"])
        role = str(frame_row["rehearsal_role"])
        truth_path = view_root / str(manifest["canonical_mask_path"])
        if sha256_file(truth_path) != str(frame_row["canonical_mask_sha256"]):
            raise ValueError(f"canonical truth SHA mismatch: {view_row_id}")
        truth = np.asarray(Image.open(truth_path), dtype=np.uint8)
        if truth.shape != shape:
            raise ValueError(f"canonical truth shape mismatch: {view_row_id}")
        a_mask = decode_packed_mask(frame_row["packed_masks"]["A"], shape)
        b_mask = decode_packed_mask(frame_row["packed_masks"]["B"], shape)
        raw_class_masks = {
            class_name: decode_packed_mask(
                frame_row["packed_masks"][f"candidate_{class_name}"], shape
            )
            for class_name in candidate_classes
        }
        if np.count_nonzero(raw_class_masks[candidate_classes[0]] & raw_class_masks[candidate_classes[1]]):
            raise ValueError(f"candidate class masks overlap: {view_row_id}")
        if not np.array_equal(
            b_mask, np.logical_or.reduce(list(raw_class_masks.values()))
        ):
            raise ValueError(f"candidate class union mismatch: {view_row_id}")
        full_truth_hazard = np.isin(truth, hazard_ids)
        residual_truth = full_truth_hazard & ~a_mask
        class_truth = {
            class_name: (truth == truth_ids_by_name[class_name]) & ~a_mask
            for class_name in candidate_classes
        }
        same_class_causal: dict[str, np.ndarray] = {}
        for class_name, current in raw_class_masks.items():
            history = histories[(session_id, sequence_id, class_name)]
            same_class_causal[class_name] = causal_two_of_three(
                current,
                history[-1] if len(history) >= 1 else None,
                history[-2] if len(history) >= 2 else None,
            )
        union_history = union_histories[(session_id, sequence_id)]
        union_causal = causal_two_of_three(
            b_mask,
            union_history[-1] if len(union_history) >= 1 else None,
            union_history[-2] if len(union_history) >= 2 else None,
        )
        for class_name in candidate_classes:
            cross = raw_class_masks[class_name] & union_causal & ~same_class_causal[class_name]
            count = int(np.count_nonzero(cross))
            cross_class_credit[class_name] += count
            if count:
                cross_class_credit_sessions[class_name].add(session_id)

        arm_class_masks = {
            arm_id: {
                class_name: np.zeros(shape, dtype=bool)
                for class_name in candidate_classes
            }
            for arm_id in arm_ids
        }
        arm_class_masks[BASELINE_ID] = {
            name: mask.copy() for name, mask in raw_class_masks.items()
        }
        for class_name in candidate_classes:
            arm_class_masks[REFERENCE_CAUSAL_ID][class_name] = (
                raw_class_masks[class_name] & union_causal
            )

        for class_name, class_mask in raw_class_masks.items():
            for component in connected_components(
                class_mask, connectivity=int(thresholds["connectivity"])
            ):
                component_id = (
                    f"{frame_row['source_id']}:{int(frame_row['frame_id'])}:"
                    f"{class_name}:{component.index}"
                )
                ledger = ledger_by_id.get(component_id)
                if ledger is None:
                    raise ValueError(f"component missing from ledger: {component_id}")
                if (
                    str(ledger["class_name"]) != class_name
                    or int(ledger["area_pixels"]) != component.area
                    or list(ledger["bbox_xyxy"]) != list(component.bbox)
                ):
                    raise ValueError(f"component ledger geometry mismatch: {component_id}")
                confidence = ledger["top1_confidence_median"]
                if confidence is not None and not math.isfinite(float(confidence)):
                    raise ValueError(f"component confidence is nonfinite: {component_id}")
                observed_component_ids.add(component_id)
                if confidence is not None and float(confidence) >= float(
                    thresholds["confidence_minimum"]
                ):
                    arm_class_masks[REFERENCE_CONFIDENCE_ID][class_name] |= component.mask
                for candidate_id in config["candidate_order"]:
                    kept, evidence = apply_frozen_candidate_to_component(
                        candidate_id=candidate_id,
                        predicted_class=class_name,
                        component_mask=component.mask,
                        same_class_causal_mask=same_class_causal[class_name],
                        confidence_median=confidence,
                        raw_area_pixels=component.area,
                        upper_band_mask=upper_mask,
                        confidence_minimum=float(thresholds["confidence_minimum"]),
                        small_fragment_max_area_pixels=int(
                            thresholds["small_fragment_max_area_pixels"]
                        ),
                    )
                    arm_class_masks[candidate_id][class_name] |= kept
                    output_decisions.append(
                        {
                            "schema_version": (
                                "blindassist.dual_loop_segmentation_conditional_gating."
                                "component_decision.v1"
                            ),
                            "protocol_id": PROTOCOL_ID,
                            "candidate_id": candidate_id,
                            "view_row_id": view_row_id,
                            "source_id": str(frame_row["source_id"]),
                            "session_id": session_id,
                            "sequence_id": sequence_id,
                            "frame_id": int(frame_row["frame_id"]),
                            "component_id": component_id,
                            "component_index": int(component.index),
                            "predicted_class": class_name,
                            "raw_area_pixels": int(component.area),
                            "raw_top1_confidence_median": confidence,
                            "gate_input_fields": [
                                "predicted_class",
                                "raw_component_mask",
                                "same_class_raw_history_masks",
                                "top1_confidence_median",
                                "raw_area_pixels",
                                "upper_head_band_geometry",
                                "frozen_thresholds"
                            ],
                            **evidence,
                        }
                    )
        if observed_component_ids - set(ledger_by_id):
            raise AssertionError("observed unexpected component id")
        for arm_id in arm_ids:
            for class_name in candidate_classes:
                if np.count_nonzero(
                    arm_class_masks[arm_id][class_name] & ~raw_class_masks[class_name]
                ):
                    raise ValueError(
                        f"arm created new class pixels: {arm_id}/{view_row_id}/{class_name}"
                    )
        frame_output = {
            "schema_version": (
                "blindassist.dual_loop_segmentation_conditional_gating.frame_metrics.v1"
            ),
            "protocol_id": PROTOCOL_ID,
            "view_row_id": view_row_id,
            "source_id": str(frame_row["source_id"]),
            "session_id": session_id,
            "sequence_id": sequence_id,
            "role": role,
            "scene_bucket": str(manifest["scene_bucket"]),
            "frame_id": int(frame_row["frame_id"]),
            "source_capture_timestamp_ns": int(manifest["source_capture_timestamp_ns"]),
            "arms": {
                arm_id: _frame_arm_metrics(
                    arm_class_masks[arm_id], residual_truth, class_truth
                )
                for arm_id in arm_ids
            },
        }
        output_frames.append(frame_output)
        role_frame_indices[role].append(len(output_frames) - 1)
        for class_name, current in raw_class_masks.items():
            history = histories[(session_id, sequence_id, class_name)]
            history.append(current.copy())
            if len(history) > 2:
                del history[:-2]
        union_history.append(b_mask.copy())
        if len(union_history) > 2:
            del union_history[:-2]

    if observed_component_ids != set(ledger_by_id):
        missing = sorted(set(ledger_by_id) - observed_component_ids)
        raise ValueError(f"unmatched component ledger rows: {missing[:3]}")

    predecessor_reproduction: dict[str, Any] = {}
    for name, specification in config["predecessor_references"].items():
        path = _resolve(repo_root, specification["path"])
        if sha256_file(path) != specification["sha256"]:
            raise ValueError(f"predecessor reference SHA mismatch: {name}")
        predecessor = read_json(path)
        selected = [
            output_frames[index]
            for role in specification["roles"]
            for index in role_frame_indices[str(role)]
        ]
        baseline = _aggregate_arm(selected, BASELINE_ID, candidate_classes)
        causal = _aggregate_arm(selected, REFERENCE_CAUSAL_ID, candidate_classes)
        confidence = _aggregate_arm(
            selected, REFERENCE_CONFIDENCE_ID, candidate_classes
        )
        _assert_confusion_equal(
            baseline["pixel"], predecessor["baseline"]["pixel"], f"{name}.baseline"
        )
        _assert_confusion_equal(
            causal["pixel"],
            _predecessor_probe(predecessor, "TEMPORAL:CAUSAL_2_OF_3")["pixel"],
            f"{name}.causal",
        )
        _assert_confusion_equal(
            confidence["pixel"],
            _predecessor_probe(
                predecessor, "CONFIDENCE:COMPONENT_MEDIAN_CONFIDENCE_GE_0_65"
            )["pixel"],
            f"{name}.confidence",
        )
        predecessor_reproduction[name] = {
            "status": "EXACT_INTEGER_CONFUSION_REPRODUCED",
            "path": str(path.relative_to(repo_root)).replace("\\", "/"),
            "sha256": specification["sha256"],
            "frame_count": len(selected),
        }

    group_fields = ("session_id", "role", "scene_bucket")
    reports = {
        arm_id: _aggregate_arm_report(
            frame_rows=output_frames,
            arm_id=arm_id,
            candidate_classes=candidate_classes,
            group_fields=group_fields,
        )
        for arm_id in arm_ids
    }
    candidate_decisions = {
        candidate_id: _decision_checks(
            reports[candidate_id], reports[BASELINE_ID], config["decision_rules"]
        )
        for candidate_id in config["candidate_order"]
    }
    pareto = _pareto_status(
        reports,
        reference_ids=REFERENCE_IDS,
        candidate_ids=config["candidate_order"],
    )
    for candidate_id in config["candidate_order"]:
        candidate_decisions[candidate_id]["pareto"] = pareto["candidates"][
            candidate_id
        ]

    decisions_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for decision in output_decisions:
        decisions_by_candidate[str(decision["candidate_id"])].append(decision)
    component_outcomes: dict[str, Any] = {}
    for candidate_id in config["candidate_order"]:
        decisions = decisions_by_candidate[candidate_id]
        actions = Counter(str(row["action"]) for row in decisions)
        split = sum(int(row["post_fragment_count"]) > 1 for row in decisions)
        component_outcomes[candidate_id] = {
            "raw_component_count": len(decisions),
            "fully_retained": actions["KEEP"],
            "partially_retained": actions["PARTIAL"],
            "removed": actions["REJECT"],
            "split_source_components": split,
            "post_fragment_count_from_raw_components": sum(
                int(row["post_fragment_count"]) for row in decisions
            ),
        }

    definition_hash = _candidate_definition_hash(config)
    session_ids = sorted(config["input_contract"]["expected_session_frame_counts"])
    held_out_folds: list[dict[str, Any]] = []
    for session_id in session_ids:
        held_out = [row for row in output_frames if row["session_id"] == session_id]
        context = [row for row in output_frames if row["session_id"] != session_id]
        fold_arms: dict[str, Any] = {}
        for candidate_id in config["candidate_order"]:
            direct_session = reports[candidate_id]["by_session_id"][session_id]
            held_metric = _add_comparison(
                _aggregate_arm(held_out, candidate_id, candidate_classes),
                _aggregate_arm(held_out, BASELINE_ID, candidate_classes),
            )
            if sha256_json(held_metric) != sha256_json(direct_session):
                raise ValueError("held-out/direct session metric mismatch")
            fold_arms[candidate_id] = {
                "held_out": held_metric,
                "development_context": _add_comparison(
                    _aggregate_arm(context, candidate_id, candidate_classes),
                    _aggregate_arm(context, BASELINE_ID, candidate_classes),
                ),
                "held_out_equals_direct_session_metrics": True,
            }
        held_out_folds.append(
            {
                "fold_id": f"HOLDOUT:{session_id}",
                "held_out_session_id": session_id,
                "context_session_ids": [value for value in session_ids if value != session_id],
                "candidate_definition_sha256": definition_hash,
                "fit_used": False,
                "training_used": False,
                "candidate_selection_used": False,
                "arms": fold_arms,
            }
        )

    sufficient_candidates = [
        candidate_id
        for candidate_id in config["candidate_order"]
        if candidate_decisions[candidate_id]["sufficient"]
    ]
    terminal = (
        config["decision_rules"]["supported_terminal"]
        if sufficient_candidates
        else config["decision_rules"]["unsupported_terminal"]
    )
    next_boundary = (
        "FREEZE_ONE_DEVELOPMENT_CANDIDATE_FOR_SEPARATE_FUTURE_CONFIRMATION_DESIGN"
        if sufficient_candidates
        else "RESIDUAL_AWARE_DDRNET_DEVELOPMENT_DESIGN_AUTHORIZED_NOT_EXECUTED"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "stage": config["stage"],
        "evidence_instance": config["evidence_instance"],
        "claim_ceiling": config["claim_ceiling"],
        "analysis_unit": "SOURCE_NATIVE_SESSION",
        "temporal_observation_unit": "MATERIALIZED_OBSERVATION_NOT_ALL_VIDEO_FRAMES",
        "candidate_definition_sha256": definition_hash,
        "candidate_order": config["candidate_order"],
        "implementation_contract": config["implementation_contract"],
        "selection_status": "NO_SELECTION_SINGLE_FROZEN_CANDIDATE_REPORTED",
        "confirmation": "NOT_ACTIVATED",
        "drives_alerts": False,
        "git_head": _current_git_head(repo_root),
        "input": {
            "frame_count": len(output_frames),
            "component_count": len(component_rows),
            "session_count": len(session_ids),
            "role_session_counts": config["input_contract"][
                "expected_role_session_counts"
            ],
            "frame_files": frame_provenance,
            "component_files": component_provenance,
            "canonical_manifest": {
                "path": str(manifest_path.relative_to(repo_root)).replace("\\", "/"),
                "sha256": manifest_spec["sha256"],
            },
            "source_session_disjointness": {
                "unique_session_ids": len(session_ids),
                "unique_view_row_ids": len({row["view_row_id"] for row in output_frames}),
                "unique_image_sha256": len(observed_image_hashes),
                "participant_route_parent_capture_independence": (
                    "NOT_EVALUABLE_MISSING_IDENTIFIERS"
                ),
                "source_family": "SANPO_REAL_V0_ONLY",
            },
        },
        "predecessor_reproduction": predecessor_reproduction,
        "cross_class_temporal_credit_removed_by_new_contract": {
            class_name: {
                "pixels": int(cross_class_credit[class_name]),
                "session_count": len(cross_class_credit_sessions[class_name]),
            }
            for class_name in candidate_classes
        },
        "arms": reports,
        "candidate_decisions": candidate_decisions,
        "component_outcomes": component_outcomes,
        "pareto": pareto,
        "held_out_stress": {
            "method": config["held_out_reporting"]["method"],
            "claim": config["held_out_reporting"]["claim"],
            "candidate_fit_count": 0,
            "candidate_selection_per_fold": False,
            "fold_count": len(held_out_folds),
            "folds": held_out_folds,
            "interpretation": (
                "Fit-free per-session stress reporting on burned Development data; "
                "not cross-validation and not independent validation."
            ),
        },
        "decision": {
            "terminal": terminal,
            "sufficient_candidate_ids": sufficient_candidates,
            "next_boundary": next_boundary,
            "residual_aware_ddrnet_training_executed": False,
            "android_or_alert_authority": "NONE",
        },
    }

    temporary = output_root.parent / f".{output_root.name}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        _write_jsonl(temporary / "frame_metrics.jsonl", output_frames)
        _write_jsonl(temporary / "component_decisions.jsonl", output_decisions)
        result["output_files"] = {
            "frame_metrics.jsonl": {
                "sha256": sha256_file(temporary / "frame_metrics.jsonl"),
                "row_count": len(output_frames),
            },
            "component_decisions.jsonl": {
                "sha256": sha256_file(temporary / "component_decisions.jsonl"),
                "row_count": len(output_decisions),
            },
        }
        _write_json(temporary / "result.json", result)
        output_root.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_root)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    config_path = _resolve(repo_root, str(args.config))
    output_root = _resolve(repo_root, str(args.output_root))
    result = run_conditional_gating(
        repo_root=repo_root,
        config_path=config_path,
        output_root=output_root,
    )
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "terminal": result["decision"]["terminal"],
                "output_root": str(output_root),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
