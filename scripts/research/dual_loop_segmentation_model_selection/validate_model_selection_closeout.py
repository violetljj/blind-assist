"""Validate the R1 closeout without repairing or rerunning the formal attempt.

This validator never runs either candidate model and never produces a repaired
formal result. It checks current Development-row self-consistency, verifies all
formal-freeze hashes, and independently confirms that the consumed fresh mask
violates the frozen evaluator's canonical 0..3 truth contract. Development
outputs were not hash-bound before truth access, and runtime aggregates have no
immutable per-frame timing rows, so neither is promoted to independent formal
evidence by this closeout.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


PROTOCOL_ID = "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1"
FRAME_SCHEMA = "blindassist.dual_loop_segmentation_model_selection_r1.frame.v1"
COMPONENT_SCHEMA = "blindassist.dual_loop_segmentation_model_selection_r1.component.v1"
PACKED_MASK_BYTES = (256 * 256 + 7) // 8
FORBIDDEN_KEYS = {"risk", "feedback", "event", "central_obstruction_agent_labels"}


class CloseoutValidationError(ValueError):
    """Raised when an R1 identity or immutable calculation does not validate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CloseoutValidationError(f"cannot read JSON: {path}") from exc
    if not isinstance(value, dict):
        raise CloseoutValidationError(f"expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise CloseoutValidationError(f"cannot read JSONL: {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CloseoutValidationError(f"blank JSONL row: {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CloseoutValidationError(f"invalid JSONL row: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise CloseoutValidationError(f"expected object: {path}:{line_number}")
            rows.append(value)
    if not rows:
        raise CloseoutValidationError(f"empty JSONL: {path}")
    return rows


def forbidden_path(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return f"{path}.{key}"
            found = forbidden_path(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = forbidden_path(child, f"{path}[{index}]")
            if found:
                return found
    return None


def close(left: Any, right: Any, *, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def ratio(numerator: int, denominator: int, *, empty: float | None) -> float | None:
    return float(numerator / denominator) if denominator else empty


def f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else float(2.0 * precision * recall / (precision + recall))


def normalize_confusion(row: dict[str, Any]) -> dict[str, Any]:
    required = {"tp", "fp", "fn", "tn", "predicted_pixels", "truth_pixels"}
    if required - row.keys():
        raise CloseoutValidationError("pixel metric row is missing confusion fields")
    tp, fp, fn, tn = (int(row[name]) for name in ("tp", "fp", "fn", "tn"))
    if min(tp, fp, fn, tn) < 0:
        raise CloseoutValidationError("pixel metric row contains a negative count")
    total = tp + fp + fn + tn
    empty = tp + fp + fn == 0
    precision = ratio(tp, tp + fp, empty=1.0 if empty else None)
    recall = ratio(tp, tp + fn, empty=1.0 if empty else None)
    iou = ratio(tp, tp + fp + fn, empty=1.0 if empty else None)
    if int(row["predicted_pixels"]) != tp + fp or int(row["truth_pixels"]) != tp + fn:
        raise CloseoutValidationError("pixel predicted/truth counts differ from confusion")
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": tp + fp,
        "truth_pixels": tp + fn,
        "pixel_count": total,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": f1(precision, recall),
        "false_positive_area_fraction": float(fp / total) if total else 0.0,
    }


def aggregate_confusion(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    normalized = [normalize_confusion(row) for row in rows]
    if not normalized:
        raise CloseoutValidationError("cannot aggregate zero confusion rows")
    totals = {
        name: sum(int(row[name]) for row in normalized)
        for name in ("tp", "fp", "fn", "tn", "predicted_pixels", "truth_pixels")
    }
    result = normalize_confusion(totals)
    precision = [row["precision"] for row in normalized if row["precision"] is not None]
    recall = [row["recall"] for row in normalized if row["recall"] is not None]
    result["mean_frame_precision"] = float(sum(precision) / len(precision)) if precision else None
    result["mean_frame_recall"] = float(sum(recall) / len(recall)) if recall else None
    return result


def component_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        counts = {
            name: int(row[name])
            for name in (
                "predicted_component_count",
                "truth_component_count",
                "hit_predicted_component_count",
                "hit_truth_component_count",
                "false_activation_component_count",
            )
        }
        if min(counts.values()) < 0:
            raise CloseoutValidationError("component metric row contains a negative count")
        if counts["hit_predicted_component_count"] > counts["predicted_component_count"]:
            raise CloseoutValidationError("component hit-predicted count exceeds predicted count")
        if counts["hit_truth_component_count"] > counts["truth_component_count"]:
            raise CloseoutValidationError("component hit-truth count exceeds truth count")
        if (
            counts["false_activation_component_count"]
            != counts["predicted_component_count"] - counts["hit_predicted_component_count"]
        ):
            raise CloseoutValidationError("component false-activation count is inconsistent")
    predicted = sum(int(row["predicted_component_count"]) for row in rows)
    truth = sum(int(row["truth_component_count"]) for row in rows)
    hit_predicted = sum(int(row["hit_predicted_component_count"]) for row in rows)
    hit_truth = sum(int(row["hit_truth_component_count"]) for row in rows)
    false_count = sum(int(row["false_activation_component_count"]) for row in rows)
    return {
        "predicted_component_count": predicted,
        "truth_component_count": truth,
        "hit_predicted_component_count": hit_predicted,
        "hit_truth_component_count": hit_truth,
        "component_precision": ratio(hit_predicted, predicted, empty=1.0 if truth == 0 else None),
        "component_recall": ratio(hit_truth, truth, empty=1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": float(false_count / len(rows)) if rows else None,
    }


def assert_mapping(
    stored: dict[str, Any],
    recomputed: dict[str, Any],
    *,
    context: str,
    optional_missing: frozenset[str] = frozenset(),
) -> None:
    for name, expected in recomputed.items():
        if name not in stored and name in optional_missing:
            continue
        if name not in stored or not close(stored[name], expected):
            raise CloseoutValidationError(
                f"{context}.{name} mismatch: stored={stored.get(name)!r} recomputed={expected!r}"
            )


def gate(actual: Any, threshold: Any, operator: str) -> dict[str, Any]:
    if actual is None:
        threshold_satisfied = False
    elif operator == ">=":
        threshold_satisfied = float(actual) >= float(threshold)
    elif operator == "<=":
        threshold_satisfied = float(actual) <= float(threshold)
    else:
        raise ValueError(operator)
    return {
        "actual": actual,
        "threshold": threshold,
        "operator": operator,
        "threshold_satisfied": bool(threshold_satisfied),
    }


def assert_identity_sets_equal(
    actual: set[tuple[str, int, str]],
    expected: set[tuple[str, int, str]],
    *,
    context: str,
) -> None:
    if actual != expected:
        raise CloseoutValidationError(f"{context}: frame identities differ from frozen YOLO trace")


def evaluator_mask_ids(path: Path) -> list[int]:
    with Image.open(path) as image:
        values = np.asarray(image.convert("L"), dtype=np.uint8)
    return [int(value) for value in np.unique(values)]


def validate_file_record(repo_root: Path, record: dict[str, Any], *, context: str) -> Path:
    relative = record.get("relative_path")
    expected_sha = record.get("sha256")
    if not isinstance(relative, str) or not isinstance(expected_sha, str):
        raise CloseoutValidationError(f"{context}: malformed frozen file record")
    path = (repo_root / relative).resolve()
    try:
        path.relative_to(repo_root)
    except ValueError as exc:
        raise CloseoutValidationError(f"{context}: frozen path escapes repository") from exc
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise CloseoutValidationError(f"{context}: frozen path missing or SHA256-mismatched: {path}")
    return path


def validate_freeze(repo_root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    if freeze.get("protocol_id") != PROTOCOL_ID or freeze.get("status") != "FORMAL_MODEL_SELECTION_INPUTS_FROZEN":
        raise CloseoutValidationError("formal freeze receipt identity/status mismatch")
    if freeze.get("fresh_holdout_truth_accessed_before_freeze") is not False:
        raise CloseoutValidationError("formal freeze receipt does not prove pre-access truth isolation")
    checked: list[str] = []
    for name in ("protocol", "dataset_role_ledger", "training_manifest", "shared_dev_yolo_trace"):
        validate_file_record(repo_root, freeze[name], context=f"freeze.{name}")
        checked.append(name)
    for name in ("manifest", "freeze_receipt", "yolo_trace"):
        validate_file_record(repo_root, freeze["fresh_holdout"][name], context=f"freeze.fresh_holdout.{name}")
        checked.append(f"fresh_holdout.{name}")
    for candidate_name, candidate in freeze.get("candidates", {}).items():
        for name in ("config", "training_report", "selected_checkpoint", "tflite", "tflite_receipt", "runtime_receipt"):
            validate_file_record(repo_root, candidate[name], context=f"freeze.candidates.{candidate_name}.{name}")
            checked.append(f"candidates.{candidate_name}.{name}")
    for name, record in freeze.get("evaluation_code", {}).items():
        validate_file_record(repo_root, record, context=f"freeze.evaluation_code.{name}")
        checked.append(f"evaluation_code.{name}")
    return {"checked_record_count": len(checked), "checked_records": checked}


def validate_runtime(runtime: dict[str, Any], *, model_sha256: str) -> dict[str, Any]:
    if runtime.get("protocol_id") != PROTOCOL_ID or runtime.get("status") != "RUNTIME_BENCHMARK_COMPLETE":
        raise CloseoutValidationError("runtime receipt identity/status mismatch")
    if runtime.get("model_sha256") != model_sha256:
        raise CloseoutValidationError("runtime receipt model SHA256 mismatch")
    contract = runtime.get("runtime_contract", {})
    if contract.get("threads") != 4 or contract.get("warmup_frames") != 20 or contract.get("measured_frames") != 200:
        raise CloseoutValidationError("runtime receipt execution contract mismatch")
    if runtime.get("corpus", {}).get("truth_pixels_read") is not False:
        raise CloseoutValidationError("runtime benchmark consumed truth")
    expected_stages = {
        "preprocess",
        "tflite_inference",
        "output_dequantize_argmax",
        "component_extraction",
        "fusion_operator",
        "total_increment",
    }
    rows = runtime.get("runtime", {})
    if set(rows) != expected_stages:
        raise CloseoutValidationError("runtime stage set mismatch")
    for name, summary in rows.items():
        if int(summary.get("count", -1)) != 200:
            raise CloseoutValidationError(f"runtime {name}: count mismatch")
        for field in ("mean", "p50", "p90", "p95", "min", "max"):
            if not math.isfinite(float(summary.get(field))) or float(summary[field]) < 0:
                raise CloseoutValidationError(f"runtime {name}: invalid {field}")
        if not (float(summary["min"]) <= float(summary["p50"]) <= float(summary["p90"]) <= float(summary["p95"]) <= float(summary["max"])):
            raise CloseoutValidationError(f"runtime {name}: percentile ordering mismatch")
    return {
        "integrity_status": "VALID_AGGREGATE_ONLY",
        "summary_recompute_status": "NOT_EVALUABLE",
        "summary_recompute_reason": "the frozen runtime receipt contains aggregate percentiles but no immutable per-frame timing rows",
        "tflite_inference_p95_ms": float(rows["tflite_inference"]["p95"]),
        "total_increment_p95_ms": float(rows["total_increment"]["p95"]),
    }


def validate_dev_candidate(
    repo_root: Path,
    *,
    candidate_name: str,
    protocol: dict[str, Any],
    freeze_candidate: dict[str, Any],
    shared_dev_yolo_trace_record: dict[str, Any],
) -> dict[str, Any]:
    candidate_root = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/dev" / candidate_name
    report_path = candidate_root / "report.json"
    frames_path = candidate_root / "frames.jsonl"
    components_path = candidate_root / "components.jsonl"
    runtime_path = candidate_root / "runtime.json"
    report = read_json(report_path)
    frames = read_jsonl(frames_path)
    components = read_jsonl(components_path)
    runtime = read_json(runtime_path)
    if report.get("protocol_id") != PROTOCOL_ID or report.get("schema_version") != "blindassist.dual_loop_segmentation_model_selection_r1.result.v1":
        raise CloseoutValidationError(f"{candidate_name}: report identity/schema mismatch")
    if report.get("phase") != "calibration" or report.get("split") != "dev" or report.get("status") != "CALIBRATION_EVALUATED":
        raise CloseoutValidationError(f"{candidate_name}: report is not completed dev calibration")
    if len(frames) != 200 or int(report.get("frame_count", -1)) != 200:
        raise CloseoutValidationError(f"{candidate_name}: dev frame denominator mismatch")
    model_sha = freeze_candidate["tflite"]["sha256"]
    if report.get("model_sha256") != model_sha:
        raise CloseoutValidationError(f"{candidate_name}: report model SHA256 differs from formal freeze")
    trace_path = validate_file_record(
        repo_root,
        shared_dev_yolo_trace_record,
        context="freeze.shared_dev_yolo_trace",
    )
    trace_rows = read_jsonl(trace_path)
    trace_keys = {
        (str(row.get("source_id")), int(row.get("frame_id")), str(row.get("image_sha256")))
        for row in trace_rows
    }
    if len(trace_keys) != len(trace_rows):
        raise CloseoutValidationError("shared dev YOLO trace contains duplicate frame identities")
    frame_keys: set[tuple[str, int, str]] = set()
    for index, row in enumerate(frames, start=1):
        if row.get("schema_version") != FRAME_SCHEMA or row.get("protocol_id") != PROTOCOL_ID:
            raise CloseoutValidationError(f"{candidate_name}: frame {index} identity/schema mismatch")
        forbidden = forbidden_path(row)
        if forbidden:
            raise CloseoutValidationError(f"{candidate_name}: frame {index} forbidden field at {forbidden}")
        key = (str(row.get("source_id")), int(row.get("frame_id")), str(row.get("image_sha256")))
        if key in frame_keys:
            raise CloseoutValidationError(f"{candidate_name}: duplicate frame identity")
        frame_keys.add(key)
        packed = row.get("packed_masks", {})
        if packed.get("shape") != [256, 256]:
            raise CloseoutValidationError(f"{candidate_name}: packed mask shape mismatch")
        for mask_name in ("candidate_hazard", "candidate_boundary_step_curb", "candidate_obstacle"):
            try:
                decoded = base64.b64decode(packed[mask_name], validate=True)
            except (KeyError, ValueError) as exc:
                raise CloseoutValidationError(f"{candidate_name}: invalid packed mask {mask_name}") from exc
            if len(decoded) != PACKED_MASK_BYTES:
                raise CloseoutValidationError(f"{candidate_name}: packed mask byte count mismatch")
        for arm in ("A", "B", "C"):
            normalized = normalize_confusion(row["arms"][arm]["pixel"])
            assert_mapping(
                row["arms"][arm]["pixel"],
                normalized,
                context=f"{candidate_name}.frame{index}.{arm}",
                optional_missing=frozenset({"pixel_count"}),
            )
        normalized_candidate = normalize_confusion(row["candidate_pixel_metrics"])
        assert_mapping(
            row["candidate_pixel_metrics"],
            normalized_candidate,
            context=f"{candidate_name}.frame{index}.candidate",
            optional_missing=frozenset({"pixel_count"}),
        )
    assert_identity_sets_equal(frame_keys, trace_keys, context=f"{candidate_name}: dev")
    component_ids: set[str] = set()
    components_by_frame: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, row in enumerate(components, start=1):
        if row.get("schema_version") != COMPONENT_SCHEMA or row.get("protocol_id") != PROTOCOL_ID:
            raise CloseoutValidationError(f"{candidate_name}: component {index} identity/schema mismatch")
        component_id = str(row.get("component_id"))
        if component_id in component_ids:
            raise CloseoutValidationError(f"{candidate_name}: duplicate component identity")
        component_ids.add(component_id)
        if int(row.get("area_pixels", 0)) <= 0 or int(row.get("truth_intersection_pixels", -1)) < 0:
            raise CloseoutValidationError(f"{candidate_name}: invalid component area/intersection")
        if bool(row.get("truth_intersects")) != (int(row["truth_intersection_pixels"]) > 0):
            raise CloseoutValidationError(f"{candidate_name}: component truth-intersection mismatch")
        components_by_frame[(str(row.get("source_id")), int(row.get("frame_id")))].append(row)
    if len(components) != int(report.get("component_count", -1)):
        raise CloseoutValidationError(f"{candidate_name}: component ledger denominator mismatch")
    expected_component_frame_keys = {(source_id, frame_id) for source_id, frame_id, _ in frame_keys}
    unexpected_component_frame_keys = set(components_by_frame) - expected_component_frame_keys
    if unexpected_component_frame_keys:
        raise CloseoutValidationError(f"{candidate_name}: component ledger contains unknown frame identities")
    recomputed_arms = {
        arm: aggregate_confusion(row["arms"][arm]["pixel"] for row in frames)
        for arm in ("A", "B", "C")
    }
    for arm, recomputed in recomputed_arms.items():
        assert_mapping(report["summary"]["arm_pixel_metrics"][arm], recomputed, context=f"{candidate_name}.summary.{arm}")
    recomputed_candidate = aggregate_confusion(row["candidate_pixel_metrics"] for row in frames)
    recomputed_components = component_summary([row["candidate_component_metrics"] for row in frames])
    assert_mapping(report["summary"]["candidate_pixel_metrics"], recomputed_candidate, context=f"{candidate_name}.summary.candidate")
    assert_mapping(report["summary"]["candidate_components"], recomputed_components, context=f"{candidate_name}.summary.components")
    delta_recall = float(recomputed_arms["C"]["recall"] - recomputed_arms["A"]["recall"])
    delta_fp = float(recomputed_arms["C"]["false_positive_area_fraction"] - recomputed_arms["A"]["false_positive_area_fraction"])
    if not close(report["summary"]["delta_recall_C_minus_A"], delta_recall) or not close(report["summary"]["delta_false_positive_area_fraction_C_minus_A"], delta_fp):
        raise CloseoutValidationError(f"{candidate_name}: delta metrics mismatch")
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in frames:
        by_source[str(row["source_id"])].append(row)
    source_metrics: dict[str, Any] = {}
    consistent_sessions = 0
    for source_id, rows in sorted(by_source.items()):
        arm_a = aggregate_confusion(row["arms"]["A"]["pixel"] for row in rows)
        arm_c = aggregate_confusion(row["arms"]["C"]["pixel"] for row in rows)
        source_delta = float(arm_c["recall"] - arm_a["recall"])
        consistent = source_delta >= float(protocol["decision_gates"]["consistent_session_delta_recall_minimum"])
        consistent_sessions += int(consistent)
        source_metrics[source_id] = {"frame_count": len(rows), "delta_recall_C_minus_A": source_delta, "consistent": consistent}
    runtime_validation = validate_runtime(runtime, model_sha256=model_sha)
    gates = protocol["decision_gates"]
    gate_observations = {
        "delta_recall": gate(delta_recall, gates["min_delta_recall"], ">="),
        "delta_false_positive_area_fraction": gate(delta_fp, gates["max_delta_false_positive_area_fraction"], "<="),
        "candidate_component_recall": gate(recomputed_components["component_recall"], gates["min_candidate_component_recall"], ">="),
        "false_activation_components_per_frame": gate(recomputed_components["false_activation_components_per_frame"], gates["max_false_activation_components_per_frame"], "<="),
        "segmentation_host_p95_ms": gate(runtime_validation["tflite_inference_p95_ms"], gates["max_segmentation_host_p95_ms"], "<="),
        "total_incremental_host_p95_ms": gate(runtime_validation["total_increment_p95_ms"], gates["max_total_incremental_host_p95_ms"], "<="),
        "consistent_sessions": gate(consistent_sessions, gates["min_consistent_sessions"], ">="),
    }
    for name, observation in gate_observations.items():
        if name in {"segmentation_host_p95_ms", "total_incremental_host_p95_ms"}:
            observation["evidence_status"] = "AGGREGATE_RECEIPT_ONLY_NOT_INDEPENDENTLY_RECOMPUTABLE"
        elif name == "candidate_component_recall":
            observation["evidence_status"] = "CURRENT_UNANCHORED_FRAME_SUMMARIES_ONLY"
        elif name == "false_activation_components_per_frame":
            observation["evidence_status"] = "CURRENT_UNANCHORED_FRAME_SUMMARIES_ONLY"
        else:
            observation["evidence_status"] = "CURRENT_UNANCHORED_ROWS_SELF_CONSISTENT"
    return {
        "model_id": freeze_candidate["model_id"],
        "report_sha256": sha256_file(report_path),
        "frames_sha256": sha256_file(frames_path),
        "components_sha256": sha256_file(components_path),
        "frame_identity_count": len(frame_keys),
        "component_identity_count": len(component_ids),
        "development_output_anchor_status": "NOT_HASH_BOUND_BEFORE_FORMAL_TRUTH_ACCESS",
        "frozen_yolo_trace_identity_match": True,
        "current_row_source_wise_metrics": source_metrics,
        "current_row_recomputed_metrics": {
            "delta_recall_C_minus_A": delta_recall,
            "delta_false_positive_area_fraction_C_minus_A": delta_fp,
            "candidate_component_recall": recomputed_components["component_recall"],
            "false_activation_components_per_frame": recomputed_components["false_activation_components_per_frame"],
            "consistent_sessions": consistent_sessions,
        },
        "runtime_validation": runtime_validation,
        "dev_gate_observations": gate_observations,
        "independent_dev_gate_terminal": "NOT_EVALUABLE",
        "independent_dev_gate_reason": (
            "Development outputs were not hash-bound before formal truth access; component recall "
            "and runtime P95 also lack independently recomputable source rows"
        ),
    }


def validate_formal_failure(repo_root: Path, freeze: dict[str, Any]) -> dict[str, Any]:
    failure_path = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/formal/failure_receipt.json"
    failure = read_json(failure_path)
    if failure.get("protocol_id") != PROTOCOL_ID or failure.get("status") != "MODEL_SELECTION_NOT_EVALUABLE":
        raise CloseoutValidationError("formal failure receipt identity/status mismatch")
    if failure.get("fresh_holdout_truth_accessed") is not True or failure.get("rerun_under_same_r1_identity") is not False:
        raise CloseoutValidationError("formal failure receipt access/no-rerun boundary mismatch")
    formal_dir = failure_path.parent
    forbidden_outputs = [name for name in ("report.json", "frames.jsonl", "components.jsonl") if (formal_dir / name).exists()]
    nested_outputs = [
        path.relative_to(formal_dir).as_posix()
        for path in formal_dir.rglob("*")
        if path.is_file() and path.name in {"report.json", "frames.jsonl", "components.jsonl"}
    ]
    if forbidden_outputs or nested_outputs:
        raise CloseoutValidationError("formal model-result outputs exist despite input-contract failure")
    manifest_path = validate_file_record(repo_root, freeze["fresh_holdout"]["manifest"], context="formal_failure.fresh_manifest")
    manifest_rows = read_jsonl(manifest_path)
    if len(manifest_rows) != int(freeze["fresh_holdout"]["formal_frame_count"]):
        raise CloseoutValidationError("fresh formal manifest denominator mismatch")
    first = manifest_rows[0]
    mask_path = (manifest_path.parent / str(first["semantic_mask_path"])).resolve()
    if not mask_path.is_file() or sha256_file(mask_path) != str(first["semantic_mask_sha256"]):
        raise CloseoutValidationError("first consumed formal mask SHA256 mismatch")
    unique_ids = evaluator_mask_ids(mask_path)
    outside = [value for value in unique_ids if value not in {0, 1, 2, 3}]
    if not outside:
        raise CloseoutValidationError("formal mask does not reproduce the recorded native/canonical mismatch")
    return {
        "failure_receipt_path": str(failure_path),
        "failure_receipt_sha256": sha256_file(failure_path),
        "fresh_manifest_sha256": sha256_file(manifest_path),
        "fresh_manifest_rows": len(manifest_rows),
        "first_consumed_mask_path": str(mask_path),
        "first_consumed_mask_sha256": sha256_file(mask_path),
        "first_consumed_mask_evaluator_observed_ids": unique_ids,
        "evaluator_ids_outside_canonical_0_3": outside,
        "formal_report_rows_created": 0,
        "failure_reproduced": True,
        "same_r1_rerun_authorized": False,
    }


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def run(repo_root: Path, output: Path) -> dict[str, Any]:
    protocol_path = repo_root / "docs/research/dual-loop/DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1_PROTOCOL_2026-07-31.json"
    freeze_path = repo_root / "artifacts.local/evidence/dual-loop-segmentation-model-selection-r1/formal_freeze_receipt.json"
    freeze_sha_path = freeze_path.with_suffix(".sha256.json")
    protocol = read_json(protocol_path)
    freeze = read_json(freeze_path)
    freeze_sha = read_json(freeze_sha_path)
    if protocol.get("protocol_id") != PROTOCOL_ID or protocol.get("status") != "DESIGN_FROZEN":
        raise CloseoutValidationError("R1 protocol identity/status mismatch")
    if freeze_sha.get("sha256") != sha256_file(freeze_path):
        raise CloseoutValidationError("formal freeze receipt sidecar SHA256 mismatch")
    freeze_validation = validate_freeze(repo_root, freeze)
    candidates = {
        name: validate_dev_candidate(
            repo_root,
            candidate_name=name,
            protocol=protocol,
            freeze_candidate=freeze["candidates"][name],
            shared_dev_yolo_trace_record=freeze["shared_dev_yolo_trace"],
        )
        for name in ("ddrnet23_slim", "segformer_b0")
    }
    formal_failure = validate_formal_failure(repo_root, freeze)
    result = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.closeout_validation.v2",
        "protocol_id": PROTOCOL_ID,
        "validation_status": "PARTIAL_VALIDATION",
        "terminal": "MODEL_SELECTION_NOT_EVALUABLE",
        "protocol_path": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "formal_freeze_path": str(freeze_path),
        "formal_freeze_sha256": sha256_file(freeze_path),
        "freeze_validation": freeze_validation,
        "candidate_dev_validation": candidates,
        "formal_failure_validation": formal_failure,
        "selection": {
            "selected_model": None,
            "formal_gate_checks_available": False,
            "reason": "fresh source-native mask violated the frozen evaluator's canonical 0..3 truth contract before formal result rows were created",
        },
        "device_benchmark_permission": {
            "authorized": False,
            "reason": "no candidate has a valid formal report plus independent gate validation",
        },
        "validation_limits": [
            "Development result files were not hash-bound before formal truth access",
            "candidate component recall cannot be independently reconstructed from the predicted-component ledger",
            "runtime P95 cannot be independently reconstructed without immutable per-frame timing rows",
        ],
        "authority_boundary": "Development pixel/component evidence only; no Android, QNN, risk-event, feedback, active-reminder, production, or safety claim",
    }
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite closeout validation: {output}")
    write_json_atomic(output, result)
    write_json_atomic(output.with_suffix(".sha256.json"), {"sha256": sha256_file(output)})
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        result = run(args.repo_root.resolve(), args.output.resolve())
    except Exception as exc:
        print(json.dumps({"validation_status": "INVALID", "terminal": "MODEL_SELECTION_NOT_EVALUABLE", "error": str(exc)}, ensure_ascii=False))
        raise
    print(json.dumps({"validation_status": result["validation_status"], "terminal": result["terminal"]}, ensure_ascii=False))
