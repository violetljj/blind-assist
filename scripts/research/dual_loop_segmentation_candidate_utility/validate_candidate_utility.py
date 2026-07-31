"""Independently validate candidate-utility R0 outputs and apply frozen gates."""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path
from typing import Any, Iterable

try:
    from . import PROTOCOL_ID
except ImportError:  # pragma: no cover - direct script execution
    from __init__ import PROTOCOL_ID


FRAME_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.frame.v1"
COMPONENT_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.component.v1"
RESULT_SCHEMA_VERSION = "blindassist.dual_loop_segmentation_candidate_utility_r0.validation.v1"
FORBIDDEN_KEYS = {"risk", "feedback", "event", "central_obstruction_agent_labels"}
PACKED_MASK_BYTES = (256 * 256 + 7) // 8


class CandidateUtilityValidationError(ValueError):
    """Raised when a result cannot be trusted or does not satisfy the contract."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CandidateUtilityValidationError(f"cannot read JSON {path}") from exc
    if not isinstance(value, dict):
        raise CandidateUtilityValidationError(f"{path}: expected JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise CandidateUtilityValidationError(f"cannot read JSONL {path}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise CandidateUtilityValidationError(f"{path}:{line_number}: blank row")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise CandidateUtilityValidationError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise CandidateUtilityValidationError(f"{path}:{line_number}: expected object")
            rows.append(row)
    if not rows:
        raise CandidateUtilityValidationError(f"{path}: empty JSONL")
    return rows


def _read_jsonl_allow_empty(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise CandidateUtilityValidationError(f"missing JSONL {path}")
    if path.stat().st_size == 0:
        return []
    return _read_jsonl(path)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)


def _forbidden_value(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in FORBIDDEN_KEYS:
                return f"{path}.{key}"
            found = _forbidden_value(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _forbidden_value(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _close(left: Any, right: Any, *, tolerance: float = 1e-8) -> bool:
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    except (TypeError, ValueError):
        return left == right


def _ratio(numerator: int, denominator: int, *, empty: float | None) -> float | None:
    return float(numerator / denominator) if denominator else empty


def _f1(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None:
        return None
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def _from_confusion(row: dict[str, Any]) -> dict[str, Any]:
    required = {"tp", "fp", "fn", "tn", "predicted_pixels", "truth_pixels"}
    if required - row.keys():
        raise CandidateUtilityValidationError("pixel metric row is missing confusion fields")
    tp, fp, fn, tn = (int(row[key]) for key in ("tp", "fp", "fn", "tn"))
    if min(tp, fp, fn, tn) < 0:
        raise CandidateUtilityValidationError("pixel metric row contains negative counts")
    total = tp + fp + fn + tn
    empty = tp + fp + fn == 0
    precision = _ratio(tp, tp + fp, empty=1.0 if empty else None)
    recall = _ratio(tp, tp + fn, empty=1.0 if empty else None)
    iou = _ratio(tp, tp + fp + fn, empty=1.0 if empty else None)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "predicted_pixels": int(row["predicted_pixels"]),
        "truth_pixels": int(row["truth_pixels"]),
        "pixel_count": total,
        "precision": precision,
        "recall": recall,
        "iou": iou,
        "f1": _f1(precision, recall),
        "false_positive_area_fraction": float(fp / total) if total else 0.0,
    }


def _aggregate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    if not values:
        raise CandidateUtilityValidationError("cannot aggregate zero rows")
    normalized = [_from_confusion(row) for row in values]
    totals = {
        key: sum(int(row[key]) for row in normalized)
        for key in ("tp", "fp", "fn", "tn", "predicted_pixels", "truth_pixels")
    }
    result = _from_confusion(totals)
    for metric in ("precision", "recall", "iou", "f1", "false_positive_area_fraction"):
        result[metric] = result[metric]
    frame_values = {
        metric: [row[metric] for row in normalized if row[metric] is not None]
        for metric in ("precision", "recall")
    }
    result["mean_frame_precision"] = (
        sum(frame_values["precision"]) / len(frame_values["precision"]) if frame_values["precision"] else None
    )
    result["mean_frame_recall"] = (
        sum(frame_values["recall"]) / len(frame_values["recall"]) if frame_values["recall"] else None
    )
    return result


def _assert_metric_row(row: dict[str, Any], *, context: str) -> dict[str, Any]:
    recomputed = _from_confusion(row)
    for key, expected in recomputed.items():
        if key in {"predicted_pixels", "truth_pixels", "pixel_count"}:
            continue
        if not _close(row.get(key), expected):
            raise CandidateUtilityValidationError(
                f"{context}: {key} inconsistent, stored={row.get(key)!r}, recomputed={expected!r}"
            )
    if int(row["predicted_pixels"]) != int(row["tp"]) + int(row["fp"]):
        raise CandidateUtilityValidationError(f"{context}: predicted pixel count inconsistent")
    if int(row["truth_pixels"]) != int(row["tp"]) + int(row["fn"]):
        raise CandidateUtilityValidationError(f"{context}: truth pixel count inconsistent")
    return recomputed


def _assert_aggregate(stored: dict[str, Any], recomputed: dict[str, Any], *, context: str) -> None:
    for key, expected in recomputed.items():
        if key not in stored:
            raise CandidateUtilityValidationError(f"{context}: missing aggregate field {key}")
        if not _close(stored[key], expected):
            raise CandidateUtilityValidationError(
                f"{context}: {key} inconsistent, stored={stored[key]!r}, recomputed={expected!r}"
            )


def _component_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        "component_precision": hit_predicted / predicted if predicted else (1.0 if truth == 0 else None),
        "component_recall": hit_truth / truth if truth else (1.0 if predicted == 0 else None),
        "false_activation_component_count": false_count,
        "false_activation_components_per_frame": false_count / len(rows) if rows else None,
    }


def _validate_temporal(report: dict[str, Any], *, require_motion_field: bool) -> None:
    temporal = report.get("temporal")
    if not isinstance(temporal, dict):
        raise CandidateUtilityValidationError("report.temporal must be an object")
    for source_id, source_values in temporal.items():
        if not isinstance(source_values, dict):
            raise CandidateUtilityValidationError(f"temporal source {source_id}: expected object")
        for class_name, summary in source_values.items():
            if not isinstance(summary, dict):
                raise CandidateUtilityValidationError(f"temporal {source_id}/{class_name}: expected object")
            for field in ("raw_adjacent_iou", "motion_warped_adjacent_iou", "motion_warp_available"):
                if field not in summary:
                    raise CandidateUtilityValidationError(f"temporal {source_id}/{class_name}: missing {field}")
            if require_motion_field and not isinstance(summary["motion_warp_available"], bool):
                raise CandidateUtilityValidationError(f"temporal {source_id}/{class_name}: motion availability must be bool")
            if summary["motion_warp_available"] and int(summary.get("motion_warped_pair_count", 0)) <= 0:
                raise CandidateUtilityValidationError(f"temporal {source_id}/{class_name}: availability/count mismatch")
            for percentile_name in ("raw_adjacent_iou", "motion_warped_adjacent_iou"):
                percentile = summary[percentile_name]
                if not isinstance(percentile, dict):
                    raise CandidateUtilityValidationError(f"temporal {source_id}/{class_name}: invalid {percentile_name}")
                for key in ("count", "median", "p90", "p95"):
                    if key not in percentile:
                        raise CandidateUtilityValidationError(
                            f"temporal {source_id}/{class_name}: missing {percentile_name}.{key}"
                        )


def _gate(actual: Any, threshold: Any, *, operator: str) -> dict[str, Any]:
    if actual is None:
        passed = False
    elif operator == ">=":
        passed = float(actual) >= float(threshold)
    elif operator == "<=":
        passed = float(actual) <= float(threshold)
    else:
        raise ValueError(operator)
    return {"actual": actual, "threshold": threshold, "operator": operator, "passed": bool(passed)}


def validate(
    *,
    protocol_path: Path,
    report_path: Path,
    frames_path: Path,
    components_path: Path,
    phase: str,
) -> dict[str, Any]:
    protocol = _read_json(protocol_path)
    report = _read_json(report_path)
    if protocol.get("protocol_id") != PROTOCOL_ID or report.get("protocol_id") != PROTOCOL_ID:
        raise CandidateUtilityValidationError("protocol id mismatch")
    if report.get("schema_version") != "blindassist.dual_loop_segmentation_candidate_utility_r0.result.v1":
        raise CandidateUtilityValidationError("unexpected evaluator report schema")
    if report.get("phase") != phase:
        raise CandidateUtilityValidationError("phase mismatch")
    frame_rows = _read_jsonl(frames_path)
    component_rows = _read_jsonl_allow_empty(components_path)
    if len(frame_rows) != int(report.get("frame_count", -1)):
        raise CandidateUtilityValidationError("frame count mismatch")
    frame_keys: set[tuple[str, int, str]] = set()
    for index, row in enumerate(frame_rows, start=1):
        forbidden = _forbidden_value(row)
        if forbidden:
            raise CandidateUtilityValidationError(f"frame row {index}: forbidden input at {forbidden}")
        if row.get("schema_version") != FRAME_SCHEMA_VERSION:
            raise CandidateUtilityValidationError(f"frame row {index}: schema mismatch")
        key = (str(row.get("source_id")), int(row.get("frame_id")), str(row.get("image_sha256")))
        if key in frame_keys:
            raise CandidateUtilityValidationError(f"frame row {index}: duplicate identity")
        frame_keys.add(key)
        packed = row.get("packed_masks")
        if not isinstance(packed, dict) or packed.get("shape") != [256, 256]:
            raise CandidateUtilityValidationError(f"frame row {index}: packed mask shape mismatch")
        for mask_name in ("candidate_hazard", "candidate_boundary_step_curb", "candidate_obstacle"):
            try:
                decoded = base64.b64decode(packed[mask_name], validate=True)
            except (KeyError, ValueError) as exc:
                raise CandidateUtilityValidationError(f"frame row {index}: invalid packed mask {mask_name}") from exc
            if len(decoded) != PACKED_MASK_BYTES:
                raise CandidateUtilityValidationError(f"frame row {index}: packed mask byte count mismatch")
        if row.get("truth_available"):
            if not isinstance(row.get("arms"), dict):
                raise CandidateUtilityValidationError(f"frame row {index}: truth frame has no arms")
            for arm in ("A", "B", "C"):
                _assert_metric_row(row["arms"][arm]["pixel"], context=f"frame {index} arm {arm}")
            _assert_metric_row(row["candidate_pixel_metrics"], context=f"frame {index} candidate")
            if row.get("unknown_nonwalkable_ablation") is not None:
                _assert_metric_row(row["unknown_nonwalkable_ablation"], context=f"frame {index} unknown")
        runtime = row.get("runtime")
        if not isinstance(runtime, dict):
            raise CandidateUtilityValidationError(f"frame row {index}: missing runtime")
        for field in ("segmentation_ms", "component_extraction_ms", "fusion_ms", "total_increment_ms"):
            if not math.isfinite(float(runtime[field])) or float(runtime[field]) < 0:
                raise CandidateUtilityValidationError(f"frame row {index}: invalid runtime {field}")

    component_ids: set[str] = set()
    for index, row in enumerate(component_rows, start=1):
        forbidden = _forbidden_value(row)
        if forbidden:
            raise CandidateUtilityValidationError(f"component row {index}: forbidden input at {forbidden}")
        if row.get("schema_version") != COMPONENT_SCHEMA_VERSION:
            raise CandidateUtilityValidationError(f"component row {index}: schema mismatch")
        component_id = str(row.get("component_id"))
        if component_id in component_ids:
            raise CandidateUtilityValidationError(f"component row {index}: duplicate component id")
        component_ids.add(component_id)
        if int(row.get("area_pixels", 0)) <= 0 or int(row.get("truth_intersection_pixels", -1)) < 0:
            raise CandidateUtilityValidationError(f"component row {index}: invalid area/intersection")
        if bool(row.get("truth_intersects")) != (int(row["truth_intersection_pixels"]) > 0):
            raise CandidateUtilityValidationError(f"component row {index}: truth intersection boolean mismatch")
        bbox = row.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4 or not (0 <= int(bbox[0]) < int(bbox[2]) <= 256 and 0 <= int(bbox[1]) < int(bbox[3]) <= 256):
            raise CandidateUtilityValidationError(f"component row {index}: invalid analysis-grid bbox")

    _validate_temporal(report, require_motion_field=True)
    truth_available = report.get("truth_status") == "source_native_pixel_truth"
    if truth_available:
        for arm in ("A", "B", "C"):
            aggregate = _aggregate([row["arms"][arm]["pixel"] for row in frame_rows])
            _assert_aggregate(report["summary"]["arm_pixel_metrics"][arm], aggregate, context=f"report arm {arm}")
        candidate_pixel = _aggregate([row["candidate_pixel_metrics"] for row in frame_rows])
        _assert_aggregate(report["summary"]["candidate_pixel_metrics"], candidate_pixel, context="report candidate pixel")
        candidate_components = _component_summary([row["candidate_component_metrics"] for row in frame_rows])
        _assert_aggregate(report["summary"]["candidate_components"], candidate_components, context="report candidate components")
        delta_recall = (
            report["summary"]["arm_pixel_metrics"]["C"]["recall"]
            - report["summary"]["arm_pixel_metrics"]["A"]["recall"]
        )
        if not _close(report["summary"]["delta_recall_C_minus_A"], delta_recall):
            raise CandidateUtilityValidationError("report delta recall is inconsistent")
        delta_fp = (
            report["summary"]["arm_pixel_metrics"]["C"]["false_positive_area_fraction"]
            - report["summary"]["arm_pixel_metrics"]["A"]["false_positive_area_fraction"]
        )
        if not _close(report["summary"]["delta_false_positive_area_fraction_C_minus_A"], delta_fp):
            raise CandidateUtilityValidationError("report delta false-positive area is inconsistent")
    else:
        if phase in {"calibration", "formal"}:
            raise CandidateUtilityValidationError("truth is required for calibration/formal validation")

    protocol_gates = protocol.get("decision_gates", {})
    if phase == "calibration":
        terminal = "CALIBRATION_VALID"
        gates: dict[str, Any] = {}
    elif phase == "temporal" or not truth_available:
        terminal = "CANDIDATE_UTILITY_NOT_EVALUABLE"
        gates = {}
    else:
        sessions: dict[str, list[dict[str, Any]]] = {}
        for row in frame_rows:
            sessions.setdefault(str(row["source_id"]), []).append(row)
        session_deltas: dict[str, float | None] = {}
        for source_id, rows in sessions.items():
            arm_a = _aggregate([row["arms"]["A"]["pixel"] for row in rows])
            arm_c = _aggregate([row["arms"]["C"]["pixel"] for row in rows])
            session_deltas[source_id] = (
                arm_c["recall"] - arm_a["recall"]
                if arm_c["recall"] is not None and arm_a["recall"] is not None
                else None
            )
        consistent = [
            source_id
            for source_id, delta in session_deltas.items()
            if delta is not None and delta >= float(protocol_gates["min_consistent_session_delta_recall"])
        ]
        summary = report["summary"]
        runtime = report["runtime"]
        gates = {
            "delta_recall": _gate(summary["delta_recall_C_minus_A"], protocol_gates["min_delta_recall"], operator=">="),
            "delta_false_positive_area_fraction": _gate(
                summary["delta_false_positive_area_fraction_C_minus_A"],
                protocol_gates["max_delta_false_positive_area_fraction"],
                operator="<=",
            ),
            "candidate_component_recall": _gate(
                summary["candidate_components"]["component_recall"],
                protocol_gates["min_candidate_component_recall"],
                operator=">=",
            ),
            "false_activation_components_per_frame": _gate(
                summary["candidate_components"]["false_activation_components_per_frame"],
                protocol_gates["max_false_activation_components_per_frame"],
                operator="<=",
            ),
            "consistent_source_sessions": {
                "actual": len(consistent),
                "threshold": protocol_gates["min_consistent_source_sessions"],
                "sessions": consistent,
                "session_deltas": session_deltas,
                "passed": len(consistent) >= int(protocol_gates["min_consistent_source_sessions"]),
            },
            "segmentation_p95_ms_host": _gate(
                runtime["segmentation_ms"]["p95"],
                protocol_gates["max_segmentation_p95_ms_host"],
                operator="<=",
            ),
            "incremental_p95_ms_host": _gate(
                runtime["total_increment_ms"]["p95"],
                protocol_gates["max_incremental_p95_ms_host"],
                operator="<=",
            ),
        }
        all_passed = all(
            value["passed"] if isinstance(value, dict) and "passed" in value else False
            for value in gates.values()
        )
        terminal = "CANDIDATE_UTILITY_SUPPORTED" if all_passed else "CURRENT_SEGMENTATION_REFERENCE_REJECTED"
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "phase": phase,
        "validation_status": "VALID",
        "terminal": terminal,
        "protocol_path": str(protocol_path),
        "report_path": str(report_path),
        "frames_path": str(frames_path),
        "components_path": str(components_path),
        "frame_count": len(frame_rows),
        "source_ids": sorted({str(row["source_id"]) for row in frame_rows}),
        "component_count": len(component_rows),
        "gate_checks": gates,
        "recomputed": {
            "truth_available": truth_available,
            "frame_identity_count": len(frame_keys),
            "component_identity_count": len(component_ids),
        },
        "authority_boundary": "pixel/component utility only; no Android, QNN, risk, event, feedback, or alert claim",
    }
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--components", type=Path, required=True)
    parser.add_argument("--phase", choices=("calibration", "formal", "temporal"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = validate(
            protocol_path=args.protocol.resolve(),
            report_path=args.report.resolve(),
            frames_path=args.frames.resolve(),
            components_path=args.components.resolve(),
            phase=args.phase,
        )
    except Exception as exc:
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "protocol_id": PROTOCOL_ID,
            "phase": args.phase,
            "validation_status": "INVALID_INPUT",
            "terminal": "CANDIDATE_UTILITY_NOT_EVALUABLE",
            "error": str(exc),
        }
        _write_json(args.output.resolve(), result)
        print(json.dumps(result, ensure_ascii=False))
        return 2
    _write_json(args.output.resolve(), result)
    print(json.dumps({"validation_status": result["validation_status"], "terminal": result["terminal"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
