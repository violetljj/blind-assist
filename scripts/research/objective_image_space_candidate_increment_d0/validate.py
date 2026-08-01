"""Independently recompute D0 aggregates and the frozen terminal from mask ledgers."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .common import (
    ContractError,
    decision,
    event_summaries,
    load_objective_view,
    load_protocol,
    percentiles,
    read_json,
    read_jsonl,
    require_hash,
    runtime_summary,
    sha256_file,
    summarize_masks,
    trapezoid_roi,
    write_json,
)


def _equal(left: Any, right: Any, path: str = "root") -> None:
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            raise ContractError(f"{path}: key mismatch")
        for key in left:
            _equal(left[key], right[key], f"{path}.{key}")
        return
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            raise ContractError(f"{path}: list length mismatch")
        for index, (lvalue, rvalue) in enumerate(zip(left, right, strict=True)):
            _equal(lvalue, rvalue, f"{path}[{index}]")
        return
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        if not math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-12):
            raise ContractError(f"{path}: {left} != {right}")
        return
    if left != right:
        raise ContractError(f"{path}: {left!r} != {right!r}")


def _class_view(
    truth_ids: np.ndarray, predicted_ids: np.ndarray, class_id: int
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.where(truth_ids == 3, 3, np.where(truth_ids == class_id, 1, 0))
    predicted = np.where(
        truth_ids == 3, 3, np.where(predicted_ids == class_id, 1, 0)
    )
    return truth.astype(np.uint8), predicted.astype(np.uint8)


def validate(
    *,
    protocol_path: Path,
    report_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    report = read_json(report_path)
    if report.get("protocol_id") != protocol["protocol_id"]:
        raise ContractError("report protocol mismatch")
    if report.get("protocol_sha256") != sha256_file(protocol_path):
        raise ContractError("report protocol SHA mismatch")
    manifest_path = Path(report["objective_manifest_path"])
    trace_path = Path(report["yolo_trace_path"])
    trace_receipt_path = Path(report["yolo_trace_receipt_path"])
    model_path = Path(report["pidnet_model_path"])
    require_hash(
        manifest_path,
        protocol["inputs"]["objective_manifest_sha256"],
        "objective manifest",
    )
    require_hash(trace_path, report["yolo_trace_sha256"], "YOLO trace")
    require_hash(
        trace_receipt_path,
        report["yolo_trace_receipt_sha256"],
        "YOLO trace receipt",
    )
    trace_receipt = read_json(trace_receipt_path)
    if (
        trace_receipt.get("trace_sha256") != report["yolo_trace_sha256"]
        or trace_receipt.get("manifest_sha256")
        != protocol["inputs"]["yolo_input_manifest_sha256"]
        or trace_receipt.get("model_sha256")
        != protocol["inputs"]["yolo_model_sha256"]
        or trace_receipt.get("labels_sha256")
        != protocol["inputs"]["yolo_labels_sha256"]
    ):
        raise ContractError("YOLO receipt binding drift")
    require_hash(model_path, protocol["inputs"]["pidnet_model_sha256"], "PIDNet")
    rows = load_objective_view(manifest_path)

    ledger_path = report_path.parent / "mask_ledger.npz"
    feature_path = report_path.parent / "frame_features.jsonl"
    require_hash(ledger_path, report["mask_ledger_sha256"], "mask ledger")
    require_hash(feature_path, report["frame_features_sha256"], "frame features")
    with np.load(ledger_path, allow_pickle=False) as ledger:
        if set(ledger.files) != {"truth_ids", "predicted_ids", "detector_masks"}:
            raise ContractError("mask ledger fields drift")
        truth_ids = np.asarray(ledger["truth_ids"], dtype=np.uint8)
        predicted_ids = np.asarray(ledger["predicted_ids"], dtype=np.uint8)
        detector_masks = np.asarray(ledger["detector_masks"], dtype=bool)
    expected_shape = (len(rows), 256, 256)
    if (
        truth_ids.shape != expected_shape
        or predicted_ids.shape != expected_shape
        or detector_masks.shape != expected_shape
    ):
        raise ContractError("mask ledger shape drift")
    if not np.isin(truth_ids, (0, 1, 2, 3)).all():
        raise ContractError("truth ledger class drift")
    if not np.isin(predicted_ids, (0, 1, 2, 3)).all():
        raise ContractError("prediction ledger class drift")

    features = read_jsonl(feature_path)
    if len(features) != len(rows):
        raise ContractError("feature row count drift")
    for index, (source, feature) in enumerate(zip(rows, features, strict=True)):
        identity = (
            str(source["source_session_id"]),
            int(source["source_frame_index"]),
            str(source["image_sha256"]),
        )
        observed = (
            str(feature["source_session_id"]),
            int(feature["source_frame_index"]),
            str(feature["image_sha256"]),
        )
        if identity != observed:
            raise ContractError(f"feature identity drift at row {index}")
        if (
            len(str(feature["raw_int8_output_sha256"])) != 64
            or float(feature["objective_operator_ms"]) < 0
            or float(feature["inference_ms"]) < 0
        ):
            raise ContractError(f"feature contract drift at row {index}")

    full = summarize_masks(truth_ids, predicted_ids, detector_masks)
    corridor = summarize_masks(
        truth_ids, predicted_ids, detector_masks, roi=trapezoid_roi()
    )
    by_class: dict[str, dict[str, Any]] = {}
    for class_id, name in ((1, "blocking_obstacle"), (2, "boundary_level_change")):
        class_truth, class_predicted = _class_view(
            truth_ids, predicted_ids, class_id
        )
        by_class[name] = summarize_masks(
            class_truth, class_predicted, detector_masks
        )
    events = event_summaries(rows, truth_ids, predicted_ids, detector_masks)
    event_distribution = {
        "delta_recall_c_minus_a": percentiles(
            [float(row["delta_recall_c_minus_a"]) for row in events]
        ),
        "added_false_positive_area_fraction": percentiles(
            [float(row["added_false_positive_area_fraction"]) for row in events]
        ),
    }
    inference_runtime = runtime_summary(
        [float(row["inference_ms"]) for row in features]
    )
    operator_runtime = runtime_summary(
        [float(row["objective_operator_ms"]) for row in features]
    )
    outcome = decision(
        protocol, full, corridor, by_class, events, operator_runtime
    )
    _equal(full, report["full_frame_valid"], "full_frame_valid")
    _equal(corridor, report["frozen_corridor"], "frozen_corridor")
    _equal(by_class, report["by_class"], "by_class")
    _equal(events, report["event_summaries"], "event_summaries")
    _equal(event_distribution, report["event_distribution"], "event_distribution")
    _equal(
        inference_runtime,
        report["runtime"]["pidnet_host_inference_descriptive_only"],
        "runtime.inference",
    )
    _equal(
        operator_runtime,
        report["runtime"]["objective_operator"],
        "runtime.operator",
    )
    _equal(outcome, report["decision"], "decision")
    if report["status"] != outcome["terminal"]:
        raise ContractError("terminal mismatch")
    if report.get("forbidden_inputs_consumed") is not False:
        raise ContractError("forbidden input firewall failed")
    if report.get("timing_claim") != "NOT_EVALUABLE_ONSET_INCOMPLETE":
        raise ContractError("timing claim boundary drift")

    result = {
        "schema_version": (
            "blindassist.objective_image_space_candidate_increment_d0.validation.v1"
        ),
        "protocol_id": protocol["protocol_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "VALID",
        "terminal": outcome["terminal"],
        "report_sha256": sha256_file(report_path),
        "mask_ledger_sha256": sha256_file(ledger_path),
        "frame_features_sha256": sha256_file(feature_path),
        "frame_count": len(rows),
        "source_session_count": len(
            {str(row["source_session_id"]) for row in rows}
        ),
        "independent_recomputation": True,
        "timing_status": "NOT_EVALUABLE_ONSET_INCOMPLETE",
    }
    write_json(output_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(
        protocol_path=args.protocol.resolve(),
        report_path=args.report.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
