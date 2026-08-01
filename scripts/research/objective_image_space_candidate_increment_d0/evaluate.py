"""Run the frozen objective image-space candidate increment D0 evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.research.dual_loop_segmentation_candidate_utility.component_metrics import (
    component_metrics,
    mask_iou,
)
from scripts.research.dual_loop_segmentation_candidate_utility.evaluate_candidate_utility import (
    box_union_mask,
    load_trace,
)
from scripts.research.riskseg_r1_p0.run_audit import (
    quantize_input,
    require_per_tensor_quantization,
)

from . import ANALYSIS_HEIGHT, ANALYSIS_WIDTH, CLASS_NAMES, PROTOCOL_ID
from .common import (
    ContractError,
    decision,
    event_summaries,
    load_objective_view,
    load_protocol,
    percentiles,
    require_hash,
    runtime_summary,
    sha256_file,
    summarize_masks,
    trapezoid_roi,
    write_json,
    write_jsonl,
)


def _safe_output(requested: Path, evidence_root: Path) -> tuple[Path, Path]:
    output = requested.resolve()
    root = evidence_root.resolve()
    try:
        output.relative_to(root)
    except ValueError as exc:
        raise ContractError("output must stay below the frozen evidence root") from exc
    if output == root or output.exists():
        raise ContractError("output must be a new child directory")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ContractError(f"temporary output exists: {temporary}")
    return output, temporary


def _class_view(
    truth_ids: np.ndarray, predicted_ids: np.ndarray, class_id: int
) -> tuple[np.ndarray, np.ndarray]:
    truth = np.where(truth_ids == 3, 3, np.where(truth_ids == class_id, 1, 0))
    predicted = np.where(
        truth_ids == 3, 3, np.where(predicted_ids == class_id, 1, 0)
    )
    return truth.astype(np.uint8), predicted.astype(np.uint8)


def _bottom_ratio(mask: np.ndarray) -> float | None:
    ys = np.nonzero(mask)[0]
    return float((int(np.max(ys)) + 1) / mask.shape[0]) if ys.size else None


def run(
    *,
    protocol_path: Path,
    manifest_path: Path,
    trace_path: Path,
    trace_receipt_path: Path,
    model_path: Path,
    output_dir: Path,
    threads: int,
) -> dict[str, Any]:
    protocol = load_protocol(protocol_path)
    inputs = protocol["inputs"]
    require_hash(manifest_path, inputs["objective_manifest_sha256"], "objective view")
    require_hash(model_path, inputs["pidnet_model_sha256"], "PIDNet model")
    receipt = json.loads(trace_receipt_path.read_text(encoding="utf-8"))
    trace_sha = sha256_file(trace_path)
    if (
        receipt.get("status") != "COMPLETE"
        or receipt.get("trace_sha256") != trace_sha
        or receipt.get("manifest_sha256") != inputs["yolo_input_manifest_sha256"]
        or receipt.get("model_sha256") != inputs["yolo_model_sha256"]
        or receipt.get("labels_sha256") != inputs["yolo_labels_sha256"]
        or int(receipt.get("frame_count", -1)) != int(protocol["data_role"]["frame_count"])
    ):
        raise ContractError("YOLO producer receipt contract mismatch")
    rows = load_objective_view(manifest_path)
    if len(rows) != int(protocol["data_role"]["frame_count"]):
        raise ContractError("objective frame count drift")
    if len({row["source_session_id"] for row in rows}) != int(
        protocol["data_role"]["source_session_count"]
    ):
        raise ContractError("objective session count drift")

    trace = load_trace(trace_path)
    expected_keys = {
        (
            str(row["source_session_id"]),
            int(row["source_frame_index"]),
            str(row["image_sha256"]),
        )
        for row in rows
    }
    if set(trace) != expected_keys:
        raise ContractError("YOLO trace is not an exact objective-view pairing")

    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=threads)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if len(input_details) != 1 or len(output_details) != 1:
        raise ContractError("PIDNet must expose exactly one input and one output")
    input_detail, output_detail = input_details[0], output_details[0]
    if (
        list(input_detail["shape"]) != [1, 288, 512, 3]
        or list(output_detail["shape"]) != [1, 288, 512, 4]
        or input_detail["dtype"] != np.int8
        or output_detail["dtype"] != np.int8
    ):
        raise ContractError("PIDNet tensor contract drift")
    require_per_tensor_quantization(input_detail, "input")
    require_per_tensor_quantization(output_detail, "output")

    output, temporary = _safe_output(
        output_dir, Path(protocol["implementation"]["evidence_root"])
    )
    temporary.mkdir(parents=True)
    count = len(rows)
    truth_ids = np.empty(
        (count, ANALYSIS_HEIGHT, ANALYSIS_WIDTH), dtype=np.uint8
    )
    predicted_ids = np.empty_like(truth_ids)
    detector_masks = np.empty_like(truth_ids, dtype=bool)
    feature_rows: list[dict[str, Any]] = []
    inference_ms: list[float] = []
    operator_ms: list[float] = []
    previous_candidate: dict[str, np.ndarray] = {}
    previous_area: dict[str, int] = {}
    replay_checked = False
    center_band = np.zeros((ANALYSIS_HEIGHT, ANALYSIS_WIDTH), dtype=bool)
    center_band[:, ANALYSIS_WIDTH // 3 : (2 * ANALYSIS_WIDTH) // 3] = True
    corridor = trapezoid_roi()

    try:
        for index, row in enumerate(rows):
            image_path = Path(str(row["image_path"]))
            mask_path = Path(str(row["oracle_mask_path"]))
            require_hash(image_path, str(row["image_sha256"]), f"image {index}")
            require_hash(mask_path, str(row["oracle_mask_sha256"]), f"mask {index}")
            with Image.open(mask_path) as opened:
                truth = np.asarray(opened.convert("L"), dtype=np.uint8)
            if truth.shape != (ANALYSIS_HEIGHT, ANALYSIS_WIDTH):
                raise ContractError(f"truth mask {index}: shape drift")
            if not np.isin(truth, (0, 1, 2, 3)).all():
                raise ContractError(f"truth mask {index}: unknown class ID")

            quantized = quantize_input(image_path, input_detail)
            interpreter.set_tensor(input_detail["index"], quantized)
            start = time.perf_counter()
            interpreter.invoke()
            raw = interpreter.get_tensor(output_detail["index"])
            elapsed_inference = (time.perf_counter() - start) * 1000.0
            if raw.dtype != np.int8 or raw.shape != (1, 288, 512, 4):
                raise ContractError("PIDNet output contract drift")
            raw_sha = hashlib.sha256(raw.tobytes(order="C")).hexdigest()
            if not replay_checked:
                interpreter.set_tensor(input_detail["index"], quantized)
                interpreter.invoke()
                replay = interpreter.get_tensor(output_detail["index"])
                if hashlib.sha256(replay.tobytes(order="C")).hexdigest() != raw_sha:
                    raise ContractError("same-backend inference is not deterministic")
                replay_checked = True
            argmax = np.argmax(raw[0], axis=-1).astype(np.uint8)
            predicted = np.asarray(
                Image.fromarray(argmax, mode="L").resize(
                    (ANALYSIS_WIDTH, ANALYSIS_HEIGHT), Image.Resampling.NEAREST
                ),
                dtype=np.uint8,
            )

            key = (
                str(row["source_session_id"]),
                int(row["source_frame_index"]),
                str(row["image_sha256"]),
            )
            trace_row = trace[key]
            detector = box_union_mask(
                trace_row["detections"],
                source_width=int(row["image_width"]),
                source_height=int(row["image_height"]),
            )
            op_start = time.perf_counter()
            valid = truth != 3
            truth_hazard = np.isin(truth, (1, 2)) & valid
            predicted_hazard = np.isin(predicted, (1, 2)) & valid
            candidate = predicted_hazard & ~detector
            residual_truth = truth_hazard & ~detector
            component = component_metrics(candidate, residual_truth)
            candidate_area = int(np.count_nonzero(candidate))
            valid_count = int(np.count_nonzero(valid))
            session = str(row["source_session_id"])
            previous = previous_candidate.get(session)
            adjacent_iou = mask_iou(previous, candidate) if previous is not None else None
            area_delta = (
                candidate_area - previous_area[session]
                if session in previous_area
                else None
            )
            elapsed_operator = (time.perf_counter() - op_start) * 1000.0
            previous_candidate[session] = candidate.copy()
            previous_area[session] = candidate_area

            truth_ids[index] = truth
            predicted_ids[index] = predicted
            detector_masks[index] = detector
            inference_ms.append(elapsed_inference)
            operator_ms.append(elapsed_operator)
            feature_rows.append(
                {
                    "schema_version": (
                        "blindassist.objective_image_space_candidate_increment_d0."
                        "frame_feature.v1"
                    ),
                    "source_session_id": session,
                    "observation_index": int(row["observation_index"]),
                    "source_frame_index": int(row["source_frame_index"]),
                    "timestamp_ns": int(row["timestamp_ns"]),
                    "image_sha256": row["image_sha256"],
                    "raw_int8_output_sha256": raw_sha,
                    "inference_ms": elapsed_inference,
                    "objective_operator_ms": elapsed_operator,
                    "candidate_area_fraction_valid": (
                        candidate_area / valid_count if valid_count else None
                    ),
                    "candidate_center_band_fraction_valid": (
                        int(np.count_nonzero(candidate & center_band)) / valid_count
                        if valid_count
                        else None
                    ),
                    "candidate_corridor_fraction_valid": (
                        int(np.count_nonzero(candidate & corridor)) / valid_count
                        if valid_count
                        else None
                    ),
                    "candidate_bottom_y_ratio": _bottom_ratio(candidate),
                    "candidate_area_delta_pixels": area_delta,
                    "candidate_adjacent_iou": adjacent_iou,
                    "candidate_component_count": component[
                        "predicted_component_count"
                    ],
                    "false_activation_component_count": component[
                        "false_activation_component_count"
                    ],
                }
            )

        ledger_path = temporary / "mask_ledger.npz"
        np.savez_compressed(
            ledger_path,
            truth_ids=truth_ids,
            predicted_ids=predicted_ids,
            detector_masks=detector_masks,
        )
        feature_path = temporary / "frame_features.jsonl"
        write_jsonl(feature_path, feature_rows)
        full = summarize_masks(truth_ids, predicted_ids, detector_masks)
        corridor_metrics = summarize_masks(
            truth_ids, predicted_ids, detector_masks, roi=corridor
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
                [
                    float(row["added_false_positive_area_fraction"])
                    for row in events
                ]
            ),
        }
        inference_runtime = runtime_summary(inference_ms)
        operator_runtime = runtime_summary(operator_ms)
        outcome = decision(
            protocol,
            full,
            corridor_metrics,
            by_class,
            events,
            operator_runtime,
        )
        report = {
            "schema_version": (
                "blindassist.objective_image_space_candidate_increment_d0.result.v1"
            ),
            "protocol_id": PROTOCOL_ID,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": outcome["terminal"],
            "authority": "CONSUMED_SESSION_DISJOINT_THESIS_DEVELOPMENT",
            "protocol_path": str(protocol_path),
            "protocol_sha256": sha256_file(protocol_path),
            "objective_manifest_path": str(manifest_path),
            "objective_manifest_sha256": sha256_file(manifest_path),
            "yolo_trace_path": str(trace_path),
            "yolo_trace_sha256": trace_sha,
            "yolo_trace_receipt_path": str(trace_receipt_path),
            "yolo_trace_receipt_sha256": sha256_file(trace_receipt_path),
            "pidnet_model_path": str(model_path),
            "pidnet_model_sha256": sha256_file(model_path),
            "mask_ledger_path": str((output / "mask_ledger.npz").resolve()),
            "mask_ledger_sha256": sha256_file(ledger_path),
            "frame_features_path": str((output / "frame_features.jsonl").resolve()),
            "frame_features_sha256": sha256_file(feature_path),
            "frame_count": count,
            "source_session_count": len(
                {str(row["source_session_id"]) for row in rows}
            ),
            "full_frame_valid": full,
            "frozen_corridor": corridor_metrics,
            "by_class": by_class,
            "event_summaries": events,
            "event_distribution": event_distribution,
            "runtime": {
                "pidnet_host_inference_descriptive_only": inference_runtime,
                "objective_operator": operator_runtime,
            },
            "decision": outcome,
            "forbidden_inputs_consumed": False,
            "timing_claim": "NOT_EVALUABLE_ONSET_INCOMPLETE",
        }
        write_json(temporary / "report.json", report)
        temporary.replace(output)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--trace-receipt", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    report = run(
        protocol_path=args.protocol.resolve(),
        manifest_path=args.manifest.resolve(),
        trace_path=args.trace.resolve(),
        trace_receipt_path=args.trace_receipt.resolve(),
        model_path=args.model.resolve(),
        output_dir=args.output_dir.resolve(),
        threads=args.threads,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "frame_count": report["frame_count"],
                "decision": report["decision"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
