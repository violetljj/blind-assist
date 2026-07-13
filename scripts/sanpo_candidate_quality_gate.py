#!/usr/bin/env python3
"""Evaluate SANPO segmentation quality without conflating device promotion.

The evaluator deliberately has three independent gates:

1. ``offline_training_quality`` evaluates the TensorFlow/Keras reference on dev;
2. ``int8_fidelity`` compares that reference with the exported full-INT8 model;
3. ``device_event`` consumes a separately produced, model-bound device report.

Passing the first two gates only makes a TFLite candidate eligible for the
device benchmark.  It never authorizes copying a model into the production app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import zip_longest
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sanpo_backend_equivalence
import sanpo_training_gate as training_gate
import train_export_sanpo_segmentation as shared


REPORT_SCHEMA = "blindassist_sanpo_candidate_quality_gate_v1"
DEVICE_REPORT_SCHEMA = "blindassist_sanpo_device_event_gate_input_v1"
SIDECAR_SUFFIX = ".sha256"
UNKNOWN_ID = shared.CLASS_IDS["unknown_nonwalkable"]
BOUNDARY_ID = shared.CLASS_IDS["boundary_step_curb"]


@dataclass(frozen=True)
class QualityThresholds:
    min_global_mean_iou: float = 0.45
    min_macro_session_mean_iou: float = 0.40
    min_worst_session_mean_iou: float = 0.25
    min_worst_scene_mean_iou: float = 0.30
    min_boundary_precision: float = 0.35
    min_boundary_recall: float = 0.50
    min_boundary_iou: float = 0.25
    min_unknown_precision: float = 0.50
    min_unknown_recall: float = 0.60
    min_dev_sessions: int = 2
    min_dev_scenes: int = 2


@dataclass(frozen=True)
class FidelityThresholds:
    min_argmax_agreement: float = 0.995
    min_per_class_prediction_iou: float = 0.97
    max_per_class_ground_truth_iou_drop: float = 0.02
    max_mean_iou_drop: float = 0.01


@dataclass(frozen=True)
class DeviceEventThresholds:
    min_event_recall: float = 0.90
    max_critical_miss_rate: float = 0.05
    max_false_alerts_per_minute: float = 0.50
    min_post_event_clearance_rate: float = 0.90
    max_repeated_alert_rate: float = 0.10
    max_p95_latency_ms: float = 100.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def confusion_matrix(predictions: Iterable[np.ndarray], targets: Iterable[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((len(shared.CLASS_NAMES), len(shared.CLASS_NAMES)), dtype=np.int64)
    sentinel = object()
    for prediction, target in zip_longest(predictions, targets, fillvalue=sentinel):
        if prediction is sentinel or target is sentinel:
            raise ValueError("prediction/target count mismatch")
        predicted = np.asarray(prediction, dtype=np.int64).reshape(-1)
        expected = np.asarray(target, dtype=np.int64).reshape(-1)
        if predicted.shape != expected.shape:
            raise ValueError("prediction/target shape mismatch")
        valid = (
            (expected >= 0)
            & (expected < len(shared.CLASS_NAMES))
            & (predicted >= 0)
            & (predicted < len(shared.CLASS_NAMES))
        )
        np.add.at(matrix, (expected[valid], predicted[valid]), 1)
    return matrix


def metrics_from_confusion(matrix: np.ndarray) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    expected_shape = (len(shared.CLASS_NAMES), len(shared.CLASS_NAMES))
    if matrix.shape != expected_shape or np.any(matrix < 0):
        raise ValueError(f"confusion matrix must be non-negative and shaped {expected_shape}")
    per_class: dict[str, dict[str, float | int | None]] = {}
    defined_ious: list[float] = []
    for class_id, name in enumerate(shared.CLASS_NAMES):
        tp = int(matrix[class_id, class_id])
        fp = int(matrix[:, class_id].sum() - tp)
        fn = int(matrix[class_id, :].sum() - tp)
        union = tp + fp + fn
        iou = _safe_ratio(tp, union)
        precision = _safe_ratio(tp, tp + fp)
        recall = _safe_ratio(tp, tp + fn)
        per_class[name] = {
            "true_positive_pixels": tp,
            "false_positive_pixels": fp,
            "false_negative_pixels": fn,
            "true_pixels": int(matrix[class_id, :].sum()),
            "predicted_pixels": int(matrix[:, class_id].sum()),
            "precision": precision,
            "recall": recall,
            "iou": iou,
        }
        if iou is not None:
            defined_ious.append(iou)
    total = int(matrix.sum())
    predicted_unknown = int(matrix[:, UNKNOWN_ID].sum())
    covered = total - predicted_unknown
    covered_correct = int(sum(matrix[index, index] for index in range(len(shared.CLASS_NAMES)) if index != UNKNOWN_ID))
    unknown = per_class[shared.CLASS_NAMES[UNKNOWN_ID]]
    boundary = per_class[shared.CLASS_NAMES[BOUNDARY_ID]]
    return {
        "confusion_matrix": matrix.tolist(),
        "per_class": per_class,
        "mean_iou": float(np.mean(defined_ious)) if defined_ious else None,
        "pixel_accuracy": _safe_ratio(int(np.trace(matrix)), total),
        "boundary": {
            "precision": boundary["precision"],
            "recall": boundary["recall"],
            "iou": boundary["iou"],
            "true_pixels": boundary["true_pixels"],
            "predicted_pixels": boundary["predicted_pixels"],
        },
        "unknown_abstention": {
            "abstain_rate": _safe_ratio(predicted_unknown, total),
            "known_coverage": _safe_ratio(covered, total),
            "unknown_precision": unknown["precision"],
            "unknown_recall": unknown["recall"],
            "unknown_iou": unknown["iou"],
            "covered_pixel_accuracy": _safe_ratio(covered_correct, covered),
            "predicted_unknown_pixels": predicted_unknown,
            "covered_pixels": covered,
        },
    }


def stratified_metrics(
    records: Sequence[shared.Record], predictions: Sequence[np.ndarray], targets: Sequence[np.ndarray]
) -> dict[str, Any]:
    if not records or len(records) != len(predictions) or len(records) != len(targets):
        raise ValueError("records, predictions and targets must be non-empty and aligned")

    def aggregate(indices: Sequence[int]) -> dict[str, Any]:
        return metrics_from_confusion(confusion_matrix((predictions[i] for i in indices), (targets[i] for i in indices)))

    sessions: dict[str, list[int]] = {}
    scenes: dict[str, list[int]] = {}
    for index, record in enumerate(records):
        sessions.setdefault(record.session_id, []).append(index)
        scenes.setdefault(record.scene_bucket or "__missing_scene_bucket__", []).append(index)
    per_session = {
        name: {"sample_count": len(indices), **aggregate(indices)} for name, indices in sorted(sessions.items())
    }
    per_scene = {
        name: {
            "sample_count": len(indices),
            "session_count": len({records[i].session_id for i in indices}),
            **aggregate(indices),
        }
        for name, indices in sorted(scenes.items())
    }
    session_mious = [item["mean_iou"] for item in per_session.values() if item["mean_iou"] is not None]
    scene_mious = [item["mean_iou"] for item in per_scene.values() if item["mean_iou"] is not None]
    return {
        "global": aggregate(list(range(len(records)))),
        "per_session": per_session,
        "per_scene": per_scene,
        "macro_session_mean_iou": float(np.mean(session_mious)) if session_mious else None,
        "worst_session_mean_iou": min(session_mious) if session_mious else None,
        "worst_scene_mean_iou": min(scene_mious) if scene_mious else None,
    }


def _at_least(value: float | int | None, threshold: float | int) -> bool:
    return value is not None and value >= threshold


def _at_most(value: float | int | None, threshold: float | int) -> bool:
    return value is not None and value <= threshold


def evaluate_training_quality(metrics: Mapping[str, Any], thresholds: QualityThresholds) -> dict[str, Any]:
    global_metrics = metrics["global"]
    boundary = global_metrics["boundary"]
    unknown = global_metrics["unknown_abstention"]
    checks = {
        "global_mean_iou": _at_least(global_metrics["mean_iou"], thresholds.min_global_mean_iou),
        "macro_session_mean_iou": _at_least(metrics["macro_session_mean_iou"], thresholds.min_macro_session_mean_iou),
        "worst_session_mean_iou": _at_least(metrics["worst_session_mean_iou"], thresholds.min_worst_session_mean_iou),
        "worst_scene_mean_iou": _at_least(metrics["worst_scene_mean_iou"], thresholds.min_worst_scene_mean_iou),
        "boundary_precision": _at_least(boundary["precision"], thresholds.min_boundary_precision),
        "boundary_recall": _at_least(boundary["recall"], thresholds.min_boundary_recall),
        "boundary_iou": _at_least(boundary["iou"], thresholds.min_boundary_iou),
        "unknown_precision": _at_least(unknown["unknown_precision"], thresholds.min_unknown_precision),
        "unknown_recall": _at_least(unknown["unknown_recall"], thresholds.min_unknown_recall),
        "dev_session_count": len(metrics["per_session"]) >= thresholds.min_dev_sessions,
        "dev_scene_count": len(metrics["per_scene"]) >= thresholds.min_dev_scenes,
        "scene_bucket_complete": "__missing_scene_bucket__" not in metrics["per_scene"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "green" if not failed else "red",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": asdict(thresholds),
    }


def quantization_fidelity(
    reference_predictions: Sequence[np.ndarray],
    quantized_predictions: Sequence[np.ndarray],
    targets: Sequence[np.ndarray],
) -> dict[str, Any]:
    if not reference_predictions or len(reference_predictions) != len(quantized_predictions) or len(targets) != len(reference_predictions):
        raise ValueError("reference, quantized and target arrays must be non-empty and aligned")
    reference_flat = np.concatenate([np.asarray(item).reshape(-1) for item in reference_predictions])
    quantized_flat = np.concatenate([np.asarray(item).reshape(-1) for item in quantized_predictions])
    agreement = float(np.mean(reference_flat == quantized_flat))
    prediction_iou: dict[str, float] = {}
    for class_id, name in enumerate(shared.CLASS_NAMES):
        reference_mask = reference_flat == class_id
        quantized_mask = quantized_flat == class_id
        union = int(np.count_nonzero(reference_mask | quantized_mask))
        intersection = int(np.count_nonzero(reference_mask & quantized_mask))
        prediction_iou[name] = float(intersection / union) if union else 1.0
    reference_metrics = metrics_from_confusion(confusion_matrix(reference_predictions, targets))
    quantized_metrics = metrics_from_confusion(confusion_matrix(quantized_predictions, targets))
    ground_truth_iou_drop: dict[str, float] = {}
    defined_drops: list[float] = []
    for name in shared.CLASS_NAMES:
        before = reference_metrics["per_class"][name]["iou"]
        after = quantized_metrics["per_class"][name]["iou"]
        drop = 0.0 if before is None and after is None else float((before or 0.0) - (after or 0.0))
        ground_truth_iou_drop[name] = drop
        defined_drops.append(drop)
    reference_mean = reference_metrics["mean_iou"] or 0.0
    quantized_mean = quantized_metrics["mean_iou"] or 0.0
    return {
        "argmax_agreement": agreement,
        "per_class_prediction_iou": prediction_iou,
        "per_class_ground_truth_iou_drop": ground_truth_iou_drop,
        "max_per_class_ground_truth_iou_drop": max(defined_drops),
        "mean_iou_drop": float(reference_mean - quantized_mean),
        "reference_ground_truth_metrics": reference_metrics,
        "quantized_ground_truth_metrics": quantized_metrics,
    }


def evaluate_fidelity(metrics: Mapping[str, Any], thresholds: FidelityThresholds) -> dict[str, Any]:
    checks = {
        "argmax_agreement": _at_least(metrics["argmax_agreement"], thresholds.min_argmax_agreement),
        "per_class_prediction_iou": min(metrics["per_class_prediction_iou"].values()) >= thresholds.min_per_class_prediction_iou,
        "per_class_ground_truth_iou_drop": _at_most(
            metrics["max_per_class_ground_truth_iou_drop"], thresholds.max_per_class_ground_truth_iou_drop
        ),
        "mean_iou_drop": _at_most(metrics["mean_iou_drop"], thresholds.max_mean_iou_drop),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "green" if not failed else "red",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": asdict(thresholds),
    }


def evaluate_device_event_report(
    payload: Mapping[str, Any] | None, model_sha256: str, thresholds: DeviceEventThresholds
) -> dict[str, Any]:
    if payload is None:
        return {
            "status": "not_evaluated",
            "passed": False,
            "failed_checks": ["device_event_report_missing"],
            "thresholds": asdict(thresholds),
            "note": "Run the separately versioned same-device continuous-sequence event benchmark.",
        }
    if payload.get("schema") != DEVICE_REPORT_SCHEMA:
        raise ValueError(f"device event report schema must be {DEVICE_REPORT_SCHEMA}")
    if payload.get("model_sha256") != model_sha256:
        raise ValueError("device event report is not bound to the evaluated TFLite SHA256")
    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("device event report requires a metrics object")
    required = {
        "event_recall",
        "critical_miss_rate",
        "false_alerts_per_minute",
        "post_event_clearance_rate",
        "repeated_alert_rate",
        "p95_latency_ms",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise ValueError(f"device event report missing metrics: {missing}")
    checks = {
        "event_recall": _at_least(metrics["event_recall"], thresholds.min_event_recall),
        "critical_miss_rate": _at_most(metrics["critical_miss_rate"], thresholds.max_critical_miss_rate),
        "false_alerts_per_minute": _at_most(metrics["false_alerts_per_minute"], thresholds.max_false_alerts_per_minute),
        "post_event_clearance_rate": _at_least(metrics["post_event_clearance_rate"], thresholds.min_post_event_clearance_rate),
        "repeated_alert_rate": _at_most(metrics["repeated_alert_rate"], thresholds.max_repeated_alert_rate),
        "p95_latency_ms": _at_most(metrics["p95_latency_ms"], thresholds.max_p95_latency_ms),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "status": "green" if not failed else "red",
        "passed": not failed,
        "checks": checks,
        "failed_checks": failed,
        "thresholds": asdict(thresholds),
        "metrics": dict(metrics),
        "report_id": payload.get("report_id"),
    }


def _quantize_input(image: np.ndarray, detail: Mapping[str, Any]) -> np.ndarray:
    scale, zero_point = detail["quantization"]
    if scale <= 0 or detail["dtype"] != np.int8:
        raise ValueError("TFLite input must use per-tensor INT8 quantization")
    quantized = np.rint(np.asarray(image, dtype=np.float32) / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)[None, ...]


def infer_tflite(tf: Any, model_path: Path, images: Sequence[np.ndarray]) -> list[np.ndarray]:
    interpreter = tf.lite.Interpreter(model_path=str(model_path), num_threads=1)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("TFLite evaluator requires exactly one input and one output")
    input_detail, output_detail = inputs[0], outputs[0]
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError("TFLite fidelity gate requires full-INT8 input/output")
    predictions: list[np.ndarray] = []
    for image in images:
        interpreter.set_tensor(input_detail["index"], _quantize_input(image, input_detail))
        interpreter.invoke()
        logits = interpreter.get_tensor(output_detail["index"])
        predictions.append(np.argmax(logits[0], axis=-1).astype(np.uint8))
    return predictions


def infer_keras(model: Any, images: Sequence[np.ndarray], batch_size: int) -> list[np.ndarray]:
    predictions: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        logits = model.predict(np.stack(images[start : start + batch_size]), verbose=0)
        predictions.extend(np.argmax(logits, axis=-1).astype(np.uint8))
    return predictions


def write_report(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    digest = sha256_file(path)
    path.with_name(path.name + SIDECAR_SUFFIX).write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = shared.project_root()
    dataset_root = shared.resolve(root, args.dataset_root).resolve()
    manifest = dataset_root / training_gate.CANONICAL_TRAINING_MANIFEST
    weights = shared.resolve(root, args.weights).resolve()
    tflite = shared.resolve(root, args.tflite).resolve() if args.tflite else None
    report_path = shared.resolve(root, args.report).resolve()
    production_assets = (root / "app" / "src" / "main" / "assets").resolve()
    if tflite is not None and tflite.is_relative_to(production_assets):
        raise ValueError("Refusing to evaluate/write a production app model path; use a benchmark-only TFLite artifact")
    if not weights.is_file():
        raise FileNotFoundError("weights file does not exist")
    if tflite is not None and not tflite.is_file():
        raise FileNotFoundError("benchmark-only TFLite does not exist")
    if args.device_event_report and tflite is None:
        raise ValueError("--device-event-report requires --tflite for SHA256 binding")
    gate_path = shared.resolve(dataset_root, args.training_gate_report).resolve()
    gate_report = training_gate.consume_training_authorization(dataset_root, gate_path)
    equivalence_path = shared.resolve(root, args.backend_equivalence_report).resolve()
    equivalence = sanpo_backend_equivalence.consume_equivalence_authorization(
        weights,
        equivalence_path,
        backbone_alpha=args.backbone_alpha,
        decoder_channels=args.decoder_channels,
        input_size=args.input_size,
        detail_output_stride=args.detail_output_stride,
        semantic_output_stride=args.semantic_output_stride,
    )
    records = shared.load_records(manifest)
    dev_records = shared.records_by_split(records, "dev")
    if not dev_records:
        raise ValueError("canonical training manifest contains no dev records")

    import tensorflow as tf

    shared.set_determinism(tf, args.seed)
    model = shared.build_mobilenetv3_lraspp(
        tf,
        args.input_size,
        backbone_alpha=args.backbone_alpha,
        decoder_channels=args.decoder_channels,
        detail_output_stride=args.detail_output_stride,
        semantic_output_stride=args.semantic_output_stride,
    )
    model.load_weights(weights)
    examples = [shared.load_example(record, args.input_size) for record in dev_records]
    images = [item[0] for item in examples]
    targets = [item[1] for item in examples]
    keras_predictions = infer_keras(model, images, args.batch_size)
    stratified = stratified_metrics(dev_records, keras_predictions, targets)
    quality_gate = evaluate_training_quality(stratified, QualityThresholds())
    fidelity_metrics = None
    if tflite is not None:
        shared.validate_int8_tflite(tf, tflite, args.input_size)
        tflite_predictions = infer_tflite(tf, tflite, images)
        fidelity_metrics = quantization_fidelity(keras_predictions, tflite_predictions, targets)
        fidelity_gate = evaluate_fidelity(fidelity_metrics, FidelityThresholds())
    else:
        fidelity_gate = {
            "status": "not_evaluated",
            "passed": False,
            "failed_checks": ["benchmark_tflite_missing"],
            "thresholds": asdict(FidelityThresholds()),
            "note": "Offline quality can be audited before export; INT8 fidelity requires --tflite.",
        }
    device_payload = None
    device_path = None
    if args.device_event_report:
        device_path = shared.resolve(root, args.device_event_report).resolve()
        device_payload = json.loads(device_path.read_text(encoding="utf-8"))
    tflite_sha256 = sha256_file(tflite) if tflite is not None else None
    device_gate = evaluate_device_event_report(device_payload, tflite_sha256 or "", DeviceEventThresholds())
    device_candidate_eligible = quality_gate["passed"] and fidelity_gate["passed"]
    benchmark_promotion_eligible = device_candidate_eligible and device_gate["passed"]
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_only": True,
        "production_model_replacement_authorized": False,
        "decision": {
            "device_benchmark_candidate_eligible": device_candidate_eligible,
            "benchmark_promotion_eligible": benchmark_promotion_eligible,
            "reason": (
                "all gates green; an explicit human-reviewed production release decision is still required"
                if benchmark_promotion_eligible
                else "one or more independent gates are red or not evaluated"
            ),
        },
        "bindings": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "training_gate_report": str(gate_path),
            "training_gate_report_sha256": gate_report["report_sha256"],
            "weights": str(weights),
            "weights_sha256": sha256_file(weights),
            "model_config": sanpo_backend_equivalence.model_config(
                args.backbone_alpha, args.decoder_channels, args.input_size,
                args.detail_output_stride, args.semantic_output_stride,
            ),
            "model_config_sha256": sanpo_backend_equivalence.model_config_sha256(
                sanpo_backend_equivalence.model_config(
                    args.backbone_alpha, args.decoder_channels, args.input_size,
                    args.detail_output_stride, args.semantic_output_stride,
                )
            ),
            "backend_equivalence_report": str(equivalence_path),
            "backend_equivalence_report_sha256": equivalence["report_sha256"],
            "tflite": str(tflite) if tflite is not None else None,
            "tflite_sha256": tflite_sha256,
            "device_event_report": str(device_path) if device_path else None,
        },
        "offline_training_quality": {"gate": quality_gate, "metrics": stratified},
        "int8_fidelity": {"gate": fidelity_gate, "metrics": fidelity_metrics},
        "device_event": {"gate": device_gate},
    }
    report["report_sha256"] = write_report(report_path, report)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run independent SANPO offline-quality, INT8-fidelity and device-event gates.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--training-gate-report", default="qa/training_gate_report.json")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--backend-equivalence-report", required=True)
    parser.add_argument("--tflite", help="Optional benchmark-only full-INT8 TFLite candidate; omit for pre-export quality audit.")
    parser.add_argument("--device-event-report", help="Optional separately generated same-device event report.")
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--input-size", type=int, default=shared.INPUT_SIZE,
        choices=sanpo_backend_equivalence.ALLOWED_INPUT_SIZES,
    )
    parser.add_argument(
        "--backbone-alpha", type=float,
        choices=sanpo_backend_equivalence.ALLOWED_BACKBONE_ALPHAS,
        default=sanpo_backend_equivalence.DEFAULT_BACKBONE_ALPHA,
    )
    parser.add_argument(
        "--decoder-channels", type=int,
        default=sanpo_backend_equivalence.DEFAULT_DECODER_CHANNELS,
    )
    parser.add_argument(
        "--detail-output-stride", type=int, choices=(4, 8),
        default=sanpo_backend_equivalence.DEFAULT_DETAIL_OUTPUT_STRIDE,
    )
    parser.add_argument(
        "--semantic-output-stride", type=int, choices=(16, 32),
        default=sanpo_backend_equivalence.DEFAULT_SEMANTIC_OUTPUT_STRIDE,
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.decoder_channels <= 0:
        parser.error("batch-size and decoder-channels must be positive")
    return args


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["decision"], ensure_ascii=False))


if __name__ == "__main__":
    main()
