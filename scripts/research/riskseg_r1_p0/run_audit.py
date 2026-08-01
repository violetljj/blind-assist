"""Run the frozen RISKSEG-R1 P0 soft dense adapter audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .core import (
    CLASS_ORDER,
    MODEL_ARMS,
    ORACLE_ARM,
    adapter_configs,
    canonical_sha256,
    fold_assignments,
    nested_oof_score,
    pool_probabilities,
    read_object,
    sha256_file,
    stable_softmax_from_int8,
    timing_against_yolo,
    write_object,
)


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def verify_bound_file(root: Path, item: dict[str, Any]) -> Path:
    path = (root / item["path"]).resolve()
    actual = sha256_file(path)
    if actual != str(item["sha256"]).lower():
        raise ValueError(f"{path}: SHA-256 {actual} != {item['sha256']}")
    return path


def verify_implementation_lock(root: Path, contract: dict[str, Any]) -> None:
    lock = contract["implementation_lock"]
    paths = {
        "core_py_sha256": root / "scripts/research/riskseg_r1_p0/core.py",
        "run_audit_py_sha256": root
        / "scripts/research/riskseg_r1_p0/run_audit.py",
        "validate_audit_py_sha256": root
        / "scripts/research/riskseg_r1_p0/validate_audit.py",
        "test_core_py_sha256": root
        / "scripts/research/riskseg_r1_p0/test_core.py",
    }
    for key, path in paths.items():
        if sha256_file(path) != str(lock[key]).lower():
            raise ValueError(f"{path}: implementation lock drift")


def bounded_new_output(root: Path, requested: Path) -> tuple[Path, Path]:
    evidence_root = (root / "artifacts.local/evidence/riskseg-r1").resolve()
    output = requested.resolve()
    try:
        output.relative_to(evidence_root)
    except ValueError as exc:
        raise ValueError(f"output must stay below {evidence_root}") from exc
    if output == evidence_root or output.exists():
        raise ValueError(f"output must be a new child directory: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    return output, temporary


def require_per_tensor_quantization(
    detail: dict[str, Any], label: str
) -> tuple[float, int]:
    parameters = detail["quantization_parameters"]
    scales = np.asarray(parameters["scales"])
    zero_points = np.asarray(parameters["zero_points"])
    if scales.size != 1 or zero_points.size != 1:
        raise ValueError(f"{label}: per-tensor quantization required")
    scale = float(scales[0])
    zero_point = int(zero_points[0])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"{label}: invalid quantization scale")
    return scale, zero_point


def quantize_input(
    image_path: Path,
    detail: dict[str, Any],
) -> np.ndarray:
    image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32)
    if image.shape != (288, 512, 3):
        raise ValueError(f"{image_path}: image shape {image.shape} != (288, 512, 3)")
    normalized = (image / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    scale, zero_point = require_per_tensor_quantization(detail, "input")
    # Match Kotlin roundToInt / java.lang.Math.round.
    quantized = np.floor(normalized / float(scale) + int(zero_point) + 0.5)
    return np.clip(quantized, -128, 127).astype(np.int8)[None, ...]


def model_trace(
    *,
    arm: str,
    model_path: Path,
    manifest: dict[str, Any],
    manifest_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    from ai_edge_litert.interpreter import Interpreter

    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"{arm}: expected one input and one output")
    input_detail, output_detail = inputs[0], outputs[0]
    if (
        list(input_detail["shape"]) != [1, 288, 512, 3]
        or list(output_detail["shape"]) != [1, 288, 512, 4]
        or input_detail["dtype"] != np.int8
        or output_detail["dtype"] != np.int8
    ):
        raise ValueError(f"{arm}: tensor contract mismatch")
    require_per_tensor_quantization(input_detail, f"{arm} input")
    output_scale, output_zero = require_per_tensor_quantization(
        output_detail, f"{arm} output"
    )
    configs = adapter_configs(contract)
    rows: list[dict[str, Any]] = []
    replay_checked = False
    for event in manifest["events"]:
        for frame in event["frames"]:
            image_path = manifest_root / frame["image_path"]
            if sha256_file(image_path) != frame["image_sha256"]:
                raise ValueError(f"{image_path}: image SHA drift")
            quantized_input = quantize_input(image_path, input_detail)
            interpreter.set_tensor(input_detail["index"], quantized_input)
            start = time.perf_counter()
            interpreter.invoke()
            raw_output = interpreter.get_tensor(output_detail["index"])
            inference_ms = (time.perf_counter() - start) * 1000.0
            if raw_output.dtype != np.int8:
                raise ValueError(f"{arm}: output dtype drift")
            raw_bytes = raw_output.tobytes(order="C")
            raw_sha = hashlib.sha256(raw_bytes).hexdigest()
            if not replay_checked:
                interpreter.set_tensor(input_detail["index"], quantized_input)
                interpreter.invoke()
                replay = interpreter.get_tensor(output_detail["index"])
                if hashlib.sha256(replay.tobytes(order="C")).hexdigest() != raw_sha:
                    raise ValueError(f"{arm}: same-backend canary is not deterministic")
                replay_checked = True
            probabilities = stable_softmax_from_int8(raw_output, float(output_scale))
            pool_start = time.perf_counter()
            scores, diagnostics = pool_probabilities(
                probabilities, contract, configs
            )
            pooling_ms = (time.perf_counter() - pool_start) * 1000.0
            argmax = np.argmax(raw_output, axis=-1).astype(np.uint8)
            rows.append(
                {
                    "schema_version": "blindassist.riskseg_r1.p0_frame_feature.v1",
                    "protocol_id": contract["protocol_id"],
                    "arm": arm,
                    "parent_event_id": event["parent_event_id"],
                    "source_session_id": event["source_session_id"],
                    "frame_index": int(frame["frame_index"]),
                    "image_sha256": frame["image_sha256"],
                    "raw_int8_output_sha256": raw_sha,
                    "argmax_mask_sha256": hashlib.sha256(
                        argmax.tobytes(order="C")
                    ).hexdigest(),
                    "output_quantization_scale": float(output_scale),
                    "output_quantization_zero_point": int(output_zero),
                    "output_shape": [1, 288, 512, 4],
                    "inference_ms": inference_ms,
                    "pooling_ms": pooling_ms,
                    "adapter_scores": scores,
                    "diagnostics": diagnostics,
                }
            )
    # ai-edge-litert releases native resources when the interpreter is
    # dereferenced; unlike TensorFlow's wrapper it has no public close().
    del interpreter
    return rows


def oracle_trace(
    *,
    manifest: dict[str, Any],
    manifest_root: Path,
    contract: dict[str, Any],
) -> list[dict[str, Any]]:
    configs = adapter_configs(contract)
    rows: list[dict[str, Any]] = []
    for event in manifest["events"]:
        for frame in event["frames"]:
            mask_path = manifest_root / frame["oracle_mask_path"]
            if sha256_file(mask_path) != frame["oracle_mask_sha256"]:
                raise ValueError(f"{mask_path}: mask SHA drift")
            mask = np.asarray(Image.open(mask_path))
            if mask.shape != (256, 256):
                raise ValueError(f"{mask_path}: mask shape {mask.shape} != (256, 256)")
            if mask.dtype != np.uint8 or np.any(mask > 3):
                raise ValueError(f"{mask_path}: oracle class contract drift")
            probabilities = np.eye(4, dtype=np.float32)[mask]
            scores, diagnostics = pool_probabilities(
                probabilities, contract, configs
            )
            rows.append(
                {
                    "schema_version": "blindassist.riskseg_r1.p0_frame_feature.v1",
                    "protocol_id": contract["protocol_id"],
                    "arm": ORACLE_ARM,
                    "parent_event_id": event["parent_event_id"],
                    "source_session_id": event["source_session_id"],
                    "frame_index": int(frame["frame_index"]),
                    "image_sha256": frame["image_sha256"],
                    "oracle_mask_sha256": frame["oracle_mask_sha256"],
                    "argmax_mask_sha256": hashlib.sha256(
                        mask.tobytes(order="C")
                    ).hexdigest(),
                    "output_shape": [1, 256, 256, 4],
                    "adapter_scores": scores,
                    "diagnostics": diagnostics,
                }
            )
    return rows


def yolo_event_scores(
    report: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for item in report["event_summaries"]:
        if item["arm"] != "A_CURRENT_YOLO_ONLY":
            continue
        rows[str(item["parent_event_id"])] = {
            "parent_event_id": item["parent_event_id"],
            "source_session_id": item["source_session_id"],
            "bucket": item["bucket"],
            "positive": bool(item["positive"]),
            "event_hit": bool(item["event_hit"]),
            "critical_miss": bool(item["critical_miss"]),
            "false_alert_event": bool(item["false_alert_event"]),
            "passed_cleared": bool(item["passed_cleared"]),
            "first_alertable_alert_frame": item["first_alertable_alert_frame"],
        }
    return rows


def old_adapter_metrics(root: Path, contract: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for arm in MODEL_ARMS:
        seed = arm.removeprefix("seed-")
        relative = contract["old_adapter_reports"][arm]["path"]
        path = (root / relative).resolve()
        expected = contract["old_adapter_reports"][arm]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"{path}: old-adapter report SHA drift")
        report = read_object(path)
        aggregate = report["arm_aggregates"]["B_LEARNED_SEGMENTATION_ONLY"]
        result[arm] = {
            "seed": int(seed),
            "hit_event_count": int(aggregate["hit_event_count"]),
            "false_alert_event_count": int(aggregate["false_alert_event_count"]),
            "cleared_event_count": int(aggregate["cleared_event_count"]),
        }
    return result


def decide(
    nested: dict[str, dict[str, Any]],
    old: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    truth = nested[ORACLE_ARM]["oof_aggregate"]
    truth_gate = {
        "hit_event_count": truth["hit_event_count"]
        >= int(contract["gates"]["truth_mask"]["minimum_hit_event_count"]),
        "false_alert_event_count": truth["false_alert_event_count"]
        <= int(contract["gates"]["truth_mask"]["maximum_false_alert_event_count"]),
        "cleared_event_count": truth["cleared_event_count"]
        >= int(contract["gates"]["truth_mask"]["minimum_cleared_event_count"]),
    }
    learned: dict[str, Any] = {}
    for arm in MODEL_ARMS:
        current = nested[arm]["oof_aggregate"]
        baseline = old[arm]
        guardrails = (
            current["hit_event_count"] >= baseline["hit_event_count"]
            and current["false_alert_event_count"]
            <= baseline["false_alert_event_count"]
            and current["cleared_event_count"] >= baseline["cleared_event_count"]
        )
        strict = (
            current["hit_event_count"] > baseline["hit_event_count"]
            or current["false_alert_event_count"]
            < baseline["false_alert_event_count"]
            or current["cleared_event_count"] > baseline["cleared_event_count"]
        )
        learned[arm] = {
            "guardrails_pass": guardrails,
            "strict_event_improvement": strict,
            "pass": guardrails and strict,
            "old_adapter": baseline,
            "soft_adapter": current,
        }
    decision_pass = learned["seed-20260801"]["pass"]
    stable_count = sum(item["pass"] for item in learned.values())
    authorize = all(truth_gate.values()) and decision_pass and stable_count >= 2
    if not all(truth_gate.values()):
        terminal = "TRUTH_MASK_SOFT_ADAPTER_FAIL_CHANGE_ACTIONABILITY_LABELS"
    elif authorize:
        terminal = "RISKSEG_R1_P1_DESIGN_AUTHORIZED"
    else:
        terminal = "RISKSEG_R1_P1_NOT_AUTHORIZED_SOFT_ADAPTER_SIGNAL_INSUFFICIENT"
    return {
        "terminal": terminal,
        "truth_mask_gate": {
            **truth_gate,
            "pass": all(truth_gate.values()),
            "metrics": truth,
        },
        "learned_seed_checks": learned,
        "passing_seed_count": stable_count,
        "decision_seed_pass": decision_pass,
        "p1_design_authorized": authorize,
        "claim_ceiling": (
            "consumed nested Development mechanism evidence only; "
            "not fresh confirmation, App promotion, or safety evidence"
        ),
    }


def main() -> None:
    args = parse_args()
    contract_path = args.contract.resolve()
    contract = read_object(contract_path)
    if (
        contract.get("schema_version")
        != "blindassist.riskseg_r1.p0_contract.v1"
        or contract.get("status") != "PRE_OUTPUT_LOCKED"
    ):
        raise ValueError("contract identity/status mismatch")
    root = contract_path.parents[3]
    verify_implementation_lock(root, contract)
    manifest_path = verify_bound_file(root, contract["frozen_manifest"])
    verify_bound_file(root, contract["frozen_manifest_receipt"])
    manifest = read_object(manifest_path)
    if (
        manifest["event_count"] != 30
        or manifest["source_session_count"] != 30
        or manifest["frame_count"] != 1920
    ):
        raise ValueError("frozen cohort cardinality drift")
    assignments = fold_assignments(manifest, contract)
    if canonical_sha256(assignments) != contract["nested_development"][
        "fold_assignment_sha256"
    ]:
        raise ValueError("fold assignment SHA drift")

    yolo_path = verify_bound_file(root, contract["yolo_reference_report"])
    yolo = yolo_event_scores(read_object(yolo_path))
    if set(yolo) != {event["parent_event_id"] for event in manifest["events"]}:
        raise ValueError("YOLO reference membership drift")
    old = old_adapter_metrics(root, contract)
    manifest_root = manifest_path.parent

    rows: list[dict[str, Any]] = []
    for arm in MODEL_ARMS:
        model_path = verify_bound_file(root, contract["models"][arm])
        rows.extend(
            model_trace(
                arm=arm,
                model_path=model_path,
                manifest=manifest,
                manifest_root=manifest_root,
                contract=contract,
            )
        )
    rows.extend(
        oracle_trace(
            manifest=manifest,
            manifest_root=manifest_root,
            contract=contract,
        )
    )

    output, temporary = bounded_new_output(root, args.output)
    temporary.mkdir(parents=True, exist_ok=False)
    trace_path = temporary / "frame_features.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
                + "\n"
            )
    nested = {
        arm: nested_oof_score(
            arm=arm,
            manifest=manifest,
            frame_rows=rows,
            yolo_events=yolo,
            contract=contract,
        )
        for arm in (*MODEL_ARMS, ORACLE_ARM)
    }
    for arm, scored in nested.items():
        scored["timing_against_yolo"] = timing_against_yolo(
            scored["oof_event_scores"], yolo
        )
    report = {
        "schema_version": "blindassist.riskseg_r1.p0_audit.v1",
        "protocol_id": contract["protocol_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "manifest_sha256": sha256_file(manifest_path),
        "fold_assignment_sha256": canonical_sha256(assignments),
        "feature_trace": {
            "path": trace_path.name,
            "sha256": sha256_file(trace_path),
            "row_count": len(rows),
        },
        "old_adapter_metrics": old,
        "nested_oof": nested,
        "decision": decide(nested, old, contract),
    }
    write_object(temporary / "report.json", report)
    temporary.replace(output)
    print(json.dumps(report["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
