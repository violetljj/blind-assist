#!/usr/bin/env python3
"""Evaluate independent metric-depth gating for door completion on NYUv2."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import (
    EXPECTED_MODEL_SHA256,
    EXPECTED_TEXT_ENCODER_SHA256,
    EXPECTED_ULTRALYTICS_VERSION,
    iou,
    sha256,
    validated_box,
)
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_nyuv2_door_depth import PRIVATE_SCHEMA, PUBLIC_SCHEMA


MANIFEST_SCHEMA = "blindassist_completion_nearness_experiment_manifest_v1"
AUTH_SCHEMA = "blindassist_completion_nearness_authorization_v1"
RUN_SCHEMA = "blindassist_completion_nearness_run_v1"
EVAL_SCHEMA = "blindassist_completion_nearness_evaluation_v1"
NYUV2_PROTOCOL_ID = "BLINDASSIST_COMPLETION_NEARNESS_NYUV2_DOOR_V1"
SUNRGBD_PROTOCOL_ID = "BLINDASSIST_COMPLETION_NEARNESS_SUNRGBD_DOOR_V1"
SUPPORTED_PROTOCOL_IDS = {NYUV2_PROTOCOL_ID, SUNRGBD_PROTOCOL_ID}
DAV2_ONNX_SHA256 = "870339770e21675830f7e2020983dda058752d237c8b86951ed1e6f9a6243d01"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _config(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA and manifest.get("protocol_id") in SUPPORTED_PROTOCOL_IDS, "manifest schema/protocol mismatch")
    _require(manifest.get("created_before_dataset_payload_access") is True, "manifest dataset precedence missing")
    _require(manifest.get("created_before_private_truth_access") is True, "manifest truth precedence missing")
    _require(manifest.get("threshold_model_prompt_or_pool_sweep") is False and manifest.get("retry_or_replay_authorized") is False, "manifest permits sweep/replay")
    provider = manifest.get("frozen_provider", {})
    decision = manifest.get("frozen_decision", {})
    _require(provider.get("proposal") == "YOLOE-26n-seg text prompt door", "proposal provider drift")
    _require(provider.get("metric_depth") == "Depth Anything V2 metric Hypersim ViT-S ONNX", "depth provider drift")
    _require(provider.get("metric_depth_input_shape") == [1, 3, 518, 686], "depth input shape drift")
    _require(provider.get("bounded_pool_size") == 10 and provider.get("selection_rule") in {"LEFTMOST_CANDIDATE_X_CENTER", "PROVIDER_SCORE_TOP1"}, "candidate contract drift")
    _require(provider.get("selected_region_aggregation") == "median of finite positive depth in bbox inset 20 percent per side", "depth aggregation drift")
    _require(decision.get("interaction_range_m") == 2.0 and decision.get("baseline_bbox_height_threshold") == 0.55, "decision threshold drift")
    _require(decision.get("target_hit_iou_threshold") == 0.30, "IoU threshold drift")
    return provider, decision


def authorize(manifest_path: Path, public_path: Path, private_path: Path, run_output: Path, journal_output: Path, authorization_output: Path) -> dict[str, Any]:
    _require(not authorization_output.exists() and not run_output.exists() and not journal_output.exists(), "formal output already exists")
    manifest, public, private = _read(manifest_path), _read(public_path), _read(private_path)
    _config(manifest)
    _require(public.get("schema_version") == PUBLIC_SCHEMA and private.get("schema_version") == PRIVATE_SCHEMA, "input schema mismatch")
    _require(private.get("public_input_sha256") == sha256(public_path), "private/public binding mismatch")
    public_ids = {case["case_id"] for case in public.get("cases", [])}
    private_cases = {case["case_id"]: case for case in private.get("cases", [])}
    _require(public_ids == set(private_cases), "input roster mismatch")
    near = sum(bool(case["true_interaction_range"]) for case in private_cases.values())
    far = len(private_cases) - near
    minimum_near = int(manifest["minimum_near_case_count"])
    minimum_far = int(manifest["minimum_far_case_count"])
    _require(near >= minimum_near and far >= minimum_far, "near/far authorization denominator insufficient")
    receipt = {
        "schema_version": AUTH_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": sha256(manifest_path),
        "public_sha256": sha256(public_path),
        "private_sha256": sha256(private_path),
        "near_case_count": near,
        "far_case_count": far,
        "minimum_near_case_count": minimum_near,
        "minimum_far_case_count": minimum_far,
        "provider_calls_before_authorization": 0,
        "run_output_path": str(run_output.resolve()),
        "journal_output_path": str(journal_output.resolve()),
        "retry_or_replay_authorized": False,
        "execution_authorized": True,
    }
    _atomic_json(authorization_output, receipt)
    return receipt


def preprocess_depth(image: Image.Image) -> np.ndarray:
    resized = np.asarray(image.convert("RGB").resize((686, 518), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    return ((resized - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def region_depth_median(depth: np.ndarray, bbox: Sequence[float], image_width: int, image_height: int) -> float | None:
    box = validated_box(list(bbox), "depth region bbox")
    inset_x = 0.20 * (box[2] - box[0])
    inset_y = 0.20 * (box[3] - box[1])
    x0 = int(np.floor((box[0] + inset_x) / image_width * depth.shape[1]))
    x1 = int(np.ceil((box[2] - inset_x) / image_width * depth.shape[1]))
    y0 = int(np.floor((box[1] + inset_y) / image_height * depth.shape[0]))
    y1 = int(np.ceil((box[3] - inset_y) / image_height * depth.shape[0]))
    x0, x1 = max(0, x0), min(depth.shape[1], x1)
    y0, y1 = max(0, y0), min(depth.shape[0], y1)
    if x1 <= x0 or y1 <= y0:
        return None
    values = depth[y0:y1, x0:x1]
    valid = values[np.isfinite(values) & (values > 0.0) & (values <= 20.0)]
    return float(np.median(valid)) if valid.size else None


def decision_for(selected: Mapping[str, Any] | None, predicted_depth_m: float | None, width: int, height: int, decision: Mapping[str, Any]) -> tuple[bool, bool]:
    if selected is None:
        return False, False
    box = validated_box(selected["bbox_xyxy"], "selected candidate")
    centered = bool(box[0] <= width / 2.0 <= box[2])
    bbox_commit = centered and (box[3] - box[1]) / height >= float(decision["baseline_bbox_height_threshold"])
    depth_commit = centered and predicted_depth_m is not None and predicted_depth_m <= float(decision["depth_gate_requires_predicted_region_depth_lte_m"])
    return bbox_commit, depth_commit


def run(manifest_path: Path, public_path: Path, authorization_path: Path, yoloe_model_path: Path, text_encoder_path: Path, depth_onnx_path: Path, journal_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not journal_path.exists() and not output_path.exists(), "formal run is immutable and already exists")
    manifest, public, authorization = _read(manifest_path), _read(public_path), _read(authorization_path)
    provider, decision = _config(manifest)
    _require(authorization.get("schema_version") == AUTH_SCHEMA and authorization.get("execution_authorized") is True, "run not authorized")
    _require(authorization.get("manifest_sha256") == sha256(manifest_path) and authorization.get("public_sha256") == sha256(public_path), "authorization binding mismatch")
    _require(authorization.get("run_output_path") == str(output_path.resolve()) and authorization.get("journal_output_path") == str(journal_path.resolve()), "authorization output path mismatch")
    _require(yoloe_model_path.is_file() and sha256(yoloe_model_path) == EXPECTED_MODEL_SHA256, "YOLOE model drift")
    _require(text_encoder_path.is_file() and sha256(text_encoder_path) == EXPECTED_TEXT_ENCODER_SHA256, "text encoder drift")
    _require(depth_onnx_path.is_file() and sha256(depth_onnx_path) == DAV2_ONNX_SHA256, "metric depth ONNX drift")

    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLOE

    _require(ultralytics.__version__ == EXPECTED_ULTRALYTICS_VERSION, "Ultralytics version drift")
    _require(not device.startswith("cuda") or torch.cuda.is_available(), "requested CUDA unavailable")
    ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device.startswith("cuda") else ["CPUExecutionProvider"]
    depth_session = ort.InferenceSession(str(depth_onnx_path.resolve()), providers=ort_providers)
    _require(depth_session.get_inputs()[0].name == "image" and depth_session.get_outputs()[0].name == "depth_m", "depth ONNX interface drift")
    yoloe = YOLOE(str(yoloe_model_path.resolve()))
    journal = {
        "schema_version": "blindassist_completion_nearness_journal_v1",
        "protocol_id": manifest["protocol_id"],
        "status": "STARTED",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_dispatched": 0,
        "cases_completed": 0,
        "proposal_calls_completed": 0,
        "depth_calls_completed": 0,
        "retry_or_replay_authorized": False,
    }
    _atomic_json(journal_path, journal)
    rows = []
    try:
        for case in public.get("cases", []):
            image_path = Path(case["query"]["image_path"])
            _require(image_path.is_file() and sha256(image_path) == case["query"]["image_sha256"], "public image binding mismatch")
            journal.update({"status": "DISPATCHING", "active_case_id": case["case_id"], "cases_dispatched": journal["cases_dispatched"] + 1})
            _atomic_json(journal_path, journal)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                width, height = image.size
                previous = Path.cwd()
                os.chdir(text_encoder_path.resolve().parent)
                try:
                    yoloe.set_classes(["door"])
                finally:
                    os.chdir(previous)
                proposal_started = time.perf_counter()
                result = yoloe.predict(source=str(image_path), verbose=False, device=device, imgsz=int(provider["proposal_imgsz"]), conf=float(provider["proposal_confidence_floor"]), max_det=int(provider["proposal_max_det"]))[0]
                if device.startswith("cuda"):
                    torch.cuda.synchronize()
                proposal_latency_ms = (time.perf_counter() - proposal_started) * 1000.0
                ranked = sorted(zip(result.boxes.conf.tolist(), result.boxes.xyxy.tolist(), strict=True), key=lambda pair: pair[0], reverse=True)
                candidates = [{"provider_rank": rank, "bbox_xyxy": [float(value) for value in box], "proposal_score": float(score)} for rank, (score, box) in enumerate(ranked[: int(provider["bounded_pool_size"])], start=1)]
                if provider["selection_rule"] == "PROVIDER_SCORE_TOP1":
                    selected = candidates[0] if candidates else None
                else:
                    selected = min(candidates, key=lambda candidate: (((candidate["bbox_xyxy"][0] + candidate["bbox_xyxy"][2]) / 2.0), candidate["provider_rank"])) if candidates else None
                depth_started = time.perf_counter()
                depth = np.asarray(depth_session.run(["depth_m"], {"image": preprocess_depth(image)})[0][0], dtype=np.float32)
                depth_latency_ms = (time.perf_counter() - depth_started) * 1000.0
                _require(depth.shape == (518, 686) and np.isfinite(depth).all() and np.all(depth > 0), "metric depth output invalid")
                predicted_depth = region_depth_median(depth, selected["bbox_xyxy"], width, height) if selected else None
                bbox_commit, depth_commit = decision_for(selected, predicted_depth, width, height, decision)
            rows.append({
                "case_id": case["case_id"],
                "image_width": width,
                "image_height": height,
                "candidates": candidates,
                "selected_candidate": selected,
                "predicted_selected_region_depth_m": predicted_depth,
                "bbox_only_completion": bbox_commit,
                "depth_gated_completion": depth_commit,
                "proposal_latency_ms": proposal_latency_ms,
                "depth_latency_ms": depth_latency_ms,
            })
            journal.update({"status": "ACTIVE", "active_case_id": None, "cases_completed": journal["cases_completed"] + 1, "proposal_calls_completed": journal["proposal_calls_completed"] + 1, "depth_calls_completed": journal["depth_calls_completed"] + 1})
            _atomic_json(journal_path, journal)
        payload = {
            "schema_version": RUN_SCHEMA,
            "protocol_id": manifest["protocol_id"],
            "manifest_sha256": sha256(manifest_path),
            "public_sha256": sha256(public_path),
            "authorization_sha256": sha256(authorization_path),
            "private_truth_access": False,
            "provider": {"yoloe_model_sha256": sha256(yoloe_model_path), "text_encoder_sha256": sha256(text_encoder_path), "depth_onnx_sha256": sha256(depth_onnx_path), "ultralytics_version": ultralytics.__version__, "onnxruntime_version": ort.__version__, "onnxruntime_providers": depth_session.get_providers(), "device": device},
            "cases": rows,
            "claim_ceiling": manifest.get("claim_ceiling"),
        }
        _atomic_json(output_path, payload)
        journal.update({"status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_sha256": sha256(output_path)})
        _atomic_json(journal_path, journal)
        return payload
    except Exception:
        journal.update({"status": "FAILED", "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(journal_path, journal)
        raise


def _confusion(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    tp = sum(bool(row[key]) and bool(row["truth_positive"]) for row in rows)
    fp = sum(bool(row[key]) and not bool(row["truth_positive"]) for row in rows)
    fn = sum(not bool(row[key]) and bool(row["truth_positive"]) for row in rows)
    tn = len(rows) - tp - fp - fn
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None}


def evaluate(manifest_path: Path, public_path: Path, private_path: Path, run_path: Path, journal_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "evaluation is immutable and already exists")
    manifest, public, private, run_payload, journal = [_read(path) for path in (manifest_path, public_path, private_path, run_path, journal_path)]
    _, decision = _config(manifest)
    _require(private.get("public_input_sha256") == sha256(public_path), "private/public binding mismatch")
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "run schema/private boundary mismatch")
    _require(run_payload.get("manifest_sha256") == sha256(manifest_path) and run_payload.get("public_sha256") == sha256(public_path), "run binding mismatch")
    _require(journal.get("status") == "COMPLETED" and journal.get("run_sha256") == sha256(run_path), "journal incomplete")
    _require(journal.get("cases_dispatched") == journal.get("cases_completed") == journal.get("proposal_calls_completed") == journal.get("depth_calls_completed"), "call accounting mismatch")
    truths = {case["case_id"]: case for case in private.get("cases", [])}
    runs = {case["case_id"]: case for case in run_payload.get("cases", [])}
    public_ids = {case["case_id"] for case in public.get("cases", [])}
    _require(public_ids == set(truths) == set(runs), "evaluation roster mismatch")
    rows = []
    for case_id in sorted(public_ids):
        truth, observed = truths[case_id], runs[case_id]
        target_truths = truth.get("legal_targets") or [{
            "target_bbox_xyxy": truth["target_bbox_xyxy"],
            "target_depth_median_m": truth["target_raw_depth_median_m"],
        }]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in target_truths]
        candidates = observed.get("candidates", [])
        available = any(iou(validated_box(candidate["bbox_xyxy"], f"{case_id} candidate"), target) >= float(decision["target_hit_iou_threshold"]) for candidate in candidates for target in targets)
        selected = observed.get("selected_candidate")
        selected_box = validated_box(selected["bbox_xyxy"], f"{case_id} selected") if selected is not None else None
        selected_ious = [iou(selected_box, target) for target in targets] if selected_box is not None else []
        matched_index = int(np.argmax(selected_ious)) if selected_ious and max(selected_ious) >= float(decision["target_hit_iou_threshold"]) else None
        selected_hit = matched_index is not None
        prediction = observed.get("predicted_selected_region_depth_m")
        matched_depth = float(target_truths[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        truth_positive = selected_hit and matched_depth <= float(decision["interaction_range_m"])
        rows.append({
            "case_id": case_id,
            "stratum": truth["stratum"],
            "target_available": available,
            "target_selected": selected_hit,
            "true_target_depth_m": matched_depth,
            "predicted_selected_region_depth_m": prediction,
            "selected_target_depth_abs_error_m": abs(float(prediction) - matched_depth) if selected_hit and prediction is not None else None,
            "truth_positive": truth_positive,
            "bbox_only_completion": bool(observed["bbox_only_completion"]),
            "depth_gated_completion": bool(observed["depth_gated_completion"]),
        })
    baseline = _confusion(rows, "bbox_only_completion")
    depth_gate = _confusion(rows, "depth_gated_completion")
    errors = [row["selected_target_depth_abs_error_m"] for row in rows if row["selected_target_depth_abs_error_m"] is not None]
    selected_rows = [row | {
        "true_near": row["true_target_depth_m"] <= float(decision["interaction_range_m"]),
        "predicted_near": row["predicted_selected_region_depth_m"] is not None and row["predicted_selected_region_depth_m"] <= float(decision["interaction_range_m"]),
    } for row in rows if row["target_selected"]]
    conditional_depth = _confusion([
        {"truth_positive": row["true_near"], "predicted_near": row["predicted_near"]}
        for row in selected_rows
    ], "predicted_near")
    if depth_gate["tp"] == baseline["tp"] == 0 and depth_gate["fp"] < baseline["fp"]:
        terminal = "DEPTH_GATE_REMOVES_FALSE_COMMIT_WITH_ZERO_END_TO_END_TRUE_COMPLETION_COVERAGE"
    elif depth_gate["fp"] < baseline["fp"] and depth_gate["tp"] >= baseline["tp"]:
        terminal = "INDEPENDENT_DEPTH_REDUCES_FALSE_COMPLETION_WITHOUT_TP_LOSS"
    elif depth_gate["fp"] < baseline["fp"]:
        terminal = "INDEPENDENT_DEPTH_FALSE_COMPLETION_REDUCTION_TRADES_OFF_TRUE_COMPLETION"
    elif depth_gate["fp"] > baseline["fp"]:
        terminal = "INDEPENDENT_DEPTH_INCREASES_FALSE_COMPLETION"
    else:
        terminal = "INDEPENDENT_DEPTH_DOES_NOT_REDUCE_FALSE_COMPLETION"
    payload = {
        "schema_version": EVAL_SCHEMA,
        "protocol_id": manifest["protocol_id"],
        "case_count": len(rows),
        "near_case_count": sum(row["stratum"] == "NEAR" for row in rows),
        "far_case_count": sum(row["stratum"] == "FAR" for row in rows),
        "target_candidate_availability_count": sum(row["target_available"] for row in rows),
        "target_selection_count": sum(row["target_selected"] for row in rows),
        "bbox_only": baseline,
        "depth_gated": depth_gate,
        "false_completion_reduction_count": baseline["fp"] - depth_gate["fp"],
        "true_completion_change_count": depth_gate["tp"] - baseline["tp"],
        "selected_target_depth_mae_m": float(np.mean(errors)) if errors else None,
        "selected_target_depth_median_abs_error_m": float(np.median(errors)) if errors else None,
        "conditional_selected_target_depth_classification": conditional_depth,
        "rows": rows,
        "inputs": {"manifest_sha256": sha256(manifest_path), "public_sha256": sha256(public_path), "private_sha256": sha256(private_path), "run_sha256": sha256(run_path), "journal_sha256": sha256(journal_path)},
        "claim_ceiling": manifest.get("claim_ceiling"),
        "terminal": terminal,
    }
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    for name in ("manifest", "public", "private", "run-output", "journal-output", "authorization-output"):
        auth.add_argument(f"--{name}", required=True, type=Path)
    execute = sub.add_parser("run")
    for name in ("manifest", "public", "authorization", "yoloe-model", "text-encoder", "depth-onnx", "journal", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("manifest", "public", "private", "run", "journal", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "authorize":
        print(json.dumps(authorize(args.manifest, args.public, args.private, args.run_output, args.journal_output, args.authorization_output), indent=2))
    elif args.command == "run":
        run(args.manifest, args.public, args.authorization, args.yoloe_model, args.text_encoder, args.depth_onnx, args.journal, args.output, args.device)
    else:
        result = evaluate(args.manifest, args.public, args.private, args.run, args.journal, args.output)
        print(json.dumps({key: result[key] for key in ("bbox_only", "depth_gated", "false_completion_reduction_count", "true_completion_change_count", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
