#!/usr/bin/env python3
"""Run and evaluate frozen stateless TartanAir S2 door completion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import DAV2_ONNX_SHA256, _atomic_json, _read, _require, preprocess_depth, region_depth_median
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import BOX_THRESHOLD, MAX_DINO_CANDIDATES, PAIR_IOU_THRESHOLD, PROMPT, TEXT_THRESHOLD, select_consensus
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_nyuv2_door_depth import PRIVATE_SCHEMA, PUBLIC_SCHEMA
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import PROTOCOL_ID
from scripts.research.goal_copilot_bridge.p0_s0_materialization.run_grounding_dino_s0_r1 import MODEL_REVISION, MODEL_REPOSITORY, WEIGHTS_FILENAME, WEIGHTS_SHA256, sha256_file
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import EXPECTED_MODEL_SHA256, EXPECTED_TEXT_ENCODER_SHA256, EXPECTED_ULTRALYTICS_VERSION, iou, sha256, validated_box


AUTH_SCHEMA = "blindassist_tartanair_current_frame_completion_authorization_v1"
RUN_SCHEMA = "blindassist_tartanair_current_frame_completion_run_v1"
EVAL_SCHEMA = "blindassist_tartanair_current_frame_completion_evaluation_v1"


def authorize(manifest_path: Path, public_path: Path, private_path: Path, run_path: Path, journal_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists() and not run_path.exists() and not journal_path.exists(), "S2 formal output already exists")
    manifest, public, private = _read(manifest_path), _read(public_path), _read(private_path)
    _require(manifest.get("protocol_id") == PROTOCOL_ID and manifest.get("created_before_provider_calls") is True, "S2 manifest drift")
    _require(public.get("schema_version") == PUBLIC_SCHEMA and private.get("schema_version") == PRIVATE_SCHEMA, "S2 input schema drift")
    _require(public.get("protocol_id") == private.get("protocol_id") == PROTOCOL_ID, "S2 input protocol drift")
    _require(private.get("public_input_sha256") == sha256(public_path), "S2 private/public binding mismatch")
    public_ids = {case["case_id"] for case in public["cases"]}
    private_cases = {case["case_id"]: case for case in private["cases"]}
    _require(public_ids == set(private_cases), "S2 roster mismatch")
    near = sum(case["stratum"] == "NEAR" for case in private_cases.values())
    far = sum(case["stratum"] == "FAR" for case in private_cases.values())
    _require(near >= 24 and far >= 24, "S2 denominator insufficient")
    receipt = {"schema_version": AUTH_SCHEMA, "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": sha256(manifest_path), "public_sha256": sha256(public_path), "private_sha256": sha256(private_path), "near_case_count": near, "far_case_count": far, "provider_calls_before_authorization": 0, "run_output_path": str(run_path.resolve()), "journal_output_path": str(journal_path.resolve()), "retry_or_replay_authorized": False, "execution_authorized": True}
    _atomic_json(output_path, receipt)
    return receipt


def run(manifest_path: Path, public_path: Path, authorization_path: Path, yoloe_model_path: Path, text_encoder_path: Path, depth_onnx_path: Path, dino_model_dir: Path, journal_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not journal_path.exists() and not output_path.exists(), "S2 formal run already exists")
    manifest, public, authorization = _read(manifest_path), _read(public_path), _read(authorization_path)
    _require(manifest.get("protocol_id") == PROTOCOL_ID and manifest.get("provider", {}).get("stateless_current_frame_only") is True, "S2 manifest/provider drift")
    _require(authorization.get("schema_version") == AUTH_SCHEMA and authorization.get("execution_authorized") is True, "S2 not authorized")
    _require(authorization.get("manifest_sha256") == sha256(manifest_path) and authorization.get("public_sha256") == sha256(public_path), "S2 authorization binding mismatch")
    _require(authorization.get("run_output_path") == str(output_path.resolve()) and authorization.get("journal_output_path") == str(journal_path.resolve()), "S2 output path drift")
    _require(yoloe_model_path.is_file() and sha256(yoloe_model_path) == EXPECTED_MODEL_SHA256, "YOLOE model drift")
    _require(text_encoder_path.is_file() and sha256(text_encoder_path) == EXPECTED_TEXT_ENCODER_SHA256, "text encoder drift")
    _require(depth_onnx_path.is_file() and sha256(depth_onnx_path) == DAV2_ONNX_SHA256, "depth model drift")
    dino_weight = dino_model_dir / WEIGHTS_FILENAME
    _require(dino_weight.is_file() and sha256_file(dino_weight) == WEIGHTS_SHA256, "DINO model drift")

    import onnxruntime as ort
    import torch
    import transformers
    import ultralytics
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    from ultralytics import YOLOE

    _require(ultralytics.__version__ == EXPECTED_ULTRALYTICS_VERSION, "Ultralytics version drift")
    _require(not device.startswith("cuda") or torch.cuda.is_available(), "requested CUDA unavailable")
    depth_session = ort.InferenceSession(str(depth_onnx_path.resolve()), providers=["CPUExecutionProvider"])
    yoloe = YOLOE(str(yoloe_model_path.resolve()))
    processor = AutoProcessor.from_pretrained(dino_model_dir, local_files_only=True)
    dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    journal = {"schema_version": "blindassist_tartanair_current_frame_completion_journal_v1", "protocol_id": PROTOCOL_ID, "status": "STARTED", "started_at_utc": datetime.now(timezone.utc).isoformat(), "cases_dispatched": 0, "cases_completed": 0, "yoloe_calls_completed": 0, "depth_calls_completed": 0, "dino_calls_completed": 0, "retry_or_replay_authorized": False}
    _atomic_json(journal_path, journal)
    rows = []
    try:
        for case in public["cases"]:
            image_path = Path(case["query"]["image_path"])
            _require(image_path.is_file() and sha256(image_path) == case["query"]["image_sha256"], "S2 public image drift")
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
            yolo_started = time.perf_counter()
            yolo_result = yoloe.predict(source=str(image_path), verbose=False, device=device, imgsz=640, conf=0.001, max_det=100)[0]
            torch.cuda.synchronize()
            yolo_latency = (time.perf_counter() - yolo_started) * 1000.0
            ranked = sorted(zip(yolo_result.boxes.conf.tolist(), yolo_result.boxes.xyxy.tolist(), strict=True), key=lambda pair: pair[0], reverse=True)[:10]
            depth_started = time.perf_counter()
            depth = np.asarray(depth_session.run(["depth_m"], {"image": preprocess_depth(image)})[0][0], dtype=np.float32)
            depth_latency = (time.perf_counter() - depth_started) * 1000.0
            yolo_candidates = [{"provider_rank": rank, "bbox_xyxy": [float(value) for value in box], "proposal_score": float(score), "predicted_region_depth_m": region_depth_median(depth, box, width, height)} for rank, (score, box) in enumerate(ranked, start=1)]
            dino_started = time.perf_counter()
            inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
            with torch.inference_mode():
                raw = dino(**inputs)
            dino_result = processor.post_process_grounded_object_detection(raw, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(height, width)])[0]
            torch.cuda.synchronize()
            dino_latency = (time.perf_counter() - dino_started) * 1000.0
            labels = dino_result.get("text_labels", dino_result.get("labels", []))
            dino_candidates = sorted(({"bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()], "score": float(score.detach().cpu()), "label": str(label)} for box, score, label in zip(dino_result["boxes"], dino_result["scores"], labels)), key=lambda row: (-row["score"], row["bbox_xyxy"]))[:MAX_DINO_CANDIDATES]
            selected = select_consensus(yolo_candidates, dino_candidates, width / 2.0)
            rows.append({"case_id": case["case_id"], "image_width": width, "image_height": height, "yoloe_candidates": yolo_candidates, "dino_candidates": dino_candidates, "selected_candidate": selected, "completion": selected is not None, "latency_ms": {"yoloe": yolo_latency, "depth": depth_latency, "dino": dino_latency}})
            journal.update({"status": "ACTIVE", "active_case_id": None, "cases_completed": journal["cases_completed"] + 1, "yoloe_calls_completed": journal["yoloe_calls_completed"] + 1, "depth_calls_completed": journal["depth_calls_completed"] + 1, "dino_calls_completed": journal["dino_calls_completed"] + 1})
            _atomic_json(journal_path, journal)
        payload = {"schema_version": RUN_SCHEMA, "protocol_id": PROTOCOL_ID, "manifest_sha256": sha256(manifest_path), "public_sha256": sha256(public_path), "authorization_sha256": sha256(authorization_path), "private_truth_access": False, "stateless_current_frame_only": True, "provider": {"yoloe_sha256": sha256(yoloe_model_path), "text_encoder_sha256": sha256(text_encoder_path), "depth_onnx_sha256": sha256(depth_onnx_path), "dino_repository": MODEL_REPOSITORY, "dino_revision": MODEL_REVISION, "dino_weights_sha256": WEIGHTS_SHA256, "dino_prompt": PROMPT, "ultralytics": ultralytics.__version__, "onnxruntime": ort.__version__, "transformers": transformers.__version__, "python": platform.python_version(), "device": device, "depth_providers": depth_session.get_providers()}, "cases": rows, "claim_ceiling": manifest["claim_ceiling"]}
        _atomic_json(output_path, payload)
        journal.update({"status": "COMPLETED", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_sha256": sha256(output_path)})
        _atomic_json(journal_path, journal)
        return payload
    except Exception:
        journal.update({"status": "FAILED", "failed_at_utc": datetime.now(timezone.utc).isoformat()})
        _atomic_json(journal_path, journal)
        raise


def evaluate(manifest_path: Path, public_path: Path, private_path: Path, run_path: Path, journal_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "S2 evaluation already exists")
    manifest, public, private, run_payload, journal = [_read(path) for path in (manifest_path, public_path, private_path, run_path, journal_path)]
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "S2 run boundary mismatch")
    _require(run_payload.get("manifest_sha256") == sha256(manifest_path) and run_payload.get("public_sha256") == sha256(public_path), "S2 run binding mismatch")
    _require(journal.get("status") == "COMPLETED" and journal.get("run_sha256") == sha256(run_path), "S2 journal incomplete")
    _require(journal["cases_completed"] == journal["yoloe_calls_completed"] == journal["depth_calls_completed"] == journal["dino_calls_completed"] == len(public["cases"]), "S2 call accounting mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    runs = {case["case_id"]: case for case in run_payload["cases"]}
    _require(set(truths) == set(runs), "S2 evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], runs[case_id]
        legal = truth["legal_targets"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        center_x = observed["image_width"] / 2.0
        opportunity = any(box[0] <= center_x <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        available = any(iou(validated_box(candidate["bbox_xyxy"], f"{case_id} candidate"), target) >= 0.30 for candidate in observed["yoloe_candidates"] for target in targets)
        selected = observed["selected_candidate"]
        matched_index = None
        if selected is not None:
            selected_box = validated_box(selected["bbox_xyxy"], f"{case_id} selected")
            overlaps = [iou(selected_box, target) for target in targets]
            matched_index = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        matched_depth = float(legal[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        false = selected is not None and not correct
        rows.append({"case_id": case_id, "stratum": truth["stratum"], "completion_opportunity": opportunity, "target_candidate_available": available, "completion_decision": selected is not None, "target_selected": matched_index is not None, "matched_target_depth_m": matched_depth, "correct_completion": correct, "false_completion": false, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    decisions = correct + false
    payload = {"schema_version": EVAL_SCHEMA, "protocol_id": PROTOCOL_ID, "case_count": len(rows), "near_case_count": sum(row["stratum"] == "NEAR" for row in rows), "far_case_count": sum(row["stratum"] == "FAR" for row in rows), "completion_opportunity_count": opportunities, "target_candidate_availability_count": sum(row["target_candidate_available"] for row in rows), "target_selection_count": sum(row["target_selected"] for row in rows), "completion_decision_count": decisions, "correct_completion_count": correct, "false_completion_count": false, "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows), "correct_completion_coverage": correct / opportunities if opportunities else None, "false_completion_fraction_of_decisions": false / decisions if decisions else None, "rows": rows, "inputs": {"manifest_sha256": sha256(manifest_path), "public_sha256": sha256(public_path), "private_sha256": sha256(private_path), "run_sha256": sha256(run_path), "journal_sha256": sha256(journal_path)}, "claim_ceiling": manifest["claim_ceiling"], "terminal": "FRESH_CURRENT_FRAME_COMPLETION_ESTABLISHED" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "FRESH_CURRENT_FRAME_COMPLETION_NOT_ESTABLISHED"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    auth = sub.add_parser("authorize")
    for name in ("manifest", "public", "private", "run", "journal", "output"):
        auth.add_argument(f"--{name}", required=True, type=Path)
    execute = sub.add_parser("run")
    for name in ("manifest", "public", "authorization", "yoloe-model", "text-encoder", "depth-onnx", "dino-model-dir", "journal", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("manifest", "public", "private", "run", "journal", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "authorize":
        print(json.dumps(authorize(args.manifest, args.public, args.private, args.run, args.journal, args.output), indent=2))
    elif args.command == "run":
        run(args.manifest, args.public, args.authorization, args.yoloe_model, args.text_encoder, args.depth_onnx, args.dino_model_dir, args.journal, args.output, args.device)
    else:
        result = evaluate(args.manifest, args.public, args.private, args.run, args.journal, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
