#!/usr/bin/env python3
"""Development-only ADE20K functional-aperture verifier on sealed S4 data.

The provider reads only public current-frame RGB-D plus the already sealed S4
proposal output.  Private target geometry is opened only by ``evaluate``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_ADE20K_FUNCTIONAL_APERTURE_DEV_V1"
RUN_SCHEMA = "blindassist_ade20k_functional_aperture_dev_run_v1"
EVAL_SCHEMA = "blindassist_ade20k_functional_aperture_dev_evaluation_v1"
MODEL_FILENAME = "yolo26x-sem-ade20k.pt"
MODEL_SHA256 = "aa51833f1aa5dc73b378d7ba99f914ef9fa82e434cb23d7b1705dd24bd33b621"
POSITIVE_LABELS = frozenset({"door", "screen door"})
NEGATIVE_LABELS = frozenset({"cabinet", "wardrobe", "chest of drawers", "counter", "refrigerator", "case", "bookcase"})
MIN_POSITIVE_FRACTION = 0.01
DEPTH_PERCENTILE = 20.0
MAX_DEPTH_M = 1.80
MIN_HEIGHT_FRACTION = 0.40
MIN_APPARENT_HEIGHT_M = 0.35
MIN_DINO_IOU = 0.85


def region_depth_percentile(depth: np.ndarray, box: Sequence[float], width: int, height: int, percentile: float = DEPTH_PERCENTILE) -> float | None:
    x1, y1, x2, y2 = validated_box(box, "depth candidate")
    sx, sy = depth.shape[1] / width, depth.shape[0] / height
    left, right = max(0, int(np.floor(x1 * sx))), min(depth.shape[1], int(np.ceil(x2 * sx)))
    top, bottom = max(0, int(np.floor(y1 * sy))), min(depth.shape[0], int(np.ceil(y2 * sy)))
    values = depth[top:bottom, left:right]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.percentile(values, percentile)) if values.size else None


def semantic_evidence(class_map: np.ndarray, box: Sequence[float], width: int, height: int, names: Mapping[int, str]) -> dict[str, Any]:
    x1, y1, x2, y2 = validated_box(box, "semantic candidate")
    sx, sy = class_map.shape[1] / width, class_map.shape[0] / height
    left, right = max(0, int(np.floor(x1 * sx))), min(class_map.shape[1], int(np.ceil(x2 * sx)))
    top, bottom = max(0, int(np.floor(y1 * sy))), min(class_map.shape[0], int(np.ceil(y2 * sy)))
    crop = class_map[top:bottom, left:right]
    total = int(crop.size)
    positive_ids = [label_id for label_id, label in names.items() if label in POSITIVE_LABELS]
    negative_ids = [label_id for label_id, label in names.items() if label in NEGATIVE_LABELS]
    positive = int(np.isin(crop, positive_ids).sum())
    negative = int(np.isin(crop, negative_ids).sum())
    fraction = positive / total if total else 0.0
    return {
        "positive_pixel_count": positive,
        "negative_pixel_count": negative,
        "positive_fraction": fraction,
        "accepted": total > 0 and fraction >= MIN_POSITIVE_FRACTION and positive > negative,
    }


def select_candidate(candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], class_map: np.ndarray, depth: np.ndarray, width: int, height: int, names: Mapping[int, str]) -> dict[str, Any] | None:
    eligible = []
    for candidate in candidates:
        box = validated_box(candidate["bbox_xyxy"], "YOLOE candidate")
        if not box[0] <= width / 2.0 <= box[2]:
            continue
        height_fraction = (box[3] - box[1]) / height
        depth_p20 = region_depth_percentile(depth, box, width, height)
        if depth_p20 is None or height_fraction < MIN_HEIGHT_FRACTION or depth_p20 > MAX_DEPTH_M or depth_p20 * height_fraction < MIN_APPARENT_HEIGHT_M:
            continue
        overlaps = [iou(box, validated_box(row["bbox_xyxy"], "DINO candidate")) for row in dino_candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        if best_iou < MIN_DINO_IOU:
            continue
        evidence = semantic_evidence(class_map, box, width, height, names)
        if evidence["accepted"]:
            eligible.append(dict(candidate) | {
                "sensor_region_depth_p20_m": depth_p20,
                "height_fraction": height_fraction,
                "apparent_height_proxy_m": depth_p20 * height_fraction,
                "dino_consensus_iou": best_iou,
                "semantic_evidence": evidence,
            })
    return max(eligible, key=lambda row: (float(row["semantic_evidence"]["positive_fraction"]), float(row["dino_consensus_iou"]), float(row["proposal_score"]))) if eligible else None


def run(public_path: Path, sealed_run_path: Path, model_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "ADE20K functional development output already exists")
    public, sealed = _read(public_path), _read(sealed_run_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sealed S4 boundary mismatch")
    _require(model_path.is_file() and sha256(model_path) == MODEL_SHA256, "ADE20K model drift")
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    _require(set(public_cases) == set(sealed_cases), "ADE20K development roster mismatch")

    import torch
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(model_path.resolve()))
    names = {int(key): str(value) for key, value in model.names.items()}
    _require(POSITIVE_LABELS <= set(names.values()) and NEGATIVE_LABELS <= set(names.values()), "ADE20K label contract drift")
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        public_case, sealed_case = public_cases[case_id], sealed_cases[case_id]
        image_path = Path(public_case["query"]["image_path"])
        depth_path = Path(public_case["range_sensor"]["depth_path"])
        _require(image_path.is_file() and sha256(image_path) == public_case["query"]["image_sha256"], "public image drift")
        _require(depth_path.is_file() and sha256(depth_path) == public_case["range_sensor"]["depth_sha256"], "public depth drift")
        with Image.open(image_path) as opened:
            width, height = opened.size
        started = time.perf_counter()
        result = model.predict(source=str(image_path), imgsz=640, device=device, verbose=False)[0]
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0
        class_map = result.semantic_mask.data.detach().cpu().numpy()
        depth = decode_depth(depth_path)
        selected = select_candidate(sealed_case["yoloe_candidates"], sealed_case["dino_candidates"], class_map, depth, width, height, names)
        rows.append({"case_id": case_id, "selected_candidate": selected, "completion": selected is not None, "semantic_latency_ms": latency_ms})
        print(f"ade20k-functional {index}/{len(public_cases)} case={case_id} completion={selected is not None}", flush=True)
    payload = {
        "schema_version": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_sha256": sha256(public_path),
        "sealed_s4_run_sha256": sha256(sealed_run_path),
        "private_truth_access": False,
        "stateless_current_frame_only": True,
        "development_only": True,
        "threshold_prompt_model_or_rule_sweep": False,
        "provider": {
            "model_filename": MODEL_FILENAME,
            "model_sha256": MODEL_SHA256,
            "ultralytics": ultralytics.__version__,
            "positive_labels": sorted(POSITIVE_LABELS),
            "negative_labels": sorted(NEGATIVE_LABELS),
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "semantic_acceptance": "positive pixel fraction >= 0.01 and positive pixel count > negative pixel count",
            "sensor_depth_percentile": DEPTH_PERCENTILE,
            "sensor_depth_max_m": MAX_DEPTH_M,
            "minimum_height_fraction": MIN_HEIGHT_FRACTION,
            "minimum_apparent_height_m": MIN_APPARENT_HEIGHT_M,
            "minimum_dino_iou": MIN_DINO_IOU,
            "device": device,
        },
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "ADE20K functional evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "ADE20K development boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    runs = {case["case_id"]: case for case in run_payload["cases"]}
    _require(set(truths) == set(runs), "ADE20K evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], runs[case_id]
        legal = truth["legal_targets"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        opportunity = any(box[0] <= 320.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        selected = observed["selected_candidate"]
        matched_index = None
        if selected is not None:
            overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), target) for target in targets]
            matched_index = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        matched_depth = float(legal[matched_index]["target_depth_median_m"]) if matched_index is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched_index is not None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    decisions = correct + false
    payload = {
        "schema_version": EVAL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "case_count": len(rows),
        "completion_opportunity_count": opportunities,
        "completion_decision_count": decisions,
        "correct_completion_count": correct,
        "false_completion_count": false,
        "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows),
        "correct_completion_coverage": correct / opportunities if opportunities else None,
        "false_completion_fraction_of_decisions": false / decisions if decisions else None,
        "rows": rows,
        "development_only": True,
        "confirmation_claim_authorized": False,
        "terminal": "DEV_FUNCTIONAL_APERTURE_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_FUNCTIONAL_APERTURE_NOT_PROMISING",
    }
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-run", "model", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_run, args.model, args.output, args.device)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
