#!/usr/bin/env python3
"""Fuse floor-connected functional regions with the frozen CLIP door labels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.clip_functional_verifier_dev import LABELS, MODEL_REPOSITORY, MODEL_REVISION, _expanded_crop
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


def run(public_path: Path, functional_prediction_path: Path, model_dir: Path, output_path: Path, device: str) -> dict[str, Any]:
    public, functional = _read(public_path), _read(functional_prediction_path)
    _require(functional.get("private_truth_access") is False and functional.get("public_sha256") == sha256(public_path), "functional/CLIP public boundary mismatch")
    import torch
    from transformers import CLIPModel, CLIPProcessor

    processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    public_cases = {case["case_id"]: case for case in public["cases"]}
    functional_cases = {case["case_id"]: case for case in functional["cases"]}
    rows = []
    for case_id in sorted(public_cases):
        candidate = functional_cases[case_id]["selected_candidate"]
        probabilities = None
        positive = False
        if candidate is not None:
            image_path = Path(public_cases[case_id]["query"]["image_path"])
            _require(sha256(image_path) == public_cases[case_id]["query"]["image_sha256"], "functional/CLIP image drift")
            with Image.open(image_path) as opened:
                crop = _expanded_crop(opened.convert("RGB"), candidate["bbox_xyxy"])
            inputs = processor(text=LABELS, images=crop, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                values = torch.softmax(model(**inputs).logits_per_image[0], dim=0).detach().cpu().tolist()
            probabilities = {label: float(value) for label, value in zip(LABELS, values, strict=True)}
            positive = int(np.argmax(values)) == 0
        selected = candidate | {"clip_label_probabilities": probabilities} if candidate is not None and positive else None
        rows.append({"case_id": case_id, "functional_candidate_present": candidate is not None, "clip_label_probabilities": probabilities, "clip_positive_argmax": positive, "selected_candidate": selected, "completion": selected is not None})
    payload = {"schema_version": "blindassist_functional_region_clip_fusion_development_prediction_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "public_sha256": sha256(public_path), "functional_prediction_sha256": sha256(functional_prediction_path), "private_truth_access": False, "threshold_prompt_model_or_rule_sweep": False, "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights_sha256": sha256(model_dir / "pytorch_model.bin"), "labels": LABELS, "decision": "positive label argmax"}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    private, prediction = _read(private_path), _read(prediction_path)
    _require(prediction.get("private_truth_access") is False and prediction.get("threshold_prompt_model_or_rule_sweep") is False, "functional/CLIP evaluation boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    rows = []
    for case_id in sorted(truths):
        legal, selected = truths[case_id]["legal_targets"], predictions[case_id]["selected_candidate"]
        targets = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in legal]
        opportunity = any(box[0] <= 320.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(targets, legal, strict=True))
        matched = None
        if selected is not None:
            overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), target) for target in targets]
            matched = int(np.argmax(overlaps)) if overlaps and max(overlaps) >= 0.30 else None
        correct = selected is not None and matched is not None and float(legal[matched]["target_depth_median_m"]) <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched is not None, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    payload = {"schema_version": "blindassist_functional_region_clip_fusion_development_evaluation_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": "DEVELOPMENT_ONLY", "prediction_sha256": sha256(prediction_path), "private_sha256": sha256(private_path), "opportunity_count": opportunities, "decision_count": sum(row["completion_decision"] for row in rows), "correct_count": correct, "false_count": false, "coverage": correct / opportunities if opportunities else None, "rows": rows, "terminal": "DEV_FUNCTIONAL_REGION_CLIP_FUSION_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_FUNCTIONAL_REGION_CLIP_FUSION_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--functional-prediction", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path, evaluation_path = args.output / "prediction.json", args.output / "evaluation.json"
    run(args.public, args.functional_prediction, args.model_dir, prediction_path, args.device)
    result = evaluate(args.private, prediction_path, evaluation_path)
    print(json.dumps({key: result[key] for key in ("opportunity_count", "decision_count", "correct_count", "false_count", "coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
