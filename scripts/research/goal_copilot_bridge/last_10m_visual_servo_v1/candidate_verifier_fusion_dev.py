#!/usr/bin/env python3
"""Development-only fusion of depth-aperture candidates and learned verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_candidate_verifier_train import expanded_crop
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_CANDIDATE_VERIFIER_FUSION_DEV_V1"
WEIGHTS_SHA256 = "6e8711a6547749a7ca2837d3d1b0a51dc711c03e3f1cfc23affee6aa762aee62"
THRESHOLD = 0.5


def run(public_path: Path, sealed_s5_run_path: Path, weights_path: Path, training_receipt_path: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "candidate verifier fusion output already exists")
    public, sealed, training = _read(public_path), _read(sealed_s5_run_path), _read(training_receipt_path)
    _require(sealed.get("private_truth_access") is False and sealed.get("public_sha256") == sha256(public_path), "sealed S5 boundary mismatch")
    _require(training.get("decision_threshold") == THRESHOLD and sha256(weights_path) == training.get("best_weights_sha256") == WEIGHTS_SHA256, "candidate verifier weights drift")

    import torch
    from torch import nn
    from torchvision import models, transforms

    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device).eval()
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(checkpoint["normalization"]["mean"], checkpoint["normalization"]["std"])])
    public_cases = {case["case_id"]: case for case in public["cases"]}
    sealed_cases = {case["case_id"]: case for case in sealed["cases"]}
    rows = []
    for case_id in sorted(public_cases):
        image_path = Path(public_cases[case_id]["query"]["image_path"])
        _require(sha256(image_path) == public_cases[case_id]["query"]["image_sha256"], "candidate verifier image drift")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        verified = []
        for candidate in sealed_cases[case_id]["yoloe_candidates"]:
            if not candidate["bbox_xyxy"][0] <= image.width / 2.0 <= candidate["bbox_xyxy"][2]:
                continue
            height_fraction = (candidate["bbox_xyxy"][3] - candidate["bbox_xyxy"][1]) / image.height
            p20, median = candidate.get("sensor_region_depth_p20_m"), candidate.get("sensor_region_depth_m")
            overlaps = [iou(validated_box(candidate["bbox_xyxy"], "candidate"), validated_box(row["bbox_xyxy"], "DINO")) for row in sealed_cases[case_id]["dino_candidates"]]
            consensus = max(overlaps, default=0.0)
            if p20 is None or median is None or height_fraction < 0.40 or not float(p20) <= 2.0 < float(median) or float(p20) * height_fraction < 0.35 or consensus < 0.85:
                continue
            tensor = transform(expanded_crop(image, candidate["bbox_xyxy"])).unsqueeze(0).to(device)
            with torch.inference_mode():
                probability = float(torch.softmax(model(tensor), dim=1)[0, 1].cpu())
            if probability >= THRESHOLD:
                verified.append(dict(candidate) | {"height_fraction": height_fraction, "depth_aperture_span_m": float(median) - float(p20), "dino_consensus_iou": consensus, "door_probability": probability})
        selected = max(verified, key=lambda row: (float(row["door_probability"]), float(row["depth_aperture_span_m"]), float(row["proposal_score"]))) if verified else None
        rows.append({"case_id": case_id, "verified_candidate_count": len(verified), "selected_candidate": selected, "completion": selected is not None})
    payload = {"schema_version": "blindassist_candidate_verifier_fusion_dev_run_v1", "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sealed_s5_run_sha256": sha256(sealed_s5_run_path), "training_receipt_sha256": sha256(training_receipt_path), "weights_sha256": WEIGHTS_SHA256, "private_truth_access": False, "stateless_current_frame_only": True, "development_only": True, "threshold_or_rule_sweep": False, "decision_threshold": THRESHOLD, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "candidate verifier fusion evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("private_truth_access") is False and run_payload.get("threshold_or_rule_sweep") is False, "candidate verifier evaluation boundary mismatch")
    truths, runs = ({case["case_id"]: case for case in payload["cases"]} for payload in (private, run_payload))
    rows = []
    for case_id in sorted(truths):
        legal, selected = truths[case_id]["legal_targets"], runs[case_id]["selected_candidate"]
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
    payload = {"schema_version": "blindassist_candidate_verifier_fusion_dev_evaluation_v1", "protocol_id": PROTOCOL_ID, "case_count": len(rows), "completion_opportunity_count": opportunities, "completion_decision_count": correct + false, "correct_completion_count": correct, "false_completion_count": false, "missed_completion_opportunity_count": sum(row["missed_completion_opportunity"] for row in rows), "correct_completion_coverage": correct / opportunities if opportunities else None, "rows": rows, "development_only": True, "confirmation_claim_authorized": False, "terminal": "DEV_CANDIDATE_VERIFIER_FUSION_PROMISING" if false == 0 and opportunities >= 8 and correct / opportunities >= 0.50 else "DEV_CANDIDATE_VERIFIER_FUSION_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "sealed-s5-run", "weights", "training-receipt", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    execute.add_argument("--device", default="cuda:0")
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.sealed_s5_run, args.weights, args.training_receipt, args.output, args.device)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_opportunity_count", "correct_completion_count", "false_completion_count", "correct_completion_coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
