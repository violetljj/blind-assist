#!/usr/bin/env python3
"""Development-only SAM 3 text-mask plus metric-ground functional regions."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence

import numpy as np
from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.functional_region_completion_dev import CONTACT_DEPTH_MAX_M, CONTACT_PIXEL_MIN, functional_observation, observer_connected_ground
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.ground_plane import ground_mask_from_depth
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.mask_depth_completion_dev import binary_mask_iou, mask_observation
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.materialize_tartanair_s2 import decode_depth
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


SOURCE_REVISION = "8f0b7f4d4e7eda2ed606ebde6702c93359ad01da"
PROMPT = "door"
CONFIDENCE_THRESHOLD = 0.5
MASK_HEIGHT_FRACTION_MIN = 0.40


def safe_ground_observation(depth: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    try:
        ground, plane = ground_mask_from_depth(depth)
        return ground, {"status": "available", **plane}
    except ValueError as error:
        return np.zeros(depth.shape, dtype=bool), {"status": "unavailable", "reason": str(error)}


def select_sam3_functional_candidate(candidates: list[dict[str, Any]], width: int) -> dict[str, Any] | None:
    eligible = [candidate for candidate in candidates if candidate["bbox_xyxy"][0] <= width / 2.0 <= candidate["bbox_xyxy"][2] and candidate["mask_height_fraction"] >= MASK_HEIGHT_FRACTION_MIN and candidate["ground_contact_pixel_count"] >= CONTACT_PIXEL_MIN and candidate["ground_contact_depth_median_m"] is not None and candidate["ground_contact_depth_median_m"] <= CONTACT_DEPTH_MAX_M]
    return max(eligible, key=lambda row: (row["proposal_score"], row["ground_contact_pixel_count"])) if eligible else None


def run_provider(public_path: Path, source_root: Path, vendor_root: Path, checkpoint_path: Path, output_path: Path, device: str, confidence_threshold: float = CONFIDENCE_THRESHOLD, role: str = "DEVELOPMENT_ONLY") -> dict[str, Any]:
    public = _read(public_path)
    revision = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
    _require(revision == SOURCE_REVISION, "SAM 3 source revision drift")
    _require(checkpoint_path.is_file() and checkpoint_path.stat().st_size > 3_000_000_000, "SAM 3 checkpoint unavailable")
    # Keep the validated workspace runtime ahead of supplemental packages so
    # the vendor directory cannot shadow torch/numpy with transitive wheels.
    sys.path.append(str(vendor_root.resolve()))
    sys.path.insert(0, str(source_root.resolve()))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    import torch

    model = build_sam3_image_model(device=device, checkpoint_path=str(checkpoint_path.resolve()), load_from_HF=False, enable_segmentation=True, enable_inst_interactivity=False, compile=False)
    _require(0.0 < confidence_threshold <= 1.0, "invalid SAM 3 confidence threshold")
    processor = Sam3Processor(model, device=device, confidence_threshold=confidence_threshold)
    masks_dir = output_path.parent / "selected_masks"
    ground_dir = output_path.parent / "ground_masks"
    masks_dir.mkdir(parents=True, exist_ok=False)
    ground_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for case in public["cases"]:
        image_path, depth_path = Path(case["query"]["image_path"]), Path(case["range_sensor"]["depth_path"])
        _require(sha256(image_path) == case["query"]["image_sha256"] and sha256(depth_path) == case["range_sensor"]["depth_sha256"], "SAM 3 public input drift")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        depth = decode_depth(depth_path)
        ground, plane = safe_ground_observation(depth)
        connected = observer_connected_ground(ground)
        ground_path = ground_dir / f"{case['case_id']}.png"
        Image.fromarray(connected.astype(np.uint8) * 255).save(ground_path)
        # SAM 3's official image-inference examples run the full forward pass
        # under CUDA bfloat16 autocast.  Without it the vision trunk mixes a
        # bfloat16 activation with float32 linear weights and fails before
        # producing a proposal.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            state = processor.set_text_prompt(prompt=PROMPT, state=processor.set_image(image))
        masks = state["masks"].squeeze(1).detach().cpu().numpy()
        boxes = state["boxes"].detach().cpu().tolist()
        scores = state["scores"].detach().cpu().tolist()
        candidates = []
        for index in sorted(range(len(scores)), key=lambda item: scores[item], reverse=True)[:10]:
            mask = masks[index].astype(bool)
            try:
                observed = mask_observation(mask, depth, boxes[index], image.width, image.height)
            except ValueError:
                continue
            candidates.append(observed | functional_observation(mask, connected, depth) | {"provider_rank": len(candidates) + 1, "proposal_score": float(scores[index]), "result_index": index})
        selected = select_sam3_functional_candidate(candidates, image.width)
        if selected is not None:
            selected_path = masks_dir / f"{case['case_id']}.png"
            Image.fromarray(masks[selected["result_index"]].astype(np.uint8) * 255).save(selected_path)
            selected = selected | {"selected_mask_path": str(selected_path.resolve()), "selected_mask_sha256": sha256(selected_path)}
        rows.append({"case_id": case["case_id"], "image_width": image.width, "image_height": image.height, "plane": plane, "connected_ground_mask_path": str(ground_path.resolve()), "connected_ground_mask_sha256": sha256(ground_path), "candidates": candidates, "selected_candidate": selected, "completion": selected is not None})
        print(f"sam3-functional {len(rows)}/{len(public['cases'])} case={case['case_id']} candidates={len(candidates)} completion={selected is not None}", flush=True)
    _require(role in {"DEVELOPMENT_ONLY", "INDEPENDENT_CONFIRMATION"}, "invalid SAM 3 evidence role")
    payload = {"schema_version": "blindassist_sam3_functional_region_prediction_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": role, "public_sha256": sha256(public_path), "private_truth_access": False, "provider": {"source_revision": SOURCE_REVISION, "checkpoint_sha256": sha256(checkpoint_path), "prompt": PROMPT, "confidence_threshold": confidence_threshold, "mask_height_fraction_min": MASK_HEIGHT_FRACTION_MIN, "contact_pixel_min": CONTACT_PIXEL_MIN, "contact_depth_max_m": CONTACT_DEPTH_MAX_M, "device": device}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def evaluate(public_path: Path, private_path: Path, prediction_path: Path, output_path: Path) -> dict[str, Any]:
    private, prediction = _read(private_path), _read(prediction_path)
    _require(prediction.get("private_truth_access") is False and prediction.get("public_sha256") == sha256(public_path), "SAM 3 evaluation boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    predictions = {case["case_id"]: case for case in prediction["cases"]}
    rows = []
    for case_id in sorted(truths):
        truth, observed = truths[case_id], predictions[case_id]
        targets = truth["legal_targets"]
        boxes = [validated_box(target["target_bbox_xyxy"], f"{case_id} target") for target in targets]
        opportunity = any(box[0] <= observed["image_width"] / 2.0 <= box[2] and float(target["target_depth_median_m"]) <= 2.0 for box, target in zip(boxes, targets, strict=True))
        selected = observed["selected_candidate"]
        matched, best_bbox_iou, best_mask_iou = None, None, None
        if selected is not None:
            bbox_overlaps = [iou(validated_box(selected["bbox_xyxy"], f"{case_id} selected"), box) for box in boxes]
            best_bbox_iou = max(bbox_overlaps) if bbox_overlaps else None
            matched = int(np.argmax(bbox_overlaps)) if bbox_overlaps and best_bbox_iou >= 0.30 else None
            selected_path = Path(selected["selected_mask_path"])
            _require(sha256(selected_path) == selected["selected_mask_sha256"], "SAM 3 selected mask drift")
            with Image.open(selected_path) as opened:
                selected_mask = np.asarray(opened.convert("L")) > 0
            mask_overlaps = []
            for target in targets:
                target_path = Path(target["target_mask_path"])
                _require(sha256(target_path) == target["target_mask_sha256"], "SAM 3 target mask drift")
                with Image.open(target_path) as opened:
                    mask_overlaps.append(binary_mask_iou(selected_mask, np.asarray(opened.convert("L")) > 0))
            best_mask_iou = max(mask_overlaps) if mask_overlaps else None
        matched_depth = float(targets[matched]["target_depth_median_m"]) if matched is not None else None
        correct = selected is not None and matched_depth is not None and matched_depth <= 2.0
        rows.append({"case_id": case_id, "completion_opportunity": opportunity, "completion_decision": selected is not None, "target_selected": matched is not None, "selected_target_bbox_iou": best_bbox_iou, "selected_target_mask_iou": best_mask_iou, "matched_target_depth_m": matched_depth, "correct_completion": correct, "false_completion": selected is not None and not correct, "missed_completion_opportunity": opportunity and not correct})
    opportunities = sum(row["completion_opportunity"] for row in rows)
    correct = sum(row["correct_completion"] for row in rows)
    false = sum(row["false_completion"] for row in rows)
    passed = false == 0 and opportunities >= 8 and correct / opportunities >= 0.50
    role = prediction.get("role")
    terminal_prefix = "CONFIRMATION" if role == "INDEPENDENT_CONFIRMATION" else "DEV"
    payload = {"schema_version": "blindassist_sam3_functional_region_evaluation_v2", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": role, "prediction_sha256": sha256(prediction_path), "private_sha256": sha256(private_path), "opportunity_count": opportunities, "decision_count": sum(row["completion_decision"] for row in rows), "correct_count": correct, "false_count": false, "coverage": correct / opportunities if opportunities else None, "rows": rows, "terminal": f"{terminal_prefix}_SAM3_FUNCTIONAL_REGION_{'PASSED' if passed else 'NOT_PASSED'}"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--confidence-threshold", type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "INDEPENDENT_CONFIRMATION"), default="DEVELOPMENT_ONLY")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    prediction_path, evaluation_path = args.output / "prediction.json", args.output / "evaluation.json"
    run_provider(args.public, args.source_root, args.vendor_root, args.checkpoint, prediction_path, args.device, args.confidence_threshold, args.role)
    result = evaluate(args.public, args.private, prediction_path, evaluation_path)
    print(json.dumps({key: result[key] for key in ("opportunity_count", "decision_count", "correct_count", "false_count", "coverage", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
