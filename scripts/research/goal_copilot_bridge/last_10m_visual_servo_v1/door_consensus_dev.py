#!/usr/bin/env python3
"""Development-only YOLOE/Grounding-DINO consensus for door completion."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _confusion, _read, _require
from scripts.research.goal_copilot_bridge.p0_s0_materialization.run_grounding_dino_s0_r1 import MODEL_REVISION, MODEL_REPOSITORY, WEIGHTS_FILENAME, WEIGHTS_SHA256, sha256_file
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_DOOR_PROPOSAL_CONSENSUS_DEV_V1"
PROMPT = "door."
BOX_THRESHOLD = 0.15
TEXT_THRESHOLD = 0.10
PAIR_IOU_THRESHOLD = 0.30
MAX_DINO_CANDIDATES = 10
RUN_SCHEMA = "blindassist_door_proposal_consensus_dev_run_v1"


def select_consensus(yoloe_candidates: Sequence[Mapping[str, Any]], dino_candidates: Sequence[Mapping[str, Any]], center_x: float) -> dict[str, Any] | None:
    rows = []
    for yoloe in yoloe_candidates:
        box = validated_box(yoloe["bbox_xyxy"], "YOLOE candidate")
        if not (box[0] <= center_x <= box[2]) or yoloe.get("predicted_region_depth_m") is None or float(yoloe["predicted_region_depth_m"]) > 2.0:
            continue
        overlaps = [iou(box, validated_box(dino["bbox_xyxy"], "DINO candidate")) for dino in dino_candidates]
        best_index = max(range(len(overlaps)), key=overlaps.__getitem__) if overlaps else None
        best_iou = overlaps[best_index] if best_index is not None else 0.0
        if best_iou >= PAIR_IOU_THRESHOLD:
            rows.append(dict(yoloe) | {"dino_consensus_iou": best_iou, "dino_candidate": dict(dino_candidates[best_index])})
    return max(rows, key=lambda row: (float(row["dino_consensus_iou"]), float(row["proposal_score"]), -int(row["provider_rank"]))) if rows else None


def run(public_path: Path, candidate_depth_run_path: Path, model_dir: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "consensus development output already exists")
    public, parent = _read(public_path), _read(candidate_depth_run_path)
    _require(parent.get("private_truth_access") is False and parent.get("public_sha256") == sha256(public_path), "candidate-depth parent boundary mismatch")
    weight_path = model_dir / WEIGHTS_FILENAME
    _require(weight_path.is_file() and sha256_file(weight_path) == WEIGHTS_SHA256, "Grounding DINO weight drift")

    import torch
    import transformers
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    _require(set(public_cases) == set(parent_cases), "consensus roster mismatch")
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        image_path = Path(public_cases[case_id]["query"]["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(raw, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(image.height, image.width)])[0]
        labels = result.get("text_labels", result.get("labels", []))
        dino = sorted(
            ({"bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()], "score": float(score.detach().cpu()), "label": str(label)} for box, score, label in zip(result["boxes"], result["scores"], labels)),
            key=lambda row: (-row["score"], row["bbox_xyxy"]),
        )[:MAX_DINO_CANDIDATES]
        selected = select_consensus(parent_cases[case_id]["candidates"], dino, image.width / 2.0)
        rows.append({"case_id": case_id, "dino_candidates": dino, "selected_candidate": selected, "completion": selected is not None})
        print(f"consensus {index}/{len(public_cases)} case={case_id} dino={len(dino)} completion={selected is not None}", flush=True)
    payload = {
        "schema_version": RUN_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "public_sha256": sha256(public_path),
        "candidate_depth_run_sha256": sha256(candidate_depth_run_path),
        "private_truth_access": False,
        "development_only": True,
        "threshold_prompt_model_or_rule_sweep": False,
        "provider": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "prompt": PROMPT, "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "pair_iou_threshold": PAIR_IOU_THRESHOLD, "device": device, "torch": torch.__version__, "transformers": transformers.__version__, "python": platform.python_version()},
        "cases": rows,
    }
    _atomic_json(output_path, payload)
    return payload


def evaluate(private_path: Path, run_path: Path, output_path: Path) -> dict[str, Any]:
    _require(not output_path.exists(), "consensus development evaluation already exists")
    private, run_payload = _read(private_path), _read(run_path)
    _require(run_payload.get("schema_version") == RUN_SCHEMA and run_payload.get("private_truth_access") is False, "consensus run boundary mismatch")
    truths = {case["case_id"]: case for case in private["cases"]}
    runs = {case["case_id"]: case for case in run_payload["cases"]}
    _require(set(truths) == set(runs), "consensus evaluation roster mismatch")
    rows = []
    for case_id in sorted(truths):
        selected = runs[case_id]["selected_candidate"]
        matched = None
        if selected is not None:
            box = validated_box(selected["bbox_xyxy"], f"{case_id} selected")
            legal = truths[case_id]["legal_targets"]
            overlaps = [iou(box, validated_box(target["target_bbox_xyxy"], f"{case_id} target")) for target in legal]
            if overlaps and max(overlaps) >= 0.30:
                matched = legal[max(range(len(overlaps)), key=overlaps.__getitem__)]
        rows.append({"case_id": case_id, "decision": selected is not None, "target_selected": matched is not None, "truth_positive": matched is not None and float(matched["target_depth_median_m"]) <= 2.0})
    confusion = _confusion(rows, "decision")
    payload = {"schema_version": "blindassist_door_proposal_consensus_dev_evaluation_v1", "protocol_id": PROTOCOL_ID, "case_count": len(rows), "completion_confusion": confusion, "target_selection_count": sum(row["target_selected"] for row in rows), "rows": rows, "development_only": True, "confirmation_claim_authorized": False, "terminal": "DEV_CONSENSUS_PROMISING" if confusion["fp"] <= 1 and confusion["tp"] >= 2 else "DEV_CONSENSUS_NOT_PROMISING"}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    execute = sub.add_parser("run")
    for name in ("public", "candidate-depth-run", "model-dir", "output"):
        execute.add_argument(f"--{name}", required=True, type=Path)
    evaluation = sub.add_parser("evaluate")
    for name in ("private", "run", "output"):
        evaluation.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args.public, args.candidate_depth_run, args.model_dir, args.output)
    else:
        result = evaluate(args.private, args.run, args.output)
        print(json.dumps({key: result[key] for key in ("completion_confusion", "target_selection_count", "terminal")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
