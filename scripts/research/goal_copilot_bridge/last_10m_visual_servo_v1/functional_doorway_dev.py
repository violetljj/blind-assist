#!/usr/bin/env python3
"""Development-only functional doorway context verifier."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import BOX_THRESHOLD, MAX_DINO_CANDIDATES, PAIR_IOU_THRESHOLD, TEXT_THRESHOLD, select_consensus
from scripts.research.goal_copilot_bridge.p0_s0_materialization.run_grounding_dino_s0_r1 import MODEL_REVISION, MODEL_REPOSITORY, WEIGHTS_FILENAME, WEIGHTS_SHA256, sha256_file
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


PROTOCOL_ID = "BLINDASSIST_FUNCTIONAL_DOORWAY_CONTEXT_DEV_V1"
FUNCTIONAL_PROMPT = "full-size room doorway . passage entrance ."


def run(public_path: Path, sensor_depth_run_path: Path, model_dir: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "functional doorway development output already exists")
    public, parent = _read(public_path), _read(sensor_depth_run_path)
    _require(parent.get("private_truth_access") is False and parent.get("public_sha256") == sha256(public_path), "functional doorway parent boundary mismatch")
    weight_path = model_dir / WEIGHTS_FILENAME
    _require(weight_path.is_file() and sha256_file(weight_path) == WEIGHTS_SHA256, "Grounding DINO weight drift")

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    _require(not device.startswith("cuda") or torch.cuda.is_available(), "requested CUDA unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    _require(set(public_cases) == set(parent_cases), "functional doorway roster mismatch")
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        image_path = Path(public_cases[case_id]["query"]["image_path"])
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, text=FUNCTIONAL_PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(raw, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(image.height, image.width)])[0]
        labels = result.get("text_labels", result.get("labels", []))
        functional = sorted(({"bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()], "score": float(score.detach().cpu()), "label": str(label)} for box, score, label in zip(result["boxes"], result["scores"], labels)), key=lambda row: (-row["score"], row["bbox_xyxy"]))[:MAX_DINO_CANDIDATES]
        selected = select_consensus(parent_cases[case_id]["candidates"], functional, image.width / 2.0)
        rows.append({"case_id": case_id, "functional_candidates": functional, "selected_candidate": selected, "completion": selected is not None})
        print(f"functional {index}/{len(public_cases)} case={case_id} proposals={len(functional)} completion={selected is not None}", flush=True)
    payload = {"schema_version": "blindassist_functional_doorway_context_dev_run_v1", "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sensor_depth_run_sha256": sha256(sensor_depth_run_path), "private_truth_access": False, "provider_public_aligned_sensor_depth": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "functional_prompt": FUNCTIONAL_PROMPT, "provider": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "pair_iou_threshold": PAIR_IOU_THRESHOLD, "device": device}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", required=True, type=Path)
    parser.add_argument("--sensor-depth-run", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args.public, args.sensor_depth_run, args.model_dir, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
