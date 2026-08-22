#!/usr/bin/env python3
"""Run frozen Grounding DINO proposals on public development inputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.door_consensus_dev import BOX_THRESHOLD, MAX_DINO_CANDIDATES, PROMPT, TEXT_THRESHOLD
from scripts.research.goal_copilot_bridge.p0_s0_materialization.run_grounding_dino_s0_r1 import MODEL_REVISION, MODEL_REPOSITORY, WEIGHTS_FILENAME, WEIGHTS_SHA256, sha256_file
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import sha256


def run(public_path: Path, model_dir: Path, output_path: Path, device: str, role: str = "DEVELOPMENT_ONLY") -> dict:
    _require(not output_path.exists(), "public DINO development output already exists")
    public = _read(public_path)
    weight = model_dir / WEIGHTS_FILENAME
    _require(weight.is_file() and sha256_file(weight) == WEIGHTS_SHA256, "public DINO weights drift")
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_dir, local_files_only=True, use_safetensors=True).to(device).eval()
    rows = []
    for case in public["cases"]:
        image_path = Path(case["query"]["image_path"])
        _require(sha256(image_path) == case["query"]["image_sha256"], "public DINO image drift")
        with Image.open(image_path) as opened:
            image = opened.convert("RGB")
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(device)
        with torch.inference_mode():
            raw = model(**inputs)
        result = processor.post_process_grounded_object_detection(raw, inputs.input_ids, threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD, target_sizes=[(image.height, image.width)])[0]
        labels = result.get("text_labels", result.get("labels", []))
        candidates = sorted(({"bbox_xyxy": [float(value) for value in box.detach().cpu().tolist()], "score": float(score.detach().cpu()), "label": str(label)} for box, score, label in zip(result["boxes"], result["scores"], labels)), key=lambda row: (-row["score"], row["bbox_xyxy"]))[:MAX_DINO_CANDIDATES]
        rows.append({"case_id": case["case_id"], "image_width": image.width, "image_height": image.height, "dino_candidates": candidates})
        print(f"dino-public {len(rows)}/{len(public['cases'])} case={case['case_id']} candidates={len(candidates)}", flush=True)
    _require(role in {"DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"}, "invalid public DINO role")
    payload = {"schema_version": "blindassist_dino_public_development_run_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(), "role": role, "public_sha256": sha256(public_path), "private_truth_access": False, "provider": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights_sha256": WEIGHTS_SHA256, "prompt": PROMPT, "box_threshold": BOX_THRESHOLD, "text_threshold": TEXT_THRESHOLD, "device": device}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--role", choices=("DEVELOPMENT_ONLY", "CONFIRMATION_ONLY"), default="DEVELOPMENT_ONLY")
    args = parser.parse_args(argv)
    result = run(args.public, args.model_dir, args.output, args.device, args.role)
    print(json.dumps({"case_count": len(result["cases"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
