#!/usr/bin/env python3
"""Development-only CLIP functional verifier for passage doors vs panels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from PIL import Image

from scripts.research.goal_copilot_bridge.last_10m_visual_servo_v1.completion_nearness import _atomic_json, _read, _require
from scripts.research.goal_copilot_bridge.p1_proposal_availability.pa3_semantic import iou, sha256, validated_box


PROTOCOL_ID = "BLINDASSIST_CLIP_FUNCTIONAL_PASSAGE_DOOR_VERIFIER_DEV_V1"
MODEL_REPOSITORY = "openai/clip-vit-base-patch32"
MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
LABELS = [
    "a full-size room door used for people to pass through",
    "a cabinet or cupboard door",
    "a drawer or furniture panel",
    "a wall or window",
    "a floor or ceiling",
]


def _expanded_crop(image: Image.Image, bbox: Sequence[float], expansion: float = 0.20) -> Image.Image:
    x0, y0, x1, y1 = validated_box(list(bbox), "CLIP crop")
    width, height = x1 - x0, y1 - y0
    crop = (max(0, int(x0 - expansion * width)), max(0, int(y0 - expansion * height)), min(image.width, int(x1 + expansion * width)), min(image.height, int(y1 + expansion * height)))
    return image.crop(crop)


def run(public_path: Path, sensor_depth_run_path: Path, sealed_semantic_run_path: Path, model_dir: Path, output_path: Path, device: str) -> dict[str, Any]:
    _require(not output_path.exists(), "CLIP functional development output already exists")
    public, parent, semantic = _read(public_path), _read(sensor_depth_run_path), _read(sealed_semantic_run_path)
    _require(parent.get("private_truth_access") is False and semantic.get("private_truth_access") is False, "CLIP parent crossed private boundary")
    _require(parent.get("public_sha256") == semantic.get("public_sha256") == sha256(public_path), "CLIP parent/public binding mismatch")
    _require((model_dir / "pytorch_model.bin").is_file(), "pinned CLIP weights missing")

    import torch
    from transformers import CLIPModel, CLIPProcessor

    _require(not device.startswith("cuda") or torch.cuda.is_available(), "requested CUDA unavailable")
    processor = CLIPProcessor.from_pretrained(model_dir, local_files_only=True)
    model = CLIPModel.from_pretrained(model_dir, local_files_only=True).to(device).eval()
    public_cases = {case["case_id"]: case for case in public["cases"]}
    parent_cases = {case["case_id"]: case for case in parent["cases"]}
    semantic_cases = {case["case_id"]: case for case in semantic["cases"]}
    _require(set(public_cases) == set(parent_cases) == set(semantic_cases), "CLIP roster mismatch")
    rows = []
    for index, case_id in enumerate(sorted(public_cases), start=1):
        with Image.open(public_cases[case_id]["query"]["image_path"]) as opened:
            image = opened.convert("RGB")
        dino = semantic_cases[case_id]["dino_candidates"]
        eligible = []
        for candidate in parent_cases[case_id]["candidates"]:
            box = validated_box(candidate["bbox_xyxy"], f"{case_id} candidate")
            if not (box[0] <= image.width / 2.0 <= box[2]) or candidate.get("predicted_region_depth_m") is None or float(candidate["predicted_region_depth_m"]) > 2.0:
                continue
            consensus = max([iou(box, item["bbox_xyxy"]) for item in dino] or [0.0])
            if consensus < 0.30:
                continue
            crop = _expanded_crop(image, box)
            inputs = processor(text=LABELS, images=crop, return_tensors="pt", padding=True).to(device)
            with torch.inference_mode():
                logits = model(**inputs).logits_per_image[0]
                probabilities = torch.softmax(logits, dim=0).detach().cpu().tolist()
            eligible.append(dict(candidate) | {"dino_consensus_iou": consensus, "clip_label_probabilities": {label: float(probability) for label, probability in zip(LABELS, probabilities, strict=True)}, "clip_positive_argmax": int(max(range(len(probabilities)), key=probabilities.__getitem__)) == 0})
        supported = [candidate for candidate in eligible if candidate["clip_positive_argmax"]]
        selected = max(supported, key=lambda candidate: (candidate["clip_label_probabilities"][LABELS[0]], candidate["dino_consensus_iou"], candidate["proposal_score"])) if supported else None
        rows.append({"case_id": case_id, "eligible_candidates": eligible, "selected_candidate": selected, "completion": selected is not None})
        print(f"clip-functional {index}/{len(public_cases)} case={case_id} eligible={len(eligible)} completion={selected is not None}", flush=True)
    payload = {"schema_version": "blindassist_clip_functional_passage_door_verifier_dev_run_v1", "protocol_id": PROTOCOL_ID, "created_at_utc": datetime.now(timezone.utc).isoformat(), "public_sha256": sha256(public_path), "sensor_depth_run_sha256": sha256(sensor_depth_run_path), "sealed_semantic_run_sha256": sha256(sealed_semantic_run_path), "private_truth_access": False, "provider_public_aligned_sensor_depth": True, "development_only": True, "threshold_prompt_model_or_rule_sweep": False, "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "weights_sha256": sha256(model_dir / "pytorch_model.bin"), "labels": LABELS, "crop_expansion": 0.20, "decision": "positive label argmax"}, "cases": rows}
    _atomic_json(output_path, payload)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    for name in ("public", "sensor-depth-run", "sealed-semantic-run", "model-dir", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)
    run(args.public, args.sensor_depth_run, args.sealed_semantic_run, args.model_dir, args.output, args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
