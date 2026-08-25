#!/usr/bin/env python3
"""Evaluate the frozen OA-V2 paired-relative slot baseline on R1C-L validation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from grail_procthor_native_m0 import sha256_file


CODE_COMMIT = "73b11c9dc83e84daeb563d0c766831f2c66b0a18"
CHECKPOINT_SHA256 = "7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb"


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _masked_image(root: Path, view: dict[str, Any]) -> Image.Image:
    rgb = np.asarray(Image.open(root / view["rgb"]).convert("RGB")).copy()
    mask = np.asarray(Image.open(root / view["owner_union_mask"]).convert("L")) > 0
    if rgb.shape[:2] != mask.shape:
        raise ValueError(f"R1C-L OA-V2 RGB/mask mismatch for {view['view_id']}")
    rgb[~mask] = 255
    return Image.fromarray(rgb)


def _preprocess(image: Image.Image) -> torch.Tensor:
    width, height = image.size
    if width >= height:
        new_width, new_height = 518, round(height * (518 / width) / 14) * 14
    else:
        new_height, new_width = 518, round(width * (518 / height) / 14) * 14
    image = image.resize((max(new_width, 14), max(new_height, 14)), Image.Resampling.BICUBIC)
    tensor = pil_to_tensor(image).float() / 255.0
    pad_h, pad_w = 518 - tensor.shape[1], 518 - tensor.shape[2]
    return F.pad(tensor, (pad_w // 2, pad_w - pad_w // 2,
                          pad_h // 2, pad_h - pad_h // 2), value=1.0)


def _pair_key(pair: dict[str, Any]) -> str:
    return "|".join(sorted((pair["reference_view_id"], pair["query_view_id"])))


def evaluate(collection_path: Path, root: Path, model_repo: Path, checkpoint: Path,
             output: Path, batch_size: int) -> dict[str, Any]:
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("R1C-L OA-V2 checkpoint identity mismatch")
    commit = subprocess.run(["git", "-C", str(model_repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != CODE_COMMIT:
        raise ValueError("R1C-L OA-V2 code identity mismatch")
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    if collection.get("role") != "validation":
        raise ValueError("R1C-L OA-V2 baseline is validation-only")
    views = {row["view_id"]: row for row in collection["views"]}
    canonical: dict[str, tuple[str, str]] = {}
    for pair in collection["pairs"]:
        canonical.setdefault(_pair_key(pair), tuple(sorted(
            (pair["reference_view_id"], pair["query_view_id"]))))

    sys.path.insert(0, str(model_repo))
    from vision_tower import VGGT_OriAny_Ref
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to(device)

    progress_path = output.with_name("oa_v2_progress.json")
    predictions: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("collection_sha256") != sha256_file(collection_path):
            raise ValueError("R1C-L OA-V2 progress identity mismatch")
        predictions = progress.get("predictions", {})
    pending = [(key, value) for key, value in sorted(canonical.items()) if key not in predictions]
    with torch.inference_mode():
        for offset in range(0, len(pending), batch_size):
            chunk = pending[offset:offset + batch_size]
            inputs = torch.stack([
                torch.stack([_preprocess(_masked_image(root, views[reference])),
                             _preprocess(_masked_image(root, views[query]))])
                for _, (reference, query) in chunk
            ]).to(device)
            logits = model(inputs)[:, 1, :360]
            degrees = torch.argmax(logits, dim=-1).cpu().tolist()
            for (key, _), degree in zip(chunk, degrees):
                cosine = math.cos(math.radians(degree))
                mode = "PRESERVE" if cosine > 1e-8 else "FLIP" if cosine < -1e-8 else "UNKNOWN"
                predictions[key] = {"relative_azimuth_degree": degree, "slot_mode": mode}
            _atomic_json(progress_path, {
                "schema": "blindassist_grail_r1c_l_oa_v2_progress_v1",
                "collection_sha256": sha256_file(collection_path),
                "completed_units": len(predictions), "total_units": len(canonical),
                "last_progress_at": datetime.now(timezone.utc).isoformat(),
                "predictions": predictions,
            })
            print(json.dumps({"completed": len(predictions), "total": len(canonical)}), flush=True)

    correct = 0
    by_type: dict[str, list[int]] = {}
    unknown = 0
    for pair in collection["pairs"]:
        mode = predictions[_pair_key(pair)]["slot_mode"]
        passed = mode in pair["valid_slot_modes"]
        correct += int(passed)
        unknown += int(mode == "UNKNOWN")
        counts = by_type.setdefault(pair["object_type"], [0, 0])
        counts[0] += int(passed)
        counts[1] += 1
    total = len(collection["pairs"])
    result = {
        "schema": "blindassist_grail_r1c_l_oa_v2_validation_baseline_v1",
        "collection_sha256": sha256_file(collection_path),
        "oa_v2_code_commit": commit, "oa_v2_checkpoint_sha256": sha256_file(checkpoint),
        "unique_unordered_view_pairs": len(canonical), "slot_correct": correct,
        "slot_total": total, "slot_accuracy": correct / max(total, 1), "unknown": unknown,
        "by_type": {key: {"correct": value[0], "total": value[1],
                          "accuracy": value[0] / max(value[1], 1)}
                    for key, value in sorted(by_type.items())},
        "final_test_accessed": False,
    }
    _atomic_json(output, result)
    print(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    evaluate(args.collection, args.root, args.model_repo, args.checkpoint,
             args.output, args.batch_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
