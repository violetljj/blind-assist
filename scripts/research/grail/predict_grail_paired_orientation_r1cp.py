#!/usr/bin/env python3
"""Persist prediction-only fixed OA-V2 outputs for the frozen R1C-P cohort."""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from PIL import Image
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit
import torch
from torch.nn import functional as F
from torchvision.transforms.functional import pil_to_tensor

from grail_grouping_r1a import predict_groups
from grail_paired_orientation_r1cp import masked_group_crop
from grail_visual_orientation_r1cv import group_members
from run_grail_m1 import negative_reference_indices, sha256_file


CODE_COMMIT = "73b11c9dc83e84daeb563d0c766831f2c66b0a18"
CHECKPOINT_SHA256 = "7b6b7f258d32b95123b9d023005ecca357d8ab944fb83476f532d3cf7a2295eb"


def _von_mises(alpha: float, x: np.ndarray, mu: float, kappa: float) -> np.ndarray:
    return np.exp(kappa * np.cos(alpha * (x - mu))) / (2 * np.pi)


def fit_alpha(distribution: np.ndarray) -> int:
    """Exact OA-V2 val_fit_alpha thresholds without importing rembg."""
    x = np.linspace(0, 2 * np.pi, 360)
    values = distribution.astype(np.float64).copy()
    values /= trapezoid(values, x) + 1e-8
    initial = [x[int(np.argmax(values))], 1]
    saved = []
    for alpha in (1.0, 2.0, 4.0):
        try:
            params, _ = curve_fit(partial(_von_mises, alpha), x, values, p0=initial)
            residual = values - _von_mises(alpha, x, *params)
            r_squared = 1 - np.sum(residual**2) / (np.sum((values - np.mean(values))**2) + 1e-8)
            saved.append((alpha, float(params[1]), float(r_squared)))
            if r_squared > 0.8:
                break
        except Exception:
            saved.append((alpha, 0.0, 0.0))
    alpha, kappa, r_squared = max(saved, key=lambda item: item[2])
    valid = ((alpha == 1 and kappa >= 0.6) or (alpha == 2 and kappa >= 0.5)
             or (alpha == 4 and kappa >= 0.25)) and r_squared >= 0.45
    return int(alpha) if valid else 0


def preprocess(images: list[Image.Image]) -> torch.Tensor:
    tensors = []
    for image in images:
        image = image.convert("RGB")
        width, height = image.size
        if width >= height:
            new_width, new_height = 518, round(height * (518 / width) / 14) * 14
        else:
            new_height, new_width = 518, round(width * (518 / height) / 14) * 14
        image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
        tensor = pil_to_tensor(image).float() / 255.0
        pad_h, pad_w = 518 - tensor.shape[1], 518 - tensor.shape[2]
        tensor = F.pad(tensor, (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2), value=1.0)
        tensors.append(tensor)
    return torch.stack(tensors)


def _prediction(logits: torch.Tensor) -> dict[str, Any]:
    azimuth = logits[:360]
    probability = torch.sigmoid(azimuth).float().cpu().numpy()
    return {
        "azimuth": int(torch.argmax(azimuth)),
        "elevation": int(torch.argmax(logits[360:540])) - 90,
        "roll": int(torch.argmax(logits[540:900])) - 180,
        "alpha": fit_alpha(probability),
        "azimuth_probability": probability.tolist(),
    }


@torch.inference_mode()
def infer(model: Any, images: list[Image.Image], device: torch.device) -> list[dict[str, Any]]:
    logits = model(preprocess(images).unsqueeze(0).to(device)).reshape(len(images), -1)
    return [_prediction(row) for row in logits]


def predict(collection_path: Path, root: Path, query_features_path: Path, reference_features_path: Path,
            model_repo: Path, checkpoint_path: Path, output: Path) -> dict[str, Any]:
    if sha256_file(checkpoint_path) != CHECKPOINT_SHA256:
        raise ValueError("OA-V2 checkpoint identity mismatch")
    commit = subprocess.run(["git", "-C", str(model_repo), "rev-parse", "HEAD"], check=True,
                            capture_output=True, text=True).stdout.strip()
    if commit != CODE_COMMIT:
        raise ValueError("OA-V2 code identity mismatch")
    sys.path.insert(0, str(model_repo))
    from vision_tower import VGGT_OriAny_Ref
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True))
    model.eval().to(device)
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    query_features = torch.load(query_features_path, weights_only=False)
    reference_features = torch.load(reference_features_path, weights_only=False)
    if query_features["collection_sha256"] != sha256_file(collection_path) or \
            reference_features["collection_sha256"] != sha256_file(collection_path):
        raise ValueError("R1C-P feature/collection identity mismatch")
    partial_path = output.with_suffix(".partial.json")
    records = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        if partial["collection_sha256"] != sha256_file(collection_path):
            raise ValueError("R1C-P prediction partial identity mismatch")
        records = partial["records"]
    completed = {record["sample_id"] for record in records}
    negative_indices = negative_reference_indices(collection["rows"])
    for number, (row, query_row, reference_row) in enumerate(zip(
            collection["rows"], query_features["rows"], reference_features["rows"]), 1):
        if row["sample_id"] in completed:
            continue
        if not (row["sample_id"] == query_row["sample_id"] == reference_row["sample_id"]):
            raise ValueError("R1C-P feature row alignment mismatch")
        query_groups = predict_groups(query_row["candidates"])
        query_image = np.asarray(Image.open(root / row["query_image"]).convert("RGB"))
        query_members = group_members(query_row["candidates"], query_groups)
        group_predictions, query_crops = [], []
        for key, indices in query_members.items():
            query_crop = masked_group_crop(query_image, row["candidates"], indices, root)
            query_crops.append(query_crop)
            independent = infer(model, [query_crop], device)[0]
            group_predictions.append({
                "group": int(key[0]), "object_type": key[1], "indices": indices,
                "independent_absolute": independent, "paired_relative": {},
            })
        references = {}
        row_index = number - 1
        for kind, reference_index in (("positive", row_index), ("negative", negative_indices[row_index])):
            source_row = collection["rows"][reference_index]
            source_features = reference_features["rows"][reference_index]
            source_groups = predict_groups(source_features["candidates"])
            source_members = group_members(source_features["candidates"], source_groups)
            source_target = next(i for i, candidate in enumerate(source_features["candidates"]) if candidate["is_target"])
            source_key = (source_groups[source_target], source_features["candidates"][source_target]["object_type"])
            source_image = np.asarray(Image.open(root / source_row["reference_full_image"]).convert("RGB"))
            source_crop = masked_group_crop(
                source_image, source_row["reference_candidates"], source_members[source_key], root
            )
            references[kind] = {
                "reference_index": reference_index,
                "reference_target_index": source_target,
                "reference_groups": source_groups,
                "reference_group": int(source_key[0]),
                "reference_group_indices": source_members[source_key],
                "reference_absolute": infer(model, [source_crop], device)[0],
            }
            for prediction, query_crop in zip(group_predictions, query_crops):
                prediction["paired_relative"][kind] = infer(model, [source_crop, query_crop], device)[1]
        records.append({
            "sample_id": row["sample_id"],
            "query_groups": query_groups,
            "references": references,
            "group_predictions": group_predictions,
        })
        checkpoint = {
            "schema": "blindassist_grail_r1c_p_prediction_checkpoint_v1",
            "prediction_role": "PREDICTION_ONLY_NO_EVALUATOR_TRUTH",
            "collection_sha256": sha256_file(collection_path),
            "records": records,
        }
        temporary = partial_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(checkpoint) + "\n", encoding="utf-8")
        temporary.replace(partial_path)
        print(json.dumps({"state": "R1CP_OAV2", "completed": number, "total": len(collection["rows"])}), flush=True)
    result = {
        "schema": "blindassist_grail_r1c_p_predictions_v1",
        "prediction_role": "PREDICTION_ONLY_NO_EVALUATOR_TRUTH",
        "collection_sha256": sha256_file(collection_path),
        "query_features_sha256": sha256_file(query_features_path),
        "reference_features_sha256": sha256_file(reference_features_path),
        "oa_v2_code_commit": commit,
        "oa_v2_checkpoint_sha256": sha256_file(checkpoint_path),
        "records": records,
    }
    output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--model-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = predict(args.collection, args.root, args.query_features, args.reference_features,
                     args.model_repo, args.checkpoint, args.output)
    print(json.dumps({"records": len(result["records"]), "output_sha256": sha256_file(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
