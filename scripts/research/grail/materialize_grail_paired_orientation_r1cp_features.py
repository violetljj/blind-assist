#!/usr/bin/env python3
"""Materialize frozen query and bilateral reference DINO features for R1C-P."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
import torch

from run_grail_m1 import VISUAL_WEIGHTS_SHA256, encode_images, expanded_crop, materialize_features, sha256_file


@torch.inference_mode()
def materialize_reference(collection_path: Path, root: Path, visual_path: Path, output: Path) -> dict:
    if sha256_file(visual_path / "model.safetensors") != VISUAL_WEIGHTS_SHA256:
        raise ValueError("R1C-P visual weights identity mismatch")
    collection_hash = sha256_file(collection_path)
    if output.exists():
        cached = torch.load(output, weights_only=False)
        if cached["collection_sha256"] != collection_hash:
            raise ValueError("R1C-P reference feature identity mismatch")
        return cached
    from transformers import AutoImageProcessor, AutoModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(visual_path, local_files_only=True)
    visual = AutoModel.from_pretrained(visual_path, local_files_only=True).to(device).eval()
    rows = []
    collection = json.loads(collection_path.read_text(encoding="utf-8"))
    for number, row in enumerate(collection["rows"], 1):
        image = Image.open(root / row["reference_full_image"]).convert("RGB")
        crops = [expanded_crop(image, candidate["bbox"]) for candidate in row["reference_candidates"]]
        embeddings, _ = encode_images(crops, processor, visual, device)
        rows.append({
            "sample_id": row["sample_id"],
            "house_index": row["house_index"],
            "reference_full_image": row["reference_full_image"],
            "candidates": [
                {**candidate, "embedding": embedding.astype("float16")}
                for candidate, embedding in zip(row["reference_candidates"], embeddings)
            ],
        })
        if number % 20 == 0:
            print(json.dumps({"state": "R1CP_REFERENCE_FEATURES", "completed": number, "total": len(collection["rows"])}), flush=True)
    result = {"schema": "blindassist_grail_r1c_p_reference_features_v1",
              "collection_sha256": collection_hash, "visual_weights_sha256": VISUAL_WEIGHTS_SHA256, "rows": rows}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    torch.save(result, temporary)
    temporary.replace(output)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--visual-model", type=Path, required=True)
    parser.add_argument("--depth-model", type=Path, required=True)
    parser.add_argument("--query-output", type=Path, required=True)
    parser.add_argument("--reference-output", type=Path, required=True)
    args = parser.parse_args()
    query = materialize_features(
        args.collection, args.root, args.query_output, args.visual_model, args.depth_model
    )
    reference = materialize_reference(args.collection, args.root, args.visual_model, args.reference_output)
    print(json.dumps({"query_rows": len(query["rows"]), "reference_rows": len(reference["rows"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
