#!/usr/bin/env python3
"""Test nearest-member DINOv2 object-context memory on consumed PV28."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
from typing import Any

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_object_context_memory_posthoc as feature  # noqa: E402
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-object-context-set-memory-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-object-context-set-memory-posthoc-result-v1"


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for dependency in protocol["dependencies"]:
        pixel.require(pixel.sha256(HERE / dependency["path"]) == dependency["sha256"], f"DEPENDENCY_HASH:{dependency['path']}")
    for key in ("cohort", "predecessor"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    predecessor = pixel.load_json(HERE / protocol["predecessor"]["path"])
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort = pixel.load_json(HERE / protocol["cohort"]["path"])
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    model_path = ROOT / protocol["model"]["path"]
    pixel.require(pixel.sha256(model_path) == protocol["model"]["sha256"], "MODEL_HASH")

    references = protocol["memory"]["reference_images"]
    positives = protocol["evaluation"]["positive_query_images"]
    negatives = protocol["evaluation"]["negative_query_images"]
    wanted = set(references + positives + negatives)
    rows = {key: value for key, value in cohort["images"].items() if key in wanted}
    pixel.require(set(rows) == wanted, "IMAGE_KEYS")
    images: dict[str, Image.Image] = {}
    receipts: dict[str, Any] = {}
    for key, row in rows.items():
        manifest_key = f"{row['scan_id']}/sequence.zip"
        archive_path = artifact_root / cohort["source_manifest"][manifest_key]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        receipts[key] = {"image_sha256": hashlib.sha256(payload).hexdigest(), "bbox_xyxy": row["bbox_xyxy"]}

    from romatch.models.transformer import vit_large

    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]), patch_size=14,
        init_values=1.0, ffn_layer="mlp", block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to("cuda:0")
    descriptors: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for key, image in images.items():
            parts = []
            for scale in protocol["descriptor"]["crop_scales"]:
                token = model.forward_features(
                    feature._tensor(feature._crop(image, rows[key]["bbox_xyxy"], float(scale)), int(protocol["descriptor"]["input_size"])).to("cuda:0")
                )["x_norm_clstoken"][0].float()
                parts.append(torch.nn.functional.normalize(token, dim=0))
            descriptors[key] = torch.nn.functional.normalize(torch.cat(parts), dim=0).cpu().numpy()
    del model, weights
    torch.cuda.empty_cache()

    scores: dict[str, float] = {}
    winning_reference: dict[str, str] = {}
    pair_scores: dict[str, dict[str, float]] = {}
    for query in positives + negatives:
        row = {reference: float(np.dot(descriptors[reference], descriptors[query])) for reference in references}
        winner = max(row, key=row.get)
        pair_scores[query] = row
        scores[query] = row[winner]
        winning_reference[query] = winner
    comparisons = [scores[p] > scores[n] for p in positives for n in negatives]
    min_positive, max_negative = min(scores[p] for p in positives), max(scores[n] for n in negatives)
    gate_met = all(comparisons)
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_PV28_OBJECT_CONTEXT_SET_MEMORY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "conclusion": (
            "L10_3RSCAN_OBJECT_CONTEXT_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_OBJECT_CONTEXT_SET_MEMORY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "scores": scores,
        "winning_reference": winning_reference,
        "pair_scores": pair_scores,
        "metrics": {
            "positive_queries": len(positives), "negative_queries": len(negatives),
            "correct_pairwise_rankings": int(sum(comparisons)), "pairwise_rankings": len(comparisons),
            "minimum_positive_score": min_positive, "maximum_negative_score": max_negative,
            "separation_margin": float(min_positive - max_negative),
        },
        "descriptor_receipts": {key: {"sha256": hashlib.sha256(value.tobytes()).hexdigest(), "dimensions": int(value.size)} for key, value in descriptors.items()},
        "image_receipts": receipts,
        "runtime": {"device": torch.cuda.get_device_name(0), "dinov2_calls": len(images) * len(protocol["descriptor"]["crop_scales"])},
        "literature_motivation": protocol["literature_motivation"],
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
