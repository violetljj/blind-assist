#!/usr/bin/env python3
"""Test a DINOv2 object-plus-context memory where pixel cycles are unreachable."""

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
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-object-context-memory-posthoc-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-object-context-memory-posthoc-result-v1"


def _crop(image: Image.Image, box: list[float], scale: float) -> Image.Image:
    x0, y0, x1, y1 = (float(value) for value in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width, height = (x1 - x0) * scale, (y1 - y0) * scale
    left, top = max(0, int(np.floor(cx - width / 2))), max(0, int(np.floor(cy - height / 2)))
    right = min(image.width, int(np.ceil(cx + width / 2)))
    bottom = min(image.height, int(np.ceil(cy + height / 2)))
    pixel.require(right > left and bottom > top, "EMPTY_CROP")
    return image.crop((left, top, right, bottom))


def _tensor(image: Image.Image, size: int) -> torch.Tensor:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    value = TF.pil_to_tensor(resized).float().div_(255.0)
    return TF.normalize(value, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).unsqueeze(0)


def run(protocol_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    for key in ("cohort", "predecessor"):
        row = protocol[key]
        pixel.require(pixel.sha256(HERE / row["path"]) == row["sha256"], f"{key.upper()}_HASH")
    predecessor = pixel.load_json(HERE / protocol["predecessor"]["path"])
    pixel.require(predecessor["conclusion"] == protocol["predecessor"]["required_conclusion"], "PREDECESSOR_CONCLUSION")
    cohort = pixel.load_json(HERE / protocol["cohort"]["path"])
    artifact_root = ROOT / protocol["source"]["artifact_root"]
    model_path = ROOT / protocol["model"]["path"]
    pixel.require(pixel.sha256(model_path) == protocol["model"]["sha256"], "MODEL_HASH")

    wanted = set(protocol["memory"]["reference_images"] + protocol["evaluation"]["positive_query_images"] + protocol["evaluation"]["negative_query_images"])
    image_rows = {key: row for key, row in cohort["images"].items() if key in wanted}
    pixel.require(set(image_rows) == wanted, "IMAGE_KEYS")
    images: dict[str, Image.Image] = {}
    image_receipts: dict[str, Any] = {}
    for key, row in image_rows.items():
        manifest_key = f"{row['scan_id']}/sequence.zip"
        archive_path = artifact_root / cohort["source_manifest"][manifest_key]["path"]
        with zipfile.ZipFile(archive_path) as archive:
            payload = archive.read(row["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            images[key] = opened.convert("RGB")
        image_receipts[key] = {
            "scan_id": row["scan_id"],
            "frame": int(row["frame"]),
            "image_sha256": hashlib.sha256(payload).hexdigest(),
            "bbox_xyxy": row["bbox_xyxy"],
        }

    from romatch.models.transformer import vit_large

    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to("cuda:0")
    descriptors: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for key, image in images.items():
            box = image_rows[key]["bbox_xyxy"]
            parts = []
            for scale in protocol["descriptor"]["crop_scales"]:
                features = model.forward_features(
                    _tensor(_crop(image, box, float(scale)), int(protocol["descriptor"]["input_size"])).to("cuda:0")
                )["x_norm_clstoken"][0].float()
                parts.append(torch.nn.functional.normalize(features, dim=0))
            descriptor = torch.nn.functional.normalize(torch.cat(parts), dim=0)
            descriptors[key] = descriptor.cpu().numpy()
    del model, weights
    torch.cuda.empty_cache()

    reference_keys = protocol["memory"]["reference_images"]
    prototype = np.mean(np.stack([descriptors[key] for key in reference_keys]), axis=0)
    prototype /= np.linalg.norm(prototype)
    scores = {key: float(np.dot(prototype, descriptors[key])) for key in wanted if key not in reference_keys}
    positive_keys = protocol["evaluation"]["positive_query_images"]
    negative_keys = protocol["evaluation"]["negative_query_images"]
    min_positive = min(scores[key] for key in positive_keys)
    max_negative = max(scores[key] for key in negative_keys)
    comparisons = [scores[p] > scores[n] for p in positive_keys for n in negative_keys]
    gate_met = all(comparisons)
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "CONSUMED_PV28_OBJECT_CONTEXT_MEMORY_POSTHOC_DEVELOPMENT",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "source": {"cohort_path": protocol["cohort"]["path"], "cohort_sha256": protocol["cohort"]["sha256"]},
        "conclusion": (
            "L10_3RSCAN_OBJECT_CONTEXT_MEMORY_POSTHOC_DEVELOPMENT_GATE_MET"
            if gate_met else "L10_3RSCAN_OBJECT_CONTEXT_MEMORY_POSTHOC_DEVELOPMENT_GATE_NOT_MET"
        ),
        "gate_met": gate_met,
        "scores": scores,
        "metrics": {
            "positive_queries": len(positive_keys),
            "negative_queries": len(negative_keys),
            "correct_pairwise_rankings": int(sum(comparisons)),
            "pairwise_rankings": len(comparisons),
            "minimum_positive_score": min_positive,
            "maximum_negative_score": max_negative,
            "separation_margin": float(min_positive - max_negative),
        },
        "descriptor_receipts": {
            key: {"sha256": hashlib.sha256(descriptors[key].tobytes()).hexdigest(), "dimensions": int(descriptors[key].size)}
            for key in descriptors
        },
        "image_receipts": image_receipts,
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
