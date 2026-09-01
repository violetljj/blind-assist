#!/usr/bin/env python3
"""Materialize frozen DINOv2 descriptors for the exact instance-head manifest."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from typing import Any
import zipfile

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import l10_3rscan_reference_pixel_field as pixel  # noqa: E402


PROTOCOL_SCHEMA = "blindassist-l10-3rscan-instance-head-descriptor-materialize-protocol-v1"
RESULT_SCHEMA = "blindassist-l10-3rscan-instance-head-descriptor-materialize-result-v1"


def _crop(image: Image.Image, box: list[float], expansion: float) -> tuple[Image.Image, list[int]]:
    x0, y0, x1, y1 = (float(value) for value in box)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    width, height = (x1 - x0) * expansion, (y1 - y0) * expansion
    left = max(0, int(np.floor(cx - width / 2.0)))
    top = max(0, int(np.floor(cy - height / 2.0)))
    right = min(image.width, int(np.ceil(cx + width / 2.0)))
    bottom = min(image.height, int(np.ceil(cy + height / 2.0)))
    pixel.require(right > left and bottom > top, "EMPTY_CROP")
    return image.crop((left, top, right, bottom)), [left, top, right, bottom]


def _tensor(image: Image.Image, size: int, torch: Any) -> tuple[Any, str]:
    resized = image.resize((size, size), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.uint8).copy()
    crop_sha256 = hashlib.sha256(array.tobytes()).hexdigest()
    value = torch.from_numpy(array).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=value.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=value.dtype).view(3, 1, 1)
    return (value - mean) / std, crop_sha256


def _atomic_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run(protocol_path: Path, descriptor_path: Path, output_path: Path) -> None:
    protocol = pixel.load_json(protocol_path)
    pixel.require(protocol.get("schema") == PROTOCOL_SCHEMA, "PROTOCOL_SCHEMA")
    pixel.require(pixel.sha256(Path(__file__)) == protocol["implementation"]["sha256"], "IMPLEMENTATION_HASH")
    manifest_row = protocol["manifest"]
    manifest_path = HERE / manifest_row["path"]
    pixel.require(pixel.sha256(manifest_path) == manifest_row["sha256"], "MANIFEST_HASH")
    manifest = pixel.load_json(manifest_path)
    pixel.require(manifest["schema"] == manifest_row["required_schema"], "MANIFEST_SCHEMA")
    pixel.require(manifest["opened_rgb_members"] == 0 and manifest["feature_calls"] == 0, "MANIFEST_ALREADY_OPENED")

    artifact_root = ROOT / protocol["source"]["artifact_root"]
    model_path = ROOT / protocol["model"]["path"]
    pixel.require(pixel.sha256(model_path) == protocol["model"]["sha256"], "MODEL_HASH")
    for scan_id, receipt in manifest["zip_receipts"].items():
        path = artifact_root / receipt["path"]
        pixel.require(path.stat().st_size == int(receipt["bytes"]), f"ZIP_BYTES:{scan_id}")
        pixel.require(pixel.sha256(path) == receipt["sha256"], f"ZIP_HASH:{scan_id}")

    import torch
    from romatch.models.transformer import vit_large

    torch.manual_seed(int(protocol["runtime"]["seed"]))
    device = str(protocol["runtime"]["device"])
    pixel.require(device.startswith("cuda") and torch.cuda.is_available(), "CUDA_REQUIRED")
    weights = torch.load(model_path, map_location="cpu", weights_only=True)
    model = vit_large(
        img_size=int(protocol["descriptor"]["input_size"]),
        patch_size=14,
        init_values=1.0,
        ffn_layer="mlp",
        block_chunks=0,
    ).eval()
    model.load_state_dict(weights)
    model = model.to(device)

    descriptors: list[np.ndarray] = []
    receipts: list[dict[str, Any]] = []
    batch_tensors: list[Any] = []
    batch_rows: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.inference_mode():
            values = model.forward_features(batch)["x_norm_clstoken"].float()
            values = torch.nn.functional.normalize(values, dim=1)
        descriptors.extend(values.cpu().numpy().astype(np.float32))
        batch_tensors.clear()
        batch_rows.clear()

    for sample in manifest["samples"]:
        zip_path = artifact_root / manifest["zip_receipts"][sample["scan_id"]]["path"]
        with zipfile.ZipFile(zip_path) as archive:
            payload = archive.read(sample["zip_member"])
        with Image.open(io.BytesIO(payload)) as opened:
            image = opened.convert("RGB")
        pixel.require(list(image.size) == sample["color_size"], f"RGB_SIZE:{sample['sample_id']}:{image.size}")
        crop, crop_box = _crop(image, sample["bbox_xyxy"], float(manifest["crop"]["expansion"]))
        tensor, crop_sha256 = _tensor(crop, int(protocol["descriptor"]["input_size"]), torch)
        batch_tensors.append(tensor)
        batch_rows.append(sample)
        receipts.append(
            {
                "sample_id": sample["sample_id"],
                "rgb_sha256": hashlib.sha256(payload).hexdigest(),
                "crop_box_xyxy": crop_box,
                "crop_sha256": crop_sha256,
            }
        )
        if len(batch_tensors) >= int(protocol["runtime"]["batch_size"]):
            flush()
    flush()
    descriptor_array = np.stack(descriptors).astype(np.float32)
    pixel.require(descriptor_array.shape == (len(manifest["samples"]), int(protocol["descriptor"]["dimension"])), "DESCRIPTOR_SHAPE")
    norms = np.linalg.norm(descriptor_array, axis=1)
    pixel.require(np.isfinite(descriptor_array).all() and np.allclose(norms, 1.0, atol=1e-5), "DESCRIPTOR_VALIDITY")
    _atomic_npz(
        descriptor_path,
        sample_ids=np.asarray([row["sample_id"] for row in manifest["samples"]]),
        descriptors=descriptor_array,
    )
    result = {
        "schema": RESULT_SCHEMA,
        "authority": "FROZEN_BACKBONE_TRAINING_AND_VALIDATION_FEATURE_MATERIALIZATION",
        "protocol_path": protocol_path.name,
        "protocol_sha256": pixel.sha256(protocol_path),
        "implementation": {"path": Path(__file__).name, "sha256": pixel.sha256(Path(__file__))},
        "manifest": manifest_row,
        "model": protocol["model"],
        "descriptor": protocol["descriptor"],
        "artifact": {
            "path": descriptor_path.resolve().relative_to(artifact_root.resolve()).as_posix(),
            "bytes": descriptor_path.stat().st_size,
            "sha256": pixel.sha256(descriptor_path),
            "shape": list(descriptor_array.shape),
        },
        "receipts": receipts,
        "opened_rgb_members": len(receipts),
        "feature_calls": len(receipts),
        "backbone_training_steps": 0,
        "score_comparisons": 0,
        "runtime": {
            "device": torch.cuda.get_device_name(torch.device(device)),
            "batch_size": int(protocol["runtime"]["batch_size"]),
            "seed": int(protocol["runtime"]["seed"]),
        },
        "next_action": protocol["next_action"],
        "conclusion": "L10_3RSCAN_INSTANCE_HEAD_DESCRIPTORS_MATERIALIZED",
        "claim_boundary": protocol["claim_boundary"],
    }
    pixel.atomic_write_json(output_path, result)
    del model, weights
    torch.cuda.empty_cache()
    print(json.dumps(result, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--descriptors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol.resolve(), args.descriptors.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
