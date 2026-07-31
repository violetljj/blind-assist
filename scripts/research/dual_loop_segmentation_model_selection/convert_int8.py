#!/usr/bin/env python3
"""Convert a verified onnx2tf SavedModel to the frozen R1 INT8 contract.

The current onnx2tf conversion preserves the official graph's NCHW input.
This script adds only a TensorFlow boundary transpose so both candidates expose
the shared utility interface: raw ``uint8-like`` RGB values as float32 during
calibration, then NHWC int8 input/output in the final TFLite artifact.
Representative data is image-only and comes exclusively from the frozen
canonical training split.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import tensorflow as tf
from PIL import Image

try:
    from .models import sha256_file
    from .train import load_shared, resolve, write_json
except ImportError:  # pragma: no cover - direct script execution
    from models import sha256_file
    from train import load_shared, resolve, write_json


INPUT_SIZE = 256
CLASS_COUNT = 4
EXPECTED_CLASS_ORDER = ["walkable", "boundary_step_curb", "obstacle", "unknown_nonwalkable"]


class _NhwcBoundary(tf.Module):
    """Expose an NHWC boundary around the already converted NCHW graph."""

    def __init__(self, source: Any) -> None:
        super().__init__()
        self.source = source
        self.source_fn = source.signatures.get("serving_default")
        if self.source_fn is None:
            raise ValueError("source SavedModel lacks serving_default")
        structured = self.source_fn.structured_input_signature
        inputs = structured[1]
        if set(inputs) != {"rgb"}:
            raise ValueError(f"source SavedModel input names are not frozen: {sorted(inputs)}")
        spec = inputs["rgb"]
        if tuple(spec.shape) != (1, 3, INPUT_SIZE, INPUT_SIZE) or spec.dtype != tf.float32:
            raise ValueError(f"source SavedModel input contract mismatch: {spec}")

    @tf.function(input_signature=[tf.TensorSpec((1, INPUT_SIZE, INPUT_SIZE, 3), tf.float32, name="rgb")])
    def serve(self, rgb: tf.Tensor) -> dict[str, tf.Tensor]:
        source_rgb = tf.transpose(rgb, (0, 3, 1, 2))
        outputs = self.source_fn(rgb=source_rgb)
        if len(outputs) != 1:
            raise ValueError(f"source SavedModel must have one output, got {list(outputs)}")
        logits = next(iter(outputs.values()))
        return {"logits": logits}


def _load_train_records(config_path: Path, dataset_root: Path) -> tuple[dict[str, Any], Path, list[Any]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("protocol_id") != "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1":
        raise ValueError("config is not bound to R1")
    if config.get("class_order") != EXPECTED_CLASS_ORDER:
        raise ValueError("config class order differs from R1")
    manifest = dataset_root / str(config["training_manifest"])
    if not manifest.is_file() or sha256_file(manifest) != str(config["dataset_manifest_sha256"]):
        raise ValueError("training manifest is missing or SHA256-mismatched")
    shared = load_shared()
    records = shared.load_records(manifest)
    train_records = [record for record in records if record.split == "train"]
    if len(train_records) != 400 or any(record.semantic_mask_path is None for record in train_records):
        raise ValueError("R1 representative data requires the canonical 400-row train split")
    return config, manifest, train_records


def _representative_dataset(records: Sequence[Any]) -> Iterable[list[np.ndarray]]:
    for record in records:
        with Image.open(record.image_path) as image:
            rgb = np.asarray(
                image.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR),
                dtype=np.float32,
            )
        if rgb.shape != (INPUT_SIZE, INPUT_SIZE, 3) or not np.isfinite(rgb).all():
            raise ValueError(f"invalid representative RGB tensor for {record.sample_id}: {rgb.shape}")
        yield [rgb[None, ...]]


def _tensor_contract(interpreter: Any, details: dict[str, Any], *, name: str) -> dict[str, Any]:
    shape = [int(value) for value in details["shape"]]
    dtype = np.dtype(details["dtype"]).name
    quantization = details.get("quantization", (0.0, 0))
    scale = float(quantization[0])
    zero_point = int(quantization[1])
    if shape != [1, INPUT_SIZE, INPUT_SIZE, 3] and name == "input":
        raise ValueError(f"TFLite input shape mismatch: {shape}")
    if shape != [1, INPUT_SIZE, INPUT_SIZE, CLASS_COUNT] and name == "output":
        raise ValueError(f"TFLite output shape mismatch: {shape}")
    if dtype != "int8" or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"TFLite {name} quantization mismatch: dtype={dtype}, scale={scale}")
    if name == "input" and not (-128 <= zero_point <= 127):
        raise ValueError(f"TFLite input zero point is outside int8: {zero_point}")
    return {
        "index": int(details["index"]),
        "shape": shape,
        "dtype": dtype,
        "quantization": {"scale": scale, "zero_point": zero_point},
        "name": str(details.get("name", "")),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    config_path = resolve(repo_root, args.config)
    dataset_root = resolve(repo_root, args.dataset_root)
    saved_model = resolve(repo_root, args.saved_model)
    output = resolve(repo_root, args.output)
    wrapper_dir = resolve(repo_root, args.wrapper_dir) if args.wrapper_dir else output.with_suffix(".nhwc_saved_model")
    if not saved_model.is_dir():
        raise FileNotFoundError(saved_model)
    if output.exists() or wrapper_dir.exists():
        raise FileExistsError(f"refusing to overwrite conversion output: {output} / {wrapper_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    config, manifest, train_records = _load_train_records(config_path, dataset_root)

    source = tf.saved_model.load(str(saved_model))
    boundary = _NhwcBoundary(source)
    tf.saved_model.save(boundary, str(wrapper_dir), signatures={"serving_default": boundary.serve})

    converter = tf.lite.TFLiteConverter.from_saved_model(str(wrapper_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: _representative_dataset(train_records)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    model_bytes = converter.convert()
    output.write_bytes(model_bytes)

    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:  # pragma: no cover - fallback for older runtimes
        from tensorflow.lite.python.interpreter import Interpreter
    interpreter = Interpreter(model_path=str(output))
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"R1 TFLite contract requires one input/output: {len(inputs)}/{len(outputs)}")
    input_contract = _tensor_contract(interpreter, inputs[0], name="input")
    output_contract = _tensor_contract(interpreter, outputs[0], name="output")
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.int8_receipt.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "status": "INT8_TFLITE_EXPORTED",
        "model_id": config["model_id"],
        "implementation_identity": config["implementation_identity"],
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "dataset_root": str(dataset_root.resolve()),
        "training_manifest": str(manifest.resolve()),
        "training_manifest_sha256": sha256_file(manifest),
        "representative_data": {
            "split": "train",
            "record_count": len(train_records),
            "mask_pixels_read": False,
            "image_only": True,
            "input_range": "0..255",
        },
        "source_saved_model": str(saved_model.resolve()),
        "source_saved_model_pb_sha256": sha256_file(saved_model / "saved_model.pb"),
        "nhwc_boundary_saved_model": str(wrapper_dir.resolve()),
        "tflite": str(output.resolve()),
        "tflite_sha256": sha256_file(output),
        "input_contract": input_contract,
        "output_contract": output_contract,
        "class_order": EXPECTED_CLASS_ORDER,
        "fresh_holdout_consumed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = output.with_suffix(".receipt.json")
    write_json(receipt_path, receipt)
    write_json(receipt_path.with_suffix(".sha256.json"), {"sha256": sha256_file(receipt_path)})
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--saved-model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wrapper-dir")
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({
        "status": result["status"],
        "model_id": result["model_id"],
        "tflite": result["tflite"],
        "tflite_sha256": result["tflite_sha256"],
    }, ensure_ascii=False))
