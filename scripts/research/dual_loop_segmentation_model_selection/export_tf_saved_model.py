#!/usr/bin/env python3
"""Export the SegFormer-B0 checkpoint through the native TensorFlow model.

The PyTorch checkpoint is the training authority. This exporter uses the
matching Hugging Face TensorFlow SegFormer implementation for the shared MiT
backbone and maps the trained four-class decoder weights explicitly. The
resulting SavedModel keeps the same raw RGB NCHW float32 boundary as the
onnx2tf path, so the common NHWC int8 converter can consume it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import tensorflow as tf
import torch
from transformers import SegformerConfig, TFSegformerForSemanticSegmentation


INPUT_SIZE = 256
CLASS_COUNT = 4
IMAGE_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGE_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state = payload.get("state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError("checkpoint does not contain a state_dict")
    if not all(isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()):
        raise ValueError("checkpoint state_dict contains a non-tensor entry")
    return state


def _decoder_arrays(state: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    mapping: dict[str, tuple[str, tuple[int, ...] | None]] = {
        "linear_0_kernel": ("core.decode_head.linear_projections.0.proj.weight", (1, 0)),
        "linear_0_bias": ("core.decode_head.linear_projections.0.proj.bias", None),
        "linear_1_kernel": ("core.decode_head.linear_projections.1.proj.weight", (1, 0)),
        "linear_1_bias": ("core.decode_head.linear_projections.1.proj.bias", None),
        "linear_2_kernel": ("core.decode_head.linear_projections.2.proj.weight", (1, 0)),
        "linear_2_bias": ("core.decode_head.linear_projections.2.proj.bias", None),
        "linear_3_kernel": ("core.decode_head.linear_projections.3.proj.weight", (1, 0)),
        "linear_3_bias": ("core.decode_head.linear_projections.3.proj.bias", None),
        "linear_fuse_kernel": ("core.decode_head.linear_fuse.weight", (2, 3, 1, 0)),
        "bn_gamma": ("core.decode_head.batch_norm.weight", None),
        "bn_beta": ("core.decode_head.batch_norm.bias", None),
        "bn_mean": ("core.decode_head.batch_norm.running_mean", None),
        "bn_var": ("core.decode_head.batch_norm.running_var", None),
        "classifier_kernel": ("core.decode_head.classifier.weight", (2, 3, 1, 0)),
        "classifier_bias": ("core.decode_head.classifier.bias", None),
    }
    arrays: dict[str, np.ndarray] = {}
    for name, (key, permutation) in mapping.items():
        if key not in state:
            raise ValueError(f"checkpoint is missing trained decoder tensor: {key}")
        value = state[key].detach().cpu().numpy().astype(np.float32, copy=False)
        if permutation is not None:
            value = np.transpose(value, permutation)
        arrays[name] = value
    if arrays["classifier_kernel"].shape != (1, 1, 256, CLASS_COUNT):
        raise ValueError(f"unexpected four-class classifier shape: {arrays['classifier_kernel'].shape}")
    return arrays


def _load_model(source_dir: Path, arrays: dict[str, np.ndarray]) -> TFSegformerForSemanticSegmentation:
    config = SegformerConfig.from_pretrained(
        str(source_dir),
        num_labels=CLASS_COUNT,
        local_files_only=True,
    )
    model = TFSegformerForSemanticSegmentation.from_pretrained(
        str(source_dir),
        from_pt=True,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    # Materialize all variables before assigning the trained decoder tensors.
    _ = model(pixel_values=tf.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=tf.float32), training=False)
    for index in range(4):
        projection = model.decode_head.mlps[index].proj
        projection.kernel.assign(arrays[f"linear_{index}_kernel"])
        projection.bias.assign(arrays[f"linear_{index}_bias"])
    model.decode_head.linear_fuse.kernel.assign(arrays["linear_fuse_kernel"])
    model.decode_head.batch_norm.gamma.assign(arrays["bn_gamma"])
    model.decode_head.batch_norm.beta.assign(arrays["bn_beta"])
    model.decode_head.batch_norm.moving_mean.assign(arrays["bn_mean"])
    model.decode_head.batch_norm.moving_variance.assign(arrays["bn_var"])
    model.decode_head.classifier.kernel.assign(arrays["classifier_kernel"])
    model.decode_head.classifier.bias.assign(arrays["classifier_bias"])
    return model


class _RawRgbNchwModule(tf.Module):
    def __init__(self, model: TFSegformerForSemanticSegmentation) -> None:
        super().__init__()
        self.model = model
        self.mean = tf.constant(IMAGE_MEAN, dtype=tf.float32)
        self.std = tf.constant(IMAGE_STD, dtype=tf.float32)

    @tf.function(input_signature=[tf.TensorSpec((1, 3, INPUT_SIZE, INPUT_SIZE), tf.float32, name="rgb")])
    def serve(self, rgb: tf.Tensor) -> dict[str, tf.Tensor]:
        normalized = (rgb / 255.0 - self.mean) / self.std
        low_resolution = self.model(pixel_values=normalized, training=False).logits
        low_nhwc = tf.transpose(low_resolution, (0, 2, 3, 1))
        high_nhwc = tf.raw_ops.ResizeBilinear(
            images=low_nhwc,
            size=tf.constant([INPUT_SIZE, INPUT_SIZE], dtype=tf.int32),
            align_corners=False,
            half_pixel_centers=True,
        )
        return {"logits": tf.transpose(high_nhwc, (0, 3, 1, 2))}


def _verify_saved_model(path: Path) -> dict[str, Any]:
    loaded = tf.saved_model.load(str(path))
    signature = loaded.signatures.get("serving_default")
    if signature is None:
        raise ValueError("SavedModel lacks serving_default")
    inputs = signature.structured_input_signature[1]
    outputs = signature.structured_outputs
    if set(inputs) != {"rgb"} or set(outputs) != {"logits"}:
        raise ValueError(f"SavedModel names are not frozen: inputs={sorted(inputs)} outputs={sorted(outputs)}")
    input_spec = inputs["rgb"]
    output_spec = outputs["logits"]
    if tuple(input_spec.shape) != (1, 3, INPUT_SIZE, INPUT_SIZE) or input_spec.dtype != tf.float32:
        raise ValueError(f"SavedModel input contract mismatch: {input_spec}")
    if tuple(output_spec.shape) != (1, CLASS_COUNT, INPUT_SIZE, INPUT_SIZE) or output_spec.dtype != tf.float32:
        raise ValueError(f"SavedModel output contract mismatch: {output_spec}")
    return {
        "shape": [int(value) for value in output_spec.shape],
        "dtype": output_spec.dtype.name,
        "input_shape": [int(value) for value in input_spec.shape],
        "input_dtype": input_spec.dtype.name,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint).resolve()
    source_dir = Path(args.source_dir).resolve()
    output = Path(args.output).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not (source_dir / "config.json").is_file() or not (source_dir / "pytorch_model.bin").is_file():
        raise FileNotFoundError(f"SegFormer source directory is incomplete: {source_dir}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite SavedModel output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = _checkpoint_state(checkpoint)
    arrays = _decoder_arrays(state)
    model = _load_model(source_dir, arrays)
    module = _RawRgbNchwModule(model)
    concrete = module.serve.get_concrete_function()

    # Track the variables directly instead of tracking the legacy tf_keras
    # child model.  This preserves the concrete graph's numerical path while
    # avoiding the serializer's DT_STRING transpose bug.  Constant folding via
    # convert_variables_to_constants_v2 is intentionally not used: it changes
    # the SegFormer backbone output by ~1e-2 after SavedModel reload.
    carrier = tf.Module()
    for index, variable in enumerate(model.variables):
        setattr(carrier, f"variable_{index:03d}", variable)
    tf.saved_model.save(
        carrier,
        str(output),
        signatures={"serving_default": concrete},
    )
    contract = _verify_saved_model(output)
    receipt = {
        "schema_version": "blindassist.dual_loop_segmentation_model_selection_r1.tf_saved_model_receipt.v1",
        "protocol_id": "DUAL_LOOP_SEGMENTATION_MODEL_SELECTION_R1",
        "status": "TF_SAVED_MODEL_EXPORTED",
        "model_id": "SegFormer-B0",
        "implementation_identity": "segformer_b0",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "source_dir": str(source_dir),
        "source_config_sha256": sha256_file(source_dir / "config.json"),
        "source_checkpoint_sha256": sha256_file(source_dir / "pytorch_model.bin"),
        "architecture_source": "transformers.TFSegformerForSemanticSegmentation + native TF decoder",
        "decoder_mapping": "trained PyTorch four-class decoder -> TF MLP/Conv2D/BatchNorm kernels",
        "saved_model": str(output),
        "saved_model_pb_sha256": sha256_file(output / "saved_model.pb"),
        "input_contract": {"shape": contract["input_shape"], "dtype": contract["input_dtype"], "range": "0..255", "layout": "NCHW"},
        "output_contract": {"shape": contract["shape"], "dtype": contract["dtype"], "layout": "NCHW", "class_count": CLASS_COUNT},
        "fresh_holdout_consumed": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    receipt_path = output.with_suffix(".receipt.json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    receipt_path.with_suffix(".sha256.json").write_text(
        json.dumps({"sha256": sha256_file(receipt_path)}, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"status": result["status"], "saved_model": result["saved_model"]}, ensure_ascii=False))
