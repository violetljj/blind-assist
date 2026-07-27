#!/usr/bin/env python3
"""Generate and verify the isolated QNN preprocessing candidate model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


SOURCE_WIDTH = 640
SOURCE_HEIGHT = 480
INPUT_SIZE = 320
RESIZED_WIDTH = 240
ROTATION_DEGREES = 90
MODEL_FILENAME = "rgba640x480_rot90_letterbox320.tflite"
CONTRACT_FILENAME = "contract.json"


class PreprocessModule(tf.Module):
    @tf.function(
        input_signature=[
            tf.TensorSpec(
                shape=[1, SOURCE_HEIGHT, SOURCE_WIDTH, 4],
                dtype=tf.uint8,
                name="rgba",
            )
        ]
    )
    def preprocess(self, rgba: tf.Tensor) -> dict[str, tf.Tensor]:
        rotated = tf.image.rot90(rgba, k=3)
        rgb = tf.cast(rotated[:, :, :, :3], tf.float32)
        resized = tf.raw_ops.ResizeNearestNeighbor(
            images=rgb,
            size=tf.constant([INPUT_SIZE, RESIZED_WIDTH], tf.int32),
            align_corners=False,
            half_pixel_centers=False,
        )
        padded = tf.pad(
            resized,
            paddings=[[0, 0], [0, 0], [40, 40], [0, 0]],
            mode="CONSTANT",
            constant_values=0,
        )
        normalized = tf.math.divide(padded, tf.constant(255.0, tf.float32))
        return {"normalized_rgb": normalized}


def synthetic_rgba() -> np.ndarray:
    y, x = np.indices((SOURCE_HEIGHT, SOURCE_WIDTH), dtype=np.int32)
    rgba = np.empty((1, SOURCE_HEIGHT, SOURCE_WIDTH, 4), dtype=np.uint8)
    rgba[0, :, :, 0] = (x * 31 + y * 7) & 0xFF
    rgba[0, :, :, 1] = (x * 13 + y * 29) & 0xFF
    rgba[0, :, :, 2] = (x * 19 + y * 17) & 0xFF
    rgba[0, :, :, 3] = 255
    return rgba


def reference_preprocess(rgba: np.ndarray) -> np.ndarray:
    output = np.zeros((1, INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    for target_y in range(INPUT_SIZE):
        source_x = target_y * 2
        for resized_x in range(RESIZED_WIDTH):
            source_y = SOURCE_HEIGHT - 1 - resized_x * 2
            output[0, target_y, 40 + resized_x, :] = (
                rgba[0, source_y, source_x, :3].astype(np.float32) / 255.0
            )
    return output


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest().upper()


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    module = PreprocessModule()
    concrete = module.preprocess.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete], module)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    model = converter.convert()

    model_path = output_dir / MODEL_FILENAME
    model_path.write_bytes(model)

    interpreter = tf.lite.Interpreter(model_content=model)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    rgba = synthetic_rgba()
    interpreter.set_tensor(input_detail["index"], rgba)
    interpreter.invoke()
    actual = interpreter.get_tensor(output_detail["index"])
    expected = reference_preprocess(rgba)
    difference = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    max_abs = float(difference.max(initial=0.0))
    mean_abs = float(difference.mean())
    if max_abs > 1e-7:
        raise RuntimeError(
            f"generated preprocessing graph exceeds tolerance: max_abs={max_abs}"
        )

    contract = {
        "schema": "blindassist_qnn_preprocess_candidate_contract_v1",
        "candidate_only": True,
        "source": {
            "shape": [1, SOURCE_HEIGHT, SOURCE_WIDTH, 4],
            "dtype": "UINT8",
            "rotation_degrees": ROTATION_DEGREES,
            "row_stride": SOURCE_WIDTH * 4,
            "pixel_stride": 4,
        },
        "output": {
            "shape": [1, INPUT_SIZE, INPUT_SIZE, 3],
            "dtype": "FLOAT32",
            "letterbox_left": 40,
            "letterbox_right": 40,
            "normalization": "value / 255.0",
        },
        "model": {
            "filename": MODEL_FILENAME,
            "sha256": sha256(model),
            "bytes": len(model),
        },
        "host_reference": {
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "acceptance_max_abs": 1e-7,
            "output_sha256": sha256(actual.tobytes()),
        },
    }
    (output_dir / CONTRACT_FILENAME).write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(contract, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts.local/experiments/qnn-preprocess-fusion-v1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args().output_dir.resolve())
