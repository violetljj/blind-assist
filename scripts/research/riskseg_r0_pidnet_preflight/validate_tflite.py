from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    CLASS_ORDER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    sha256_file,
)


def _quantize(values: np.ndarray, scale: float, zero_point: int, dtype: np.dtype) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"invalid quantization scale: {scale}")
    info = np.iinfo(dtype)
    quantized = np.rint(values / scale + zero_point)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def _dequantize(values: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    return (values.astype(np.float32) - np.float32(zero_point)) * np.float32(scale)


def _natural_input(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        rgb = image.convert("RGB").resize(
            (INPUT_WIDTH, INPUT_HEIGHT),
            resample=Image.Resampling.BILINEAR,
        )
        array = np.asarray(rgb, dtype=np.float32) / np.float32(255.0)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    return ((array - mean) / std)[None, ...]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--natural-image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    from ai_edge_litert.interpreter import Interpreter

    model_path = args.model.resolve()
    natural_image = args.natural_image.resolve()
    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError(f"expected one input/output, got {len(inputs)}/{len(outputs)}")
    input_detail = inputs[0]
    output_detail = outputs[0]
    expected_input_shape = [1, INPUT_HEIGHT, INPUT_WIDTH, 3]
    expected_output_shape = [1, INPUT_HEIGHT, INPUT_WIDTH, 4]
    if input_detail["shape"].tolist() != expected_input_shape:
        raise ValueError(f"unexpected input shape: {input_detail['shape'].tolist()}")
    if output_detail["shape"].tolist() != expected_output_shape:
        raise ValueError(f"unexpected output shape: {output_detail['shape'].tolist()}")
    if input_detail["dtype"] != np.int8 or output_detail["dtype"] != np.int8:
        raise ValueError(
            f"full INT8 surface required, got {input_detail['dtype']}/{output_detail['dtype']}"
        )
    input_scale, input_zero = input_detail["quantization"]
    output_scale, output_zero = output_detail["quantization"]

    tensor_details = interpreter.get_tensor_details()
    float_activation_tensors = [
        detail["name"]
        for detail in tensor_details
        if detail["dtype"] in (np.float16, np.float32, np.float64)
        and int(np.prod(detail["shape"], dtype=np.int64)) > 0
    ]
    if float_activation_tensors:
        raise ValueError(
            "full-integer model contains float tensors: "
            + ", ".join(float_activation_tensors[:20])
        )

    canaries = {
        "synthetic_zero_normalized": np.zeros(
            expected_input_shape,
            dtype=np.float32,
        ),
        "train_rgb_non_eval": _natural_input(natural_image),
    }
    canary_receipts: dict[str, dict] = {}
    for name, values in canaries.items():
        quantized_input = _quantize(
            values,
            float(input_scale),
            int(input_zero),
            np.dtype(np.int8),
        )
        interpreter.set_tensor(input_detail["index"], quantized_input)
        interpreter.invoke()
        quantized_output = interpreter.get_tensor(output_detail["index"])
        dequantized = _dequantize(
            quantized_output,
            float(output_scale),
            int(output_zero),
        )
        if not np.isfinite(dequantized).all():
            raise ValueError(f"{name} output contains non-finite values")
        argmax = np.argmax(dequantized, axis=-1)
        unique = sorted(int(value) for value in np.unique(argmax))
        if any(value < 0 or value >= len(CLASS_ORDER) for value in unique):
            raise ValueError(f"{name} argmax class outside 0..3: {unique}")
        canary_receipts[name] = {
            "input_float_range": [float(values.min()), float(values.max())],
            "input_int8_range": [
                int(quantized_input.min()),
                int(quantized_input.max()),
            ],
            "output_int8_range": [
                int(quantized_output.min()),
                int(quantized_output.max()),
            ],
            "output_float_range": [
                float(dequantized.min()),
                float(dequantized.max()),
            ],
            "finite": True,
            "argmax_unique_classes": unique,
        }

    operations = interpreter._get_ops_details()  # noqa: SLF001
    receipt = {
        "schema_version": "blindassist.riskseg_r0.pidnet_tflite_validation.v1",
        "protocol_id": "RISKSEG_R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "TFLITE_FULL_INT8_CANARIES_PASS",
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "input": {
            "name": input_detail["name"],
            "shape": expected_input_shape,
            "dtype": str(input_detail["dtype"]),
            "quantization_scale": float(input_scale),
            "quantization_zero_point": int(input_zero),
            "layout": "NHWC",
            "color_order": "RGB",
            "normalization": "(rgb/255 - ImageNet mean) / ImageNet std",
        },
        "output": {
            "name": output_detail["name"],
            "shape": expected_output_shape,
            "dtype": str(output_detail["dtype"]),
            "quantization_scale": float(output_scale),
            "quantization_zero_point": int(output_zero),
            "layout": "NHWC",
            "class_order": list(CLASS_ORDER),
        },
        "tensor_count": len(tensor_details),
        "float_tensor_count": len(float_activation_tensors),
        "operation_count": len(operations),
        "operation_types": sorted({str(operation["op_name"]) for operation in operations}),
        "natural_image_path": str(natural_image),
        "natural_image_sha256": sha256_file(natural_image),
        "canaries": canary_receipts,
    }
    args.output.resolve().write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

