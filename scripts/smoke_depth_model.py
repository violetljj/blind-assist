from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from inspect_tflite import create_interpreter, resolve_model_path, tensor_shape


DEFAULT_MODEL = ".downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite"
DEFAULT_DATASET = "test-artifacts.local/datasets/blindassist-evalset-20260527-impl"


def decode_and_resize(image_path: Path, width: int, height: int) -> np.ndarray:
    import tensorflow as tf

    image = tf.io.decode_image(image_path.read_bytes(), channels=3, expand_animations=False)
    image = tf.image.resize(image, (height, width), method="bilinear")
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()[np.newaxis, ...]


def output_size(shape: list[int]) -> tuple[int, int]:
    if len(shape) == 2:
        return shape[0], shape[1]
    if len(shape) == 3:
        return shape[1], shape[2]
    if len(shape) == 4 and shape[-1] == 1:
        return shape[1], shape[2]
    if len(shape) == 4 and shape[1] == 1:
        return shape[2], shape[3]
    raise AssertionError(f"Unsupported depth output shape: {shape}")


def manifest_images(dataset_root: Path, limit: int) -> list[Path]:
    manifest = dataset_root / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"BlindAssist manifest not found: {manifest}")
    images = []
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("image_path") or row.get("image") or row.get("file_name") or row.get("relative_path")
        if not value:
            continue
        candidate = dataset_root / value
        if not candidate.is_file():
            candidate = dataset_root / "images" / "test" / Path(value).name
        if candidate.is_file():
            images.append(candidate)
        if len(images) >= limit:
            break
    if not images:
        raise AssertionError(f"No evalset images found under {dataset_root / 'images' / 'test'}")
    return images


def summarize(values: np.ndarray) -> dict[str, Any]:
    flat = values.astype(np.float32).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return {"finite": 0, "min": None, "max": None, "mean": None, "all_zero": True}
    return {
        "finite": int(finite.size),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "mean": float(np.mean(finite)),
        "all_zero": bool(np.allclose(finite, 0.0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a depth model on BlindAssist EvalSet images.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET)
    parser.add_argument("--image-limit", type=int, default=20)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = resolve_model_path(project_root, args.model)
    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_absolute():
        dataset_root = project_root / dataset_root

    interpreter, backend = create_interpreter(model_path)
    interpreter.allocate_tensors()
    input_tensor = interpreter.get_input_details()[0]
    output_tensor = interpreter.get_output_details()[0]
    input_shape = tensor_shape(input_tensor)
    output_shape = tensor_shape(output_tensor)
    if len(input_shape) != 4 or input_shape[0] != 1 or input_shape[-1] != 3:
        raise AssertionError(f"Depth input must be NHWC [1,H,W,3], got {input_shape}")
    output_size(output_shape)

    images = manifest_images(dataset_root, args.image_limit)
    output_index = output_tensor["index"]
    rows = []
    for image_path in images:
        input_array = decode_and_resize(image_path, input_shape[2], input_shape[1])
        interpreter.set_tensor(input_tensor["index"], input_array)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        stats = summarize(output)
        if stats["finite"] <= 0 or stats["all_zero"]:
            raise AssertionError(f"Invalid depth output for {image_path.name}: {stats}")
        rows.append({"image": image_path.name, **stats})

    payload = {
        "model": str(model_path.resolve()),
        "backend": backend,
        "dataset_root": str(dataset_root.resolve()),
        "image_count": len(rows),
        "input_shape": input_shape,
        "output_shape": output_shape,
        "images": rows,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.json_output:
        output_path = Path(args.json_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
