from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "app/src/main/assets/yolo11n_fp16_320.tflite"
DEFAULT_INPUT_SHAPE = (1, 320, 320, 3)
DEFAULT_OUTPUT_SHAPE = (1, 84, 2100)
DEFAULT_DTYPE = "float32"


def configure_local_caches(project_root: Path) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)


def parse_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid shape: {value}") from error
    if not shape:
        raise argparse.ArgumentTypeError("shape must contain at least one dimension")
    return shape


def dtype_name(dtype: Any) -> str:
    return getattr(dtype, "__name__", str(dtype).replace("<class '", "").replace("'>", ""))


def create_interpreter(model_path: Path) -> tuple[Any, str]:
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=str(model_path)), "ai-edge-litert"
    except Exception:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(model_path)), "tensorflow"
        except Exception as tensorflow_error:
            raise RuntimeError(
                "Could not import ai-edge-litert or tensorflow to inspect the TFLite model."
            ) from tensorflow_error


def tensor_shape(tensor: dict[str, Any]) -> list[int]:
    shape = tensor.get("shape")
    return [int(part) for part in shape.tolist()] if hasattr(shape, "tolist") else list(shape)


def tensor_summary(tensor: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": str(tensor.get("name", "")),
        "shape": tensor_shape(tensor),
        "dtype": dtype_name(tensor.get("dtype")),
        "index": int(tensor.get("index", -1)),
    }


def output_layout(shape: list[int]) -> str:
    if len(shape) != 3:
        return "unknown"

    dim1 = shape[1]
    dim2 = shape[2]
    if dim1 >= 5 and dim1 <= dim2:
        return "yolo_raw_channels_first"
    if dim2 >= 5 and dim2 < dim1:
        return "yolo_raw_channels_last"
    if dim2 >= 6 and dim1 <= 1000:
        return "end2end_detections"
    return "unknown"


def assert_tensor(
    kind: str,
    tensor: dict[str, Any],
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> None:
    actual_shape = tuple(tensor_summary(tensor)["shape"])
    actual_dtype = tensor_summary(tensor)["dtype"]
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{kind} shape mismatch: expected {list(expected_shape)}, got {list(actual_shape)}"
        )
    if actual_dtype != expected_dtype:
        raise AssertionError(
            f"{kind} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
        )


def resolve_model_path(project_root: Path, value: str) -> Path:
    model_path = Path(value)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    return model_path


def inspect_model(
    model_path: Path,
    expected_input_shape: tuple[int, ...],
    expected_output_shape: tuple[int, ...],
    expected_input_dtype: str,
    expected_output_dtype: str,
    assert_expected: bool,
) -> dict[str, Any]:
    interpreter, backend = create_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise AssertionError("model has no input tensors")
    if not output_details:
        raise AssertionError("model has no output tensors")

    input_tensor = input_details[0]
    output_tensor = output_details[0]
    if assert_expected:
        assert_tensor("input", input_tensor, expected_input_shape, expected_input_dtype)
        assert_tensor("output", output_tensor, expected_output_shape, expected_output_dtype)

    inputs = [tensor_summary(tensor) for tensor in input_details]
    outputs = [tensor_summary(tensor) for tensor in output_details]
    return {
        "model": str(model_path.resolve()),
        "size_bytes": model_path.stat().st_size,
        "backend": backend,
        "inputs": inputs,
        "outputs": outputs,
        "primary_output_layout": output_layout(outputs[0]["shape"]),
        "assertions": "passed" if assert_expected else "not_requested",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and optionally assert TFLite input/output tensor details."
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="TFLite model paths. Defaults to the committed BlindAssist YOLO11n asset.",
    )
    parser.add_argument("--expected-input-shape", type=parse_shape, default=DEFAULT_INPUT_SHAPE)
    parser.add_argument("--expected-output-shape", type=parse_shape, default=DEFAULT_OUTPUT_SHAPE)
    parser.add_argument("--expected-input-dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--expected-output-dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--allow-any-shape",
        action="store_true",
        help="Inspect candidate models without asserting the default BlindAssist tensor contract.",
    )
    parser.add_argument(
        "--json-output",
        help="Optional JSON file for machine-readable inspection results.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)

    model_values = args.models or [DEFAULT_MODEL]
    results = []
    for value in model_values:
        model_path = resolve_model_path(project_root, value)
        result = inspect_model(
            model_path=model_path,
            expected_input_shape=args.expected_input_shape,
            expected_output_shape=args.expected_output_shape,
            expected_input_dtype=args.expected_input_dtype,
            expected_output_dtype=args.expected_output_dtype,
            assert_expected=not args.allow_any_shape,
        )
        results.append(result)

        print(f"backend={result['backend']}")
        print(f"model={result['model']}")
        print(f"size_bytes={result['size_bytes']}")
        for tensor in result["inputs"]:
            print(f"input name={tensor['name']} shape={tensor['shape']} dtype={tensor['dtype']}")
        for tensor in result["outputs"]:
            print(f"output name={tensor['name']} shape={tensor['shape']} dtype={tensor['dtype']}")
        print(f"primary_output_layout={result['primary_output_layout']}")
        print(f"assertions={result['assertions']}")

    if args.json_output:
        output_path = Path(args.json_output)
        if not output_path.is_absolute():
            output_path = project_root / output_path
        write_json(output_path, results)


if __name__ == "__main__":
    main()
