from __future__ import annotations

import argparse
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
    except Exception as litert_error:
        try:
            import tensorflow as tf

            return tf.lite.Interpreter(model_path=str(model_path)), "tensorflow"
        except Exception as tensorflow_error:
            raise RuntimeError(
                "Could not import ai-edge-litert or tensorflow to inspect the TFLite model."
            ) from tensorflow_error


def assert_tensor(
    kind: str,
    details: list[dict[str, Any]],
    expected_shape: tuple[int, ...],
    expected_dtype: str,
) -> dict[str, Any]:
    if not details:
        raise AssertionError(f"model has no {kind} tensors")

    tensor = details[0]
    actual_shape = tuple(int(part) for part in tensor["shape"].tolist())
    actual_dtype = dtype_name(tensor["dtype"])
    if actual_shape != expected_shape:
        raise AssertionError(
            f"{kind} shape mismatch: expected {list(expected_shape)}, got {list(actual_shape)}"
        )
    if actual_dtype != expected_dtype:
        raise AssertionError(
            f"{kind} dtype mismatch: expected {expected_dtype}, got {actual_dtype}"
        )
    return tensor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print and assert TFLite input/output tensor details."
    )
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--expected-input-shape", type=parse_shape, default=DEFAULT_INPUT_SHAPE)
    parser.add_argument("--expected-output-shape", type=parse_shape, default=DEFAULT_OUTPUT_SHAPE)
    parser.add_argument("--expected-input-dtype", default=DEFAULT_DTYPE)
    parser.add_argument("--expected-output-dtype", default=DEFAULT_DTYPE)
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = project_root / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")

    interpreter, backend = create_interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor = assert_tensor(
        "input",
        input_details,
        expected_shape=args.expected_input_shape,
        expected_dtype=args.expected_input_dtype,
    )
    output_tensor = assert_tensor(
        "output",
        output_details,
        expected_shape=args.expected_output_shape,
        expected_dtype=args.expected_output_dtype,
    )

    print(f"backend={backend}")
    print(f"model={model_path.resolve()}")
    print(
        f"input name={input_tensor['name']} "
        f"shape={input_tensor['shape'].tolist()} dtype={dtype_name(input_tensor['dtype'])}"
    )
    print(
        f"output name={output_tensor['name']} "
        f"shape={output_tensor['shape'].tolist()} dtype={dtype_name(output_tensor['dtype'])}"
    )
    print("assertions=passed")


if __name__ == "__main__":
    main()
