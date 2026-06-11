from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inspect_tflite import create_interpreter, dtype_name, resolve_model_path, tensor_summary


DEFAULT_MODEL = ".downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite"


def depth_layout(shape: list[int]) -> str:
    if len(shape) == 2:
        return "depth_hw"
    if len(shape) == 3:
        return "depth_bhw"
    if len(shape) == 4 and shape[-1] == 1:
        return "depth_bhwc"
    if len(shape) == 4 and shape[1] == 1:
        return "depth_bchw"
    return "unknown"


def inspect_depth_model(model_path: Path) -> dict[str, Any]:
    interpreter, backend = create_interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    if not input_details:
        raise AssertionError("depth model has no input tensors")
    if not output_details:
        raise AssertionError("depth model has no output tensors")

    primary_input = tensor_summary(input_details[0])
    primary_output = tensor_summary(output_details[0])
    if primary_input["dtype"] != "float32":
        raise AssertionError(f"depth input dtype must be float32, got {primary_input['dtype']}")
    if primary_output["dtype"] != "float32":
        raise AssertionError(f"depth output dtype must be float32, got {primary_output['dtype']}")
    if not (
        len(primary_input["shape"]) == 4
        and primary_input["shape"][0] == 1
        and primary_input["shape"][-1] == 3
    ):
        raise AssertionError(f"depth input must be NHWC [1,H,W,3], got {primary_input['shape']}")
    layout = depth_layout(primary_output["shape"])
    if layout == "unknown":
        raise AssertionError(f"depth output must be a dense map, got {primary_output['shape']}")

    return {
        "model": str(model_path.resolve()),
        "size_bytes": model_path.stat().st_size,
        "backend": backend,
        "inputs": [tensor_summary(tensor) for tensor in input_details],
        "outputs": [tensor_summary(tensor) for tensor in output_details],
        "primary_output_layout": layout,
        "assertions": "passed",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a candidate monocular depth TFLite model.")
    parser.add_argument("model", nargs="?", default=DEFAULT_MODEL)
    parser.add_argument("--json-output")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    model_path = resolve_model_path(project_root, args.model)
    result = inspect_depth_model(model_path)

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
        write_json(output_path, result)


if __name__ == "__main__":
    main()
