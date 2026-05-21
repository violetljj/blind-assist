from __future__ import annotations

import argparse
import os
from pathlib import Path


def configure_local_caches(project_root: Path) -> None:
    matplotlib_cache = project_root / ".cache" / "matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    matplotlib_cache.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Print TFLite input/output tensor details.")
    parser.add_argument("model", nargs="?", default="app/src/main/assets/yolo11n_fp16_320.tflite")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    configure_local_caches(project_root)

    import tensorflow as tf

    model_path = Path(args.model)
    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    print(f"model={model_path.resolve()}")
    for tensor in interpreter.get_input_details():
        print(f"input name={tensor['name']} shape={tensor['shape'].tolist()} dtype={tensor['dtype']}")
    for tensor in interpreter.get_output_details():
        print(f"output name={tensor['name']} shape={tensor['shape'].tolist()} dtype={tensor['dtype']}")


if __name__ == "__main__":
    main()
