from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

from scripts.research.riskseg_r0_pidnet_preflight.modeling import sha256_file


ONNX2TF_TEST_SAMPLE_NAME = (
    "calibration_image_sample_data_20x128x128x3_float32.npy"
)
IMAGENET_MEAN_NHWC = "[[[[0.485,0.456,0.406]]]]"
IMAGENET_STD_NHWC = "[[[[0.229,0.224,0.225]]]]"


def _read_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export-receipt", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--calibration-array", type=Path, required=True)
    parser.add_argument("--onnx2tf-test-sample", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    export_receipt_path = args.export_receipt.resolve()
    calibration_receipt_path = args.calibration_receipt.resolve()
    calibration_array = args.calibration_array.resolve()
    test_sample = args.onnx2tf_test_sample.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    export_receipt = _read_object(export_receipt_path)
    if (
        export_receipt.get("schema_version")
        != "blindassist.riskseg_r0.pidnet_trained_export.v1"
    ):
        raise ValueError("unexpected trained export receipt schema")
    simplified = export_receipt.get("onnx_simplified")
    if not isinstance(simplified, dict):
        raise ValueError("export receipt has no simplified ONNX binding")
    onnx_path = (export_receipt_path.parent / str(simplified["path"])).resolve()
    if sha256_file(onnx_path) != simplified.get("sha256"):
        raise ValueError("simplified ONNX hash mismatch")
    onnx.checker.check_model(onnx.load(onnx_path), full_check=True)

    calibration_receipt = _read_object(calibration_receipt_path)
    if (
        calibration_receipt.get("schema_version")
        != "blindassist.riskseg_r0.pidnet_calibration.v1"
        or calibration_receipt.get("role")
        != "train_only_quantization_calibration"
    ):
        raise ValueError("unexpected calibration receipt contract")
    if sha256_file(calibration_array) != calibration_receipt.get("array_sha256"):
        raise ValueError("calibration array hash mismatch")
    calibration = np.load(calibration_array, allow_pickle=False)
    expected_shape = tuple(calibration_receipt.get("array_shape", ()))
    if calibration.shape != expected_shape or calibration.dtype != np.float32:
        raise ValueError(
            f"calibration array contract mismatch: {calibration.shape}/{calibration.dtype}"
        )
    if calibration.ndim != 4 or calibration.shape[1:] != (288, 512, 3):
        raise ValueError(f"unexpected calibration layout: {calibration.shape}")
    if float(calibration.min()) < 0.0 or float(calibration.max()) > 1.0:
        raise ValueError("calibration array must be NHWC RGB in [0,1]")

    if test_sample.name != ONNX2TF_TEST_SAMPLE_NAME:
        raise ValueError(
            f"onnx2tf test sample must be named {ONNX2TF_TEST_SAMPLE_NAME}"
        )
    test_values = np.load(test_sample, allow_pickle=False)
    if test_values.shape != (20, 128, 128, 3) or test_values.dtype != np.float32:
        raise ValueError(
            f"unexpected onnx2tf test sample: {test_values.shape}/{test_values.dtype}"
        )

    source_onnx_sha256 = sha256_file(onnx_path)
    with tempfile.TemporaryDirectory(
        prefix="riskseg-r0-onnx2tf-",
        dir=output_dir.parent,
    ) as temporary_dir_raw:
        temporary_dir = Path(temporary_dir_raw)
        converter_input = temporary_dir / onnx_path.name
        shutil.copy2(onnx_path, converter_input)
        command = [
            sys.executable,
            "-m",
            "onnx2tf",
            "-i",
            str(converter_input),
            "-o",
            str(output_dir),
            "-oiqt",
            "-cind",
            "input_rgb_normalized",
            str(calibration_array),
            IMAGENET_MEAN_NHWC,
            IMAGENET_STD_NHWC,
            "-iqd",
            "int8",
            "-oqd",
            "int8",
            "-nuo",
            "-v",
            "error",
        ]
        subprocess.run(command, cwd=test_sample.parent, check=True)
    if sha256_file(onnx_path) != source_onnx_sha256:
        raise ValueError("onnx2tf mutated the source ONNX despite isolated copy")
    candidates = sorted(output_dir.glob("*_full_integer_quant.tflite"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one full-integer TFLite, found {[p.name for p in candidates]}"
        )
    model_path = candidates[0]
    receipt = {
        "schema_version": "blindassist.riskseg_r0.pidnet_trained_int8.v1",
        "protocol_id": "RISKSEG-R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": int(export_receipt["seed"]),
        "export_receipt": {
            "path": str(export_receipt_path),
            "sha256": sha256_file(export_receipt_path),
        },
        "onnx_simplified": {
            "path": str(onnx_path),
            "sha256": source_onnx_sha256,
            "converter_input": "isolated byte copy because onnx2tf rewrites ONNX metadata",
        },
        "calibration_receipt": {
            "path": str(calibration_receipt_path),
            "sha256": sha256_file(calibration_receipt_path),
        },
        "calibration_array": {
            "path": str(calibration_array),
            "sha256": sha256_file(calibration_array),
            "shape": list(calibration.shape),
            "dtype": str(calibration.dtype),
        },
        "onnx2tf_test_sample": {
            "path": str(test_sample),
            "sha256": sha256_file(test_sample),
            "role": "converter_internal_parity_only_not_quantization_calibration",
        },
        "converter": {
            "package": "onnx2tf",
            "version": importlib.metadata.version("onnx2tf"),
            "quantization": "full_integer_W8A8",
            "input_dtype": "int8",
            "output_dtype": "int8",
            "quantization_granularity": "per_channel_default",
            "normalization_mean_nhwc": IMAGENET_MEAN_NHWC,
            "normalization_std_nhwc": IMAGENET_STD_NHWC,
        },
        "tflite": {
            "path": model_path.name,
            "sha256": sha256_file(model_path),
            "size_bytes": model_path.stat().st_size,
        },
    }
    receipt_path = output_dir / "conversion_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
