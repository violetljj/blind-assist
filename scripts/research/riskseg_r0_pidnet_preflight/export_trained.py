from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxsim import simplify

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    CLASS_ORDER,
    DeploymentWrapper,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    build_pidnet_s,
    load_trained_deployment_checkpoint,
    official_repo_commit,
    set_deterministic_seed,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    official_repo = args.official_repo.resolve()
    checkpoint = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    seed = int(payload.get("seed", -1)) if isinstance(payload, dict) else -1
    if seed < 0:
        raise ValueError("training checkpoint does not contain a valid seed")
    set_deterministic_seed(seed)

    model = build_pidnet_s(official_repo=official_repo, augment=False)
    load_report = load_trained_deployment_checkpoint(
        model=model,
        checkpoint_path=checkpoint,
    )
    wrapper = DeploymentWrapper(model).eval()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = wrapper(example).detach().cpu().numpy()
    expected_shape = (1, len(CLASS_ORDER), INPUT_HEIGHT, INPUT_WIDTH)
    if torch_output.shape != expected_shape:
        raise ValueError(f"unexpected PyTorch output shape: {torch_output.shape}")
    if not np.isfinite(torch_output).all():
        raise ValueError("trained PyTorch output contains non-finite values")

    stem = f"pidnet_s_512x288_4class_seed_{seed}"
    onnx_path = output_dir / f"{stem}.onnx"
    torch.onnx.export(
        wrapper,
        (example,),
        onnx_path,
        input_names=["input_rgb_normalized"],
        output_names=["logits"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model, full_check=True)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_output = session.run(
        ["logits"],
        {"input_rgb_normalized": example.numpy()},
    )[0]
    if ort_output.shape != torch_output.shape:
        raise ValueError(
            f"ONNX output shape {ort_output.shape} != PyTorch {torch_output.shape}"
        )
    if not np.isfinite(ort_output).all():
        raise ValueError("trained ONNX Runtime output contains non-finite values")
    max_abs_error = float(np.max(np.abs(ort_output - torch_output)))
    if max_abs_error > 1e-4:
        raise ValueError(f"ONNX parity max abs error too large: {max_abs_error}")

    simplified_path = output_dir / f"{stem}_simplified.onnx"
    simplified_model, simplifier_check = simplify(onnx_model)
    if not simplifier_check:
        raise ValueError("onnxsim structural equivalence check failed")
    onnx.checker.check_model(simplified_model, full_check=True)
    onnx.save(simplified_model, simplified_path)
    simplified_session = ort.InferenceSession(
        str(simplified_path),
        providers=["CPUExecutionProvider"],
    )
    simplified_output = simplified_session.run(
        ["logits"],
        {"input_rgb_normalized": example.numpy()},
    )[0]
    simplified_max_abs_error = float(
        np.max(np.abs(simplified_output - torch_output))
    )
    if simplified_max_abs_error > 1e-4:
        raise ValueError(
            "simplified ONNX parity max abs error too large: "
            f"{simplified_max_abs_error}"
        )

    receipt = {
        "schema_version": "blindassist.riskseg_r0.pidnet_trained_export.v1",
        "protocol_id": "RISKSEG-R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "official_repo_commit": official_repo_commit(official_repo),
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "load_report": load_report,
        "input": {
            "name": "input_rgb_normalized",
            "shape_nchw": [1, 3, INPUT_HEIGHT, INPUT_WIDTH],
            "dtype": "float32_before_quantization",
            "color_order": "RGB",
            "normalization": "(rgb/255 - ImageNet mean) / ImageNet std",
        },
        "output": {
            "name": "logits",
            "shape_nchw": [1, len(CLASS_ORDER), INPUT_HEIGHT, INPUT_WIDTH],
            "dtype": "float32_before_quantization",
            "class_order": list(CLASS_ORDER),
            "upsample": "bilinear_align_corners_false",
        },
        "onnx_raw": {
            "path": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "opset": 17,
            "checker_full_check": True,
            "onnxruntime_provider": "CPUExecutionProvider",
            "torch_onnx_max_abs_error": max_abs_error,
        },
        "onnx_simplified": {
            "path": simplified_path.name,
            "sha256": sha256_file(simplified_path),
            "opset": 17,
            "onnxsim_check": True,
            "checker_full_check": True,
            "onnxruntime_provider": "CPUExecutionProvider",
            "torch_onnx_max_abs_error": simplified_max_abs_error,
            "deployment_role": "full_integer_tflite_conversion_input",
        },
        "torch_version": torch.__version__,
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
    }
    receipt_path = output_dir / "export_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
