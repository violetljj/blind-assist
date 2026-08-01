from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from scripts.research.riskseg_r0_pidnet_preflight.modeling import (
    CLASS_ORDER,
    DeploymentWrapper,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    build_pidnet_s,
    load_imagenet_backbone,
    official_repo_commit,
    set_deterministic_seed,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    official_repo = args.official_repo.resolve()
    pretrained = args.pretrained.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_deterministic_seed(args.seed)

    model = build_pidnet_s(official_repo=official_repo, augment=False)
    load_report = load_imagenet_backbone(
        model=model,
        checkpoint_path=pretrained,
    )
    wrapper = DeploymentWrapper(model).eval()
    example = torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH, dtype=torch.float32)
    with torch.inference_mode():
        torch_output = wrapper(example).detach().cpu().numpy()
    if torch_output.shape != (1, 4, INPUT_HEIGHT, INPUT_WIDTH):
        raise ValueError(f"unexpected PyTorch output shape: {torch_output.shape}")
    if not np.isfinite(torch_output).all():
        raise ValueError("PyTorch preflight output contains non-finite values")

    checkpoint_path = output_dir / "pidnet_s_preflight_init.pt"
    torch.save(
        {
            "schema_version": "blindassist.riskseg_r0.pidnet_preflight_init.v1",
            "seed": args.seed,
            "class_order": list(CLASS_ORDER),
            "state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    onnx_path = output_dir / "pidnet_s_512x288_4class_preflight.onnx"
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
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(
        ["logits"],
        {"input_rgb_normalized": example.numpy()},
    )[0]
    if ort_output.shape != torch_output.shape:
        raise ValueError(
            f"ONNX output shape {ort_output.shape} != PyTorch {torch_output.shape}"
        )
    if not np.isfinite(ort_output).all():
        raise ValueError("ONNX Runtime output contains non-finite values")
    max_abs_error = float(np.max(np.abs(ort_output - torch_output)))
    if max_abs_error > 1e-4:
        raise ValueError(f"ONNX parity max abs error too large: {max_abs_error}")

    license_path = official_repo / "LICENSE"
    model_source_path = official_repo / "models" / "pidnet.py"
    model_utils_path = official_repo / "models" / "model_utils.py"
    receipt = {
        "schema_version": "blindassist.riskseg_r0.pidnet_preflight_export.v1",
        "protocol_id": "RISKSEG_R0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "official_repo_commit": official_repo_commit(official_repo),
        "official_license": "MIT",
        "official_license_sha256": sha256_file(license_path),
        "official_model_source_sha256": sha256_file(model_source_path),
        "official_model_utils_sha256": sha256_file(model_utils_path),
        "pretrained_path": str(pretrained),
        "pretrained_sha256": sha256_file(pretrained),
        "pretrained_recovery_note": (
            "official README identifies PIDNet_S_ImageNet.pth.tar, but its original "
            "Google Drive link is unavailable; the byte artifact was recovered from "
            "Zenodo record 14606189 and matched its registered size and MD5"
        ),
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
            "shape_nchw": [1, 4, INPUT_HEIGHT, INPUT_WIDTH],
            "dtype": "float32_before_quantization",
            "class_order": list(CLASS_ORDER),
            "upsample": "bilinear_align_corners_false",
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
        },
        "onnx": {
            "path": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "opset": 17,
            "checker_full_check": True,
            "onnxruntime_provider": "CPUExecutionProvider",
            "torch_onnx_max_abs_error": max_abs_error,
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

