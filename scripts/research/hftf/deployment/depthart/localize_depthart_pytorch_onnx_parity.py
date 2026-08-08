#!/usr/bin/env python3
"""Localize DepthART PyTorch/ONNX drift on the frozen G4-D canary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import torch

try:
    from .export_depthart_camera_external import ExternalCameraMetric, install_timm_compat
except ImportError:
    from export_depthart_camera_external import ExternalCameraMetric, install_timm_compat


SCHEMA = "blindassist_depthart_pytorch_onnx_stage_localization_v1"
RTOL = 3e-5
ATOL = 3e-6
FROZEN_ORT_VERSION = "1.27.0"

ANCHORS = [
    ("patch_conv1", "/patch_embed/patch_embed.0/c/Conv_output_0"),
    ("patch_bn1", "/patch_embed/patch_embed.0/bn/BatchNormalization_output_0"),
    ("patch_gelu1", "/patch_embed/patch_embed.1/Mul_1_output_0"),
    ("patch_conv2", "/patch_embed/patch_embed.2/c/Conv_output_0"),
    ("patch_bn2", "/patch_embed/patch_embed.2/bn/BatchNormalization_output_0"),
    ("patch_embed", "/patch_embed/patch_embed.3/Mul_1_output_0"),
    ("daa1_cam_dw", "/daa1/proj_cam_dw/Conv_output_0"),
    ("daa1_cam_pw", "/daa1/proj_cam_pw/Conv_output_0"),
    ("daa1_norm_x", "/daa1/cross_attention/norm_attnx/LayerNormalization_output_0"),
    ("daa1_norm_ctx", "/daa1/cross_attention/norm_attnctx/LayerNormalization_output_0"),
    ("daa1_kv", "/daa1/cross_attention/kv/Add_output_0"),
    ("daa1_q", "/daa1/cross_attention/q/Add_output_0"),
    ("daa1_out", "/daa1/cross_attention/out/Add_output_0"),
    ("daa1_ls1", "/daa1/cross_attention/ls1/Mul_output_0"),
    ("daa1_mlp_norm", "/daa1/cross_attention/mlp/norm/LayerNormalization_output_0"),
    ("daa1_mlp_proj1", "/daa1/cross_attention/mlp/proj1/Add_output_0"),
    ("daa1_mlp_act", "/daa1/cross_attention/mlp/act/Mul_1_output_0"),
    ("daa1_mlp_proj2", "/daa1/cross_attention/mlp/proj2/Add_output_0"),
    ("daa1_ls2", "/daa1/cross_attention/ls2/Mul_output_0"),
    ("daa1_attention", "/daa1/cross_attention/Add_1_output_0"),
    ("daa1", "/daa1/Reshape_3_output_0"),
    ("stage1", "/norm0/BatchNormalization_output_0"),
    ("daa2", "/daa2/Reshape_3_output_0"),
    ("stage2", "/norm2/BatchNormalization_output_0"),
    ("daa3", "/daa3/Reshape_3_output_0"),
    ("stage3", "/norm4/BatchNormalization_output_0"),
    ("daa4", "/daa4/Reshape_3_output_0"),
    ("stage4", "/norm6/BatchNormalization_output_0"),
    ("depth_head", "/depth_head/output_conv2/output_conv2.3/Sigmoid_output_0"),
    ("scale_head", "/sfh/ReduceMean_output_0"),
    ("depth", "depth"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    if a.shape != b.shape or a.size == 0:
        raise ValueError(f"invalid comparison shapes: {a.shape} vs {b.shape}")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("non-finite comparison input")
    difference = np.abs(a - b)
    return {
        "elements": int(a.size),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(a - b)))),
        "bit_exact": bool(np.array_equal(a, b)),
        "allclose": bool(np.allclose(a, b, rtol=RTOL, atol=ATOL)),
    }


def load_canary(canary_dir: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    receipt = json.loads((canary_dir / "canary-receipt.json").read_text(encoding="utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "image",
        "camera_prompt_4",
        "camera_prompt_8",
        "camera_prompt_16",
        "camera_prompt_32",
        "depth",
    ):
        identity = receipt["files"][name]
        path = canary_dir / identity["path"]
        if sha256(path) != identity["sha256"]:
            raise ValueError(f"canary hash mismatch: {name}")
        arrays[name] = np.fromfile(path, dtype=np.float32).reshape(identity["shape"])
    return receipt, arrays


def add_probe_outputs(source: Path, destination: Path) -> None:
    model = onnx.load(source)
    known = {
        value.name: value
        for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
    }
    missing = [tensor for _, tensor in ANCHORS if tensor not in known]
    if missing:
        model = onnx.shape_inference.infer_shapes(model)
        known = {
            value.name: value
            for value in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)
        }
    existing = {value.name for value in model.graph.output}
    for _, tensor in ANCHORS:
        if tensor in existing:
            continue
        if tensor not in known:
            raise ValueError(f"ONNX anchor not found: {tensor}")
        output = model.graph.output.add()
        output.CopyFrom(known[tensor])
    onnx.checker.check_model(model)
    onnx.save(model, destination)


def capture_model(
    model: torch.nn.Module,
    wrapper: torch.nn.Module,
    tensors: list[torch.Tensor],
) -> dict[str, np.ndarray]:
    captured: dict[str, np.ndarray] = {}
    modules = {
        "patch_conv1": model.pretrained.patch_embed[0].c,
        "patch_bn1": model.pretrained.patch_embed[0].bn,
        "patch_gelu1": model.pretrained.patch_embed[1],
        "patch_conv2": model.pretrained.patch_embed[2].c,
        "patch_bn2": model.pretrained.patch_embed[2].bn,
        "patch_embed": model.pretrained.patch_embed,
        "daa1_cam_dw": model.daa1.proj_cam_dw,
        "daa1_cam_pw": model.daa1.proj_cam_pw,
        "daa1_norm_x": model.daa1.cross_attention.norm_attnx,
        "daa1_norm_ctx": model.daa1.cross_attention.norm_attnctx,
        "daa1_kv": model.daa1.cross_attention.kv,
        "daa1_q": model.daa1.cross_attention.q,
        "daa1_out": model.daa1.cross_attention.out,
        "daa1_ls1": model.daa1.cross_attention.ls1,
        "daa1_mlp_norm": model.daa1.cross_attention.mlp.norm,
        "daa1_mlp_proj1": model.daa1.cross_attention.mlp.proj1,
        "daa1_mlp_act": model.daa1.cross_attention.mlp.act,
        "daa1_mlp_proj2": model.daa1.cross_attention.mlp.proj2,
        "daa1_ls2": model.daa1.cross_attention.ls2,
        "daa1_attention": model.daa1.cross_attention,
        "daa1": model.daa1,
        "stage1": model.pretrained.norm0,
        "daa2": model.daa2,
        "stage2": model.pretrained.norm2,
        "daa3": model.daa3,
        "stage3": model.pretrained.norm4,
        "daa4": model.daa4,
        "stage4": model.pretrained.norm6,
        "depth_head": model.depth_head,
        "scale_head": model.sfh,
    }
    handles = []
    for name, module in modules.items():
        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, key: str = name) -> None:
            if not isinstance(output, torch.Tensor):
                raise TypeError(f"non-tensor anchor output: {key}")
            captured[key] = output.detach().float().cpu().numpy()

        handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode():
            output = wrapper(*tensors)
        captured["depth"] = output.detach().float().cpu().numpy()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def first_failure(comparisons: dict[str, dict[str, Any]]) -> str | None:
    return next((name for name, _ in ANCHORS if not comparisons[name]["allclose"]), None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--cuda-bin", type=Path, required=True)
    parser.add_argument("--primitive-onnx", type=Path, required=True)
    parser.add_argument("--canary-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if ort.__version__ != FROZEN_ORT_VERSION:
        raise RuntimeError(f"frozen ORT {FROZEN_ORT_VERSION} required, got {ort.__version__}")
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    extension_dir = args.extension_dir.resolve()
    cuda_bin = args.cuda_bin.resolve()
    primitive_onnx = args.primitive_onnx.resolve()
    canary_dir = args.canary_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is required before opening an evidence version; "
            "the frozen PyTorch oracle was generated on CUDA"
        )
    # The canonical parity path is correctness-first. cuDNN enables TF32 for
    # convolutions by default on recent PyTorch builds, while ORT's CPU Conv
    # evaluates the exported FP32 graph without TF32 input truncation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    canary_receipt, arrays = load_canary(canary_dir)

    install_timm_compat()
    extension_files = list(extension_dir.glob("depthart_selective_scan_cuda*.pyd"))
    if len(extension_files) != 1:
        raise ValueError(f"expected one SelectiveScan CUDA extension in {extension_dir}")
    dll_handles = []
    if sys.platform == "win32":
        for directory in (Path(torch.__file__).parent / "lib", cuda_bin, extension_dir):
            dll_handles.append(os.add_dll_directory(str(directory)))
    sys.path.insert(0, str(extension_dir))
    sys.path.insert(0, str(source / "metric"))
    sys.path.insert(0, str(source / "deploy" / "shared"))
    sys.path.insert(0, str(source / "deploy" / "shared" / "selective_scan"))
    from depthart_selective_scan import install_depthart  # type: ignore
    from export_helpers import install_exportable_sdpa  # type: ignore
    from model import load_model  # type: ignore
    from network import tvimblock  # type: ignore

    model = load_model(checkpoint, "S", "indoor", "cuda").eval()
    install_depthart(tvimblock)
    wrapper = ExternalCameraMetric(model).cuda().eval()
    input_names = ("image", "camera_prompt_4", "camera_prompt_8", "camera_prompt_16", "camera_prompt_32")
    torch_inputs = [torch.from_numpy(arrays[name]).cuda() for name in input_names]
    native = capture_model(model, wrapper, torch_inputs)
    saved_oracle = compare(native["depth"], arrays["depth"])

    install_exportable_sdpa()
    export_replay = capture_model(model, wrapper, torch_inputs)

    output_dir.mkdir(parents=True, exist_ok=False)
    probe_onnx = output_dir / "primitive-stage-probes.onnx"
    add_probe_outputs(primitive_onnx, probe_onnx)
    session = ort.InferenceSession(str(probe_onnx), providers=["CPUExecutionProvider"])
    requested = [tensor for _, tensor in ANCHORS]
    ort_values = session.run(requested, {name: arrays[name] for name in input_names})
    onnx_outputs = {
        name: np.asarray(value, dtype=np.float32)
        for (name, _), value in zip(ANCHORS, ort_values)
    }

    native_vs_export = {
        name: compare(native[name], export_replay[name]) for name, _ in ANCHORS
    }
    export_vs_onnx = {
        name: compare(export_replay[name], onnx_outputs[name]) for name, _ in ANCHORS
    }
    native_vs_onnx = {
        name: compare(native[name], onnx_outputs[name]) for name, _ in ANCHORS
    }
    for family, values in (
        ("native", native),
        ("export_replay", export_replay),
        ("onnx", onnx_outputs),
    ):
        family_dir = output_dir / family
        family_dir.mkdir()
        for name, _ in ANCHORS:
            np.ascontiguousarray(values[name], dtype=np.float32).tofile(family_dir / f"{name}.raw")

    receipt = {
        "schema": SCHEMA,
        "status": "LOCALIZED" if first_failure(native_vs_onnx) is not None else "PASS",
        "authority": "SYNTHETIC_PYTORCH_ONNX_NUMERICAL_DIAGNOSTIC_ONLY",
        "tolerance": {"rtol": RTOL, "atol": ATOL},
        "runtime": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "numpy": np.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "provider": "CPUExecutionProvider",
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
        },
        "assets": {
            "checkpoint_sha256": sha256(checkpoint),
            "selective_scan_cuda_extension_sha256": sha256(extension_files[0]),
            "primitive_onnx_sha256": sha256(primitive_onnx),
            "probe_onnx_sha256": sha256(probe_onnx),
            "canary_generator": canary_receipt["generator"],
            "image_sha256": canary_receipt["files"]["image"]["sha256"],
        },
        "saved_native_oracle": saved_oracle,
        "native_vs_export_replay": native_vs_export,
        "export_replay_vs_onnx": export_vs_onnx,
        "native_vs_onnx": native_vs_onnx,
        "first_native_vs_export_failure": first_failure(native_vs_export),
        "first_export_replay_vs_onnx_failure": first_failure(export_vs_onnx),
        "first_native_vs_onnx_failure": first_failure(native_vs_onnx),
        "explicit_exclusions": [
            "HTP_PARITY",
            "G4_E_PARTITION_PURITY",
            "G4_F_PERFORMANCE",
            "DA2_REPLACEMENT",
        ],
    }
    receipt_path = output_dir / "pytorch-onnx-stage-localization-receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
