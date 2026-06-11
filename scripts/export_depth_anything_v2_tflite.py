from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_SRC = ".downloads/depth-lab/src/Depth-Anything-V2-main"
DEFAULT_CHECKPOINT = ".downloads/depth-lab/checkpoints/depth_anything_v2_vits.pth"
DEFAULT_OUTPUT = ".downloads/depth-lab/exports/depth_anything_v2_small_fp32.tflite"
DEFAULT_WORK_DIR = ".downloads/depth-lab/exports/work"


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path


def model_config(encoder: str) -> dict[str, Any]:
    configs: dict[str, dict[str, Any]] = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }
    return configs[encoder]


def export_onnx(
    src_root: Path,
    checkpoint: Path,
    onnx_path: Path,
    encoder: str,
    input_size: int,
    opset: int,
    layout: str,
) -> None:
    import torch
    import torch.nn as nn

    sys.path.insert(0, str(src_root))
    from depth_anything_v2.dpt import DepthAnythingV2  # type: ignore

    class NhwcDepthAnything(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 1, 1, 3))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 1, 1, 3))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            image = (image - self.mean) / self.std
            nchw = image.permute(0, 3, 1, 2).contiguous()
            depth = self.model(nchw)
            return depth.unsqueeze(-1)

    class NchwDepthAnything(nn.Module):
        def __init__(self, model: nn.Module) -> None:
            super().__init__()
            self.model = model
            self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1))

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            image = (image - self.mean) / self.std
            depth = self.model(image)
            return depth.unsqueeze(1)

    model = DepthAnythingV2(**model_config(encoder))
    state = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    wrapper: nn.Module
    if layout == "nhwc":
        wrapper = NhwcDepthAnything(model).eval()
        dummy = torch.rand(1, input_size, input_size, 3, dtype=torch.float32)
    elif layout == "nchw":
        wrapper = NchwDepthAnything(model).eval()
        dummy = torch.rand(1, 3, input_size, input_size, dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        str(onnx_path),
        input_names=["image"],
        output_names=["depth"],
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )


def convert_onnx_to_tflite(onnx_path: Path, tf_output_dir: Path, output_path: Path) -> Path:
    if tf_output_dir.exists():
        shutil.rmtree(tf_output_dir)
    tf_output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(tf_output_dir),
        "-n",
    ]
    subprocess.run(command, check=True)

    candidates = sorted(tf_output_dir.rglob("*.tflite"), key=lambda path: path.stat().st_size, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"onnx2tf did not produce a .tflite file under {tf_output_dir}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidates[0], output_path)
    return candidates[0]


def write_metadata(output_path: Path, payload: dict[str, Any]) -> None:
    metadata_path = output_path.with_suffix(output_path.suffix + ".json")
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Depth Anything V2 Small to an experimental NHWC float32 TFLite.")
    parser.add_argument("--src-root", default=DEFAULT_SRC)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--input-size", type=int, default=252)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--layout", choices=["nhwc", "nchw"], default="nhwc")
    parser.add_argument("--onnx-only", action="store_true")
    args = parser.parse_args()

    if args.input_size % 14 != 0:
        raise ValueError("--input-size must be divisible by 14 for Depth Anything V2 patch layout")

    project_root = Path(__file__).resolve().parents[1]
    src_root = resolve(project_root, args.src_root)
    checkpoint = resolve(project_root, args.checkpoint)
    output = resolve(project_root, args.output)
    work_dir = resolve(project_root, args.work_dir)
    onnx_path = work_dir / f"depth_anything_v2_{args.encoder}_{args.input_size}_{args.layout}.onnx"
    tf_output_dir = work_dir / f"onnx2tf_{args.layout}"

    if not src_root.is_dir():
        raise FileNotFoundError(f"Depth Anything V2 source root not found: {src_root}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Depth Anything V2 checkpoint not found: {checkpoint}")

    export_onnx(src_root, checkpoint, onnx_path, args.encoder, args.input_size, args.opset, args.layout)
    copied_from = None
    if not args.onnx_only:
        copied_from = convert_onnx_to_tflite(onnx_path, tf_output_dir, output)

    payload = {
        "source": "Depth Anything V2",
        "encoder": args.encoder,
        "input_size": args.input_size,
        "layout": args.layout,
        "opset": args.opset,
        "checkpoint": str(checkpoint.resolve()),
        "onnx": str(onnx_path.resolve()),
        "tflite": str(output.resolve()) if not args.onnx_only else None,
        "copied_from": str(copied_from.resolve()) if copied_from else None,
    }
    if not args.onnx_only:
        write_metadata(output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
