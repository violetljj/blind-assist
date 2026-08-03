#!/usr/bin/env python3
"""Run one metric-depth source on explicit RGB target observations.

The manifest owns target selection through a per-frame torso ROI.  Keeping that
selection fixed lets the companion evaluator compare depth sources before a
detector or tracker is introduced.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import types
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil

MODEL_UNIDEPTH = "unidepth-v2-vits14"
MODEL_VDA = "video-depth-anything-metric-vits-stream"
MODEL_METRIC3D = "metric3d-v2-vits-onnx"
MODEL_METRIC3D_PYTORCH = "metric3d-v2-vits-pytorch"
MODEL_MOGE2_VITS_NORMAL = "moge-2-vits-normal"
MODEL_DAV2_METRIC_HYPERSIM_VITS = "depth-anything-v2-metric-hypersim-vits"
MODEL_CHOICES = (
    MODEL_UNIDEPTH,
    MODEL_VDA,
    MODEL_METRIC3D,
    MODEL_METRIC3D_PYTORCH,
    MODEL_MOGE2_VITS_NORMAL,
    MODEL_DAV2_METRIC_HYPERSIM_VITS,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    required = {
        "sequence_id",
        "frame_index",
        "timestamp_ns",
        "frame_path",
        "scenario",
        "camera_motion",
        "torso_roi_xyxy_px",
    }
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"line {line_number} missing fields: {missing}")
        truth_depth = row.get("truth_depth_m")
        truth_direction = row.get("truth_direction")
        if truth_depth is not None and float(truth_depth) <= 0:
            raise ValueError(
                f"line {line_number} truth_depth_m must be null or positive"
            )
        if truth_depth is None and truth_direction not in {
            "approach",
            "recede",
            "stable_or_lateral",
        }:
            raise ValueError(
                f"line {line_number} needs truth_depth_m or truth_direction"
            )
        roi = row["torso_roi_xyxy_px"]
        if not isinstance(roi, list) or len(roi) != 4:
            raise ValueError(
                f"line {line_number} torso_roi_xyxy_px must have four values"
            )
        frame_path = Path(str(row["frame_path"]))
        if not frame_path.is_absolute():
            frame_path = (path.parent / frame_path).resolve()
        row["frame_path"] = str(frame_path)
        rows.append(row)
    if not rows:
        raise ValueError("manifest contains no observations")
    rows.sort(
        key=lambda row: (
            str(row["sequence_id"]),
            int(row["timestamp_ns"]),
            int(row["frame_index"]),
        )
    )
    return rows


def validate_roi(
    roi: list[Any], image_shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x0, y0, x1, y1 = (round(float(value)) for value in roi)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"ROI {(x0, y0, x1, y1)} outside image {(width, height)}")
    return x0, y0, x1, y1


def robust_roi_median(
    depth: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[float | None, int, float]:
    x0, y0, x1, y1 = roi
    values = np.asarray(depth[y0:y1, x0:x1], dtype=np.float64).reshape(-1)
    valid = values[np.isfinite(values) & (values > 0)]
    valid_fraction = len(valid) / len(values) if len(values) else 0.0
    if not len(valid):
        return None, 0, valid_fraction
    if len(valid) >= 10:
        lower, upper = np.quantile(valid, [0.1, 0.9])
        trimmed = valid[(valid >= lower) & (valid <= upper)]
        if len(trimmed):
            valid = trimmed
    return float(np.median(valid)), len(valid), valid_fraction


def intrinsics_matrix(row: dict[str, Any]) -> np.ndarray:
    values = row.get("intrinsics_fx_fy_cx_cy")
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("intrinsics_fx_fy_cx_cy is required and must have four values")
    fx, fy, cx, cy = (float(value) for value in values)
    if not all(math.isfinite(value) for value in (fx, fy, cx, cy)):
        raise ValueError("intrinsics must be finite")
    if fx <= 0 or fy <= 0:
        raise ValueError("focal lengths must be positive")
    return np.asarray(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def horizontal_fov_degrees(row: dict[str, Any], width: int) -> float:
    intrinsics = intrinsics_matrix(row)
    return math.degrees(2.0 * math.atan(float(width) / (2.0 * float(intrinsics[0, 0]))))


class DepthSource(ABC):
    model_id: str

    def reset_sequence(self) -> None:
        """Reset temporal state before a new sequence."""

    @abstractmethod
    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        raise NotImplementedError


class UniDepthSource(DepthSource):
    model_id = MODEL_UNIDEPTH

    def __init__(
        self,
        repo: Path,
        model_name: str,
        resolution_level: int,
        device: str,
    ) -> None:
        sys.path.insert(0, str(repo))
        # UniDepth imports its optional training visualizer during package
        # initialization.  Inference never calls it, so avoid requiring a
        # working wandb/protobuf stack in this isolated runtime.
        wandb_stub = types.ModuleType("wandb")
        wandb_stub.log = lambda *args, **kwargs: None
        wandb_stub.Image = lambda value: value
        sys.modules["wandb"] = wandb_stub
        import torch
        from unidepth.models import UniDepthV2

        self.torch = torch
        self.device = torch.device(device)
        self.model = UniDepthV2.from_pretrained(model_name)
        self.model.resolution_level = resolution_level
        self.model = self.model.to(self.device).eval()

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        camera = None
        if row.get("intrinsics_fx_fy_cx_cy") is not None:
            camera = self.torch.from_numpy(intrinsics_matrix(row))
        prediction = self.model.infer(tensor, camera)
        depth = prediction["depth"].squeeze().detach().cpu().numpy()
        metadata: dict[str, Any] = {}
        confidence = prediction.get("confidence")
        if confidence is not None:
            metadata["confidence_map"] = confidence.squeeze().detach().cpu().numpy()
        return depth, metadata


class VideoDepthAnythingSource(DepthSource):
    model_id = MODEL_VDA

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        input_size: int,
        device: str,
    ) -> None:
        sys.path.insert(0, str(repo))
        import torch
        from video_depth_anything.video_depth_stream import VideoDepthAnything

        self.torch = torch
        self.device = device
        self.input_size = input_size
        self.model = VideoDepthAnything(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(device).eval()

    def reset_sequence(self) -> None:
        self.model.transform = None
        self.model.frame_id_list = []
        self.model.frame_cache_list = []
        self.model.id = -1

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        depth = self.model.infer_video_depth_one(
            rgb,
            input_size=self.input_size,
            device=self.device,
            fp32=self.device == "cpu",
        )
        return np.asarray(depth), {}


class Metric3DOnnxSource(DepthSource):
    model_id = MODEL_METRIC3D
    input_height = 616
    input_width = 1064

    def __init__(self, model_path: Path, provider: str) -> None:
        import onnxruntime as ort

        providers = (
            ["CPUExecutionProvider"]
            if provider == "cpu"
            else ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input = self.session.get_inputs()[0]
        self.dtype = np.float16 if self.input.type == "tensor(float16)" else np.float32

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = rgb.shape[:2]
        scale = min(
            self.input_height / height,
            self.input_width / width,
        )
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_height = self.input_height - resized_height
        pad_width = self.input_width - resized_width
        pad_top = pad_height // 2
        pad_left = pad_width // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_height - pad_top,
            pad_left,
            pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        tensor = np.ascontiguousarray(padded.transpose(2, 0, 1)[None], dtype=self.dtype)
        outputs = self.session.run(None, {self.input.name: tensor})
        canonical = np.squeeze(outputs[0]).astype(np.float32)
        canonical = canonical[
            pad_top : self.input_height - (pad_height - pad_top),
            pad_left : self.input_width - (pad_width - pad_left),
        ]
        canonical = cv2.resize(
            canonical, (width, height), interpolation=cv2.INTER_LINEAR
        )
        intrinsics = intrinsics_matrix(row)
        canonical_to_real = float(intrinsics[0, 0]) * scale / 1000.0
        return canonical * canonical_to_real, {
            "onnx_providers": self.session.get_providers(),
        }


class Metric3DPytorchSource(DepthSource):
    model_id = MODEL_METRIC3D_PYTORCH
    input_height = 616
    input_width = 1064

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        device: str,
        precision: str = "fp32",
    ) -> None:
        import torch

        self.torch = torch
        self.device = torch.device(device)
        self.precision = precision
        if precision not in {"fp32", "tf32", "fp16", "bf16"}:
            raise ValueError(f"unsupported Metric3D precision: {precision}")
        if self.device.type != "cuda" and precision != "fp32":
            raise ValueError("non-fp32 Metric3D precision requires CUDA")
        if precision == "tf32":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.model = torch.hub.load(
            str(repo),
            "metric3d_vit_small",
            source="local",
            pretrain=False,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state["model_state_dict"], strict=False)
        self.model = self.model.to(self.device).eval()
        self.mean = torch.tensor([123.675, 116.28, 103.53], device=self.device).float()[
            :, None, None
        ]
        self.std = torch.tensor([58.395, 57.12, 57.375], device=self.device).float()[
            :, None, None
        ]

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        height, width = rgb.shape[:2]
        scale = min(
            self.input_height / height,
            self.input_width / width,
        )
        resized_height = int(height * scale)
        resized_width = int(width * scale)
        resized = cv2.resize(
            rgb,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
        pad_height = self.input_height - resized_height
        pad_width = self.input_width - resized_width
        pad_top = pad_height // 2
        pad_left = pad_width // 2
        padded = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_height - pad_top,
            pad_left,
            pad_width - pad_left,
            cv2.BORDER_CONSTANT,
            value=[123.675, 116.28, 103.53],
        )
        tensor = (
            self.torch.from_numpy(np.ascontiguousarray(padded.transpose(2, 0, 1)))
            .float()
            .to(self.device)
        )
        tensor = ((tensor - self.mean) / self.std)[None]
        autocast_dtype = {
            "fp16": self.torch.float16,
            "bf16": self.torch.bfloat16,
        }.get(self.precision)
        with (
            self.torch.inference_mode(),
            self.torch.autocast(
                device_type=self.device.type,
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None,
            ),
        ):
            prediction, _, _ = self.model.inference({"input": tensor})
        canonical = prediction.squeeze()
        canonical = canonical[
            pad_top : self.input_height - (pad_height - pad_top),
            pad_left : self.input_width - (pad_width - pad_left),
        ]
        canonical = self.torch.nn.functional.interpolate(
            canonical[None, None],
            (height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze()
        intrinsics = intrinsics_matrix(row)
        canonical_to_real = float(intrinsics[0, 0]) * scale / 1000.0
        depth = canonical * canonical_to_real
        depth = self.torch.clamp(depth, 0, 300)
        return depth.detach().cpu().numpy(), {
            "runtime": "pytorch",
            "device": str(self.device),
            "precision": self.precision,
        }


class MoGe2Source(DepthSource):
    model_id = MODEL_MOGE2_VITS_NORMAL

    def __init__(
        self,
        repo: Path,
        utils3d_repo: Path,
        checkpoint: Path,
        device: str,
        resolution_level: int = 9,
    ) -> None:
        sys.path[:0] = [str(utils3d_repo), str(repo)]
        import torch
        from moge.model.v2 import MoGeModel

        self.torch = torch
        self.device = torch.device(device)
        self.resolution_level = resolution_level
        self.model = MoGeModel.from_pretrained(checkpoint)
        self.model = self.model.to(self.device).eval()

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        tensor = self.torch.from_numpy(rgb.copy()).permute(2, 0, 1)
        tensor = tensor.to(self.device, dtype=self.torch.float32) / 255.0
        prediction = self.model.infer(
            tensor,
            resolution_level=self.resolution_level,
            fov_x=horizontal_fov_degrees(row, rgb.shape[1]),
            use_fp16=self.device.type == "cuda",
        )
        depth = prediction["depth"].detach().cpu().numpy()
        mask = prediction.get("mask")
        valid_fraction = None
        if mask is not None:
            mask_array = mask.detach().cpu().numpy().astype(bool)
            depth = np.where(mask_array, depth, np.nan)
            valid_fraction = float(np.mean(mask_array))
        metadata: dict[str, Any] = {
            "runtime": "pytorch",
            "device": str(self.device),
            "precision": "fp16_autocast" if self.device.type == "cuda" else "fp32",
            "resolution_level": self.resolution_level,
            "published_intrinsics_fov_constraint": True,
        }
        if valid_fraction is not None:
            metadata["model_valid_fraction"] = valid_fraction
        return np.asarray(depth, dtype=np.float32), metadata


class DepthAnythingV2MetricSource(DepthSource):
    model_id = MODEL_DAV2_METRIC_HYPERSIM_VITS

    def __init__(
        self,
        repo: Path,
        checkpoint: Path,
        device: str,
        input_size: int = 518,
        precision: str = "fp32",
    ) -> None:
        sys.path.insert(0, str(repo / "metric_depth"))
        import torch
        from depth_anything_v2.dpt import DepthAnythingV2

        self.torch = torch
        self.device = torch.device(device)
        self.input_size = input_size
        self.precision = precision
        if precision not in {"fp32", "fp16"}:
            raise ValueError(
                f"unsupported Depth Anything V2 precision: {precision}"
            )
        if self.device.type != "cuda" and precision != "fp32":
            raise ValueError("DA V2 FP16 requires CUDA")
        self.model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
            max_depth=20.0,
        )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model = self.model.to(self.device).eval()

    def infer(
        self, rgb: np.ndarray, row: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # The official infer_image API accepts OpenCV BGR input.
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with self.torch.autocast(
            device_type=self.device.type,
            dtype=self.torch.float16,
            enabled=self.precision == "fp16",
        ):
            depth = self.model.infer_image(bgr, input_size=self.input_size)
        return np.asarray(depth, dtype=np.float32), {
            "runtime": "pytorch",
            "device": str(self.device),
            "precision": self.precision,
            "input_size": self.input_size,
            "training_domain": "hypersim_indoor",
            "max_depth_m": 20.0,
        }


def build_source(args: argparse.Namespace) -> DepthSource:
    if args.model == MODEL_UNIDEPTH:
        return UniDepthSource(
            args.unidepth_repo,
            args.unidepth_model_name,
            args.unidepth_resolution_level,
            args.device,
        )
    if args.model == MODEL_VDA:
        return VideoDepthAnythingSource(
            args.vda_repo,
            args.vda_checkpoint,
            args.vda_input_size,
            args.device,
        )
    if args.model == MODEL_METRIC3D:
        return Metric3DOnnxSource(args.metric3d_onnx, args.onnx_provider)
    if args.model == MODEL_MOGE2_VITS_NORMAL:
        return MoGe2Source(
            args.moge_repo,
            args.utils3d_repo,
            args.moge_checkpoint,
            args.device,
            args.moge_resolution_level,
        )
    if args.model == MODEL_DAV2_METRIC_HYPERSIM_VITS:
        return DepthAnythingV2MetricSource(
            args.depth_anything_repo,
            args.depth_anything_checkpoint,
            args.device,
            args.depth_anything_input_size,
            args.depth_anything_precision,
        )
    return Metric3DPytorchSource(
        args.metric3d_repo,
        args.metric3d_checkpoint,
        args.device,
    )


def produce(rows: list[dict[str, Any]], source: DepthSource) -> list[dict[str, Any]]:
    output = []
    previous_sequence = None
    process = psutil.Process()
    for row in rows:
        sequence = str(row["sequence_id"])
        if sequence != previous_sequence:
            source.reset_sequence()
            previous_sequence = sequence
        bgr = cv2.imread(str(row["frame_path"]), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cannot read frame: {row['frame_path']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        roi = validate_roi(row["torso_roi_xyxy_px"], rgb.shape)
        torch_runtime = getattr(source, "torch", None)
        if torch_runtime is not None and torch_runtime.cuda.is_available():
            torch_runtime.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        depth, metadata = source.infer(rgb, row)
        latency_ms = (time.perf_counter() - started) * 1000.0
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"depth shape {depth.shape} differs from RGB {rgb.shape[:2]}"
            )
        predicted, valid_pixels, valid_fraction = robust_roi_median(depth, roi)
        observation = {
            key: row[key]
            for key in (
                "sequence_id",
                "frame_index",
                "timestamp_ns",
                "scenario",
                "camera_motion",
            )
        }
        if row.get("truth_depth_m") is not None:
            observation["truth_depth_m"] = float(row["truth_depth_m"])
        if row.get("truth_direction") is not None:
            observation["truth_direction"] = str(row["truth_direction"])
        observation.update(
            {
                "model_id": source.model_id,
                "predicted_depth_m": predicted,
                "latency_ms": latency_ms,
                "torso_roi_xyxy_px": list(roi),
                "roi_valid_pixels_after_trim": valid_pixels,
                "roi_valid_fraction_before_trim": valid_fraction,
                "process_rss_mib": (process.memory_info().rss / (1024.0 * 1024.0)),
            }
        )
        if torch_runtime is not None and torch_runtime.cuda.is_available():
            observation["cuda_peak_allocated_mib"] = (
                torch_runtime.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            )
        confidence_map = metadata.pop("confidence_map", None)
        if confidence_map is not None:
            confidence, _, _ = robust_roi_median(confidence_map, roi)
            observation["roi_median_confidence"] = confidence
        observation.update(metadata)
        output.append(observation)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, nargs="+", required=True)
    parser.add_argument("--model", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--unidepth-repo", type=Path)
    parser.add_argument(
        "--unidepth-model-name",
        default="lpiccinelli/unidepth-v2-vits14",
    )
    parser.add_argument("--unidepth-resolution-level", type=int, default=0)
    parser.add_argument("--vda-repo", type=Path)
    parser.add_argument("--vda-checkpoint", type=Path)
    parser.add_argument("--vda-input-size", type=int, default=392)
    parser.add_argument("--metric3d-onnx", type=Path)
    parser.add_argument("--metric3d-repo", type=Path)
    parser.add_argument("--metric3d-checkpoint", type=Path)
    parser.add_argument("--moge-repo", type=Path)
    parser.add_argument("--utils3d-repo", type=Path)
    parser.add_argument("--moge-checkpoint", type=Path)
    parser.add_argument("--moge-resolution-level", type=int, default=9)
    parser.add_argument("--depth-anything-repo", type=Path)
    parser.add_argument("--depth-anything-checkpoint", type=Path)
    parser.add_argument("--depth-anything-input-size", type=int, default=518)
    parser.add_argument(
        "--depth-anything-precision",
        choices=("fp32", "fp16"),
        default="fp32",
    )
    parser.add_argument("--onnx-provider", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.model == MODEL_UNIDEPTH and args.unidepth_repo is None:
        parser.error("--unidepth-repo is required for UniDepth")
    if args.model == MODEL_VDA and (
        args.vda_repo is None or args.vda_checkpoint is None
    ):
        parser.error("--vda-repo and --vda-checkpoint are required for VDA")
    if args.model == MODEL_METRIC3D and args.metric3d_onnx is None:
        parser.error("--metric3d-onnx is required for Metric3D")
    if args.model == MODEL_METRIC3D_PYTORCH and (
        args.metric3d_repo is None or args.metric3d_checkpoint is None
    ):
        parser.error(
            "--metric3d-repo and --metric3d-checkpoint are required "
            "for PyTorch Metric3D"
        )
    if args.model == MODEL_MOGE2_VITS_NORMAL and (
        args.moge_repo is None
        or args.utils3d_repo is None
        or args.moge_checkpoint is None
    ):
        parser.error(
            "--moge-repo, --utils3d-repo, and --moge-checkpoint are required for MoGe-2"
        )
    if args.model == MODEL_DAV2_METRIC_HYPERSIM_VITS and (
        args.depth_anything_repo is None or args.depth_anything_checkpoint is None
    ):
        parser.error(
            "--depth-anything-repo and --depth-anything-checkpoint are "
            "required for Depth Anything V2 Metric"
        )
    return args


def main() -> None:
    args = parse_args()
    rows = [row for manifest in args.manifest for row in load_manifest(manifest)]
    rows.sort(
        key=lambda row: (
            str(row["sequence_id"]),
            int(row["timestamp_ns"]),
            int(row["frame_index"]),
        )
    )
    output = produce(rows, build_source(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
