"""Frozen DepthART metric-depth Teacher used by the AG-ST label factory."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

from download_b0_arkitscenes_assets import require


class DepthArtMetricTeacher:
    """Small inference-only wrapper around the locally frozen DepthART-S model."""

    def __init__(self, source: Path, checkpoint: Path, device: str = "cuda") -> None:
        import torch

        require(source.is_dir(), f"DepthART source missing: {source}")
        require(checkpoint.is_file(), f"DepthART checkpoint missing: {checkpoint}")
        require(device.startswith("cuda") and torch.cuda.is_available(), "DepthART Teacher requires CUDA")

        deployment = Path(__file__).resolve().parents[1] / "hftf" / "deployment" / "depthart"
        sys.path.insert(0, str(deployment))
        from export_depthart_camera_external import install_timm_compat

        install_timm_compat()
        metric = source / "metric"
        common_spec = importlib.util.spec_from_file_location(
            "blindassist_ag_st_depthart_common", metric / "common.py"
        )
        require(common_spec is not None and common_spec.loader is not None, "DepthART common loader missing")
        common = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common)
        model_spec = importlib.util.spec_from_file_location(
            "blindassist_ag_st_depthart_model", metric / "model.py"
        )
        require(model_spec is not None and model_spec.loader is not None, "DepthART model loader missing")
        model_module = importlib.util.module_from_spec(model_spec)
        sys.path.insert(0, str(metric))
        try:
            model_spec.loader.exec_module(model_module)
        finally:
            sys.path.pop(0)

        self._torch = torch
        self._make_K = common.make_K
        self._preprocess = common.preprocess
        self._device = device
        self._model = model_module.load_model(checkpoint, "S", "indoor", device).eval()
        self._model.requires_grad_(False)

    def infer(self, rgb_u8: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
        """Return native-raster metric depth without reading source depth."""

        rgb = np.asarray(rgb_u8, dtype=np.uint8)
        matrix = np.asarray(intrinsics, dtype=np.float32)
        require(rgb.ndim == 3 and rgb.shape[2] == 3, "DepthART RGB shape invalid")
        require(matrix.shape == (3, 3), "DepthART intrinsics shape invalid")
        height, width = rgb.shape[:2]
        # The official helper consumes BGR because its CLI reads through OpenCV.
        bgr = rgb[..., ::-1].copy()
        tensor, camera = self._preprocess(
            bgr,
            self._make_K(matrix[0, 0], matrix[1, 1], matrix[0, 2], matrix[1, 2]),
            width,
            height,
        )
        with self._torch.inference_mode():
            prediction = self._model(tensor.to(self._device), camera.to(self._device))
        depth = prediction[0].detach().float().cpu().numpy().astype(np.float32)
        require(depth.shape == (height, width), "DepthART native output shape drift")
        require(np.isfinite(depth).all() and np.all(depth > 0), "DepthART output invalid")
        return depth
