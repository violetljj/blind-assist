"""Official UniDepth V2 / Metric3D v2 / MoGe-2 inference, with local-only weights.

Models are returned on CPU. The caller owns CUDA placement, timing and release.
No evaluator labels are read here; public_K is the input image's public camera K.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


def _image_and_k(rgb, public_K):
    image = np.asarray(rgb.convert("RGB"), dtype=np.uint8).copy()
    k = np.asarray(public_K, dtype=np.float32).copy()
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError("public_K must be finite 3x3 K with positive focal lengths")
    return image, k


def _native(depth, height, width):
    value = depth.detach().float().cpu().numpy().squeeze()
    if value.shape != (height, width):
        raise ValueError(f"Expected native depth {(height, width)}, got {value.shape}")
    return np.ascontiguousarray(value, dtype=np.float32)


def load_model(name, source_dir, weight_dir):
    """Return CPU model, infer(PIL RGB, K)->native optical-Z meters, and metadata.

    source_dir is the corresponding official repository checkout. weight_dir is
    UniDepth's directory containing config.json/model.safetensors, or Metric3D's
    directory containing metric_depth_vit_large_800k.pth (or the checkpoint file).
    MoGe-2 expects its metric checkpoint model.pt (or the checkpoint file).
    All checkpoints must already exist: this function never downloads a model.
    CUDA inference uses FP32 parameters and autocast FP16; CPU uses FP32.
    """
    import torch

    source = Path(source_dir).resolve()
    weights = Path(weight_dir).resolve()
    sys.path.insert(0, str(source))
    normalized = name.lower().replace("-", "_")
    if normalized in ("unidepth", "unidepth_v2", "unidepth_v2_vitl"):
        from safetensors.torch import load_file
        from unidepth.models import UniDepthV2

        config = json.loads((weights / "config.json").read_text(encoding="utf-8"))
        # Full pretrained checkpoint includes the encoder: forbid a second fetch.
        config["model"]["pixel_encoder"]["pretrained"] = None
        model = UniDepthV2(config)
        info = model.load_state_dict(load_file(str(weights / "model.safetensors")), strict=True)
        model.float().eval()

        def infer(rgb, public_K):
            image, k = _image_and_k(rgb, public_K)
            device = next(model.parameters()).device
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).to(device)
            camera = torch.from_numpy(k).to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                prediction = model.infer(tensor, camera, normalize=True)
            # Official depth is points[:, -1:], while radius is Euclidean range.
            return _native(prediction["depth"], image.shape[0], image.shape[1])

        meta = {
            "model": "UniDepthV2 ViT-L/14", "public_intrinsics": True,
            "output": "optical_z_meters", "preprocess": "official infer defaults",
            "shape_constraints": model.shape_constraints,
            "resolution_level": "unset (official full default bounds)",
            "interpolation_mode": model.interpolation_mode,
            "checkpoint_missing_keys": list(info.missing_keys),
            "checkpoint_unexpected_keys": list(info.unexpected_keys),
        }
    elif normalized in ("metric3d", "metric3d_v2", "metric3d_v2_vitl"):
        import cv2
        import torch.nn.functional as F

        spec = importlib.util.spec_from_file_location("rgb_metric3d_hubconf", source / "hubconf.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model = module.metric3d_vit_large(pretrain=False)
        checkpoint = weights if weights.is_file() else weights / "metric_depth_vit_large_800k.pth"
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        info = model.load_state_dict(state["model_state_dict"], strict=False)
        # Official training initialization removes mask_token; the full depth
        # checkpoint consequently omits it. DensePredModel calls encoder(input)
        # with masks=None, so prepare_tokens_with_masks never reads that token.
        # Preserve this one official inference-only exception, record it below,
        # and reject every other absent model parameter.
        allowed_missing = {"depth_model.encoder.mask_token"}
        essential_missing = set(info.missing_keys) - allowed_missing
        if essential_missing:
            raise RuntimeError(f"Metric3D checkpoint has missing model keys: {sorted(essential_missing)}")
        del state
        model.float().eval()

        def infer(rgb, public_K):
            image, k = _image_and_k(rgb, public_K)
            height, width = image.shape[:2]
            target_h, target_w = 616, 1064
            # Exact official hubconf demo recipe: isotropic scale, rounded-down
            # image dimensions, ImageNet-mean pad, fx scaled by the same factor.
            scale = min(target_h / height, target_w / width)
            resized = cv2.resize(image, (int(width * scale), int(height * scale)),
                                 interpolation=cv2.INTER_LINEAR)
            rh, rw = resized.shape[:2]
            top, left = (target_h - rh) // 2, (target_w - rw) // 2
            bottom, right = target_h - rh - top, target_w - rw - left
            padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                        cv2.BORDER_CONSTANT, value=[123.675, 116.28, 103.53])
            device = next(model.parameters()).device
            tensor = torch.from_numpy(padded.transpose(2, 0, 1).copy()).float()
            mean = torch.tensor([123.675, 116.28, 103.53])[:, None, None]
            std = torch.tensor([58.395, 57.12, 57.375])[:, None, None]
            tensor = ((tensor - mean) / std)[None].to(device)
            with torch.inference_mode(), torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                depth, _, _ = model.inference({"input": tensor})
            depth = depth.squeeze()[top:target_h - bottom, left:target_w - right]
            depth = F.interpolate(depth[None, None].float(), (height, width),
                                  mode="bilinear", align_corners=False).squeeze()
            # Canonical focal is 1000 px. Do not multiply by depth-normalization
            # bounds: the official model.inference output already applies those.
            depth = (depth * (float(k[0, 0]) * scale / 1000.0)).clamp(0, 300)
            return _native(depth, height, width)

        meta = {
            "model": "Metric3D v2 ViT-L (RAFT 8 iterations)", "public_intrinsics": True,
            "output": "optical_z_meters", "preprocess": "official hubconf demo",
            "input_size_hw": [616, 1064], "canonical_focal_px": 1000.0,
            "postprocess": "unpad -> bilinear native resize -> fx*scale/1000 -> clamp[0,300]",
            "xformers": "official torch attention fallback when unavailable",
            "checkpoint_missing_keys": list(info.missing_keys),
            "checkpoint_unexpected_keys": list(info.unexpected_keys),
            "checkpoint_allowed_missing_keys": sorted(allowed_missing),
            "checkpoint_missing_key_reason": "official training removes unused mask_token; inference masks=None",
        }
    elif normalized in ("moge2", "moge_2", "moge_v2"):
        dependency_dir = source.parent.parent / "deps"
        if dependency_dir.is_dir():
            sys.path.insert(0, str(dependency_dir))
        from moge.model.v2 import MoGeModel

        checkpoint = weights if weights.is_file() else weights / "model.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = state["model_config"]
        if not config.get("scale_head"):
            raise ValueError("MoGe-2 checkpoint must include its learned metric scale_head")
        model = MoGeModel(**config)
        info = model.load_state_dict(state["model"], strict=True)
        del state
        model.float().eval()

        def infer(rgb, public_K):
            image, _ = _image_and_k(rgb, public_K)
            device = next(model.parameters()).device
            tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().to(device) / 255.0
            # Official default FOV recovery and token-resolution settings.
            # The metric scale is the learned head, never fitted to evaluator GT.
            with torch.inference_mode():
                prediction = model.infer(
                    tensor, apply_mask=True, force_projection=True, fov_x=None,
                    resolution_level=9, use_fp16=device.type == "cuda",
                )
            # Masked pixels stay +inf as in official infer; no fabricated depth.
            return _native(prediction["depth"], image.shape[0], image.shape[1])

        meta = {
            "model": "MoGe-2 ViT-L metric (normal variant)", "public_intrinsics": False,
            "output": "optical_z_meters", "preprocess": "official infer defaults",
            "fov_x": None, "resolution_level": 9,
            "num_tokens_range": model.num_tokens_range,
            "apply_mask": True, "force_projection": True,
            "invalid_depth": "positive infinity from official predicted mask",
            "metric_scale": "learned scale_head output exp; applied by official infer",
            "checkpoint_missing_keys": list(info.missing_keys),
            "checkpoint_unexpected_keys": list(info.unexpected_keys),
        }
    else:
        raise ValueError(f"Unknown local backbone adapter {name!r}")
    meta.update({
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "parameter_dtype": "float32", "cuda_autocast_dtype": "float16",
        "source_dir": str(source), "weight_dir": str(weights),
    })
    return model, infer, meta
