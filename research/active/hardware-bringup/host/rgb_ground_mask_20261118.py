"""Frozen RGB-only ground evidence for the 2026-11-18 paired development replay."""
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import interpolate
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


ROOT = Path(__file__).resolve().parents[4]
REVISION = "489d5cd81a0b59fab9b7ea758d3548ebe99677da"
MODEL = ROOT / "artifacts.local/models/nvidia--segformer-b0-finetuned-ade-512-512/snapshots" / REVISION
GROUND_CLASSES = ("floor", "road", "sidewalk")
MIN_PROBABILITY = 0.80


class GroundMask:
    """Consistent operating point for indoor and outdoor RGB. True marks a pixel to remove."""

    def __init__(self, device="cuda"):
        self.device = torch.device(device)
        self.processor = SegformerImageProcessor.from_pretrained(MODEL, local_files_only=True)
        self.model, status = SegformerForSemanticSegmentation.from_pretrained(
            MODEL, local_files_only=True, use_safetensors=True, output_loading_info=True
        )
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if status.get(key):
                raise RuntimeError(f"SegFormer checkpoint did not load cleanly: {key}={status[key]}")
        self.model.eval().to(self.device)
        self.class_ids = [self.model.config.label2id[name] for name in GROUND_CLASSES]

    @torch.inference_mode()
    def predict(self, rgb):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("Expected an RGB uint8 image with shape (height, width, 3)")
        pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"].to(self.device)
        logits = self.model(pixel_values=pixels).logits.float()
        probability = logits.softmax(dim=1)[:, self.class_ids].sum(dim=1, keepdim=True)
        full_size = interpolate(probability, size=rgb.shape[:2], mode="bilinear", align_corners=False)
        return (full_size[0, 0] >= MIN_PROBABILITY).cpu().numpy()
